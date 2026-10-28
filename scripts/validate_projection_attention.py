"""Native BF16/FP8 QKV projection -> self-attention, one-launch chain validation."""
import argparse
import ctypes
import datetime
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile
import contextlib
import time
from types import SimpleNamespace

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from native_pi0.dense import ResidentDense
from native_pi0.projection_attention import ResidentProjectionAttention
from native_pi0.launch_trace import LaunchTrace
from scripts.validate_attention import reference


def nrmse(a,b):return float((a.float()-b.float()).square().mean().sqrt()/b.float().square().mean().sqrt().clamp_min(1e-12))


def diagnostics(chain):
    data=chain.read_diagnostics();n=chain._queue.nodes;epoch=chain._queue.epoch
    np.testing.assert_array_equal(data['counts'],[n,n,n,0,epoch])
    np.testing.assert_array_equal(data['values'],np.full(n,epoch,dtype=np.uint32))
    np.testing.assert_array_equal(data['visits'],np.full(n,epoch,dtype=np.uint64))
    assert int(data['exits'].sum())==n
    for node,parents in enumerate(chain._queue.parents):
        if parents:assert data['starts'][node]>=max(data['ends'][p] for p in parents)
    if chain.attention.splits>1:
        state=chain.attention.read_split_counts()
        assert state['epoch']==epoch and all(v==(epoch<<32)|chain.attention.splits for v in state['counts'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--quick',action='store_true');parser.add_argument('--profile',action='store_true')
    parser.add_argument('--report',type=Path);parser.add_argument('--build-directory',type=Path)
    parser.add_argument('--output-projection',action='store_true')
    parser.add_argument('--node-order',choices=('ordered','permuted'),default='ordered')
    args=parser.parse_args()
    if args.report and not args.profile:parser.error('Reports require launch-count gates')
    torch.set_num_threads(8);torch.manual_seed(701);rng=np.random.default_rng(701);records=[];dense_records=[]
    paths=['native_pi0/projection_attention.cu','native_pi0/projection_attention.py','native_pi0/dense.cu','native_pi0/dense.py',
        'native_pi0/tcgen_probe.cu','native_pi0/attention.cu','native_pi0/attention.py','native_pi0/dag_probe.cu','native_pi0/dag_probe.py',
        'native_pi0/dag_scheduler.cuh','native_pi0/launch_trace.cpp','native_pi0/launch_trace.py','scripts/validate_projection_attention.py','scripts/validate_attention.py',
        'third_party/Megakernels/include/megakernel.cuh','third_party/Megakernels/include/util.cuh','third_party/Megakernels/include/controller/controller.cuh',
        'third_party/Megakernels/include/controller/instruction_fetch.cuh']
    source_hashes={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths}
    if args.build_directory:args.build_directory.mkdir(parents=True,exist_ok=True)
    with (contextlib.nullcontext(args.build_directory) if args.build_directory else tempfile.TemporaryDirectory()) as directory:
        library=Path(directory).resolve()/'projection_attention.so'
        command=['nvcc','-O3','-lineinfo','-std=c++20','-arch=sm_100a','-DKITTENS_HOPPER','-DKITTENS_BLACKWELL','-DNATIVE_COMPUTE_CONSUMER_WARPS=4',
            '--expt-relaxed-constexpr','--expt-extended-lambda','-I'+str(ROOT/'third_party/Megakernels/include'),'-I'+str(ROOT/'third_party/Megakernels/ThunderKittens/include'),
            '-shared','-Xcompiler','-fPIC',str(ROOT/'native_pi0/projection_attention.cu'),str(ROOT/'native_pi0/launch_trace.cpp'),
            '-lcupti','-Xlinker','-rpath','-Xlinker','/usr/local/cuda/lib64','-lcuda','-o',str(library)]
        subprocess.run(command,check=True)
        for m,k,n in ((1,17,33),(51,1024,1024),(129,192,384)):
            for dtype in (torch.bfloat16,torch.float8_e4m3fn):
                for output_dtype in (torch.bfloat16,torch.float32):
                    x=torch.randn((m,k),device='cuda').to(dtype);w=(torch.randn((n,k),device='cuda')*.05).to(dtype)
                    with ResidentDense(library,x,w,alpha=1.2345,bias=torch.randn(n)*.02,residual=torch.randn(m,n)*.1,
                                       output_dtype=output_dtype,workers=4) as projection:
                        stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
                        with torch.cuda.stream(stream):projection.output.fill_(float('nan'));projection.forward()
                        stream.synchronize()
                        expected=projection.a.float().cpu()@projection.b.float().cpu().T*projection.alpha
                        expected+=projection.residual.float().cpu();expected+=projection.bias.float().cpu()
                        actual=projection.output.cpu();gold=expected.to(output_dtype)
                        error=max(nrmse(actual,gold),nrmse(actual[:m,:n],gold[:m,:n]))
                        assert error<=(.005 if output_dtype==torch.bfloat16 else 3e-5),error
                        entry={'m':m,'n':n,'k':k,'dtype':str(dtype),'output_dtype':str(output_dtype),'bias':True,'residual':True,'alpha':projection.alpha,'nrmse':error}
                        if args.profile:
                            with LaunchTrace(library) as trace:projection.forward();torch.cuda.synchronize()
                            assert trace.count==1,trace.count;entry['cuda_launch_count']=trace.count
                        dense_records.append(entry)
        print('Dense BF16/FP8 -> BF16/FP32 bias/residual: 12 padded/descales cases pass',flush=True)
        cases=[('small-bf16',1,3,192,2,1,64,torch.bfloat16,1,4,False,True),
               ('cross-batch-fp8',2,65,192,2,2,72,torch.float8_e4m3fn,1,16,True,False),
               ('gqa-split',1,51,1024,8,1,256,torch.float8_e4m3fn,8,148,True,True)]
        if not args.quick:cases += [('vision',3,256,1152,16,16,72,torch.float8_e4m3fn,1,148,False,False),
                                   ('prefix',1,768,2048,8,1,256,torch.float8_e4m3fn,1,148,True,False),
                                   ('prefix-bf16',1,129,2048,8,1,256,torch.bfloat16,2,16,True,True)]
        for label,b,rows,k,h,hk,d,dtype,splits,workers,rope,masked in cases:
            m=b*rows;n=(h+2*hk)*d
            x=torch.randn((m,k),device='cuda').to(dtype);w=(torch.randn((n,k),device='cuda')*.05).to(dtype)
            bias=torch.randn(n,device='cuda',dtype=torch.bfloat16)*.02
            angle=torch.randn((b,rows,d),device='cuda');tables=(angle.cos().bfloat16(),angle.sin().bfloat16()) if rope else None
            mask=None
            if masked:mask=torch.rand((b,rows,rows),device='cuda')>.15;mask[:,0]=False
            with contextlib.ExitStack() as stages:
                projection=stages.enter_context(ResidentDense(library,x,w,alpha=.125 if dtype==torch.float8_e4m3fn else 1.,bias=bias,workers=workers))
                output_projection=None
                if args.output_projection:
                    output_projection=stages.enter_context(ResidentDense(library,torch.zeros((m,h*d),device='cuda',dtype=torch.bfloat16),
                        (torch.randn((k,h*d),device='cuda')*.05).bfloat16(),alpha=.5,bias=torch.randn(k)*.02,residual=torch.randn(m,k)*.1,workers=workers))
                nodes=(projection.output.shape[0]//128)*(projection.output.shape[1]//128)+b*h*((rows+63)//64)*splits
                if output_projection is not None:nodes+=(output_projection.output.shape[0]//128)*(output_projection.output.shape[1]//128)
                order=rng.permutation(nodes)
                with ResidentProjectionAttention(library,projection,batches=b,rows=rows,heads=h,kv_heads=hk,head_dim=d,
                    q_rope=tables,k_rope=tables,mask=mask,workers=workers,splits=splits,node_order=order if args.node_order=='permuted' else None,output_projection=output_projection) as chain:
                    projection_error=whole_error=separate_error=0.;projection_exact=True;attention_exact=True;output_exact=True;output_projection_error=0.
                    for replay in range(3):
                        stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
                        with torch.cuda.stream(stream):
                            changed=torch.randn((m,k),device='cuda').to(dtype);projection.inputs.copy_(changed)
                            projection.output.fill_(float('nan'));chain.output.fill_(float('nan'))
                            if output_projection is not None:output_projection.inputs.fill_(float('nan'))
                            if splits>1:chain.attention.poison_partials()
                            chain.forward()
                        stream.synchronize();diagnostics(chain)
                        projected=projection.output.clone();output=chain.output.clone();attended=chain.attention.output.clone()
                        assert torch.isfinite(output).all()
                        gold=projection.a.float().cpu()@projection.b.float().cpu().T*projection.alpha
                        if projection.residual is not None:gold+=projection.residual.float().cpu()
                        if projection.bias is not None:gold+=projection.bias.float().cpu()
                        error=nrmse(projected.cpu(),gold.bfloat16());assert error<=.005,(label,error);projection_error=max(projection_error,error)
                        qkv=gold.bfloat16()[:m]
                        expected=SimpleNamespace(q=qkv[:,:h*d].view(b,rows,h,d),k=qkv[:,h*d:(h+hk)*d].view(b,rows,hk,d),
                            v=qkv[:,(h+hk)*d:(h+2*hk)*d].view(b,rows,hk,d),bias=chain.attention.bias,q_rope=chain.attention.q_rope,
                            k_rope=chain.attention.k_rope,mask=chain.attention.mask,prefix=None,scale=chain.attention.scale)
                        gold_attention=reference(expected)
                        def project_output(value):
                            o=output_projection;a=torch.zeros(o.a.shape,dtype=torch.float32)
                            a[:m,:h*d]=value.bfloat16().float().reshape(m,h*d)
                            result=a@o.b.float().cpu().T*o.alpha
                            if o.residual is not None:result+=o.residual.float().cpu()
                            if o.bias is not None:result+=o.bias.float().cpu()
                            return result.bfloat16()
                        whole_gold=gold_attention
                        if output_projection is not None:
                            stage_gold=project_output(attended.cpu())
                            error=nrmse(output_projection.output.cpu(),stage_gold);assert error<=.005,(label,error)
                            output_projection_error=max(output_projection_error,error)
                            whole_gold=project_output(gold_attention)[:m,:k]
                        error=nrmse(output.cpu(),whole_gold);assert error<=.01,(label,error);whole_error=max(whole_error,error)
                        projection.forward();chain.attention.forward()
                        if output_projection is not None:output_projection.forward()
                        torch.cuda.synchronize()
                        equal=torch.equal(projected.view(torch.int16),projection.output.view(torch.int16));projection_exact &= equal
                        assert equal,'Projection differs from separate native launch'
                        error=nrmse(output,chain.output);assert error<=1e-4,(label,error);separate_error=max(separate_error,error)
                        attention_exact &= torch.equal(attended.view(torch.int16),chain.attention.output.view(torch.int16))
                        output_exact &= torch.equal(output.view(torch.int16),chain.output.view(torch.int16))
                        separate_output=chain.output.clone()
                    record={'label':label,'batch':b,'rows':rows,'k':k,'heads':h,'kv_heads':hk,'head_dim':d,'dtype':str(dtype),'kv_splits':splits,
                        'workers':workers,'nodes':nodes,'plan':chain._queue.plan,'node_order':args.node_order,'replays':3,
                        'projection_nrmse':projection_error,'whole_chain_nrmse':whole_error,'separate_native_nrmse':separate_error,
                        'projection_bit_exact_separate':projection_exact,'attention_bit_exact_separate':attention_exact,
                        'output_projection':args.output_projection,'output_projection_nrmse':output_projection_error,'output_bit_exact_separate':output_exact,
                        'projection_sha256':hashlib.sha256(projected.contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest(),
                        'output_sha256':hashlib.sha256(output.contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest(),
                        'diagnostics':'Every node once; exact queue/epoch counts; parent-before-child timestamps; split counts exact'}
                    if args.profile:
                        with LaunchTrace(library) as trace:chain.forward();torch.cuda.synchronize()
                        assert trace.count==1 and 'ProjectionAttentionGlobals' in trace.symbol,(trace.count,trace.symbol)
                        diagnostics(chain);record.update(cuda_launch_count=trace.count,kernel_symbol=trace.symbol)
                        with LaunchTrace(library) as trace:
                            projection.forward();chain.attention.forward()
                            if output_projection is not None:output_projection.forward()
                            torch.cuda.synchronize()
                        assert trace.count==2+int(args.output_projection),trace.count;record['split_cuda_launch_count']=trace.count
                        with LaunchTrace(library) as trace:chain.forward();chain.forward();torch.cuda.synchronize()
                        assert trace.count==2,trace.count;record['two_forward_negative_control']=2;diagnostics(chain)
                    if args.report:
                        def separate():
                            projection.forward();chain.attention.forward()
                            if output_projection is not None:output_projection.forward()
                        def measure(call):
                            for _ in range(5):call()
                            torch.cuda.synchronize();start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);samples=[];wall=[]
                            for _ in range(30):
                                torch.cuda.synchronize();t=time.perf_counter_ns();start.record();call();end.record();end.synchronize()
                                samples.append(start.elapsed_time(end));wall.append((time.perf_counter_ns()-t)/1e6)
                            def stats(v):return {'median_ms':statistics.median(v),'p10_ms':float(np.percentile(v,10)),'p90_ms':float(np.percentile(v,90)),'samples_ms':v}
                            return {'gpu':stats(samples),'synchronized_wall':stats(wall)}
                        record['integrated']=measure(chain.forward);diagnostics(chain)
                        assert torch.equal(chain.output.view(torch.int16),output.view(torch.int16))
                        assert torch.equal(projection.output.view(torch.int16),projected.view(torch.int16))
                        record['separate']=measure(separate)
                        assert torch.equal(chain.output.view(torch.int16),separate_output.view(torch.int16))
                        record['post_timing_outputs']='bit-exact per-path replay'
                    print(label,'projection',projection_error,'whole',whole_error,'vs separate',separate_error,flush=True);records.append(record)
    if args.report:
        assert all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==value for p,value in source_hashes.items()),'Source changed during validation; rebuild required'
        report={'timestamp_utc':datetime.datetime.now(datetime.UTC).isoformat(),'seed':701,'scope':('Synthetic BF16/FP8 QKV -> attention -> BF16 output projection with bias/residual' if args.output_projection else 'Synthetic BF16/FP8 QKV -> attention')+'; no KV-cache generation, pretrained or complete model',
            'timed_boundary':'Resident inputs/weights/parameters to reused attention output; excludes setup/references/transfers; event intervals include host submission gaps',
            'build_command':command,'upstreams':json.loads((ROOT/'upstreams.json').read_text()),
            'source_sha256':source_hashes,'dense_results':dense_records,'results':records}
        args.report.parent.mkdir(parents=True,exist_ok=True);args.report.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
