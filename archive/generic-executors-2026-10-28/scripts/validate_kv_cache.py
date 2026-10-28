"""Validate fused rotated KV generation, immutable prefix reuse and suffix attention."""
import argparse
import contextlib
import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import statistics
import time

import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from native_pi0.attention import ResidentAttention
from native_pi0.dense import ResidentDense
from native_pi0.projection_attention import ResidentProjectionAttention
from native_pi0.launch_trace import LaunchTrace
from native_pi0.cached_prefix_expert import ResidentCachedPrefixExpert


def nrmse(a,b):return float((a.float()-b.float()).square().mean().sqrt()/b.float().square().mean().sqrt().clamp_min(1e-12))


def transform(x,bias,tables):
    value=x.float().cpu()
    if bias is not None:value+=bias.float().cpu()
    if tables:
        half=value.shape[-1]//2;lo,hi=value[...,:half].clone(),value[...,half:].clone()
        c,s=[t.float().cpu()[...,:half].unsqueeze(2) for t in tables]
        value=torch.cat((lo*c-hi*s,hi*c+lo*s),-1)
    return value.to(x.dtype)


def gold_suffix(op):
    q=transform(op.q,op.bias[0],op.q_rope).float()
    k=transform(op.k,op.bias[1],op.k_rope).float();v=transform(op.v,op.bias[2],[]).float()
    p=op.cached_prefix_rows
    k=torch.cat((op.kv_cache[0][:,:p].float().cpu(),k),1)
    v=torch.cat((op.kv_cache[1][:,:p].float().cpu(),v),1)
    group=q.shape[2]//k.shape[2];k=k.repeat_interleave(group,2);v=v.repeat_interleave(group,2)
    scores=(q.transpose(1,2)@k.transpose(1,2).transpose(-1,-2))*op.scale
    mask=torch.ones((q.shape[0],q.shape[1],k.shape[1]),dtype=torch.bool);mask[:,0,p+1:]=False
    if op.mask is not None:mask &= op.mask.cpu()
    probability=torch.softmax(scores.masked_fill(~mask[:,None],float('-inf')),dim=-1).nan_to_num(0)
    return (probability@v.transpose(1,2)).transpose(1,2)


def check_cache(op):
    p=op.cached_prefix_rows;count=op.raw_key_rows
    for actual,expected in zip(op.kv_cache,(transform(op.k,op.bias[1],op.k_rope),transform(op.v,op.bias[2],[]))):
        error=nrmse(actual[:,p:p+count].cpu(),expected)
        assert error<=1e-4,('KV cache transform',error)
        assert torch.isfinite(actual[:,p:p+count]).all()
    if op.splits>1:
        state=op.read_split_counts()
        assert state['epoch']==op.epoch and all(v==(op.epoch<<32)|op.splits for v in state['counts'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile',action='store_true');parser.add_argument('--quick',action='store_true')
    parser.add_argument('--report',type=Path);parser.add_argument('--build-directory',type=Path)
    parser.add_argument('--node-order',choices=('ordered','permuted'),default='ordered')
    args=parser.parse_args();torch.set_num_threads(8);torch.manual_seed(171)
    paths=['native_pi0/attention.cu','native_pi0/attention.py','native_pi0/dense.cu','native_pi0/dense.py',
        'native_pi0/projection_attention.cu','native_pi0/projection_attention.py','native_pi0/cached_prefix_expert.py','native_pi0/tcgen_probe.cu',
        'native_pi0/dag_probe.cu','native_pi0/dag_probe.py','native_pi0/dag_scheduler.cuh','native_pi0/launch_trace.cpp','native_pi0/launch_trace.py',
        'scripts/validate_kv_cache.py','third_party/Megakernels/include/megakernel.cuh','third_party/Megakernels/include/util.cuh',
        'third_party/Megakernels/include/controller/controller.cuh','third_party/Megakernels/include/controller/instruction_fetch.cuh']
    hashes={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths};records=[]
    if args.build_directory:args.build_directory.mkdir(parents=True,exist_ok=True)
    with (contextlib.nullcontext(args.build_directory) if args.build_directory else tempfile.TemporaryDirectory()) as directory:
        library=Path(directory).resolve()/'kv_cache.so'
        command=['nvcc','-O3','-lineinfo','-std=c++20','-arch=sm_100a','-DKITTENS_HOPPER','-DKITTENS_BLACKWELL','-DNATIVE_COMPUTE_CONSUMER_WARPS=4',
            '--expt-relaxed-constexpr','--expt-extended-lambda','-I'+str(ROOT/'third_party/Megakernels/include'),'-I'+str(ROOT/'third_party/Megakernels/ThunderKittens/include'),
            '-shared','-Xcompiler','-fPIC',str(ROOT/'native_pi0/projection_attention.cu'),str(ROOT/'native_pi0/launch_trace.cpp'),
            '-lcupti','-Xlinker','-rpath','-Xlinker','/usr/local/cuda/lib64','-lcuda','-o',str(library)]
        subprocess.run(command,check=True)
        cases=[('tail',2,65,3,4,1,64,torch.bfloat16,2,8,16),('empty-parts',1,1,3,2,1,72,torch.float16,8,16,1)]
        if not args.quick:cases += [('prefix-boundary',1,129,51,8,1,256,torch.bfloat16,1,8,148),
            ('full-prefix',1,768,51,8,1,256,torch.bfloat16,1,8,148),('fp16',1,67,17,4,2,128,torch.float16,2,4,16)]
        for label,b,p,n,h,hk,d,dtype,psplits,splits,workers in cases:
            def rand(shape):return torch.randn(shape,device='cuda').to(dtype)
            def rope(rows):
                angle=torch.randn((b,rows,d),device='cuda');return angle.cos().to(dtype),angle.sin().to(dtype)
            cache=(torch.full((b,p+n+7,hk,d),float('nan'),device='cuda',dtype=dtype),torch.full((b,p+n+7,hk,d),float('nan'),device='cuda',dtype=dtype))
            pq,pk,pv=rand((b,p,h,d)),rand((b,p,hk,d)),rand((b,p,hk,d));pr=rope(p)
            pb=[rand((heads,d))*.1 for heads in (h,hk,hk)]
            with ResidentAttention(library,pq,pk,pv,bias=pb,q_rope=pr,k_rope=pr,kv_cache=cache,workers=workers,splits=psplits) as producer:
                if args.profile:
                    with LaunchTrace(library) as trace:producer.forward();torch.cuda.synchronize()
                    assert trace.count==1,trace.count
                else:producer.forward();torch.cuda.synchronize()
                check_cache(producer)
                assert all(torch.isnan(t[:,p:]).all() for t in cache)
                frozen=[t[:,:p].clone() for t in cache]
                q,k,v=rand((b,n,h,d)),rand((b,n,hk,d)),rand((b,n,hk,d));r=rope(n)
                bias=[rand((heads,d))*.1 for heads in (h,hk,hk)]
                mask=torch.ones((b,n,p+n),device='cuda',dtype=torch.bool);mask[:,0,:min(p,3)]=False
                with ResidentAttention(library,q,k,v,bias=bias,q_rope=r,k_rope=r,mask=mask,kv_cache=cache,cached_prefix_rows=p,workers=workers,splits=splits) as suffix:
                    maximum=0.
                    # Raw prefix tensors may now be discarded or overwritten.
                    pk.fill_(float('nan'));pv.fill_(float('nan'))
                    for step in range(10):
                        stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
                        with torch.cuda.stream(stream):
                            suffix.q.copy_(rand(suffix.q.shape));suffix.k.copy_(rand(suffix.k.shape));suffix.v.copy_(rand(suffix.v.shape))
                            suffix.output.fill_(float('nan'))
                            for t in cache:t[:,p:p+n].fill_(float('nan'))
                            suffix.poison_partials();suffix.forward()
                        stream.synchronize();check_cache(suffix)
                        error=nrmse(suffix.output.cpu(),gold_suffix(suffix));assert error<=.01,(label,step,error);maximum=max(maximum,error)
                        for t,snapshot in zip(cache,frozen):assert torch.equal(t[:,:p].view(torch.int16),snapshot.view(torch.int16))
                        assert all(torch.isnan(t[:,p+n:]).all() for t in cache)
                    # π0 state query must ignore action K/V in this same step.
                    state=suffix.output[:,0].clone();suffix.k[:,1:].add_(2);suffix.v[:,1:].mul_(3)
                    suffix.forward();torch.cuda.synchronize();assert torch.equal(state.view(torch.int16),suffix.output[:,0].view(torch.int16))
                    record={'label':label,'batch':b,'prefix_rows':p,'suffix_rows':n,'heads':h,'kv_heads':hk,'head_dim':d,'dtype':str(dtype),
                        'prefix_splits':psplits,'suffix_splits':splits,'workers':workers,'suffix_replays':10,'max_nrmse':maximum,
                        'prefix_cache':'bit-exact immutable across suffix replays','cache_tail':'NaN sentinels untouched','raw_prefix_inputs':'poisoned after generation',
                        'state_action_mask':'state unchanged after mutating action K/V'}
                    if args.profile:
                        with LaunchTrace(library) as trace:suffix.forward();torch.cuda.synchronize()
                        assert trace.count==1,trace.count;record['suffix_cuda_launch_count']=1
                        record['prefix_cuda_launch_count']=1
                    print(label,'cache generation/reuse NRMSE',maximum,flush=True);records.append(record)
        # Integrate existing prefix cache with native QKV and O projections.
        b,p,n,h,hk,d,kdim=1,65,3,2,1,64,192
        cache=(torch.randn(b,p+n,hk,d,device='cuda',dtype=torch.bfloat16),torch.randn(b,p+n,hk,d,device='cuda',dtype=torch.bfloat16))
        saved=[x[:,:p].clone() for x in cache]
        with ResidentDense(library,torch.randn(n,kdim,device='cuda').to(torch.float8_e4m3fn),(torch.randn((h+2*hk)*d,kdim,device='cuda')*.05).to(torch.float8_e4m3fn),workers=4) as projection:
            with ResidentDense(library,torch.zeros(n,h*d,device='cuda',dtype=torch.bfloat16),(torch.randn(kdim,h*d,device='cuda')*.05).bfloat16(),workers=4) as output:
                with ResidentProjectionAttention(library,projection,batches=b,rows=n,heads=h,kv_heads=hk,head_dim=d,workers=4,splits=8,
                    output_projection=output,kv_cache=cache,cached_prefix_rows=p) as chain:
                    result=chain.forward().clone();torch.cuda.synchronize()
                    projection.forward();chain.attention.forward();output.forward();torch.cuda.synchronize()
                    assert torch.equal(result.view(torch.int16),chain.output.view(torch.int16))
                    for x,y in zip(cache,saved):assert torch.equal(x[:,:p].view(torch.int16),y.view(torch.int16))
                    if args.profile:
                        with LaunchTrace(library) as trace:chain.forward();torch.cuda.synchronize()
                        assert trace.count==1,trace.count
                    print('Cached-prefix QKV/attention/O: bit-exact separate native outputs',flush=True)
        integrated={'prefix_rows':p,'suffix_rows':n,'separate_outputs':'bit-exact','cuda_launch_count':1 if args.profile else None}
        pair_records=[]
        pairs=[('tail-pair',2,65,3,2,4,1,64,192,128,2,8,16)]
        if not args.quick:pairs += [('full-pair',1,768,51,8,8,1,256,2048,1024,1,8,148),('one-worker',1,129,3,2,2,1,72,192,192,8,16,1)]
        for label,b,p,n,ph,eh,hk,d,pkdim,ekdim,ps,es,workers in pairs:
            with contextlib.ExitStack() as stack:
                cache=tuple(torch.full((b,p+n+7,hk,d),float('nan'),device='cuda',dtype=torch.bfloat16) for _ in range(2))
                def make_chain(rows,heads,kdim,splits,cached):
                    dtype=torch.bfloat16 if label=='one-worker' and not cached else torch.float8_e4m3fn
                    q=stack.enter_context(ResidentDense(library,torch.randn(b*rows,kdim,device='cuda').to(dtype),
                        (torch.randn((heads+2*hk)*d,kdim,device='cuda')*.05).to(dtype),bias=torch.randn((heads+2*hk)*d)*.02,workers=workers))
                    o=stack.enter_context(ResidentDense(library,torch.zeros(b*rows,heads*d,device='cuda',dtype=torch.bfloat16),
                        (torch.randn(kdim,heads*d,device='cuda')*.05).bfloat16(),residual=torch.randn(b*rows,kdim)*.1,workers=workers))
                    angle=torch.randn(b,rows,d,device='cuda');tables=angle.cos().bfloat16(),angle.sin().bfloat16()
                    return stack.enter_context(ResidentProjectionAttention(library,q,batches=b,rows=rows,heads=heads,kv_heads=hk,head_dim=d,workers=workers,
                        splits=splits,q_rope=tables,k_rope=tables,output_projection=o,kv_cache=cache,cached_prefix_rows=cached))
                prefix=make_chain(p,ph,pkdim,ps,0);expert=make_chain(n,eh,ekdim,es,p)
                nodes=prefix._queue.nodes+expert._queue.nodes;order=torch.randperm(nodes).numpy()
                with ResidentCachedPrefixExpert(library,prefix,expert,workers=workers,node_order=order if args.node_order=='permuted' else None) as pair:
                    def check_pair():
                        data=pair.read_diagnostics();epoch=pair._queue.epoch
                        assert list(data['counts'])==[nodes,nodes,nodes,0,epoch]
                        assert all(v==epoch for v in data['values']) and all(v==epoch for v in data['visits'])
                        for node,parents in enumerate(pair._queue.parents):
                            if parents:assert data['starts'][node]>=max(data['ends'][parent] for parent in parents)
                        check_cache(prefix.attention);check_cache(expert.attention)
                    for replay in range(3):
                        stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
                        with torch.cuda.stream(stream):
                            for c in (prefix,expert):
                                c.projection.inputs.copy_(torch.randn_like(c.projection.inputs.float()).to(c.projection.a.dtype))
                                c.projection.output.fill_(float('nan'));c.output.fill_(float('nan'))
                                if c.attention.splits>1:c.attention.poison_partials()
                            for value in cache:value.fill_(float('nan'))
                            pair.forward()
                        stream.synchronize();check_pair()
                        result=pair.output.clone();prefix_result=prefix.output.clone();stored=[x.clone() for x in cache]
                        assert torch.isfinite(result).all()
                        assert nrmse(expert.attention.output.cpu(),gold_suffix(expert.attention))<=.01
                        prefix.forward();expert.forward();torch.cuda.synchronize()
                        assert torch.equal(result.view(torch.int16),expert.output.view(torch.int16))
                        assert torch.equal(prefix_result.view(torch.int16),prefix.output.view(torch.int16))
                        for actual,expected in zip(cache,stored):
                            assert torch.equal(actual[:,:p+n].view(torch.int16),expected[:,:p+n].view(torch.int16))
                            assert torch.isnan(actual[:,p+n:]).all()
                    record={'label':label,'prefix_rows':p,'suffix_rows':n,'prefix_heads':ph,'expert_heads':eh,'kv_heads':hk,'head_dim':d,
                        'nodes':nodes,'workers':workers,'replays':3,'node_order':args.node_order,'outputs_and_kv':'bit-exact separate child chains',
                        'output_sha256':hashlib.sha256(result.contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()}
                    if args.profile:
                        with LaunchTrace(library) as trace:pair.forward();torch.cuda.synchronize()
                        assert trace.count==1,trace.count;record['cuda_launch_count']=1;check_pair()
                        with LaunchTrace(library) as trace:prefix.forward();expert.forward();torch.cuda.synchronize()
                        assert trace.count==2,trace.count;record['separate_child_launch_count']=2
                    if args.profile and args.report:
                        def separate():prefix.forward();expert.forward()
                        def measure(call):
                            for _ in range(5):call()
                            torch.cuda.synchronize();a,z=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);times=[];wall=[]
                            for _ in range(30):
                                torch.cuda.synchronize();t=time.perf_counter_ns();a.record();call();z.record();z.synchronize()
                                times.append(a.elapsed_time(z));wall.append((time.perf_counter_ns()-t)/1e6)
                            def stats(values):return {'median_ms':statistics.median(values),'p10_ms':float(torch.tensor(values).quantile(.1)),'p90_ms':float(torch.tensor(values).quantile(.9)),'samples_ms':values}
                            return {'gpu':stats(times),'synchronized_wall':stats(wall)}
                        record['integrated']=measure(pair.forward);check_pair()
                        assert torch.equal(pair.output.view(torch.int16),result.view(torch.int16))
                        record['separate']=measure(separate)
                    pair_records.append(record);print(label,'single-launch cached prefix/expert pair: exact outputs and cache',flush=True)
    assert all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==v for p,v in hashes.items()),'Source changed during validation'
    if args.report:
        args.report.parent.mkdir(parents=True,exist_ok=True)
        args.report.write_text(json.dumps({'timestamp_utc':datetime.datetime.now(datetime.UTC).isoformat(),'seed':171,'scope':'Synthetic native rotated KV generation and prefix reuse; ten separate suffix calls, not a complete denoising model',
            'source_sha256':hashes,'build_command':command,'results':records,'integrated':integrated,'pairs':pair_records,
            'pair_scope':'One prefix/expert attention-layer pair per launch; not a full transformer or ten-step denoising model'},indent=2)+'\n')


if __name__=='__main__':main()
