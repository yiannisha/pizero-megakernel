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

import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from native_pi0.attention import ResidentAttention
from native_pi0.dense import ResidentDense
from native_pi0.projection_attention import ResidentProjectionAttention
from native_pi0.launch_trace import LaunchTrace


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
    args=parser.parse_args();torch.set_num_threads(8);torch.manual_seed(171)
    paths=['native_pi0/attention.cu','native_pi0/attention.py','native_pi0/dense.cu','native_pi0/dense.py',
        'native_pi0/projection_attention.cu','native_pi0/projection_attention.py','native_pi0/tcgen_probe.cu',
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
    assert all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==v for p,v in hashes.items()),'Source changed during validation'
    if args.report:
        args.report.parent.mkdir(parents=True,exist_ok=True)
        args.report.write_text(json.dumps({'timestamp_utc':datetime.datetime.now(datetime.UTC).isoformat(),'seed':171,'scope':'Synthetic native rotated KV generation and prefix reuse; ten separate suffix calls, not a complete denoising model',
            'source_sha256':hashes,'build_command':command,'results':records,'integrated':integrated},indent=2)+'\n')


if __name__=='__main__':main()
