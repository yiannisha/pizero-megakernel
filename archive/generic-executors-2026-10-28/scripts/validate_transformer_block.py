"""Check complete native attention/GeGLU blocks against split native and CPU math.

Timing comparisons here are component comparisons against split native stages.
They do not establish model-level speedups over FlashRT or any Torch baseline.
"""
import argparse
import ctypes
from contextlib import ExitStack
import datetime
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from native_pi0.normalization import ResidentNorm
from native_pi0.dense import ResidentDense
from native_pi0.primitives import ResidentNVFP4
from native_pi0.projection_attention import ResidentProjectionAttention
from native_pi0.transformer_block import ResidentTransformerBlock
from native_pi0.quantization import pack_nvfp4,interleave_gate_up,PackedNVFP4,unswizzle_scales
from native_pi0.launch_trace import LaunchTrace
from scripts.validate_dag_matmul import check_diagnostics


def nrmse(a,b):
    a,b=torch.as_tensor(a).float(),torch.as_tensor(b).float()
    return float((a-b).square().mean().sqrt()/b.square().mean().sqrt().clamp_min(1e-12))


def decode(data,scales,global_scale):
    x=data.cpu().numpy();rows,half=x.shape
    return torch.from_numpy(PackedNVFP4(x,unswizzle_scales(scales.cpu().numpy(),rows,half*2//16),global_scale,(rows,half*2)).dequantize())


def norm_gold(norm):
    x=norm.inputs.float().cpu()
    y=x*torch.rsqrt(x.square().mean(-1,keepdim=True)+1e-6)
    if norm.gamma is not None:y*=norm.gamma.cpu()
    return y.to(torch.bfloat16).float()


def timing(fn):
    for _ in range(3):fn()
    torch.cuda.synchronize();samples=[]
    for _ in range(15):
        start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        start.record();fn();end.record();end.synchronize();samples.append(start.elapsed_time(end))
    return {'median_ms':statistics.median(samples),'samples_ms':samples}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--library',type=Path);parser.add_argument('--build-directory',type=Path,default=Path('/tmp/native-transformer-block'))
    parser.add_argument('--report',type=Path);parser.add_argument('--quick',action='store_true');parser.add_argument('--profile',action='store_true')
    parser.add_argument('--node-order',choices=('ordered','permuted'),default='ordered')
    args=parser.parse_args();torch.set_num_threads(8);torch.manual_seed(719);rng=np.random.default_rng(719)
    paths=[str(p.relative_to(ROOT)) for p in (ROOT/'native_pi0').glob('*.cu')]+[
        'native_pi0/transformer_block.py','native_pi0/normalization.py','native_pi0/dense.py','native_pi0/primitives.py',
        'native_pi0/projection_attention.py','native_pi0/attention.py','native_pi0/dag_probe.py','native_pi0/dag_scheduler.cuh',
        'scripts/validate_transformer_block.py','third_party/Megakernels/include/megakernel.cuh']
    hashes={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths}
    library=args.library
    if library is None:
        args.build_directory.mkdir(parents=True,exist_ok=True);library=args.build_directory/'transformer_block.so'
        subprocess.run(['nvcc','-O3','-lineinfo','-std=c++20','-arch=sm_100a','-DKITTENS_HOPPER','-DKITTENS_BLACKWELL',
            '-DNATIVE_COMPUTE_CONSUMER_WARPS=4','--expt-relaxed-constexpr','--expt-extended-lambda',
            '-I'+str(ROOT/'third_party/Megakernels/include'),'-I'+str(ROOT/'third_party/Megakernels/ThunderKittens/include'),
            '-shared','-Xcompiler','-fPIC',str(ROOT/'native_pi0/transformer_block.cu'),str(ROOT/'native_pi0/launch_trace.cpp'),
            '-lcupti','-Xlinker','-rpath','-Xlinker','/usr/local/cuda/lib64','-lcuda','-o',str(library)],check=True)
    library=library.resolve();records=[]
    lib=ctypes.CDLL(str(library));group_rows=lib.transformer_block_norm_rows;group_rows.restype=ctypes.c_int
    norm_group_rows=group_rows()
    cases=[('small-bf16',1,3,128,2,1,64,256,False,1,1)]
    if not args.quick:cases += [('batch-tail-fp8',2,65,256,2,1,72,512,True,2,16),
        ('expert-core',1,51,1024,8,1,256,4096,True,8,148),('prefix-core',1,768,2048,8,1,256,16384,True,1,148)]
    for label,b,rows,width,heads,hk,hd,ff,fp8,splits,workers in cases:
        m=b*rows;mp=(m+127)//128*128;dtype=torch.float8_e4m3fn if fp8 else torch.bfloat16
        def rand(shape,scale=1.):return (torch.randn(shape,device='cuda')*scale).to(torch.bfloat16)
        with ExitStack() as stack:
            x=rand((mp,width));x[m:]=0
            pre=stack.enter_context(ResidentNorm(library,x,precision='fp8' if fp8 else 'bf16',gamma=1+torch.randn(width)*.1,workers=workers))
            qkv=stack.enter_context(ResidentDense(library,pre.output[:m],rand(((heads+2*hk)*hd,width),.02).to(dtype),workers=workers))
            o=stack.enter_context(ResidentDense(library,torch.zeros((m,heads*hd),device='cuda',dtype=torch.bfloat16),rand((width,heads*hd),.02),workers=workers))
            angle=torch.randn((b,rows,hd),device='cuda');rope=(angle.cos().to(torch.bfloat16),angle.sin().to(torch.bfloat16))
            cached_rows=768 if label=='expert-core' else 0;cache=None
            if cached_rows:
                cache=tuple(torch.full((b,cached_rows+rows+7,hk,hd),float('nan'),device='cuda',dtype=torch.bfloat16) for _ in range(2))
                for t in cache:t[:,:cached_rows].copy_(rand((b,cached_rows,hk,hd)))
                frozen_cache=[t[:,:cached_rows].clone() for t in cache]
            attention=stack.enter_context(ResidentProjectionAttention(library,qkv,batches=b,rows=rows,heads=heads,kv_heads=hk,head_dim=hd,
                workers=workers,splits=splits,q_rope=rope,k_rope=rope,output_projection=o,kv_cache=cache,cached_prefix_rows=cached_rows))
            post=stack.enter_context(ResidentNorm(library,torch.zeros_like(x),precision='fp4',gamma=1+torch.randn(width)*.1,workers=workers,global_scale=.125))
            a=pack_nvfp4(np.zeros((mp,width),np.float32),global_scale=.125)
            w=pack_nvfp4(interleave_gate_up(rng.normal(0,.02,(ff,width)).astype(np.float32),rng.normal(0,.02,(ff,width)).astype(np.float32)))
            gate=stack.enter_context(ResidentNVFP4(library,a,w,workers=workers,output_dtype=torch.uint8,output_geglu=True,output_global_scale=.125))
            da=pack_nvfp4(np.zeros((mp,ff),np.float32),global_scale=.125)
            dw=pack_nvfp4(rng.normal(0,.02,(width,ff)).astype(np.float32))
            down=stack.enter_context(ResidentNVFP4(library,da,dw,workers=workers,output_dtype=torch.bfloat16))
            nodes=attention._queue.nodes+2*(mp//norm_group_rows)+(mp//128)*(gate.b.shape[0]//128+width//128)
            order=rng.permutation(nodes)
            block=stack.enter_context(ResidentTransformerBlock(library,attention,pre,post,gate,down,workers=workers,
                node_order=order if args.node_order=='permuted' else None))
            def separate():
                pre.forward();attention.forward();post.forward();gate.forward();down.forward()
            max_errors={k:0. for k in ('pre_norm','qkv','attention','o_residual','post_norm','gate_geglu','down_residual')}
            for replay in range(3):
                stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    block.inputs[:m].copy_(rand((m,width)));block.inputs[m:]=0
                    for t in (qkv.output,attention.attention.output,o.output,down.output):t.fill_(float('nan'))
                    post.output.fill_(255);post.output_scales.fill_(255);gate.output.fill_(255);gate.output_scales.fill_(255)
                    block.forward()
                stream.synchronize();check_diagnostics(block)
                snapshots=[t.clone() for t in (pre.output,qkv.output,attention.attention.output,o.output,post.output,post.output_scales,gate.output,gate.output_scales,down.output)]
                cache_snapshots=None if cache is None else [t[:,:cached_rows+rows].clone() for t in cache]
                separate();torch.cuda.synchronize()
                for actual,expected in zip((pre.output,qkv.output,attention.attention.output,o.output,post.output,post.output_scales,gate.output,gate.output_scales,down.output),snapshots):
                    assert torch.equal(actual.view(torch.uint8),expected.view(torch.uint8)),(label,replay,'split mismatch',actual.shape)
                if cache is not None:
                    for t,frozen,snapshot in zip(cache,frozen_cache,cache_snapshots):
                        assert torch.equal(t[:,:cached_rows],frozen)
                        assert torch.equal(t[:,:cached_rows+rows],snapshot)
                        assert torch.isnan(t[:,cached_rows+rows:]).all()
                errors={}
                ng=norm_gold(pre)
                errors['pre_norm']=nrmse(pre.output.float().cpu(),ng.to(dtype).float())
                errors['qkv']=nrmse(qkv.output.cpu(),(qkv.a.float().cpu()@qkv.b.float().cpu().T).to(torch.bfloat16))
                from scripts.validate_kv_cache import transform
                at=attention.attention;q=transform(at.q,None,at.q_rope).float();k=transform(at.k,None,at.k_rope).float();v=at.v.float().cpu()
                if cache is not None:
                    from scripts.validate_kv_cache import gold_suffix,check_cache
                    ag=gold_suffix(at);check_cache(at)
                else:
                    k=k.repeat_interleave(heads//hk,2);v=v.repeat_interleave(heads//hk,2)
                    ag=(torch.softmax((q.transpose(1,2)@k.transpose(1,2).transpose(-1,-2))*at.scale,-1)@v.transpose(1,2)).transpose(1,2)
                errors['attention']=nrmse(at.output.cpu(),ag)
                og=((o.a.float().cpu()@o.b.float().cpu().T).to(torch.bfloat16).float()+block.inputs.float().cpu()).to(torch.bfloat16)
                errors['o_residual']=nrmse(o.output.cpu(),og)
                pg=norm_gold(post);packed=pack_nvfp4(pg.numpy(),global_scale=post.global_scale)
                errors['post_norm']=nrmse(decode(post.output,post.output_scales,post.global_scale),packed.dequantize())
                ga=decode(gate.a,gate.sa,gate.activation_global_scale);gw=decode(gate.b,gate.sb,gate.weight_global_scale)
                proj=ga@gw.T;g=proj[:,0::2];u=proj[:,1::2]
                geglu=.5*g*(1+torch.tanh(.7978845608028654*(g+.044715*g*g*g)))*u
                errors['gate_geglu']=nrmse(decode(gate.output,gate.output_scales,gate.output_global_scale),pack_nvfp4(geglu.numpy(),global_scale=gate.output_global_scale).dequantize())
                dg=((decode(down.a,down.sa,down.activation_global_scale)@decode(down.b,down.sb,down.weight_global_scale).T).to(torch.bfloat16).float()+o.output.float().cpu()).to(torch.bfloat16)
                errors['down_residual']=nrmse(down.output.cpu(),dg)
                for key,error in errors.items():
                    assert error<=(.03 if key in ('post_norm','gate_geglu') else .01),(label,key,error)
                    max_errors[key]=max(max_errors[key],error)
            record={'label':label,'logical_rows':m,'width':width,'ffn_width':ff,'workers':workers,'nodes':nodes,
                'node_order':args.node_order,'norm_group_rows':norm_group_rows,'attention_keys':cached_rows+rows,'cached_prefix_rows':cached_rows,
                'residual_rounding':'BF16 projection before BF16 residual addition',
                'split_outputs':'all intermediate/final buffers bit-exact','cpu_component_nrmse':max_errors,
                'output_sha256':hashlib.sha256(block.output.cpu().view(torch.uint8).numpy().tobytes()).hexdigest()}
            if args.profile:
                with LaunchTrace(library) as trace:block.forward();torch.cuda.synchronize()
                assert trace.count==1;record['cuda_launch_count']=trace.count
                with LaunchTrace(library) as trace:separate();torch.cuda.synchronize()
                assert trace.count==5;record['split_cuda_launch_count']=trace.count
                record['integrated_gpu']=timing(block.forward);record['split_native_gpu']=timing(separate)
                block.forward();torch.cuda.synchronize();snapshot=block.output.clone();separate();torch.cuda.synchronize()
                assert torch.equal(block.output,snapshot[:m])
            print(label,'complete block exact vs split',max_errors,flush=True);records.append(record)
    assert hashes=={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths},'Sources changed during validation'
    report={'timestamp_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'source_sha256':hashes,
        'library_sha256':hashlib.sha256(library.read_bytes()).hexdigest(),'compiled_in_run':args.library is None,
        'scope':'One RMS pre-norm attention and NVFP4 GeGLU block; synthetic inputs, no checkpoint or end-to-end claim',
        'timing_baseline':'Five split native VM launches, not FlashRT, Torch, or realtime-vla','results':records}
    if args.report:args.report.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
