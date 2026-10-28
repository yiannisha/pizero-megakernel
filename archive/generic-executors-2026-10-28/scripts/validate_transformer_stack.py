"""Synthetic full-depth prefix/expert stacks in one original native VM launch.

Weights are random. The serial baseline is native blocks plus final norm,
not FlashRT, Torch, realtime-vla, or complete policy inference.
"""
import argparse
from contextlib import ExitStack
import datetime
import hashlib
import json
from pathlib import Path
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
from native_pi0.transformer_stack import ResidentTransformerStack
from native_pi0.quantization import pack_nvfp4,interleave_gate_up
from native_pi0.launch_trace import LaunchTrace
from scripts.validate_dag_matmul import check_diagnostics
from scripts.validate_transformer_block import decode,norm_gold,nrmse,timing
from scripts.validate_kv_cache import check_cache,gold_suffix,transform


def make_block(owners,library,rng,*,batch,rows,width,ff,heads,hk,hd,fp8,splits,workers,cached_rows,write_cache):
    m=batch*rows;mp=(m+127)//128*128;dtype=torch.float8_e4m3fn if fp8 else torch.bfloat16
    def rand(shape,scale=1.):return (torch.randn(shape,device='cuda')*scale).to(torch.bfloat16)
    x=torch.zeros((mp,width),device='cuda',dtype=torch.bfloat16)
    pre=owners.enter_context(ResidentNorm(library,x,precision='fp8' if fp8 else 'bf16',gamma=1+torch.randn(width)*.1,workers=workers))
    qkv=owners.enter_context(ResidentDense(library,pre.output[:m],rand(((heads+2*hk)*hd,width),.02).to(dtype),workers=workers))
    o=owners.enter_context(ResidentDense(library,torch.zeros((m,heads*hd),device='cuda',dtype=torch.bfloat16),rand((width,heads*hd),.02),workers=workers))
    angle=torch.randn((batch,rows,hd),device='cuda');rope=(angle.cos().to(torch.bfloat16),angle.sin().to(torch.bfloat16))
    cache=None
    if write_cache or cached_rows:
        cache=tuple(torch.full((batch,cached_rows+rows+7,hk,hd),float('nan'),device='cuda',dtype=torch.bfloat16) for _ in range(2))
        if cached_rows:
            for t in cache:t[:,:cached_rows].copy_(rand((batch,cached_rows,hk,hd)))
    attention=owners.enter_context(ResidentProjectionAttention(library,qkv,batches=batch,rows=rows,heads=heads,kv_heads=hk,head_dim=hd,
        workers=workers,splits=splits,q_rope=rope,k_rope=rope,output_projection=o,kv_cache=cache,cached_prefix_rows=cached_rows))
    post=owners.enter_context(ResidentNorm(library,torch.zeros_like(x),precision='fp4',gamma=1+torch.randn(width)*.1,workers=workers,global_scale=.125))
    a=pack_nvfp4(np.zeros((mp,width),np.float32),global_scale=.125)
    w=pack_nvfp4(interleave_gate_up(rng.normal(0,.02,(ff,width)).astype(np.float32),rng.normal(0,.02,(ff,width)).astype(np.float32)))
    gate=owners.enter_context(ResidentNVFP4(library,a,w,workers=workers,output_dtype=torch.uint8,output_geglu=True,output_global_scale=.125))
    a=pack_nvfp4(np.zeros((mp,ff),np.float32),global_scale=.125);w=pack_nvfp4(rng.normal(0,.02,(width,ff)).astype(np.float32))
    down=owners.enter_context(ResidentNVFP4(library,a,w,workers=workers,output_dtype=torch.bfloat16))
    return owners.enter_context(ResidentTransformerBlock(library,attention,pre,post,gate,down,workers=workers))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--library',type=Path);parser.add_argument('--build-directory',type=Path,default=Path('/tmp/native-transformer-stack'))
    parser.add_argument('--quick',action='store_true');parser.add_argument('--profile',action='store_true');parser.add_argument('--report',type=Path)
    parser.add_argument('--node-order',choices=('ordered','permuted'),default='ordered')
    args=parser.parse_args();torch.set_num_threads(8);torch.manual_seed(739);rng=np.random.default_rng(739)
    paths=[str(p.relative_to(ROOT)) for p in (ROOT/'native_pi0').glob('*.cu')]+[str(p.relative_to(ROOT)) for p in (ROOT/'native_pi0').glob('*.py')]+[
        'native_pi0/dag_scheduler.cuh','scripts/validate_transformer_stack.py','scripts/validate_transformer_block.py',
        'third_party/Megakernels/include/megakernel.cuh','third_party/Megakernels/include/controller/instruction_fetch.cuh']
    hashes={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths}
    library=args.library
    if library is None:
        args.build_directory.mkdir(parents=True,exist_ok=True);library=args.build_directory/'transformer_stack.so'
        subprocess.run(['nvcc','-O3','-lineinfo','-std=c++20','-arch=sm_100a','-DKITTENS_HOPPER','-DKITTENS_BLACKWELL',
            '-DNATIVE_COMPUTE_CONSUMER_WARPS=4','--expt-relaxed-constexpr','--expt-extended-lambda',
            '-I'+str(ROOT/'third_party/Megakernels/include'),'-I'+str(ROOT/'third_party/Megakernels/ThunderKittens/include'),
            '-shared','-Xcompiler','-fPIC',str(ROOT/'native_pi0/transformer_block.cu'),str(ROOT/'native_pi0/launch_trace.cpp'),
            '-lcupti','-Xlinker','-rpath','-Xlinker','/usr/local/cuda/lib64','-lcuda','-o',str(library)],check=True)
    library=library.resolve();records=[]
    cases=[('one-worker-depth18',18,1,3,128,256,2,1,64,1,1,0,False)]
    if not args.quick:cases += [('mixed-tail',3,2,65,256,512,2,1,72,2,16,0,True),
        ('expert-depth18',18,1,51,1024,4096,8,1,256,8,148,768,True),
        ('prefix-depth18',18,1,768,2048,16384,8,1,256,1,148,0,True)]
    for label,depth,b,rows,width,ff,heads,hk,hd,splits,workers,cached_rows,fp8 in cases:
        with ExitStack() as owners:
            blocks=[]
            for layer in range(depth):
                blocks.append(make_block(owners,library,rng,batch=b,rows=rows,width=width,ff=ff,heads=heads,hk=hk,hd=hd,
                    fp8=fp8 and (label!='mixed-tail' or layer%2==0),splits=splits,workers=workers,cached_rows=cached_rows,write_cache=True))
                if (layer+1)%3==0:print(label,'prepared layers',layer+1,flush=True)
            final=owners.enter_context(ResidentNorm(library,torch.zeros_like(blocks[-1].down.output),precision='bf16',gamma=1+torch.randn(width)*.1,workers=workers))
            if label=='one-worker-depth18':
                try:ResidentTransformerStack(library,blocks,blocks[0].pre_norm,workers=workers)
                except ValueError as error:assert 'distinct context' in str(error),str(error)
                else:raise AssertionError('A child normalization cannot also be the final normalization')
            nodes=sum(block._queue.nodes for block in blocks)+final.logical_rows//blocks[0].norm_group_rows
            order=rng.permutation(nodes)
            stack=owners.enter_context(ResidentTransformerStack(library,blocks,final,workers=workers,node_order=order if args.node_order=='permuted' else None))
            def serial():
                for block in blocks:block.forward()
                final.forward()
            caches=[block.attention.attention.kv_cache for block in blocks]
            frozen=[[t[:,:cached_rows].clone() for t in cache] for cache in caches]
            maximum=0.;attention_max=0.;down_max=0.
            for replay in range(3):
                stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    stack.inputs[:b*rows].copy_(torch.randn((b*rows,width),device='cuda').to(torch.bfloat16));stack.inputs[b*rows:]=0
                    final.output.fill_(float('nan'))
                    for block,cache in zip(blocks,caches):
                        block.down.output.fill_(float('nan'));block.attention.output_projection.output.fill_(float('nan'))
                        for t in cache:t[:,cached_rows:cached_rows+rows]=float('nan')
                    stack.forward()
                stream.synchronize();check_diagnostics(stack)
                snapshots=[block.down.output.clone() for block in blocks];final_snapshot=final.output.clone()
                cache_snapshots=[[t[:,:cached_rows+rows].clone() for t in cache] for cache in caches]
                serial();torch.cuda.synchronize()
                for block,snapshot in zip(blocks,snapshots):assert torch.equal(block.down.output,snapshot),(label,replay,'layer output')
                assert torch.equal(final.output,final_snapshot),(label,replay,'final norm')
                for layer,(block,cache,old,reference) in enumerate(zip(blocks,caches,frozen,cache_snapshots)):
                    check_cache(block.attention.attention)
                    for actual,prefix,expected in zip(cache,old,reference):
                        assert torch.equal(actual[:,:cached_rows],prefix),(label,replay,layer,'prefix cache mutation')
                        assert torch.equal(actual[:,:cached_rows+rows],expected),(label,replay,layer,'cache mismatch')
                        assert torch.isnan(actual[:,cached_rows+rows:]).all(),(label,replay,layer,'capacity guard')
                error=nrmse(final.output.cpu(),norm_gold(final));maximum=max(maximum,error);assert error<=.005,(label,error)
                # Independent CPU attention/down checks at both ends of the stack.
                for block in (blocks[0],blocks[-1]):
                    at=block.attention.attention
                    if cached_rows:gold=gold_suffix(at)
                    else:
                        q=transform(at.q,None,at.q_rope).float();k=transform(at.k,None,at.k_rope).float();v=at.v.float().cpu()
                        k=k.repeat_interleave(heads//hk,2);v=v.repeat_interleave(heads//hk,2)
                        gold=(torch.softmax((q.transpose(1,2)@k.transpose(1,2).transpose(-1,-2))*at.scale,-1)@v.transpose(1,2)).transpose(1,2)
                    error=nrmse(at.output.cpu(),gold);attention_max=max(attention_max,error);assert error<=.01,(label,'attention',error)
                    d=block.down
                    gold=((decode(d.a,d.sa,d.activation_global_scale)@decode(d.b,d.sb,d.weight_global_scale).T).to(torch.bfloat16).float()+d.residual.float().cpu()).to(torch.bfloat16)
                    error=nrmse(d.output.cpu(),gold);down_max=max(down_max,error);assert error<=.005,(label,'down',error)
                print(label,'replay',replay+1,'all layer outputs and caches exact',flush=True)
            record={'label':label,'depth':depth,'batch':b,'rows':rows,'width':width,'ffn_width':ff,'cached_prefix_rows':cached_rows,
                'workers':workers,'nodes':nodes,'node_order':args.node_order,'serial_layer_outputs_and_caches':'bit-exact all layers, three replays',
                'final_norm_nrmse':maximum,'sampled_attention_nrmse':attention_max,'sampled_down_nrmse':down_max,
                'output_sha256':hashlib.sha256(stack.output.cpu().view(torch.uint8).numpy().tobytes()).hexdigest()}
            if args.profile:
                with LaunchTrace(library) as trace:stack.forward();torch.cuda.synchronize()
                assert trace.count==1;record['cuda_launch_count']=trace.count
                with LaunchTrace(library) as trace:serial();torch.cuda.synchronize()
                assert trace.count==depth+1;record['serial_cuda_launch_count']=trace.count
                record['integrated_gpu']=timing(stack.forward);record['serial_native_gpu']=timing(serial)
                stack.forward();torch.cuda.synchronize();snapshot=stack.output.clone();serial();torch.cuda.synchronize();assert torch.equal(stack.output,snapshot)
            print(label,'final output SHA256',record['output_sha256'],flush=True)
            records.append(record)
    assert hashes=={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths},'Sources changed during validation'
    report={'timestamp_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'source_sha256':hashes,
        'library_sha256':hashlib.sha256(library.read_bytes()).hexdigest(),'compiled_in_run':args.library is None,
        'scope':'Synthetic transformer layer stacks and final norm; expert caches prefilled independently; not a complete policy or pretrained parity',
        'timing_baseline':'Serial native blocks plus final norm; not FlashRT, Torch, realtime-vla, or full-model inference','results':records}
    if args.report:args.report.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
