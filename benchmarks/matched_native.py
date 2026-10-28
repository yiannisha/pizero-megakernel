"""Matched π0 native/OpenPI latency on one B200 model and input set."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
from types import MethodType,SimpleNamespace

import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'third_party/openpi/src')]
from benchmarks.openpi_backend import capture_prefix,capture_suffix,fixed_sample
from native_pi0.model import NativePi0
from openpi.models_pytorch.pi0_pytorch import PI0Pytorch


def nrmse(a,b):
    return float((a.float()-b.float()).square().mean().sqrt()/b.float().square().mean().sqrt().clamp_min(1e-12))


def measure(fn,warmup,iterations):
    for _ in range(warmup):fn()
    torch.cuda.synchronize()
    gpu,wall,others=[],[],set()
    for _ in range(iterations):
        query=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True)
        others.update(int(pid) for pid in query.split() if pid.isdigit() and int(pid)!=os.getpid())
        start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        t0=time.perf_counter_ns();start.record();fn();end.record();end.synchronize()
        gpu.append(start.elapsed_time(end));wall.append((time.perf_counter_ns()-t0)/1e6)
    return {'gpu_median_ms':statistics.median(gpu),'gpu_samples_ms':gpu,
            'wall_median_ms':statistics.median(wall),'wall_samples_ms':wall,
            'other_compute_pids':sorted(others),'gpu_isolation_verified':not others}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--library',type=Path,required=True)
    parser.add_argument('--backends',nargs='+',choices=('native','eager','graph','compiled'),default=['native'])
    parser.add_argument('--views',type=int,choices=(1,2,3),default=3)
    parser.add_argument('--prompt',type=int,default=0)
    parser.add_argument('--chunk',type=int,default=50)
    parser.add_argument('--steps',type=int,default=10)
    parser.add_argument('--workers',type=int,default=148)
    parser.add_argument('--native-variants',nargs='+',choices=('fused','no_gelu','no_skinny','no_qkv','neither'),default=['fused'])
    parser.add_argument('--warmup',type=int,default=2)
    parser.add_argument('--iterations',type=int,default=7)
    parser.add_argument('--seed',type=int,default=101)
    parser.add_argument('--checkpoint',type=Path)
    parser.add_argument('--report',type=Path,required=True)
    args=parser.parse_args()
    if args.prompt<0 or min(args.chunk,args.steps,args.warmup,args.iterations,args.workers)<1:parser.error('Invalid fixed workload')
    torch.manual_seed(args.seed);torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    config=SimpleNamespace(pi05=False,paligemma_variant='gemma_2b',action_expert_variant='gemma_300m',dtype='bfloat16',
        action_dim=32,action_horizon=args.chunk,pytorch_compile_mode=None)
    with torch.device('cuda'):model=PI0Pytorch(config).eval()
    if args.checkpoint:
        from safetensors.torch import load_model
        load_model(model,str(args.checkpoint),strict=True)
    model.paligemma_with_expert.paligemma.language_model.config._attn_implementation='eager'
    model.paligemma_with_expert.gemma_expert.model.config._attn_implementation='eager'
    model.embed_prefix=MethodType(capture_prefix,model);model.embed_suffix=MethodType(capture_suffix,model)
    images=torch.randn((args.views,224,224,3),device='cuda')
    state=torch.randn((1,32),device='cuda');noise=torch.randn((1,args.chunk,32),device='cuda')
    tokens=torch.randint(0,257152,(1,args.prompt),device='cuda')
    imgs=[images[i].permute(2,0,1)[None] for i in range(args.views)]
    masks=[torch.ones(1,device='cuda',dtype=torch.bool) for _ in imgs]
    token_mask=torch.ones_like(tokens,dtype=torch.bool)
    def eager():return fixed_sample(model,imgs,masks,tokens,token_mask,state,noise,steps=args.steps)
    with torch.no_grad():reference=eager().clone();torch.cuda.synchronize()
    result={'workload':{'views':args.views,'prompt_len':args.prompt,'chunk':args.chunk,'flow_steps':args.steps,'seed':args.seed,'native_workers':args.workers,
                       'checkpoint':str(args.checkpoint) if args.checkpoint else None,'same_model_and_inputs':True},
            'environment':{'gpu':torch.cuda.get_device_name(),'capability':torch.cuda.get_device_capability(),
                           'torch':torch.__version__,'cuda':torch.version.cuda},
            'library_sha256':hashlib.sha256(args.library.read_bytes()).hexdigest(),
            'source_sha256':{name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in
                ('benchmarks/matched_native.py','benchmarks/openpi_backend.py','native_pi0/model.py','native_pi0/dense.py')},
            'backends':{}}
    scale=float(reference.float().square().mean().sqrt().clamp_min(1e-12))
    def record(name,fn,extra=None):
        with torch.no_grad():output=fn().clone();torch.cuda.synchronize()
        delta=(output-reference).float()
        validation={'nrmse':nrmse(output,reference),'normalized_max_abs':float(delta.abs().max()/scale),
                    'cosine':float(torch.nn.functional.cosine_similarity(output.float().flatten(),reference.float().flatten(),dim=0)),
                    'finite':bool(torch.isfinite(output).all())}
        with torch.no_grad():timing=measure(fn,args.warmup,args.iterations)
        result['backends'][name]={'validation':validation,'timing':timing,**(extra or {})}
        print(name,validation['nrmse'],timing['gpu_median_ms'],flush=True)
        args.report.parent.mkdir(parents=True,exist_ok=True)
        args.report.write_text(json.dumps(result,indent=2)+'\n')
    if 'native' in args.backends:
        for variant in args.native_variants:
            with NativePi0.from_model(model,num_views=args.views,prompt_len=args.prompt,chunk_size=args.chunk,
                                      flow_steps=args.steps,library=args.library,workers=args.workers,
                                      fuse_vision_gelu=variant not in ('no_gelu','neither'),
                                      skinny_expert=variant not in ('no_skinny','neither'),
                                      fuse_qkv_rope=variant!='no_qkv') as native:
                record('native_'+variant,lambda:native.forward(images,state,noise,tokens),
                       {'single_native_kernel':True,'program':native.plan})
    if 'eager' in args.backends:record('eager',eager)
    if 'graph' in args.backends:
        stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream),torch.no_grad():
            for _ in range(args.warmup):eager()
        torch.cuda.current_stream().wait_stream(stream);torch.cuda.synchronize()
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph,stream=stream),torch.no_grad():graph_output=eager()
        def replay():graph.replay();return graph_output
        record('graph',replay)
    if 'compiled' in args.backends:
        compiled=torch.compile(eager,backend='inductor',mode='max-autotune-no-cudagraphs')
        record('compiled',compiled)
        from torch._dynamo.utils import counters
        from torch._inductor import metrics
        result['backends']['compiled']['compile_stats']={'unique_graphs':counters['stats']['unique_graphs'],
            'graph_breaks':dict(counters['graph_break']),'generated_kernels':metrics.generated_kernel_count}
        args.report.write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
