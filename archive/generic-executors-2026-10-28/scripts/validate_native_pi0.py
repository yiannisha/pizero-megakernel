"""Run the complete native reference against the same pinned OpenPI model."""
import argparse
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace,MethodType

import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'third_party/openpi/src')]
from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
from benchmarks.openpi_backend import capture_prefix,capture_suffix,fixed_sample
from native_pi0.model import NativePi0
from native_pi0.launch_trace import LaunchTrace


def nrmse(a,b):return float((a.float()-b.float()).square().mean().sqrt()/b.float().square().mean().sqrt().clamp_min(1e-12))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--views',type=int,default=3);p.add_argument('--chunk',type=int,default=50)
    p.add_argument('--steps',type=int,default=10);p.add_argument('--prompt',type=int,default=0)
    p.add_argument('--library',type=Path,default=Path('/tmp/native_pi0_reference.so'));p.add_argument('--report',type=Path)
    p.add_argument('--benchmark',action='store_true')
    args=p.parse_args();torch.manual_seed(101);torch.set_num_threads(8)
    config=SimpleNamespace(pi05=False,paligemma_variant='gemma_2b',action_expert_variant='gemma_300m',dtype='bfloat16',
        action_dim=32,action_horizon=args.chunk,pytorch_compile_mode=None)
    print('Constructing actual pinned OpenPI model',flush=True)
    with torch.device('cuda'):model=PI0Pytorch(config).eval()
    model.paligemma_with_expert.paligemma.language_model.config._attn_implementation='eager'
    model.paligemma_with_expert.gemma_expert.model.config._attn_implementation='eager'
    model.embed_prefix=MethodType(capture_prefix,model);model.embed_suffix=MethodType(capture_suffix,model)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    images=torch.randn((args.views,224,224,3),device='cuda');state=torch.randn((1,32),device='cuda')
    noise=torch.randn((1,args.chunk,32),device='cuda');tokens=torch.randint(0,257152,(1,args.prompt),device='cuda')
    captures={};hooks=[]
    def hook(name):
        def capture(module,inputs,output):
            value=output[0] if isinstance(output,tuple) else output
            captures.setdefault(name,[]).append(value.detach().clone())
        return capture
    pg=model.paligemma_with_expert
    hooks.append(pg.paligemma.vision_tower.vision_model.embeddings.register_forward_hook(hook('patch')))
    for i,layer in enumerate(pg.paligemma.vision_tower.vision_model.encoder.layers):hooks.append(layer.register_forward_hook(hook(f'vision.{i}')))
    for i,layer in enumerate(pg.paligemma.language_model.layers):hooks.append(layer.register_forward_hook(hook(f'prefix.{i}')))
    for i,layer in enumerate(pg.gemma_expert.model.layers):hooks.append(layer.register_forward_hook(hook(f'expert.{i}')))
    first=pg.paligemma.language_model.layers[0]
    for name,module in [('norm1',first.input_layernorm),('norm2',first.post_attention_layernorm),('q',first.self_attn.q_proj),
        ('k',first.self_attn.k_proj),('v',first.self_attn.v_proj),('o',first.self_attn.o_proj),('down',first.mlp.down_proj)]:
        hooks.append(module.register_forward_hook(hook('debug.'+name)))
    for name,module in [('input',first.input_layernorm),('residual1',first.post_attention_layernorm),('attention',first.self_attn.o_proj),('activation',first.mlp.down_proj)]:
        hooks.append(module.register_forward_pre_hook(lambda module,inputs,name=name:hook('debug.'+name)(module,inputs,inputs)))
    first=pg.paligemma.vision_tower.vision_model.encoder.layers[0]
    for name,module in [('norm1',first.layer_norm1),('norm2',first.layer_norm2),('q',first.self_attn.q_proj),('k',first.self_attn.k_proj),
        ('v',first.self_attn.v_proj),('o',first.self_attn.out_proj),('fc1',first.mlp.fc1),('down',first.mlp.fc2)]:
        hooks.append(module.register_forward_hook(hook('visiondebug.'+name)))
    for name,module in [('input',first.layer_norm1),('residual1',first.layer_norm2),('attention',first.self_attn.out_proj),('activation',first.mlp.fc2)]:
        hooks.append(module.register_forward_pre_hook(lambda module,inputs,name=name:hook('visiondebug.'+name)(module,inputs,inputs)))
    print('Running complete OpenPI eager reference',flush=True)
    with torch.no_grad():
        gold=fixed_sample(model,[images[i].permute(2,0,1)[None] for i in range(args.views)],
            [torch.ones(1,device='cuda',dtype=torch.bool) for _ in range(args.views)],tokens,
            torch.ones_like(tokens,dtype=torch.bool),state,noise,steps=args.steps)
    torch.cuda.synchronize()
    for h in hooks:h.remove()
    print('Building complete native π0 program',flush=True);start=time.perf_counter()
    with NativePi0.from_model(model,num_views=args.views,prompt_len=args.prompt,chunk_size=args.chunk,flow_steps=args.steps,library=args.library) as native:
        print('Native setup seconds',time.perf_counter()-start,flush=True)
        with LaunchTrace(args.library) as trace:
            actual=native.forward(images,state,noise,tokens);torch.cuda.synchronize()
        assert trace.count==1,trace.count
        errors={}
        for name,value in native.landmarks.items():
            if name.startswith('actions.'):continue
            if name.startswith('expert.'):
                _,step,layer=name.split('.');expected=captures[f'expert.{layer}'][int(step)]
            elif name.startswith(('vision.','visiondebug.')) or name=='patch':expected=torch.cat(captures[name],dim=0)
            else:expected=captures[name][0]
            expected=expected.reshape(-1,expected.shape[-1]);value=value.reshape(-1,value.shape[-1])
            errors[name]=nrmse(value,expected.to(value.dtype))
            if name=='patch' or name.startswith(('debug.','visiondebug.')) or name.endswith('.0') or name.endswith('.17') or name.endswith('.26'):print(name,errors[name],flush=True)
        error=nrmse(actual,gold);print('COMPLETE π0 native/eager action NRMSE',error,'one launch',trace.count,flush=True)
        scale=gold.float().square().mean().sqrt().clamp_min(1e-12)
        normalized_max=float((actual-gold).abs().max()/scale)
        cosine=float(torch.nn.functional.cosine_similarity(actual.float().flatten(),gold.float().flatten(),dim=0))
        limits={'nrmse':2*torch.finfo(torch.bfloat16).eps,'normalized_max_abs':8*torch.finfo(torch.bfloat16).eps,'cosine_min':.999}
        passed=error<=limits['nrmse'] and normalized_max<=limits['normalized_max_abs'] and cosine>=limits['cosine_min']
        report={'views':args.views,'chunk':args.chunk,'flow_steps':args.steps,'prompt_len':args.prompt,
            'cuda_launch_count':trace.count,'kernel_symbol':trace.symbol,'action_nrmse':error,'landmark_nrmse':errors,
            'normalized_max_abs':normalized_max,'cosine':cosine,'numerical_limits':limits,'numerical_gate_passed':passed,
            'native_stages':len(native.stages),'native_tasks':native.queue.nodes,'precision':native.precision_policy,
            'same_weights':True,'scope':'Complete π0 including vision, prefix, state/action/time heads and all Euler steps; random initialized actual OpenPI weights'}
        if args.benchmark:
            from scripts.benchmark_transformer_workers import GpuMonitor
            imgs=[images[i].permute(2,0,1)[None] for i in range(args.views)]
            masks=[torch.ones(1,device='cuda',dtype=torch.bool) for _ in range(args.views)]
            token_mask=torch.ones_like(tokens,dtype=torch.bool)
            with torch.no_grad(),GpuMonitor() as monitor:
                report['native_timing']=monitor.measure(lambda:native.forward(images,state,noise,tokens))
                report['eager_timing']=monitor.measure(lambda:fixed_sample(model,imgs,masks,tokens,token_mask,state,noise,steps=args.steps))
            print('Complete model medians ms native/eager',report['native_timing']['median_ms'],report['eager_timing']['median_ms'],flush=True)
        if args.report:args.report.write_text(json.dumps(report,indent=2)+'\n')
        assert passed,('Complete action parity',error,normalized_max,cosine)


if __name__=='__main__':main()
