"""Run the complete native reference against the same pinned OpenPI model."""
import argparse
from contextlib import nullcontext
import hashlib
import math
import subprocess
import statistics
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
    p.add_argument('--library',type=Path,default=Path('/tmp/pi0_native_model.so'));p.add_argument('--report',type=Path)
    p.add_argument('--benchmark',action='store_true');p.add_argument('--diagnose-attention',action='store_true');p.add_argument('--skip-trace',action='store_true')
    args=p.parse_args();torch.manual_seed(101);torch.set_num_threads(8)
    source_paths=['native_pi0/model_attention.cuh','native_pi0/model.cu','native_pi0/model.py','native_pi0/native.cu','native_pi0/program.py','native_pi0/attention.cu','native_pi0/dense.cu','native_pi0/norm.cu','native_pi0/tcgen_probe.cu','scripts/validate_native_pi0.py']
    hashes={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in source_paths}
    library_hash=hashlib.sha256(args.library.read_bytes()).hexdigest()
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
    if args.diagnose_attention:
        q,k,v=[captures['visiondebug.'+name][0].view(1,256,16,72).transpose(1,2) for name in ('q','k','v')]
        target=captures['visiondebug.attention'][0]
        def compare_attention(label,result):
            print('Vision oracle rounding',label,nrmse(result.transpose(1,2).reshape_as(target).to(target.dtype),target),flush=True)
        with torch.no_grad():
            p32=torch.softmax((q.float()@k.float().transpose(-1,-2))/math.sqrt(72),dim=-1)
            compare_attention('FP32 probabilities',p32@v.float())
            scores32=(q.float()@k.float().transpose(-1,-2))/math.sqrt(72)
            unnormalized=torch.exp(scores32-scores32.amax(-1,keepdim=True))
            compare_attention('BF16 unnormalized probabilities',(unnormalized.to(v.dtype)@v).float()/unnormalized.sum(-1,keepdim=True))
            from torch.nn.attention import sdpa_kernel,SDPBackend
            for backend in (SDPBackend.FLASH_ATTENTION,SDPBackend.CUDNN_ATTENTION,SDPBackend.MATH):
                with sdpa_kernel(backend):
                    compare_attention(str(backend),torch.nn.functional.scaled_dot_product_attention(q,k,v))
            compare_attention('BF16 normalized probabilities',p32.to(v.dtype)@v)
            scores=(q@k.transpose(-1,-2))/math.sqrt(72)
            compare_attention('eager BF16 scores',torch.softmax(scores.float(),dim=-1).to(v.dtype)@v)
    print('Building complete native π0 program',flush=True);start=time.perf_counter()
    with NativePi0.from_model(model,num_views=args.views,prompt_len=args.prompt,chunk_size=args.chunk,flow_steps=args.steps,library=args.library) as native:
        print('Native setup seconds',time.perf_counter()-start,flush=True)
        with (nullcontext(None) if args.skip_trace else LaunchTrace(args.library)) as trace:
            actual=native.forward(images,state,noise,tokens);torch.cuda.synchronize()
        if trace is not None:assert trace.count==1,trace.count
        diagnostics={}
        if args.diagnose_attention:
            with torch.no_grad():
                from torch.nn.attention import sdpa_kernel,SDPBackend
                def vision_attention(tensors):
                    q,k,v=[value.reshape(args.views,256,16,72).transpose(1,2) for value in tensors]
                    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                        return torch.nn.functional.scaled_dot_product_attention(q,k,v).transpose(1,2).reshape(-1,1152)
                native_vision=vision_attention([native.landmarks['visiondebug.'+part] for part in ('q','k','v')])
                gold_vision=vision_attention([torch.cat(captures['visiondebug.'+part],dim=0) for part in ('q','k','v')])
                native_target=native.landmarks['visiondebug.attention'][:args.views*256]
                gold_target=torch.cat(captures['visiondebug.attention'],dim=0).reshape(-1,1152)
                diagnostics['vision_native_vs_sdpa_on_native_qkv']=nrmse(native_target,native_vision)
                diagnostics['vision_gold_vs_sdpa_on_gold_qkv']=nrmse(gold_target,gold_vision)
                prefix_rows=args.views*256+args.prompt
                rotary=pg.paligemma.language_model.rotary_emb
                positions=torch.arange(prefix_rows,device=images.device)[None]
                cos,sin=rotary(torch.empty((1,prefix_rows,256),device=images.device,dtype=torch.bfloat16),positions)
                def prefix_attention(tensors):
                    q,k,v=[value.reshape(1,prefix_rows,heads,256).transpose(1,2)
                           for value,heads in zip(tensors,(8,1,1))]
                    def rotate(x):return torch.cat((-x[...,128:],x[...,:128]),dim=-1)
                    q=q*cos.unsqueeze(1)+rotate(q)*sin.unsqueeze(1)
                    k=k*cos.unsqueeze(1)+rotate(k)*sin.unsqueeze(1)
                    k=k.repeat_interleave(8,dim=1);v=v.repeat_interleave(8,dim=1)
                    score=(q@k.transpose(-1,-2))*.0625
                    prob=torch.softmax(score,dim=-1,dtype=torch.float32).to(q.dtype)
                    return (prob@v).transpose(1,2).reshape(-1,2048)
                native_prefix=prefix_attention([native.landmarks['debug.'+part] for part in ('q','k','v')])
                gold_prefix=prefix_attention([captures['debug.'+part][0] for part in ('q','k','v')])
                native_target=native.landmarks['debug.attention'][:prefix_rows]
                gold_target=captures['debug.attention'][0].reshape(-1,2048)
                diagnostics['prefix_native_vs_eager_on_native_qkv']=nrmse(native_target,native_prefix)
                diagnostics['prefix_gold_vs_eager_on_gold_qkv']=nrmse(gold_target,gold_prefix)
                print('Attention input-matched diagnostics',diagnostics,flush=True)
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
        error=nrmse(actual,gold);print('COMPLETE π0 native/eager action NRMSE',error,'launch trace',None if trace is None else trace.count,flush=True)
        scale=gold.float().square().mean().sqrt().clamp_min(1e-12)
        normalized_max=float((actual-gold).abs().max()/scale)
        cosine=float(torch.nn.functional.cosine_similarity(actual.float().flatten(),gold.float().flatten(),dim=0))
        limits={'nrmse':2*torch.finfo(torch.bfloat16).eps,'normalized_max_abs':8*torch.finfo(torch.bfloat16).eps,'cosine_min':.999}
        passed=error<=limits['nrmse'] and normalized_max<=limits['normalized_max_abs'] and cosine>=limits['cosine_min']
        report={'views':args.views,'chunk':args.chunk,'flow_steps':args.steps,'prompt_len':args.prompt,
            'cuda_launch_count':None if trace is None else trace.count,'kernel_symbol':None if trace is None else trace.symbol,'action_nrmse':error,'landmark_nrmse':errors,
            'normalized_max_abs':normalized_max,'cosine':cosine,'numerical_limits':limits,'numerical_gate_passed':passed,
            'source_sha256':hashes,'library_sha256':library_hash,'native_program':native.plan,'native_stages':len(native.stages),'native_tasks':native.plan['compute_tiles'],'precision':native.precision_policy,
            'same_weights':True,'scope':'Complete π0 including vision, prefix, state/action/time heads and all Euler steps; random initialized actual OpenPI weights'}
        if diagnostics:report['attention_input_matched_diagnostics']=diagnostics
        if args.benchmark:
            imgs=[images[i].permute(2,0,1)[None] for i in range(args.views)]
            masks=[torch.ones(1,device='cuda',dtype=torch.bool) for _ in range(args.views)]
            token_mask=torch.ones_like(tokens,dtype=torch.bool)
            def measure(fn):
                fn();torch.cuda.synchronize();samples=[];others=set()
                for _ in range(7):
                    query=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True)
                    import os
                    others.update(int(pid) for pid in query.split() if pid.isdigit() and int(pid)!=os.getpid())
                    start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                    start.record();fn();end.record();end.synchronize();samples.append(start.elapsed_time(end))
                return {'median_ms':statistics.median(samples),'samples_ms':samples,'other_compute_pids':sorted(others),'gpu_isolation_verified':not others}
            with torch.no_grad():
                report['native_timing']=measure(lambda:native.forward(images,state,noise,tokens))
                report['eager_timing']=measure(lambda:fixed_sample(model,imgs,masks,tokens,token_mask,state,noise,steps=args.steps))
            print('Complete model medians ms native/eager',report['native_timing']['median_ms'],report['eager_timing']['median_ms'],flush=True)
        # Verify replay consumes changed resident inputs and retains native output
        # across repeated executions, without any host reset of dependency flags.
        original=actual.clone();native.forward(images,state,noise,tokens);torch.cuda.synchronize()
        assert torch.equal(native.output,original),'Same-input native replay changed output'
        changed_noise=noise+.125
        with torch.no_grad():
            changed_gold=fixed_sample(model,[images[i].permute(2,0,1)[None] for i in range(args.views)],
                [torch.ones(1,device='cuda',dtype=torch.bool) for _ in range(args.views)],tokens,torch.ones_like(tokens,dtype=torch.bool),state,changed_noise,steps=args.steps)
            changed=native.forward(images,state,changed_noise,tokens);torch.cuda.synchronize()
        replay_error=nrmse(changed,changed_gold)
        report['changed_noise_action_nrmse']=replay_error;report['same_input_replay_bit_exact']=True
        report['changed_input_changes_output']=not torch.equal(changed,original)
        passed=passed and replay_error<=limits['nrmse'] and report['changed_input_changes_output']
        report['numerical_gate_passed']=passed
        if args.report:args.report.write_text(json.dumps(report,indent=2)+'\n')
        assert passed,('Complete action parity',error,normalized_max,cosine)


if __name__=='__main__':main()
