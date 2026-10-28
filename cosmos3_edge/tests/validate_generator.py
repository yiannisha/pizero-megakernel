"""Native complete generation transformer, given prepared hidden states/KV."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch
import torch.nn.functional as F

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from cosmos3_edge.model import NativeGeneratorTransformer
from cosmos3_edge.weights import CheckpointWeights
from cosmos3_edge.tests.validate_rope import official_math
from scripts.validate_dag_matmul import check_diagnostics


def metrics(a,b):
    a,b=a.double(),b.double()
    return dict(relative_l2=float((a-b).norm()/b.norm().clamp_min(1e-30)),cosine=float((a*b).sum()/(a.norm()*b.norm()).clamp_min(1e-30)))


def rms(x,w):
    y=x.float();return (y*torch.rsqrt(y.square().mean(-1,keepdim=True)+1e-5)*w.float()).bfloat16()


@torch.no_grad()
def reference(model,source,official):
    hidden=source;checks=[]
    for layer in model.layers:
        x=rms(hidden,layer['norm1'].gamma)
        q,k,v=[F.linear(x,p.b).view(model.rows,heads,128) for p,heads in zip(layer['projections'],(16,8,8))]
        prep=layer['preparation'];q,k=official['apply_rotary_pos_emb_partial'](rms(q,prep.qw),rms(k,prep.kw),prep.cos,prep.sin)
        count=layer['prefix_rows'];kc=torch.cat((layer['cache_k'][:count],k));vc=torch.cat((layer['cache_v'][:count],v))
        out=F.scaled_dot_product_attention(q.transpose(0,1)[None],kc.repeat_interleave(2,dim=1).transpose(0,1)[None],
            vc.repeat_interleave(2,dim=1).transpose(0,1)[None],is_causal=False)[0].transpose(0,1).reshape(model.rows,2048)
        hidden=hidden+F.linear(out,layer['output_projection'].b)
        hidden=hidden+F.linear(F.relu(F.linear(rms(hidden,layer['norm2'].gamma),layer['up'].b)).square(),layer['down'].b)
        record=dict(layer=layer['index'],hidden=metrics(layer['down'].output[:model.rows],hidden),
                    generated_k=metrics(layer['cache_k'][count:],k),generated_v=metrics(layer['cache_v'][count:],v))
        assert record['hidden']['relative_l2']<.03 and record['hidden']['cosine']>=.999,record
        checks.append(record)
    expected=rms(hidden,model.final_norm.gamma)
    return expected,checks


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,default=ROOT/'.cache/cosmos3-edge')
    parser.add_argument('--layers',type=int,choices=(1,2,28),default=1)
    parser.add_argument('--rows',type=int,default=17)
    parser.add_argument('--prefix',type=int,default=5)
    parser.add_argument('--splits',type=int,default=1)
    parser.add_argument('--sanitizer',action='store_true')
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    torch.set_num_threads(8);torch.manual_seed(257);torch.cuda.init();torch.backends.cuda.matmul.allow_tf32=False
    subprocess.run(['make','-C',str(ROOT/'cosmos3_edge')],check=True)
    weights=CheckpointWeights(args.checkpoint);official=official_math()
    indices=[0] if args.layers==1 else [0,27] if args.layers==2 else list(range(28))
    hidden=(torch.randn((args.rows,2048),device='cuda')*.1).bfloat16()
    kv=[(torch.randn((args.prefix,8,128),device='cuda',dtype=torch.bfloat16),torch.randn((args.prefix,8,128),device='cuda',dtype=torch.bfloat16)) for _ in indices]
    positions=torch.randint(0,30000,(3,args.rows),dtype=torch.int64)
    with NativeGeneratorTransformer(ROOT/'cosmos3_edge/build/native.so',weights,hidden,kv,positions,
        layer_indices=indices,attention_splits=args.splits,permute_nodes=True) as model:
        checks=[]
        frozen=[(l['cache_k'][:args.prefix].clone(),l['cache_v'][:args.prefix].clone()) for l in model.layers]
        for replay in range(3):
            source=(torch.randn_like(hidden)*.1).bfloat16();model.inputs.copy_(source)
            for layer in model.layers:
                for op in layer['projections']+[layer['output_projection'],layer['up'],layer['down']]:op.output[:args.rows].fill_(float('nan'))
                layer['norm1'].output[:args.rows].fill_(float('nan'));layer['norm2'].output[:args.rows].fill_(float('nan'))
                layer['preparation'].q_output.fill_(float('nan'));layer['attention_buffer'][:args.rows].fill_(float('nan'))
                layer['cache_k'][args.prefix:].fill_(float('nan'));layer['cache_v'][args.prefix:].fill_(float('nan'))
                if args.splits>1:layer['attention'].poison_partials()
            model.output.fill_(float('nan'))
            stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):model.forward()
            stream.synchronize();check_diagnostics(model.engine)
            torch.testing.assert_close(model.inputs,source,rtol=0,atol=0)
            for layer,(k,v) in zip(model.layers,frozen):
                torch.testing.assert_close(layer['cache_k'][:args.prefix],k,rtol=0,atol=0)
                torch.testing.assert_close(layer['cache_v'][:args.prefix],v,rtol=0,atol=0)
                if args.splits>1:
                    counters=layer['attention'].read_split_counts()
                    assert all(c>>32==model.engine._queue.epoch and c&0xffffffff==args.splits for c in counters['counts'])
            expected,per_layer=reference(model,source,official)
            comparison=metrics(model.output,expected)
            assert comparison['relative_l2']<.03 and comparison['cosine']>=.999,comparison
            checks.append(dict(final=comparison,per_layer=per_layer));print(f'Generator {args.layers} layers, {args.rows} rows, replay {replay}: {comparison}',flush=True)
        record=dict(layers=indices,rows=args.rows,understanding_rows=args.prefix,splits=args.splits,
                    checks=checks,nodes=model.engine._queue.nodes,queue_plan=model.engine._queue.plan,
                    static_prefix='bit-exact',node_order='permuted',input_unchanged=True)
        if not args.sanitizer:
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as profile:
                model.forward();torch.cuda.synchronize()
            kernels=[e.name for e in profile.events() if e.device_type==torch.autograd.DeviceType.CUDA]
            assert len(kernels)==1 and 'EdgeEngineGlobals' in kernels[0],kernels
            record.update(cuda_launch_count=1,kernel_symbol=kernels[0])
            for _ in range(5):model.forward()
            torch.cuda.synchronize();start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            times=[]
            for _ in range(30):
                start.record();model.forward();end.record();end.synchronize();times.append(start.elapsed_time(end))
            record['gpu_ms']=dict(zip(('p10','p50','p90'),map(float,np.percentile(times,[10,50,90]))))
        print(json.dumps(record),flush=True)
    if args.output:
        sources=['cosmos3_edge/model.py','cosmos3_edge/engine.cuh','cosmos3_edge/engine.py','cosmos3_edge/gemm.cuh',
                 'cosmos3_edge/gemm.py','native_pi0/norm.cu','native_pi0/normalization.py','native_pi0/tcgen_probe.cu',
                 'cosmos3_edge/tests/validate_generator.py']
        report=dict(timestamp_utc=datetime.datetime.now(datetime.UTC).isoformat(),gpu=torch.cuda.get_device_name(),
                    scope='BF16 generation transformer with checkpoint weights, synthetic hidden states and prepared synthetic KV; not full denoising or application fidelity',
                    source_sha256={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in sources},results=[record])
        args.output.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
