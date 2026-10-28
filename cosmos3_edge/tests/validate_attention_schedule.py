"""Validate QKV → Q/K prep → joint attention in one original VM launch."""
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
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from cosmos3_edge.attention import NativeJointAttention
from cosmos3_edge.attention_schedule import GenerationAttentionSchedule
from cosmos3_edge.gemm import NativeProjection
from cosmos3_edge.rope import NativeQKNormRope, mrope_tables
from cosmos3_edge.tests.validate_rope import official_math
from scripts.validate_dag_matmul import check_diagnostics


def metrics(a, b):
    a, b = a.double(), b.double()
    return dict(relative_l2=float((a-b).norm()/b.norm().clamp_min(1e-30)),
                cosine=float((a*b).sum()/(a.norm()*b.norm()).clamp_min(1e-30)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--quick', action='store_true')
    parser.add_argument('--sanitizer', action='store_true')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    torch.set_num_threads(8); torch.manual_seed(239); torch.cuda.init()
    torch.backends.cuda.matmul.allow_tf32 = False
    subprocess.run(['make', '-C', str(ROOT/'cosmos3_edge')], check=True)
    library = ROOT/'cosmos3_edge/build/native.so'; official = official_math()
    shapes = [(17, 17, 128, 5, 1), (17, 9, 128, 5, 8)] if args.quick else [
        (17, 17, 128, 5, 1), (60, 60, 2048, 128, 8),
        (6300, 60, 2048, 128, 8), (6300, 6300, 2048, 128, 1)]
    records = []
    for rows, queries, hidden, prefix, splits in shapes:
        for precision in ('bf16', 'fp8'):
            a_scale, w_scale = (1., 1.) if precision == 'bf16' else (.125, .03125)
            x = torch.randn((rows, hidden)).bfloat16()
            a = x if precision == 'bf16' else (x.float()/a_scale).to(torch.float8_e4m3fn)
            weights = [(torch.randn((width, hidden))*.02).bfloat16() for width in (2048, 1024, 1024)]
            if precision == 'fp8':
                weights = [(w.float()/w_scale).to(torch.float8_e4m3fn) for w in weights]
            with ExitStack() as stack:
                projections = [stack.enter_context(NativeProjection(library, a, w, precision=precision,
                    activation_scale=a_scale, weight_scale=w_scale)) for w in weights]
                raw_q, raw_k, raw_v = [p.output.view(rows, heads, 128) for p, heads in zip(projections, (16, 8, 8))]
                kc = torch.randn((prefix+rows, 8, 128), dtype=torch.bfloat16, device='cuda'); vc = torch.randn_like(kc)
                frozen_k, frozen_v = kc[:prefix].clone(), vc[:prefix].clone()
                qw, kw = [(1+torch.randn(128)*.1).bfloat16().cuda() for _ in range(2)]
                cos, sin = mrope_tables(torch.randint(0, 30000, (3, rows)))
                prep = stack.enter_context(NativeQKNormRope(library, raw_q, raw_k, qw, kw, cos, sin, v=raw_v,
                                                           k_output=kc[prefix:], v_output=vc[prefix:]))
                attention = stack.enter_context(NativeJointAttention(library, prep.q_output[-queries:], kc, vc, splits=splits))
                batch = 192
                projection_nodes = sum((p.a.shape[0]//128)*(p.b.shape[0]//128) for p in projections)
                nodes = projection_nodes+(rows*24+batch-1)//batch+1+16*((queries+63)//64)*splits
                graph = stack.enter_context(GenerationAttentionSchedule(library, projections, prep, attention,
                    node_order=np.random.default_rng(239).permutation(nodes)))
                checks = []
                for epoch in range(3):
                    current = torch.randn((rows, hidden)).bfloat16()
                    if precision == 'fp8':
                        current = (current.float()/a_scale).to(torch.float8_e4m3fn)
                    for p in projections:
                        p.a[:rows].copy_(current); p.physical_output.fill_(float('nan'))
                    prep.q_output.fill_(float('nan')); kc[prefix:].fill_(float('nan')); vc[prefix:].fill_(float('nan'))
                    attention.output.fill_(float('nan'))
                    if splits > 1:
                        attention.poison_partials()
                    stream = torch.cuda.Stream(); stream.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(stream):
                        graph.forward()
                    stream.synchronize(); check_diagnostics(graph)
                    snapshots = [p.output.clone() for p in projections]+[prep.q_output.clone(), kc.clone(), vc.clone(), attention.output.clone()]
                    torch.testing.assert_close(kc[:prefix], frozen_k, rtol=0, atol=0)
                    torch.testing.assert_close(vc[:prefix], frozen_v, rtol=0, atol=0)
                    if splits > 1:
                        counts = attention.read_split_counts()
                        assert counts['epoch'] == graph._queue.epoch
                        assert all(c>>32 == graph._queue.epoch and (c & 0xffffffff) == splits for c in counts['counts'])
                    # Compare with separately launched components, including epoch
                    # interleaving: the next graph call must safely advance past it.
                    for p in projections:
                        p.forward()
                    prep.forward(); attention.forward(); torch.cuda.synchronize()
                    actual = [p.output for p in projections]+[prep.q_output, kc, vc, attention.output]
                    for value, expected in zip(actual, snapshots):
                        torch.testing.assert_close(value, expected, rtol=0, atol=0)
                    independent = [((p.a.float()@p.b.float().T)*(a_scale*w_scale)).bfloat16()[:rows].view(rows, heads, 128)
                                   for p, heads in zip(projections, (16, 8, 8))]
                    nq, nk = official['Nemotron3DenseVLRMSNorm'](128).cuda(), official['Nemotron3DenseVLRMSNorm'](128).cuda()
                    with torch.no_grad():
                        nq.weight.copy_(qw); nk.weight.copy_(kw)
                        eq, ek = official['apply_rotary_pos_emb_partial'](nq(independent[0]), nk(independent[1]), prep.cos, prep.sin)
                    rk = torch.cat((frozen_k, ek), dim=0).repeat_interleave(2, dim=1)
                    rv = torch.cat((frozen_v, independent[2]), dim=0).repeat_interleave(2, dim=1)
                    reference = F.scaled_dot_product_attention(eq[-queries:].transpose(0, 1)[None], rk.transpose(0, 1)[None], rv.transpose(0, 1)[None], is_causal=False)[0].transpose(0, 1)
                    comparison = metrics(snapshots[-1], reference)
                    assert comparison['relative_l2'] < .03 and comparison['cosine'] >= .999, comparison
                    checks.append(comparison)
                record = dict(rows=rows, queries=queries, hidden=hidden, understanding_rows=prefix,
                              projection_precision=precision, splits=splits, nodes=nodes,
                              queue_plan=graph._queue.plan, checks=checks, split_outputs='bit-exact',
                              static_prefix='bit-exact', node_order='permuted', epoch_interleaving='passed')
                if not args.sanitizer:
                    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]) as profile:
                        graph.forward(); torch.cuda.synchronize()
                    kernels = [e.name for e in profile.events() if e.device_type == torch.autograd.DeviceType.CUDA]
                    assert len(kernels) == 1 and 'GenerationAttentionGlobals' in kernels[0], kernels
                    record.update(cuda_launch_count=1, kernel_symbol=kernels[0])
                    for _ in range(5):
                        graph.forward()
                    torch.cuda.synchronize(); start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    times = []
                    for _ in range(30):
                        start.record(); graph.forward(); end.record(); end.synchronize(); times.append(start.elapsed_time(end))
                    record['gpu_ms'] = dict(zip(('p10', 'p50', 'p90'), map(float, np.percentile(times, [10, 50, 90]))))
                records.append(record); print(json.dumps(record), flush=True)
    if args.output:
        sources = ['cosmos3_edge/attention_schedule.cuh', 'cosmos3_edge/attention_schedule.py', 'cosmos3_edge/rope.cuh',
                   'cosmos3_edge/gemm.cuh', 'cosmos3_edge/dense_dag.cuh', 'native_pi0/attention.cu',
                   'native_pi0/tcgen_probe.cu', 'cosmos3_edge/tests/validate_attention_schedule.py']
        report = dict(timestamp_utc=datetime.datetime.now(datetime.UTC).isoformat(), gpu=torch.cuda.get_device_name(),
                      scope='One-launch QKV/preparation/joint attention on synthetic inputs and weights; not a complete layer or model',
                      source_sha256={p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in sources}, results=records)
        args.output.write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
