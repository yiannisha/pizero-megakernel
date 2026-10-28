"""Validate native Cosmos Q/K prep against pinned official and FlashRT math."""
import argparse
import ast
import ctypes
import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from cosmos3_edge.rope import NativeQKNormRope, mrope_tables
from cosmos3_edge.spec import EdgeSpec, SOURCES

OFFICIAL = 'third_party/cosmos-framework/cosmos_framework/model/generator/reasoner/nemotron_3_dense_vl/nemotron_3_dense_vl.py'


def official_math():
    # Execute the actual pinned math classes/functions without importing the
    # full Transformers training package. The dynamic frequency-update decorator
    # is identity for this fixed default-RoPE configuration.
    tree = ast.parse((ROOT/OFFICIAL).read_text())
    names = {'rotate_half', 'apply_rotary_pos_emb_partial', 'Nemotron3DenseVLRMSNorm', 'MultiModalRotaryEmbedding'}
    selected = [node for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names]
    scope = dict(torch=torch, nn=torch.nn, dynamic_rope_update=lambda fn: fn, Nemotron3DenseVLTextConfig=object)
    exec(compile(ast.Module(body=selected, type_ignores=[]), OFFICIAL, 'exec'), scope)
    return scope


def relative(a, b):
    return float((a.double()-b.double()).norm()/b.double().norm().clamp_min(1e-30))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--quick', action='store_true')
    parser.add_argument('--vectors-per-task', type=int, default=192)
    parser.add_argument('--sanitizer', action='store_true')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    torch.set_num_threads(8); torch.manual_seed(227)
    # Initialize the CUDA context before registering test DSOs under memcheck.
    torch.cuda.init()
    subprocess.run(['make', '-C', str(ROOT/'cosmos3_edge'), 'ROPE_VECTORS='+str(args.vectors_per_task), 'all', 'build/rope_reference.so'], check=True)
    library = ROOT/'cosmos3_edge/build/native.so'
    query = ctypes.CDLL(str(library)).edge_rope_vectors_per_task
    query.argtypes = []; query.restype = ctypes.c_int
    assert query() == args.vectors_per_task
    reference = ctypes.CDLL(str(ROOT/'cosmos3_edge/build/rope_reference.so')).reference_edge_rope
    reference.argtypes = [ctypes.c_void_p]*8+[ctypes.c_int]*5+[ctypes.c_void_p]; reference.restype = ctypes.c_int
    official = official_math(); s = EdgeSpec.from_checkpoint()
    config = json.loads((SOURCES/'transformer__config.json').read_text())
    config = SimpleNamespace(head_dim=128, rope_theta=config['rope_theta'], max_position_embeddings=131072, mrope_section=list(s.rope_axes))
    embedding = official['MultiModalRotaryEmbedding'](config)
    records = []
    for rows in ([3, 17] if args.quick else [3, 60, 6300]):
        positions = torch.randint(0, 30000, (3, rows), dtype=torch.int64)
        cos, sin = mrope_tables(positions)
        official_cos, official_sin = embedding(torch.empty((1, rows, 2048), dtype=torch.bfloat16), positions[:, None, :])
        torch.testing.assert_close(cos, official_cos[0], rtol=0, atol=0)
        torch.testing.assert_close(sin, official_sin[0], rtol=0, atol=0)
        # Strided GPU views reproduce projection buffers with neighboring fields.
        q = torch.randn((rows, 17, 128), device='cuda', dtype=torch.bfloat16)[:, :16]
        k = torch.randn((rows, 9, 128), device='cuda', dtype=torch.bfloat16)[:, :8]
        v = torch.randn_like(k)
        qw, kw = (1+torch.randn(128)*.1).bfloat16().cuda(), (1+torch.randn(128)*.1).bfloat16().cuda()
        for mode in ('fused', 'reference'):
            kc = torch.full((rows+5, 8, 128), -123., dtype=torch.bfloat16, device='cuda')
            vc = torch.full_like(kc, -123.)
            with NativeQKNormRope(library, q, k, qw, kw, cos, sin, v=v,
                                  k_output=kc[2:rows+2], v_output=vc[2:rows+2], rounding=mode) as op:
                maximum = 0.
                for epoch in range(3):
                    q.normal_(); k.normal_(); v.normal_()
                    op.q_output.fill_(float('nan')); op.k_output.fill_(float('nan')); op.v_output.fill_(float('nan'))
                    stream = torch.cuda.Stream(); stream.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(stream):
                        op.forward()
                    stream.synchronize()
                    normq, normk = official['Nemotron3DenseVLRMSNorm'](128).cuda(), official['Nemotron3DenseVLRMSNorm'](128).cuda()
                    with torch.no_grad():
                        normq.weight.copy_(qw); normk.weight.copy_(kw)
                        eq, ek = official['apply_rotary_pos_emb_partial'](normq(q), normk(k), op.cos, op.sin)
                    error = max(relative(op.q_output, eq), relative(op.k_output, ek)); maximum = max(error, maximum)
                    assert error < (.001 if mode == 'reference' else .005), error
                    torch.testing.assert_close(op.v_output, v, atol=0, rtol=0)
                    assert torch.all(kc[:2] == -123.) and torch.all(kc[rows+2:] == -123.)
                    assert torch.all(vc[:2] == -123.) and torch.all(vc[rows+2:] == -123.)
                    if mode == 'fused':
                        fq, fk = torch.empty_like(op.q_output), torch.empty_like(op.k_output)
                        error = reference(q.data_ptr(), k.data_ptr(), qw.data_ptr(), kw.data_ptr(), op.cos.data_ptr(), op.sin.data_ptr(),
                                          fq.data_ptr(), fk.data_ptr(), rows, 16, 8, q.stride(0), k.stride(0), torch.cuda.current_stream().cuda_stream)
                        assert error == 0, error
                        torch.testing.assert_close(op.q_output, fq, rtol=0, atol=0)
                        torch.testing.assert_close(op.k_output, fk, rtol=0, atol=0)
                record = dict(rows=rows, rounding=mode, q_heads=16, kv_heads=8, vectors_per_task=query(),
                              official_mrope_tables='exact', relative_l2_vs_official=maximum,
                              flashrt_fused='exact' if mode == 'fused' else 'not applicable',
                              v_copy='exact', cache_prefix_tail='preserved', epochs=3)
                if not args.sanitizer:
                    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]) as profile:
                        op.forward(); torch.cuda.synchronize()
                    kernels = [e.name for e in profile.events() if e.device_type == torch.autograd.DeviceType.CUDA]
                    assert len(kernels) == 1 and 'megakernel' in kernels[0], kernels
                    record['cuda_launch_count'] = len(kernels)
                    for _ in range(5):
                        op.forward()
                    torch.cuda.synchronize(); start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    times = []
                    for _ in range(30):
                        start.record(); op.forward(); end.record(); end.synchronize(); times.append(start.elapsed_time(end))
                    record['gpu_ms'] = dict(zip(('p10', 'p50', 'p90'), map(float, np.percentile(times, [10, 50, 90]))))
                records.append(record); print(json.dumps(record), flush=True)
        # Separate understanding-key normalization must not mutate causal keys.
        original = k.clone()
        with NativeQKNormRope(library, None, k, None, kw, cos, sin, workers=16) as key_only:
            _, result, _ = key_only.forward()
            torch.cuda.synchronize(); torch.testing.assert_close(k, original, rtol=0, atol=0)
            with torch.no_grad():
                _, expected = official['apply_rotary_pos_emb_partial'](normq(q), normk(k), key_only.cos, key_only.sin)
            assert relative(result, expected) < .005
    if args.output:
        sources = ['cosmos3_edge/rope.cuh', 'cosmos3_edge/rope.py', 'cosmos3_edge/tests/validate_rope.py',
                   'cosmos3_edge/tests/rope_reference.cu', OFFICIAL, 'third_party/FlashRT/csrc/kernels/cosmos3_edge_misc.cu']
        report = dict(timestamp_utc=datetime.datetime.now(datetime.UTC).isoformat(), gpu=torch.cuda.get_device_name(),
                      scope='Standalone native QK preparation on synthetic activations; not attention/layer/model fidelity',
                      source_sha256={p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in sources}, results=records)
        args.output.write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
