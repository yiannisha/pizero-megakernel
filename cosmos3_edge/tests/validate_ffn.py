"""Validate one-launch native ReLU² FFN scheduling on synthetic Cosmos shapes."""
import argparse
import ctypes
import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from cosmos3_edge.ffn import NativeFFN
from native_pi0.primitives import ResidentNVFP4
from native_pi0.quantization import PackedNVFP4, pack_nvfp4, swizzle_scales
from native_pi0.launch_trace import LaunchTrace
from scripts.validate_dag_matmul import check_diagnostics, check_packed_storage


def physical(packed):
    return PackedNVFP4(packed.data, packed.scales, packed.global_scale,
                      (packed.data.shape[0], packed.data.shape[1]*2)).dequantize()


def stats(values):
    return dict(zip(('p10_ms', 'p50_ms', 'p90_ms'), map(float, np.percentile(values, [10, 50, 90]))))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--quick', action='store_true')
    parser.add_argument('--checkpoint', type=Path, help='Pinned local checkpoint; activations remain synthetic')
    parser.add_argument('--sanitizer', action='store_true', help='Skip CUPTI and timings under compute-sanitizer')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    torch.set_num_threads(8)
    subprocess.run(['make', '-C', str(ROOT/'cosmos3_edge'), 'build/native_test.so'], check=True)
    library = ROOT/'cosmos3_edge/build/native_test.so'
    module = ctypes.CDLL(str(library))
    oracle = module.reference_fp4_output
    oracle.argtypes = [ctypes.c_void_p]*3+[ctypes.c_int]*2+[ctypes.c_float, ctypes.c_int]
    oracle.restype = ctypes.c_int
    split = module.validation_split_dag_matmul
    split.argtypes = [ctypes.c_void_p]*2
    split.restype = ctypes.c_int
    rng = np.random.default_rng(20261028)
    shapes = [(3, 128, 256, 4), (129, 256, 384, 16)]
    if not args.quick:
        shapes += [(60, 2048, 9216, 64), (6300, 2048, 9216, 148)]
    checkpoint = None
    if args.checkpoint:
        from cosmos3_edge.weights import CheckpointWeights
        checkpoint = CheckpointWeights(args.checkpoint)
        shapes = [(60, 2048, 9216, 64)]
        if not args.quick:
            shapes.append((6300, 2048, 9216, 148))
    records = []
    for m, hidden, intermediate, workers in shapes:
        activation = pack_nvfp4(rng.normal(size=(m, hidden)), global_scale=.125, row_alignment=128, column_alignment=64)
        up = pack_nvfp4(rng.normal(0, .02, size=(intermediate, hidden)), global_scale=.125, scale_policy='ceil')
        down = pack_nvfp4(rng.normal(0, .02, size=(hidden, intermediate)), global_scale=.125, scale_policy='ceil')
        reference_weights = None
        if checkpoint is not None:
            layer = 27 if m == 60 else 0
            prefix = f'layers.{layer}.mlp_moe_gen'
            reference_weights = [checkpoint.load(prefix+suffix).float().numpy()
                                 for suffix in ('.up_proj.weight', '.down_proj.weight')]
            up, down = [pack_nvfp4(w, global_scale=.125, scale_policy='ceil') for w in reference_weights]
        residual = torch.from_numpy(rng.normal(0, .1, size=(m, hidden)).astype(np.float32)).bfloat16()
        nodes = (activation.data.shape[0]//128)*((intermediate+hidden)//128)
        order = rng.permutation(nodes)
        with NativeFFN(library, activation, up, down, hidden_scale=1., residual=residual,
                       workers=workers, node_order=order) as ffn:
            maximum = 0.
            bf16_comparisons = []
            for epoch in range(3):
                activation = pack_nvfp4(rng.normal(size=(m, hidden)), global_scale=.125, row_alignment=128, column_alignment=64)
                ffn.up.a.copy_(torch.from_numpy(activation.data))
                ffn.up.sa.copy_(torch.from_numpy(swizzle_scales(activation.scales)))
                # Poison all producer storage: stale outputs must not satisfy dependencies.
                ffn.up.output.fill_(255); ffn.up.output_scales.fill_(255)
                ffn.down.output.fill_(float('nan'))
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    ffn.forward()
                stream.synchronize()
                check_diagnostics(ffn.graph)
                for stage in (ffn.up, ffn.down):
                    check_packed_storage(stage.a); check_packed_storage(stage.b)
                check_packed_storage(ffn.up.output)
                with ResidentNVFP4(library, activation, up, workers=workers) as plain:
                    projection = plain.forward().cpu().numpy()
                cpu_projection = physical(activation) @ physical(up).T
                np.testing.assert_allclose(projection, cpu_projection, atol=1e-5*max(1., float(np.abs(cpu_projection).max())), rtol=3e-5)
                packed = np.empty_like(ffn.up.output.cpu().numpy())
                scales = np.empty((packed.shape[0], intermediate//16), np.uint8)
                error = oracle(projection.ctypes.data, packed.ctypes.data, scales.ctypes.data,
                               packed.shape[0], intermediate, 1., 3)
                assert error == 0, error
                np.testing.assert_array_equal(ffn.up.output.cpu().numpy(), packed)
                np.testing.assert_array_equal(ffn.up.output_scales.cpu().numpy(), swizzle_scales(scales))
                reference_hidden = PackedNVFP4(packed, scales, np.float32(1.), (packed.shape[0], intermediate))
                expected = physical(reference_hidden) @ physical(down).T
                expected[:m] += residual.float().numpy()
                actual = ffn.down.output.float().cpu().numpy()
                gold = torch.from_numpy(expected).bfloat16().float().numpy()
                error = float(np.linalg.norm(actual.astype(np.float64)-gold)/max(np.linalg.norm(gold.astype(np.float64)), 1e-30))
                maximum = max(maximum, error)
                assert np.isfinite(actual).all() and error <= .005, error
                np.testing.assert_array_equal(actual[m:], 0.)
                if reference_weights is not None:
                    # Framework reference math is outside the native measured boundary.
                    x = torch.from_numpy(physical(activation)[:m]).bfloat16().cuda()
                    wu, wd = [torch.from_numpy(w).bfloat16().cuda() for w in reference_weights]
                    y = torch.relu(torch.nn.functional.linear(x, wu)).square()
                    bf16 = (torch.nn.functional.linear(y, wd)+residual.cuda()).float().cpu().numpy()
                    diff = actual[:m].astype(np.float64)-bf16.astype(np.float64)
                    relative = float(np.linalg.norm(diff)/max(np.linalg.norm(bf16.astype(np.float64)), 1e-30))
                    cosine = float(np.sum(actual[:m].astype(np.float64)*bf16)/max(np.linalg.norm(actual[:m].astype(np.float64))*np.linalg.norm(bf16.astype(np.float64)), 1e-30))
                    bf16_comparisons.append(dict(relative_l2=relative, cosine=cosine,
                        initial_gate_passed=relative<.03 and cosine>=.999))
                snapshot = ffn.down.output.clone()
                error = split(ffn.graph.handle, torch.cuda.current_stream().cuda_stream)
                assert error == 0, error
                torch.testing.assert_close(ffn.down.output, snapshot, atol=0, rtol=0)
            record = dict(m=m, hidden=hidden, intermediate=intermediate, workers=workers,
                          epochs=3, permuted_dag=True, producer_bytes='exact', producer_scales='exact',
                          down_vs_split='exact', final_nrmse=maximum, final_nrmse_limit=.005,
                          nodes=nodes, queue_plan=ffn.graph._queue.plan,
                          weights='pinned checkpoint' if checkpoint is not None else 'synthetic',
                          activations='synthetic', bf16_comparisons=bf16_comparisons)
            if not args.sanitizer:
                with LaunchTrace(library) as trace:
                    ffn.forward(); torch.cuda.synchronize()
                assert trace.count == 1 and 'DagMatmul' in trace.symbol, (trace.count, trace.symbol)
                check_diagnostics(ffn.graph)
                record.update(cuda_launch_count=trace.count, kernel_symbol=trace.symbol)
                for _ in range(5):
                    ffn.forward()
                torch.cuda.synchronize()
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                gpu, wall = [], []
                for _ in range(30):
                    torch.cuda.synchronize(); t = time.perf_counter_ns()
                    start.record(); ffn.forward(); end.record(); end.synchronize()
                    gpu.append(start.elapsed_time(end)); wall.append((time.perf_counter_ns()-t)/1e6)
                record.update(gpu=stats(gpu), synchronized_wall=stats(wall))
            records.append(record)
            print(json.dumps(record), flush=True)
    if args.output:
        sources = ['cosmos3_edge/native.cu', 'cosmos3_edge/ffn.py', 'cosmos3_edge/Makefile',
                   'cosmos3_edge/tests/validate_ffn.py', 'native_pi0/nvfp4_probe.cu',
                   'native_pi0/tcgen_probe.cu', 'native_pi0/dag_matmul.cu',
                   'native_pi0/dag_scheduler.cuh', 'native_pi0/fp4_output_reference.cu']
        report = dict(timestamp_utc=datetime.datetime.now(datetime.UTC).isoformat(),
                      scope='One-launch ReLU2 FFN with synthetic activations; checkpoint weights when explicitly selected; not full model fidelity',
                      boundary='GPU packed input and weights to BF16 output; includes residual; excludes setup and input packing',
                      environment=dict(gpu=torch.cuda.get_device_name(), torch=torch.__version__, cuda=torch.version.cuda),
                      source_sha256={p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in sources}, results=records)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
