"""Check dense FFN native scheduling, BF16 fidelity, and prepared-FP8 inputs."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from cosmos3_edge.ffn import NativeDenseFFN
from cosmos3_edge.weights import CheckpointWeights
from scripts.validate_dag_matmul import check_diagnostics


def metrics(actual, expected):
    a, b = actual.double(), expected.double()
    return dict(relative_l2=float((a-b).norm()/b.norm().clamp_min(1e-30)),
                cosine=float((a*b).sum()/(a.norm()*b.norm()).clamp_min(1e-30)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--quick', action='store_true')
    parser.add_argument('--sanitizer', action='store_true')
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    torch.set_num_threads(8); torch.manual_seed(207)
    torch.backends.cuda.matmul.allow_tf32 = False
    subprocess.run(['make', '-C', str(ROOT/'cosmos3_edge')], check=True)
    library = ROOT/'cosmos3_edge/build/native.so'
    checkpoint = CheckpointWeights(args.checkpoint) if args.checkpoint else None
    shapes = [(3, 128, 256), (129, 256, 384)]
    if checkpoint is not None:
        shapes = [(60, 2048, 9216)]
    if not args.quick:
        shapes.append((6300, 2048, 9216))
    records = []
    for m, h, intermediate in shapes:
        x = torch.randn((m, h)).bfloat16()
        residual = (torch.randn((m, h))*.1).bfloat16()
        if checkpoint is None:
            wu, wd = (torch.randn((intermediate, h))*.02).bfloat16(), (torch.randn((h, intermediate))*.02).bfloat16()
        else:
            prefix = 'layers.27.mlp_moe_gen' if m == 60 else 'layers.0.mlp_moe_gen'
            wu, wd = [checkpoint.load(prefix+suffix) for suffix in ('.up_proj.weight', '.down_proj.weight')]
        for precision in ('bf16', 'fp8'):
            a_scale, w_scale = (1., 1.) if precision == 'bf16' else (.125, .03125)
            a, w = (x, wu) if precision == 'bf16' else ((x.float()/a_scale).to(torch.float8_e4m3fn), (wu.float()/w_scale).to(torch.float8_e4m3fn))
            nodes = ((m+127)//128)*((h+intermediate)//128)
            with NativeDenseFFN(library, a, w, wd, up_precision=precision, activation_scale=a_scale,
                                weight_scale=w_scale, residual=residual, workers=16 if m<200 else 148,
                                node_order=np.random.default_rng(207).permutation(nodes)) as ffn:
                records_epoch = []
                for epoch in range(3):
                    x = torch.randn((m, h), generator=torch.Generator().manual_seed(207+epoch)).bfloat16()
                    current = x if precision == 'bf16' else (x.float()/a_scale).to(torch.float8_e4m3fn)
                    ffn.up.a[:m].copy_(current)
                    a_gpu, w_gpu = ffn.up.a.float(), ffn.up.b.float()
                    hidden = ((a_gpu@w_gpu.T)*(a_scale*w_scale)).relu().square().bfloat16()
                    expected = (hidden.float()@ffn.down.b.float().T + ffn.down.residual.float()).bfloat16().float()
                    ffn.up.physical_output.fill_(float('nan')); ffn.down.physical_output.fill_(float('nan'))
                    stream = torch.cuda.Stream(); stream.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(stream):
                        ffn.forward()
                    stream.synchronize(); diagnostic = check_diagnostics(ffn.graph)
                    actual = ffn.down.physical_output.float()
                    comparison = metrics(actual, expected)
                    assert comparison['relative_l2'] <= .001, comparison
                    assert torch.isfinite(actual).all()
                    # Independent official-order BF16 reference outside forward.
                    reference = (torch.nn.functional.linear(torch.nn.functional.linear(x.cuda(), wu.cuda()).relu().square(), wd.cuda())+residual.cuda()).float()
                    fidelity = metrics(ffn.output.float(), reference)
                    if precision == 'bf16':
                        assert fidelity['relative_l2'] < .03 and fidelity['cosine'] >= .999, fidelity
                    snapshot = ffn.down.physical_output.clone()
                    ffn.up.forward(); ffn.down.a.copy_(ffn.up.physical_output); ffn.down.forward()
                    torch.testing.assert_close(ffn.down.physical_output, snapshot, atol=0, rtol=0)
                    records_epoch.append(dict(native_math=comparison, bf16_reference=fidelity, split_output='exact'))
                record = dict(m=m, hidden=h, intermediate=intermediate, up_precision=precision,
                              weight_source='checkpoint' if checkpoint else 'synthetic', activations='synthetic', epochs=records_epoch,
                              diagnostics='all nodes once; dependency timestamp ordering; exact epoch counters')
                record['stage_activity'] = []
                for stage in range(2):
                    mask = ffn.graph.node_stage == stage
                    starts, ends = diagnostic['starts'][mask], diagnostic['ends'][mask]
                    owners = diagnostic['owners'][mask]
                    record['stage_activity'].append(dict(stage=stage,
                        active_ctas=int(len(np.unique(owners))),
                        consumer_span_us=float((ends.max()-starts.min())/1000),
                        median_consumer_interval_us=float(np.median((ends-starts).astype(np.float64)))/1000))
                if not args.sanitizer:
                    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]) as profile:
                        ffn.forward(); torch.cuda.synchronize()
                    kernels = [e.name for e in profile.events() if e.device_type == torch.autograd.DeviceType.CUDA]
                    assert len(kernels) == 1 and 'megakernel' in kernels[0], kernels
                    record.update(cuda_launch_count=1, kernel_symbol=kernels[0])
                    for _ in range(5):
                        ffn.forward()
                    torch.cuda.synchronize(); start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    times = []
                    for _ in range(30):
                        start.record(); ffn.forward(); end.record(); end.synchronize(); times.append(start.elapsed_time(end))
                    record['gpu_ms'] = dict(zip(('p10', 'p50', 'p90'), map(float, np.percentile(times, [10, 50, 90]))))
                records.append(record); print(json.dumps(record), flush=True)
    if args.output:
        report = dict(timestamp_utc=datetime.datetime.now(datetime.UTC).isoformat(),
                      scope='Dense native FFN, synthetic activations; not full model parity or calibrated precision selection',
                      gpu=torch.cuda.get_device_name(),
                      source_sha256={p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in (
                          'cosmos3_edge/dense_dag.cuh', 'cosmos3_edge/gemm.cuh', 'cosmos3_edge/gemm.py',
                          'cosmos3_edge/schedule.py', 'cosmos3_edge/ffn.py', 'native_pi0/tcgen_probe.cu',
                          'cosmos3_edge/tests/validate_dense_ffn.py')}, results=records)
        args.output.write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
