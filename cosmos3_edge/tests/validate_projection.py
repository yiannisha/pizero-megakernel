"""BF16/FP8 resident native projection validation on all Cosmos shapes."""
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
from cosmos3_edge.gemm import NativeProjection
from cosmos3_edge.spec import EdgeSpec


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--quick', action='store_true')
    parser.add_argument('--sanitizer', action='store_true')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    subprocess.run(['make', '-C', str(ROOT/'cosmos3_edge')], check=True)
    library = ROOT/'cosmos3_edge/build/native.so'
    generator = torch.Generator().manual_seed(205)
    records = []
    shapes = [('tail', 3, 130, 192, True)] if args.quick else [
        (name, m, n, k, name in ('o', 'down'))
        for m in (60, 6300) for name, (n, k) in EdgeSpec.from_checkpoint().projections.items()]
    for precision in ('bf16', 'fp8'):
        for name, m, n, k, extras in shapes:
            x = torch.randn((m, k), generator=generator)*.1
            w = torch.randn((n, k), generator=generator)*.02
            scale_a, scale_b = (1., 1.) if precision == 'bf16' else (.125, .03125)
            if precision == 'fp8':
                x, w = (x/scale_a).to(torch.float8_e4m3fn), (w/scale_b).to(torch.float8_e4m3fn)
            else:
                x, w = x.bfloat16(), w.bfloat16()
            bias = (torch.randn(n, generator=generator)*.01).bfloat16() if extras else None
            residual = (torch.randn((m, n), generator=generator)*.01).bfloat16() if extras else None
            with NativeProjection(library, x, w, precision=precision,
                                  activation_scale=scale_a, weight_scale=scale_b,
                                  bias=bias, residual=residual, relu2=name in ('up', 'tail'),
                                  workers=16 if args.quick else 148) as op:
                a = op.a.float(); b = op.b.float()
                expected = (a@b.T)*(scale_a*scale_b)
                if op.residual is not None:
                    expected += op.residual.float()
                if op.bias is not None:
                    expected += op.bias.float()
                if name in ('up', 'tail'):
                    expected = expected.relu().square()
                gold = expected.bfloat16().float()
                maximum = 0.
                for _ in range(3):
                    op.physical_output.fill_(float('nan'))
                    stream = torch.cuda.Stream(); stream.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(stream):
                        op.forward()
                    stream.synchronize()
                    actual = op.physical_output.float()
                    assert torch.isfinite(actual).all()
                    error = float((actual-gold).double().norm()/gold.double().norm().clamp_min(1e-30))
                    maximum = max(error, maximum)
                    assert error <= .001, error
                    assert not torch.count_nonzero(actual[:, n:])
                record = dict(precision=precision, projection=name, m=m, n=n, k=k,
                              bias=extras, residual=extras, relu2=name in ('up', 'tail'),
                              relative_l2_vs_dequantized_reference=maximum, limit=.001)
                if not args.sanitizer:
                    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]) as profile:
                        op.forward(); torch.cuda.synchronize()
                    kernels = [e.name for e in profile.events() if e.device_type == torch.autograd.DeviceType.CUDA]
                    assert len(kernels) == 1 and 'megakernel' in kernels[0], kernels
                    record.update(cuda_launch_count=len(kernels), kernel_symbol=kernels[0])
                    for _ in range(5):
                        op.forward()
                    torch.cuda.synchronize()
                    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    times = []
                    for _ in range(30):
                        start.record(); op.forward(); end.record(); end.synchronize()
                        times.append(start.elapsed_time(end))
                    record['gpu_ms'] = dict(zip(('p10', 'p50', 'p90'), map(float, np.percentile(times, [10, 50, 90]))))
                records.append(record)
                print(json.dumps(record), flush=True)
    if args.output:
        sources = ['cosmos3_edge/gemm.cuh', 'cosmos3_edge/gemm.py', 'cosmos3_edge/native.cu',
                   'cosmos3_edge/tests/validate_projection.py', 'native_pi0/tcgen_probe.cu']
        report = dict(timestamp_utc=datetime.datetime.now(datetime.UTC).isoformat(),
                      scope='Synthetic BF16/FP8 projection operator parity; not model fidelity or calibrated FP8 policy',
                      boundary='Preconverted GPU operands to BF16 output; setup and quantization excluded',
                      environment=dict(gpu=torch.cuda.get_device_name(), torch=torch.__version__, cuda=torch.version.cuda),
                      source_sha256={p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in sources}, results=records)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
