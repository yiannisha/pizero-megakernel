"""Native head-128 joint GQA with cached understanding KV and generation prep."""
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

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from cosmos3_edge.attention import NativeJointAttention
from cosmos3_edge.rope import NativeQKNormRope, mrope_tables


def metrics(actual, expected):
    a, b = actual.double(), expected.double()
    return dict(relative_l2=float((a-b).norm()/b.norm().clamp_min(1e-30)),
                cosine=float((a*b).sum()/(a.norm()*b.norm()).clamp_min(1e-30)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--quick', action='store_true')
    parser.add_argument('--sanitizer', action='store_true')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    torch.set_num_threads(8); torch.manual_seed(231)
    subprocess.run(['make', '-C', str(ROOT/'cosmos3_edge')], check=True)
    library = ROOT/'cosmos3_edge/build/native.so'
    shapes = [(17, 17, 5)] if args.quick else [(17, 17, 5), (6300, 60, 128), (6300, 6300, 128)]
    records = []
    for generated, queries, prefix in shapes:
        q = torch.randn((generated, 16, 128), dtype=torch.bfloat16, device='cuda')
        k = torch.randn((generated, 8, 128), dtype=torch.bfloat16, device='cuda')
        v = torch.randn_like(k)
        ku, vu = torch.randn((prefix, 8, 128), dtype=torch.bfloat16, device='cuda'), torch.randn((prefix, 8, 128), dtype=torch.bfloat16, device='cuda')
        kc = torch.empty((prefix+generated, 8, 128), dtype=torch.bfloat16, device='cuda'); vc = torch.empty_like(kc)
        weights = [(1+torch.randn(128)*.1).bfloat16().cuda() for _ in range(3)]
        cu, su = mrope_tables(torch.randint(0, 30000, (3, prefix)))
        cg, sg = mrope_tables(torch.randint(0, 30000, (3, generated)))
        with NativeQKNormRope(library, None, ku, None, weights[2], cu, su, v=vu,
                             k_output=kc[:prefix], v_output=vc[:prefix]) as conditioning:
            conditioning.forward(); torch.cuda.synchronize()
        frozen_k, frozen_v = kc[:prefix].clone(), vc[:prefix].clone()
        with NativeQKNormRope(library, q, k, weights[0], weights[1], cg, sg, v=v,
                             k_output=kc[prefix:], v_output=vc[prefix:]) as prep:
            selected_q = prep.q_output[-queries:]
            for splits in (1, 8):
                with NativeJointAttention(library, selected_q, kc, vc, splits=splits) as attention:
                    checks = []
                    for epoch in range(3):
                        q.normal_(); k.normal_(); v.normal_()
                        prep.q_output.fill_(float('nan')); kc[prefix:].fill_(float('nan')); vc[prefix:].fill_(float('nan'))
                        if splits > 1:
                            attention.poison_partials()
                        stream = torch.cuda.Stream(); stream.wait_stream(torch.cuda.current_stream())
                        with torch.cuda.stream(stream):
                            prep.forward(); attention.forward()
                        stream.synchronize()
                        torch.testing.assert_close(kc[:prefix], frozen_k, atol=0, rtol=0)
                        torch.testing.assert_close(vc[:prefix], frozen_v, atol=0, rtol=0)
                        reference = F.scaled_dot_product_attention(selected_q.transpose(0, 1).unsqueeze(0),
                            kc.repeat_interleave(2, dim=1).transpose(0, 1).unsqueeze(0),
                            vc.repeat_interleave(2, dim=1).transpose(0, 1).unsqueeze(0), is_causal=False)[0].transpose(0, 1)
                        comparison = metrics(attention.output, reference)
                        assert comparison['relative_l2'] < .01 and comparison['cosine'] >= .999, comparison
                        checks.append(comparison)
                        if splits > 1:
                            counts = attention.read_split_counts()
                            assert counts['epoch'] == attention.epoch
                            assert all((c>>32) == attention.epoch and (c & 0xffffffff) == splits for c in counts['counts'])
                    record = dict(generated_rows=generated, query_rows=queries, understanding_rows=prefix, splits=splits,
                                  checks=checks, static_prefix='bit-exact across three generation updates',
                                  scope='attention only; QK prep is a separate native launch in this test')
                    if not args.sanitizer:
                        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]) as profile:
                            attention.forward(); torch.cuda.synchronize()
                        kernels = [e.name for e in profile.events() if e.device_type == torch.autograd.DeviceType.CUDA]
                        assert len(kernels) == 1 and 'megakernel' in kernels[0], kernels
                        record['attention_cuda_launch_count'] = len(kernels)
                        for _ in range(5):
                            attention.forward()
                        torch.cuda.synchronize(); start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                        times = []
                        for _ in range(30):
                            start.record(); attention.forward(); end.record(); end.synchronize(); times.append(start.elapsed_time(end))
                        record['attention_gpu_ms'] = dict(zip(('p10', 'p50', 'p90'), map(float, np.percentile(times, [10, 50, 90]))))
                    records.append(record); print(json.dumps(record), flush=True)
    if args.output:
        sources = ['cosmos3_edge/attention.py', 'cosmos3_edge/rope.cuh', 'cosmos3_edge/rope.py',
                   'native_pi0/attention.cu', 'native_pi0/attention.py', 'cosmos3_edge/tests/validate_joint_attention.py']
        report = dict(timestamp_utc=datetime.datetime.now(datetime.UTC).isoformat(), gpu=torch.cuda.get_device_name(),
                      scope='Native long-sequence attention and standalone QK prep on synthetic activations; not a complete layer or model',
                      source_sha256={p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in sources}, results=records)
        args.output.write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
