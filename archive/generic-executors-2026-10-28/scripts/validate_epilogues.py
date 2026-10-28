"""Validate fused BF16 NVFP4 epilogues in the original native VM."""
import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from native_pi0.primitives import ResidentNVFP4
from native_pi0.quantization import pack_nvfp4


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--consumer-warps',type=int,choices=(4,16),default=4)
    parser.add_argument('--k-tile',type=int,choices=(64,128,256),default=64)
    parser.add_argument('--profile',action='store_true')
    args=parser.parse_args()
    torch.set_num_threads(8)
    rng=np.random.default_rng(47)
    with tempfile.TemporaryDirectory() as directory:
        library=Path(directory)/'fp4.so'
        subprocess.run(['nvcc','-O3','-std=c++20','-arch=sm_100a','-DKITTENS_HOPPER','-DKITTENS_BLACKWELL',
            '-DNATIVE_COMPUTE_CONSUMER_WARPS='+str(args.consumer_warps),
            '-DNATIVE_FP4_K_TILE='+str(args.k_tile),
            '--expt-relaxed-constexpr','--expt-extended-lambda',
            '-I'+str(ROOT/'third_party/Megakernels/include'),
            '-I'+str(ROOT/'third_party/Megakernels/ThunderKittens/include'),
            '-shared','-Xcompiler','-fPIC',str(ROOT/'native_pi0/nvfp4_probe.cu'),'-lcuda','-o',str(library)],check=True)
        for m,n,k,workers in ((128,128,64,1),(256,384,192,4),(51,2560,1024,16),(768,1024,256,16)):
            a=pack_nvfp4(rng.normal(size=(m,k)),global_scale=.125,row_alignment=128)
            b=pack_nvfp4(rng.normal(size=(n,k)),global_scale=.25,scale_policy='ceil')
            padded_m=a.data.shape[0]
            with ResidentNVFP4(library,a,b,workers=workers) as baseline:
                fp32=baseline.forward().cpu()
            expected=a.dequantize()@b.dequantize().T
            np.testing.assert_allclose(fp32.numpy()[:m],expected,atol=1e-5*max(1,float(np.abs(expected).max())),rtol=3e-5)
            np.testing.assert_array_equal(fp32.numpy()[m:],0)
            for bias_enabled,residual_enabled in ((False,False),(True,False),(False,True),(True,True)):
                bias=torch.from_numpy(rng.normal(size=n).astype(np.float32)).bfloat16() if bias_enabled else None
                residual=torch.from_numpy(rng.normal(size=(padded_m,n)).astype(np.float32)).bfloat16() if residual_enabled else None
                expected=fp32.clone()
                if residual is not None: expected+=residual.float()
                if bias is not None: expected+=bias.float()
                expected=expected.bfloat16()
                with ResidentNVFP4(library,a,b,workers=workers,output_dtype=torch.bfloat16,bias=bias,residual=residual) as operation:
                    for _ in range(3):
                        actual=operation.forward().cpu()
                        torch.testing.assert_close(actual,expected,rtol=0,atol=0)
                    if args.profile:
                        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as profile:
                            operation.forward();torch.cuda.synchronize()
                        kernels=[e.name for e in profile.events() if e.device_type==torch.autograd.DeviceType.CUDA]
                        if len(kernels)!=1 or 'megakernel' not in kernels[0]:
                            raise AssertionError(f'Expected one VM launch: {kernels}')
                print(f'BF16 epilogue M={m} N={n} K={k} bias={bias_enabled} residual={residual_enabled}: bit-exact',flush=True)

if __name__=='__main__': main()
