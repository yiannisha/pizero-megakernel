"""Validate BF16/FP8 TCGen05/TMA inside the pinned native VM."""
import ctypes
import argparse
import subprocess
import tempfile
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--consumer-warps',type=int,choices=(4,16),default=16)
    args=parser.parse_args()
    with tempfile.TemporaryDirectory() as directory:
        library = Path(directory) / 'tcgen.so'
        subprocess.run(['nvcc', '-O3', '-std=c++20', '-arch=sm_100a', '-DKITTENS_HOPPER',
            '-DNATIVE_COMPUTE_CONSUMER_WARPS='+str(args.consumer_warps),
            '-DKITTENS_BLACKWELL', '--expt-relaxed-constexpr', '--expt-extended-lambda',
            '-I'+str(ROOT/'third_party/Megakernels/include'),
            '-I'+str(ROOT/'third_party/Megakernels/ThunderKittens/include'),
            '-shared', '-Xcompiler', '-fPIC', str(ROOT/'native_pi0/tcgen_probe.cu'),
            '-lcuda', '-o', str(library)], check=True)
        loaded = ctypes.CDLL(str(library))
        fn = loaded.run_tcgen_probe
        fn.argtypes = [ctypes.c_void_p]*3+[ctypes.c_int]
        fn.restype = ctypes.c_int
        torch.manual_seed(17)
        for repetitions in (1, 2, 17, 65):
            a = torch.randn(128,64).bfloat16()
            b = torch.randn(128,64).bfloat16()
            expected = (a.float() @ b.float().T).numpy()
            output = np.empty((repetitions,128,128), np.float32)
            err = fn(a.data_ptr(), b.data_ptr(), output.ctypes.data, repetitions)
            if err: raise RuntimeError(f'CUDA error {err}')
            for result in output:
                np.testing.assert_allclose(result, expected, atol=3e-5, rtol=2e-5)
            print(f'BF16 TMA/TCGen05 passed: {repetitions} VM instructions', flush=True)
        fp8 = loaded.run_fp8_probe
        fp8.argtypes = [ctypes.c_void_p]*3+[ctypes.c_int, ctypes.c_float, ctypes.c_float]
        fp8.restype = ctypes.c_int
        for repetitions in (1, 2, 17, 65):
            for a_scale, w_scale in ((1., 1.), (.0123, .0456), (2., .25)):
                a = (torch.randn(128,64)*4).to(torch.float8_e4m3fn)
                b = (torch.randn(128,64)*4).to(torch.float8_e4m3fn)
                alpha = np.float32(a_scale)*np.float32(w_scale)
                expected = (a.float() @ b.float().T).numpy()*alpha
                output = np.empty((repetitions,128,128), np.float32)
                err = fp8(a.data_ptr(), b.data_ptr(), output.ctypes.data, repetitions, a_scale, w_scale)
                if err: raise RuntimeError(f'FP8 CUDA error {err}')
                for result in output:
                    np.testing.assert_allclose(result, expected, atol=1e-4*float(alpha), rtol=2e-5)
            print(f'FP8 TMA/TCGen05 with FP32 descale passed: {repetitions} VM instructions', flush=True)
        loop = loaded.run_k_probe
        loop.argtypes = [ctypes.c_void_p]*3+[ctypes.c_int]*3+[ctypes.c_float]*2
        loop.restype = ctypes.c_int
        for precision, dtype in enumerate((torch.bfloat16, torch.float8_e4m3fn)):
            for k in (128, 192, 256, 1024):
                a = torch.randn(128,k).to(dtype)
                b = torch.randn(128,k).to(dtype)
                alpha = np.float32(.0123)*np.float32(.0456)
                expected = (a.float() @ b.float().T).numpy()*alpha
                output = np.empty((17,128,128), np.float32)
                err = loop(a.data_ptr(), b.data_ptr(), output.ctypes.data, 17, k, precision, .0123, .0456)
                if err: raise RuntimeError(f'K-loop CUDA error {err}')
                for result in output:
                    np.testing.assert_allclose(result, expected, atol=1e-4*float(alpha), rtol=3e-5)
                print(f'{dtype} double-buffered K={k} passed: 17 VM instructions', flush=True)
        grid = loaded.run_grid_probe
        grid.argtypes = [ctypes.c_void_p]*3+[ctypes.c_int]*6+[ctypes.c_float]*2
        grid.restype = ctypes.c_int
        for precision, dtype in enumerate((torch.bfloat16, torch.float8_e4m3fn)):
            for m, n, k, workers in ((256,384,192,4), (768,1024,256,16), (51,2560,1024,16)):
                padded_m = (m+127)//128*128
                a = torch.zeros(padded_m,k, dtype=dtype)
                a[:m] = torch.randn(m,k).to(dtype)
                b = torch.randn(n,k).to(dtype)
                alpha = np.float32(.0123)*np.float32(.0456)
                expected = (a.float() @ b.float().T).numpy()*alpha
                output = np.empty((3,padded_m,n), np.float32)
                err = grid(a.data_ptr(), b.data_ptr(), output.ctypes.data, 3, padded_m, n, k, workers, precision, .0123, .0456)
                if err: raise RuntimeError(f'Grid CUDA error {err}')
                for result in output:
                    np.testing.assert_allclose(result, expected, atol=1e-4*float(alpha), rtol=3e-5)
                print(f'{dtype} tiled M={m} N={n} K={k} passed: {workers} CTAs, three repetitions', flush=True)

if __name__ == '__main__': main()
