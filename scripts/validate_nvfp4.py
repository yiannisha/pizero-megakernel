"""Validate SM100 block-scaled FP4 math in the original Megakernels VM."""
import ctypes
import argparse
import subprocess
import sys
import tempfile
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from native_pi0.quantization import pack_nvfp4, swizzle_scales

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mma-pipeline',choices=('serial','overlap'),default='serial')
    parser.add_argument('--consumer-warps',type=int,choices=(4,16),default=16)
    parser.add_argument('--k-tile',type=int,choices=(64,128,256),default=64)
    args=parser.parse_args()
    with tempfile.TemporaryDirectory() as directory:
        library=Path(directory)/'fp4.so'
        subprocess.run(['nvcc','-O3','-std=c++20','-arch=sm_100a','-DKITTENS_HOPPER','-DKITTENS_BLACKWELL',
            '-DNVFP4_OVERLAP_MMA='+str(int(args.mma_pipeline=='overlap')),
            '-DNATIVE_COMPUTE_CONSUMER_WARPS='+str(args.consumer_warps),
            '-DNATIVE_FP4_K_TILE='+str(args.k_tile),
            '--expt-relaxed-constexpr','--expt-extended-lambda',
            '-I'+str(ROOT/'third_party/Megakernels/include'),
            '-I'+str(ROOT/'third_party/Megakernels/ThunderKittens/include'),
            '-shared','-Xcompiler','-fPIC',str(ROOT/'native_pi0/nvfp4_probe.cu'),'-lcuda','-o',str(library)],check=True)
        loaded=ctypes.CDLL(str(library))
        fn=loaded.run_nvfp4_probe
        fn.argtypes=[ctypes.c_void_p]*5+[ctypes.c_int,ctypes.c_float,ctypes.c_int]
        fn.restype=ctypes.c_int
        rng=np.random.default_rng(11)
        for reps,case,k in ((1,'random',64),(2,'random',64),(17,'random',64),(65,'random',64),(17,'zero',64),(17,'varied',64),(17,'random',128),(17,'random',192),(17,'random',256),(17,'random',1024)):
            x,y=rng.normal(size=(128,k)),rng.normal(size=(128,k))
            if case=='zero': x[:]=0
            if case=='varied':
                x*=np.exp2(np.arange(128)[:,None]%12-6)
                y*=np.tile([.01,.1,1,10],(128,16)).reshape(128,64)
                x[::3,:16]=0
            a=pack_nvfp4(x,global_scale=.125)
            b=pack_nvfp4(y,global_scale=.25,scale_policy='ceil')
            sa,sb=swizzle_scales(a.scales),swizzle_scales(b.scales)
            output=np.empty((reps,128,128),np.float32)
            err=fn(a.data.ctypes.data,b.data.ctypes.data,sa.ctypes.data,sb.ctypes.data,output.ctypes.data,reps,
                   float(np.float32(a.global_scale*b.global_scale)),k)
            if err: raise RuntimeError(f'NVFP4 CUDA error {err}')
            expected=a.dequantize()@b.dequantize().T
            tolerance=1e-5*max(1,float(np.abs(expected).max()))
            for result in output: np.testing.assert_allclose(result,expected,atol=tolerance,rtol=3e-5)
            print(f'NVFP4 SM100 TMA/scale-copy/MMA passed: {reps} VM instructions, {case}, K={k}',flush=True)
        grid=loaded.run_nvfp4_grid_probe
        grid.argtypes=[ctypes.c_void_p]*5+[ctypes.c_int]*5+[ctypes.c_float]
        grid.restype=ctypes.c_int
        for m,n,k,workers in ((256,384,192,4),(768,1024,256,16),(51,2560,1024,16)):
            a=pack_nvfp4(rng.normal(size=(m,k)),global_scale=.125,row_alignment=128)
            b=pack_nvfp4(rng.normal(size=(n,k)),global_scale=.25,scale_policy='ceil')
            sa,sb=swizzle_scales(a.scales),swizzle_scales(b.scales)
            padded_m=a.data.shape[0]
            output=np.empty((3,padded_m,n),np.float32)
            err=grid(a.data.ctypes.data,b.data.ctypes.data,sa.ctypes.data,sb.ctypes.data,output.ctypes.data,
                     3,padded_m,n,k,workers,float(np.float32(a.global_scale*b.global_scale)))
            if err: raise RuntimeError(f'NVFP4 grid CUDA error {err}')
            expected=a.dequantize()@b.dequantize().T
            tolerance=1e-5*max(1,float(np.abs(expected).max()))
            for result in output:
                np.testing.assert_allclose(result[:m],expected,atol=tolerance,rtol=3e-5)
                np.testing.assert_array_equal(result[m:],0)
            print(f'NVFP4 M={m} N={n} K={k} passed: {workers} CTAs, three repetitions',flush=True)

if __name__=='__main__': main()
