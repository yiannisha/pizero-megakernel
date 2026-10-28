"""Validate native compact GeGLU FP4 against FlashRT's SM100 visitor math."""
import argparse
import ctypes
import datetime
import hashlib
import json
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from native_pi0.primitives import ResidentNVFP4
from native_pi0.launch_trace import LaunchTrace
from native_pi0.quantization import pack_nvfp4,swizzle_scales,interleave_gate_up


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--consumer-warps',type=int,choices=(4,16),default=4)
    parser.add_argument('--k-tile',type=int,choices=(64,128,256),default=64)
    parser.add_argument('--quick',action='store_true',help='Only small debug shapes')
    parser.add_argument('--profile',action='store_true')
    parser.add_argument('--benchmark-output',type=Path)
    args=parser.parse_args()
    torch.set_num_threads(8)
    rng=np.random.default_rng(71)
    records=[]
    with tempfile.TemporaryDirectory() as directory:
        library=Path(directory)/'fp4.so'
        subprocess.run(['nvcc','-O3','-std=c++20','-arch=sm_100a','-DKITTENS_HOPPER','-DKITTENS_BLACKWELL',
            '-DNATIVE_COMPUTE_CONSUMER_WARPS='+str(args.consumer_warps),
            '-DNATIVE_FP4_K_TILE='+str(args.k_tile),
            '--expt-relaxed-constexpr','--expt-extended-lambda',
            '-I'+str(ROOT/'third_party/Megakernels/include'),
            '-I'+str(ROOT/'third_party/Megakernels/ThunderKittens/include'),
            '-I'+str(ROOT/'third_party/FlashRT/csrc'),'-I'+str(ROOT/'third_party/cutlass/include'),
            '-shared','-Xcompiler','-fPIC',str(ROOT/'native_pi0/nvfp4_probe.cu'),
            str(ROOT/'native_pi0/fp4_output_reference.cu'),str(ROOT/'native_pi0/launch_trace.cpp'),
            '-lcupti','-Xlinker','-rpath','-Xlinker','/usr/local/cuda/lib64','-lcuda','-o',str(library)],check=True)
        oracle=ctypes.CDLL(str(library)).reference_fp4_output
        oracle.argtypes=[ctypes.c_void_p]*3+[ctypes.c_int]*2+[ctypes.c_float,ctypes.c_int]
        oracle.restype=ctypes.c_int
        shapes=[(128,128,64,1),(256,384,192,4)]
        if not args.quick: shapes.extend([(51,8192,1024,64),(768,32768,2048,148)])
        for m,n,k,workers in shapes:
            a=pack_nvfp4(rng.normal(size=(m,k)),global_scale=.125,row_alignment=128)
            gate=rng.normal(0,.1,size=(n//2,k)).astype(np.float32)
            up=rng.normal(0,.1,size=(n//2,k)).astype(np.float32)
            b=pack_nvfp4(interleave_gate_up(gate,up),global_scale=.25,scale_policy='ceil')
            padded_m=a.data.shape[0]
            with ResidentNVFP4(library,a,b,workers=workers) as baseline:
                projection=baseline.forward().cpu().numpy()
            expected=a.dequantize()@b.dequantize().T
            np.testing.assert_allclose(projection[:m],expected,atol=1e-5*max(1,float(np.abs(expected).max())),rtol=3e-5)
            np.testing.assert_array_equal(projection[m:],0)
            for global_scale in (1.,8.):
                q=np.empty((padded_m,n//4),np.uint8);sf=np.empty((padded_m,n//32),np.uint8)
                error=oracle(projection.ctypes.data,q.ctypes.data,sf.ctypes.data,padded_m,n,global_scale,2)
                if error: raise RuntimeError(f'Compact oracle CUDA error {error}')
                with ResidentNVFP4(library,a,b,workers=workers,output_dtype=torch.uint8,
                                   output_global_scale=global_scale,output_geglu=True) as operation:
                    for _ in range(3):
                        np.testing.assert_array_equal(operation.forward().cpu().numpy(),q)
                        np.testing.assert_array_equal(operation.output_scales.cpu().numpy(),swizzle_scales(sf))
                    np.testing.assert_array_equal(q[m:],0)
                    np.testing.assert_array_equal(sf[m:],0)
                    if args.profile:
                        with LaunchTrace(library) as trace:
                            operation.forward();torch.cuda.synchronize()
                        if trace.count!=1 or 'megakernel' not in trace.symbol: raise AssertionError((trace.count,trace.symbol))
                        if m==128 and global_scale==1:
                            with LaunchTrace(library) as negative_trace:
                                operation.forward();operation.forward();torch.cuda.synchronize()
                            if negative_trace.count!=2: raise AssertionError(('two-launch control',negative_trace.count))
                    if args.benchmark_output:
                        for _ in range(5): operation.forward()
                        torch.cuda.synchronize()
                        start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                        gpu,wall=[],[]
                        for _ in range(30):
                            torch.cuda.synchronize();t0=time.perf_counter_ns()
                            start.record();operation.forward();end.record();end.synchronize()
                            wall.append((time.perf_counter_ns()-t0)/1e6);gpu.append(start.elapsed_time(end))
                        def stats(values):
                            return {'median_ms':statistics.median(values),'p10_ms':float(np.percentile(values,10)),
                                    'p90_ms':float(np.percentile(values,90)),'samples_ms':values}
                        records.append({'m':m,'padded_m':padded_m,'n_interleaved':n,'n_compact':n//2,'k':k,'workers':workers,
                                        'output_global_scale':global_scale,'gpu':stats(gpu),'synchronized_wall':stats(wall),
                                        'resources':operation.resources,'cuda_launch_count':trace.count if args.profile else None,
                                        'launch_trace_method':'CUPTI driver callback' if args.profile else None,
                                        'kernel_symbol':trace.symbol if args.profile else None,
                                        'oracle_packed_bytes':'exact','oracle_scale_bytes':'exact'})
                print(f'Compact GeGLU M={m} N={n} -> {n//2} K={k} gs={global_scale}: byte-exact',flush=True)
    if args.benchmark_output:
        sources=['native_pi0/patches/megakernels-dynamic-instructions.patch','third_party/Megakernels/include/util.cuh','third_party/Megakernels/include/controller/controller.cuh','third_party/Megakernels/include/controller/instruction_fetch.cuh','native_pi0/patches/megakernels-serial-barrier-init.patch','third_party/Megakernels/include/megakernel.cuh','native_pi0/nvfp4_probe.cu','native_pi0/tcgen_probe.cu','native_pi0/primitives.py',
                 'native_pi0/fp4_output_reference.cu','native_pi0/quantization.py','native_pi0/launch_trace.cpp','native_pi0/launch_trace.py','scripts/validate_geglu.py']
        report={'timestamp_utc':datetime.datetime.now(datetime.UTC).isoformat(),'seed':71,'consumer_warps':args.consumer_warps,'k_tile':args.k_tile,
                'scope':'Synthetic native interleaved gate/up GEMM + compact GeGLU-to-FP4, not model latency or pretrained fidelity',
                'timed_boundary':'resident packed operands to compact FP4 data/scales; excludes reference, setup, transfers; CUDA events include submission gaps',
                'environment':{'gpu':torch.cuda.get_device_name(),'torch':torch.__version__,'cuda':torch.version.cuda},
                'upstreams':json.loads((ROOT/'upstreams.json').read_text()),
                'source_sha256':{name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in sources},'results':records}
        args.benchmark_output.parent.mkdir(parents=True,exist_ok=True)
        args.benchmark_output.write_text(json.dumps(report,indent=2)+'\n')

if __name__=='__main__': main()
