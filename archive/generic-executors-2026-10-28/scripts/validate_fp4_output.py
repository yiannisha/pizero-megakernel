"""Validate direct FP4 epilogues against pinned FlashRT/CUTLASS device math."""
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
from native_pi0.quantization import pack_nvfp4,swizzle_scales


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--consumer-warps',type=int,choices=(4,16),default=4)
    parser.add_argument('--k-tile',type=int,choices=(64,128,256),default=64)
    parser.add_argument('--profile',action='store_true')
    parser.add_argument('--benchmark-output',type=Path,help='Optional 30-sample resident timing report; run without sanitizer')
    args=parser.parse_args()
    torch.set_num_threads(8)
    rng=np.random.default_rng(61)
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
            str(ROOT/'native_pi0/fp4_output_reference.cu'),'-lcuda','-o',str(library)],check=True)
        oracle=ctypes.CDLL(str(library)).reference_fp4_output
        oracle.argtypes=[ctypes.c_void_p]*3+[ctypes.c_int]*2+[ctypes.c_float,ctypes.c_int]
        oracle.restype=ctypes.c_int
        for m,n,k,workers in ((128,128,64,1),(256,384,192,4),(51,2560,1024,16),(768,1024,256,16),(256,4304,1152,64)):
            logical_n=n
            a=pack_nvfp4(rng.normal(size=(m,k)),global_scale=.125,row_alignment=128)
            b=pack_nvfp4(rng.normal(size=(n,k)),global_scale=.25,scale_policy='ceil',row_alignment=128)
            n=b.data.shape[0]
            padded_m=a.data.shape[0]
            with ResidentNVFP4(library,a,b,workers=workers) as baseline:
                projection=baseline.forward().cpu()
            expected=a.dequantize()@b.dequantize().T
            np.testing.assert_allclose(projection.numpy()[:m,:logical_n],expected,atol=1e-5*max(1,float(np.abs(expected).max())),rtol=3e-5)
            np.testing.assert_array_equal(projection.numpy()[:,logical_n:],0)
            for gelu,bias_enabled,residual_enabled,global_scale in ((False,False,False,1.),(False,True,True,.125),(True,False,False,1.),(True,True,True,.125),(True,True,False,1.)):
                bias=torch.from_numpy(rng.normal(size=n).astype(np.float32)).bfloat16() if bias_enabled else None
                residual=torch.from_numpy(rng.normal(size=(padded_m,n)).astype(np.float32)).bfloat16() if residual_enabled else None
                if bias is not None: bias[logical_n:]=0
                if residual is not None: residual[:,logical_n:]=0
                values=projection.clone()
                if residual is not None: values+=residual.float()
                if bias is not None: values+=bias.float()
                q=np.empty((padded_m,n//2),np.uint8);sf=np.empty((padded_m,n//16),np.uint8)
                error=oracle(values.numpy().ctypes.data,q.ctypes.data,sf.ctypes.data,padded_m,n,global_scale,int(gelu))
                if error: raise RuntimeError(f'Oracle CUDA error {error}')
                with ResidentNVFP4(library,a,b,workers=workers,output_dtype=torch.uint8,bias=bias,residual=residual,
                                   output_global_scale=global_scale,output_gelu=gelu) as operation:
                    for _ in range(3):
                        np.testing.assert_array_equal(operation.forward().cpu().numpy(),q)
                        np.testing.assert_array_equal(operation.output_scales.cpu().numpy(),swizzle_scales(sf))
                    if args.profile:
                        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as profile:
                            operation.forward();torch.cuda.synchronize()
                        kernels=[e.name for e in profile.events() if e.device_type==torch.autograd.DeviceType.CUDA]
                        if len(kernels)!=1 or 'megakernel' not in kernels[0]: raise AssertionError(kernels)
                    if args.benchmark_output:
                        for _ in range(5): operation.forward()
                        torch.cuda.synchronize()
                        start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                        gpu,wall=[],[]
                        for _ in range(30):
                            torch.cuda.synchronize()
                            t0=time.perf_counter_ns()
                            start.record();operation.forward();end.record();end.synchronize()
                            wall.append((time.perf_counter_ns()-t0)/1e6)
                            gpu.append(start.elapsed_time(end))
                        def stats(samples):
                            return {'median_ms':statistics.median(samples),'p10_ms':float(np.percentile(samples,10)),
                                    'p90_ms':float(np.percentile(samples,90)),'samples_ms':samples}
                        records.append({'m':m,'padded_m':padded_m,'n':logical_n,'padded_n':n,'k':k,'workers':workers,'gelu':gelu,
                                        'bias':bias_enabled,'residual':residual_enabled,'output_global_scale':global_scale,
                                        'gpu':stats(gpu),'synchronized_wall':stats(wall),'resources':operation.resources,
                                        'cuda_launch_count':len(kernels) if args.profile else None,
                                        'oracle_packed_bytes':'exact','oracle_scale_bytes':'exact'})
                print(f'FP4 epilogue M={m} N={n} K={k} GELU={gelu} bias={bias_enabled} residual={residual_enabled}: byte-exact',flush=True)
    if args.benchmark_output:
        sources=['native_pi0/patches/megakernels-dynamic-instructions.patch','third_party/Megakernels/include/util.cuh','third_party/Megakernels/include/controller/controller.cuh','third_party/Megakernels/include/controller/instruction_fetch.cuh','native_pi0/patches/megakernels-serial-barrier-init.patch','third_party/Megakernels/include/megakernel.cuh','native_pi0/nvfp4_probe.cu','native_pi0/tcgen_probe.cu','native_pi0/primitives.py',
                 'native_pi0/fp4_output_reference.cu','native_pi0/quantization.py','scripts/validate_fp4_output.py']
        report={'timestamp_utc':datetime.datetime.now(datetime.UTC).isoformat(),
                'scope':'Synthetic resident projection plus direct FP4 epilogue; not complete pi0 or pretrained fidelity',
                'timed_boundary':'packed GPU operands to reused FP4 data and scale buffers; excludes oracle, setup and transfers; CUDA events include submission gaps',
                'consumer_warps':args.consumer_warps,'k_tile':args.k_tile,'seed':61,'upstreams':json.loads((ROOT/'upstreams.json').read_text()),
                'environment':{'gpu':torch.cuda.get_device_name(),'torch':torch.__version__,'cuda':torch.version.cuda},
                'source_sha256':{name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in sources},'results':records}
        args.benchmark_output.parent.mkdir(parents=True,exist_ok=True)
        args.benchmark_output.write_text(json.dumps(report,indent=2)+'\n')

if __name__=='__main__': main()
