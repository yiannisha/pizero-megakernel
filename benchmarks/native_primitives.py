"""Development NVFP4 projection timing; this is not an end-to-end pi0 result."""
import argparse
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
from native_pi0.quantization import pack_nvfp4


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workload',choices=('expert','vision','prefix'),default='expert')
    parser.add_argument('--workers',type=int,nargs='+',default=[4,16,148])
    parser.add_argument('--iterations',type=int,default=30)
    parser.add_argument('--mma-pipeline',choices=('serial','overlap'),default='serial')
    parser.add_argument('--consumer-warps',type=int,choices=(4,16),default=16)
    parser.add_argument('--k-tile',type=int,choices=(64,128,256),default=64)
    parser.add_argument('--epilogue',choices=('fp32','bf16','bias','bias-residual'),default='fp32')
    parser.add_argument('--validation-only',action='store_true',help='No timing or CUPTI profiling; suitable for compute-sanitizer')
    parser.add_argument('--output',type=Path,default=ROOT/'results/native_fp4_primitive.json')
    args=parser.parse_args()
    if args.iterations<1: parser.error('iterations must be positive')
    torch.set_num_threads(8)
    m,n,k={'expert':(51,2560,1024),'vision':(256,3456,1152),'prefix':(768,2560,2048)}[args.workload]
    rng=np.random.default_rng(23)
    a=pack_nvfp4(rng.normal(size=(m,k)),global_scale=.125,row_alignment=128)
    b=pack_nvfp4(rng.normal(size=(n,k)),global_scale=.25,scale_policy='ceil')
    expected=a.dequantize()@b.dequantize().T
    padded_m=a.data.shape[0]
    reference=np.zeros((padded_m,n),np.float32)
    reference[:m]=expected
    bias=None if args.epilogue not in ('bias','bias-residual') else torch.from_numpy(rng.normal(size=n).astype(np.float32)).bfloat16()
    residual=None if args.epilogue!='bias-residual' else torch.from_numpy(rng.normal(size=(padded_m,n)).astype(np.float32)).bfloat16()
    if residual is not None: reference+=residual.float().numpy()
    if bias is not None: reference+=bias.float().numpy()
    if args.epilogue!='fp32': reference=torch.from_numpy(reference).bfloat16().float().numpy()
    records=[]
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
        for workers in args.workers:
            with ResidentNVFP4(library,a,b,workers=workers,
                              output_dtype=torch.float32 if args.epilogue=='fp32' else torch.bfloat16,
                              bias=bias,residual=residual) as operation:
                for _ in range(5): operation.forward()
                torch.cuda.synchronize()
                actual=operation.output.float().cpu().numpy()
                np.testing.assert_allclose(actual,reference,atol=1e-5*max(1,float(np.abs(reference).max())),rtol=3e-5)
                # Stream interoperability is checked outside timing.
                stream=torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream): operation.forward()
                stream.synchronize()
                np.testing.assert_array_equal(operation.output.float().cpu().numpy(),actual)
                if args.validation_only:
                    records.append({'workers':workers,'correctness':'passed','cuda_launch_count':None,'resources':operation.resources,
                                    'timing_and_launch_profile':'not_run_validation_only'})
                    print(f'{args.workload} workers={workers}: resident correctness and stream checks passed',flush=True)
                    continue
                gpu,wall=[],[]
                start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                for _ in range(args.iterations):
                    torch.cuda.synchronize()
                    t0=time.perf_counter_ns()
                    start.record(); operation.forward(); end.record(); end.synchronize()
                    wall.append((time.perf_counter_ns()-t0)/1e6)
                    gpu.append(start.elapsed_time(end))
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as profile:
                    operation.forward();torch.cuda.synchronize()
                kernel_events=[e for e in profile.events() if e.device_type==torch.autograd.DeviceType.CUDA]
                kernels=[e.name for e in kernel_events]
                if len(kernels)!=1 or 'megakernel' not in kernels[0]:
                    raise AssertionError(f'Expected one native VM launch, observed {kernels}')
                error=actual[:m]-reference[:m]
                def stats(samples):
                    return {'median_ms':statistics.median(samples),'p10_ms':float(np.percentile(samples,10)),
                            'p90_ms':float(np.percentile(samples,90)),'samples_ms':samples}
                records.append({'workers':workers,'gpu':stats(gpu),'synchronized_wall':stats(wall),'resources':operation.resources,
                                'cuda_launch_count':len(kernels),'kernel_names':kernels,
                                'profiled_kernel_duration_us':kernel_events[0].device_time_total,
                                'nrmse_vs_dequantized':float(np.linalg.norm(error)/np.linalg.norm(reference[:m]))})
                print(f'{args.workload} workers={workers}: GPU median {statistics.median(gpu):.4f} ms',flush=True)
    sources=[ROOT/'native_pi0/patches/megakernels-dynamic-instructions.patch',ROOT/'third_party/Megakernels/include/util.cuh',ROOT/'third_party/Megakernels/include/controller/controller.cuh',ROOT/'third_party/Megakernels/include/controller/instruction_fetch.cuh',ROOT/'native_pi0/patches/megakernels-serial-barrier-init.patch',ROOT/'third_party/Megakernels/include/megakernel.cuh',ROOT/'native_pi0/nvfp4_probe.cu',ROOT/'native_pi0/tcgen_probe.cu',ROOT/'native_pi0/primitives.py',Path(__file__)]
    result={'scope':'one synthetic NVFP4 projection, not a model or calibrated policy benchmark',
            'timestamp_utc':datetime.datetime.now(datetime.UTC).isoformat(),'workload':args.workload,
            'mma_pipeline':args.mma_pipeline,
            'consumer_warps':args.consumer_warps,'k_tile':args.k_tile,
            'epilogue':args.epilogue,
            'shape':{'m':m,'padded_m':a.data.shape[0],'n':n,'k':k},
            'precision':'W4A4 E2M1, UE4M3 per16, FP32 accumulation; '+args.epilogue+' output epilogue; nearest activation scales, ceil weight scales',
            'timed_boundary':'GPU-resident packed operands to reused GPU output; excludes setup, packing, transfers, correctness and profiling',
            'environment':{'gpu':torch.cuda.get_device_name(),'torch':torch.__version__,'cuda':torch.version.cuda},
            'upstreams':json.loads((ROOT/'upstreams.json').read_text()),
            'source_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
            'results':records}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')

if __name__=='__main__': main()
