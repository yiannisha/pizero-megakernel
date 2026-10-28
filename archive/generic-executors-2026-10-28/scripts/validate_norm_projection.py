"""Validate one-launch, on-demand native normalization -> NVFP4 projection."""
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
from native_pi0.norm_projection import ResidentNormProjection
from native_pi0.quantization import PackedNVFP4,pack_nvfp4,unswizzle_scales
from native_pi0.launch_trace import LaunchTrace


def nrmse(actual,expected):
    difference=actual.astype(np.float64)-expected.astype(np.float64)
    return float(np.linalg.norm(difference)/max(np.linalg.norm(expected.astype(np.float64)),1e-30))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile',action='store_true')
    parser.add_argument('--producer-rows',type=int,choices=(1,4,8),default=1)
    parser.add_argument('--benchmark-output',type=Path)
    args=parser.parse_args()
    if args.benchmark_output and not args.profile:parser.error('Timing requires --profile launch checks')
    torch.set_num_threads(8);torch.manual_seed(97);records=[]
    with tempfile.TemporaryDirectory() as directory:
        library=Path(directory)/'norm_projection.so'
        subprocess.run(['nvcc','-O3','-lineinfo','-std=c++20','-arch=sm_100a','-DKITTENS_HOPPER','-DKITTENS_BLACKWELL',
            '-DNATIVE_COMPUTE_CONSUMER_WARPS=4','-DNATIVE_FP4_K_TILE=256','-DNATIVE_NORM_PROJECTION_VALIDATION',
            '-DNATIVE_NORM_PRODUCER_ROWS='+str(args.producer_rows),
            '--expt-relaxed-constexpr','--expt-extended-lambda',
            '-I'+str(ROOT/'third_party/Megakernels/include'),'-I'+str(ROOT/'third_party/Megakernels/ThunderKittens/include'),
            '-shared','-Xcompiler','-fPIC',str(ROOT/'native_pi0/norm_projection.cu'),str(ROOT/'native_pi0/launch_trace.cpp'),
            '-lcupti','-Xlinker','-rpath','-Xlinker','/usr/local/cuda/lib64','-lcuda','-o',str(library)],check=True)
        split=ctypes.CDLL(str(library)).validation_split_norm_projection
        split.argtypes=[ctypes.c_void_p]*2;split.restype=ctypes.c_int

        # CPU matmul references use dequantized E2M1 + decoded scale operands.
        # Also compare the producer bytes and residual with the standalone native
        # producer, whose separate suite checks pinned FlashRT/OpenPI math.
        for rows,k,n,workers in ((1,64,128,1),(3,192,384,4),(51,1024,8192,16),
                                 (129,192,384,148),(256,1152,4352,148),(768,2048,2560,148),
                                 (256,192,384,1)):
            for kind,dtype,out_dtype in (('rms',torch.bfloat16,torch.float32),('layer',torch.float16,torch.bfloat16)):
                x=torch.randn(rows,k).to(dtype)
                residual=(torch.randn(rows,k)*.1).to(dtype)
                gamma=1+torch.randn(k)*.05
                beta=torch.randn(k)*.02 if kind=='layer' else None
                inverse=1+torch.rand(k)*.1
                weight=pack_nvfp4(torch.randn(n,k).numpy()*.05,global_scale=.5)
                weight_unscaled=PackedNVFP4(weight.data,weight.scales,np.float32(1),weight.logical_shape).dequantize()
                with ResidentNormProjection(library,x,weight,workers=workers,output_dtype=out_dtype,kind=kind,
                                            gamma=gamma,beta=beta,inverse_scale=inverse,residual=residual,global_scale=.025) as op:
                    maximum=0.;chain_maximum=0.
                    for replay in range(3):
                        stream=torch.cuda.Stream()
                        with torch.cuda.stream(stream):
                            # Poison logical activation bytes; readiness must be
                            # recomputed for every epoch, including changed inputs.
                            op.norm.output[:rows].fill_(255)
                            op.inputs.copy_((x.float()+replay*.125).to(dtype))
                            op.residual.copy_((residual.float()-replay*.03125).to(dtype))
                            op.forward()
                        stream.synchronize()
                        actual=op.output.float().cpu().numpy()
                        packed=op.norm.output.cpu().numpy().copy();sf=op.norm.output_scales.cpu().numpy().copy()
                        res=op.residual_output.clone()
                        linear_sf=unswizzle_scales(sf,packed.shape[0],k//16)
                        a=PackedNVFP4(packed,linear_sf,np.float32(1),(packed.shape[0],k)).dequantize()
                        gold=(a@weight_unscaled.T)*np.float32(.025*weight.global_scale)
                        if out_dtype==torch.bfloat16:gold=torch.from_numpy(gold).bfloat16().float().numpy()
                        error=nrmse(actual,gold);maximum=max(maximum,error)
                        projection_limit=1e-5 if out_dtype==torch.float32 else .005
                        assert error<=projection_limit,(rows,k,n,kind,replay,error)
                        assert np.all(actual[rows:]==0)
                        op.norm.forward();torch.cuda.synchronize()
                        np.testing.assert_array_equal(packed,op.norm.output.cpu().numpy())
                        np.testing.assert_array_equal(sf,op.norm.output_scales.cpu().numpy())
                        assert torch.equal(res,op.residual_output)
                        # Independent CPU normalization/packing validates the
                        # entire chain without treating native producer output as gold.
                        summed=(op.inputs.float().cpu()+op.residual.float().cpu()).to(dtype).float().numpy()
                        mean=np.mean(summed,axis=1,keepdims=True,dtype=np.float32) if kind=='layer' else np.float32(0)
                        centered=summed-mean
                        normalized=centered/np.sqrt(np.mean(centered*centered,axis=1,keepdims=True,dtype=np.float32)+np.float32(1e-6))
                        normalized=normalized*gamma.numpy()
                        if beta is not None:normalized+=beta.numpy()
                        normalized*=inverse.numpy()
                        normalized=torch.from_numpy(normalized).to(dtype).float().numpy()
                        cpu=pack_nvfp4(normalized,global_scale=.025,row_alignment=128,column_alignment=64)
                        cpu_a=PackedNVFP4(cpu.data,cpu.scales,np.float32(1),(rows,k)).dequantize()
                        chain_gold=(cpu_a@weight_unscaled.T)*np.float32(.025*weight.global_scale)
                        if out_dtype==torch.bfloat16:chain_gold=torch.from_numpy(chain_gold).bfloat16().float().numpy()
                        chain_error=nrmse(actual[:rows],chain_gold);chain_maximum=max(chain_maximum,chain_error)
                        assert chain_error<=.005,(rows,k,n,kind,replay,'whole chain',chain_error)
                    metadata={'rows':rows,'k':k,'n':n,'workers':workers,'kind':kind,'dtype':str(dtype),'output_dtype':str(out_dtype),
                              'split_norm_workers':op.norm.workers,
                              'replays':3,'producer_vs_standalone':'byte-exact','poisoned_input_bytes':'recomputed',
                              'gemm_nrmse':maximum,'gemm_nrmse_limit':projection_limit,'whole_chain_nrmse':chain_maximum,'nrmse_limit':.005}
                    if args.profile:
                        with LaunchTrace(library) as trace:op.forward();torch.cuda.synchronize()
                        assert trace.count==1 and 'megakernel' in trace.symbol,(trace.count,trace.symbol)
                        metadata.update(cuda_launch_count=trace.count,kernel_symbol=trace.symbol)
                        expected=op.output.clone()
                        with LaunchTrace(library) as trace:
                            error=split(op.handle,torch.cuda.current_stream().cuda_stream);torch.cuda.synchronize()
                        assert not error,error
                        assert trace.count==2,trace.count
                        torch.testing.assert_close(op.output,expected,rtol=0,atol=0)
                        metadata['split_baseline_cuda_launch_count']=trace.count
                    if args.benchmark_output:
                        for _ in range(5):op.forward()
                        torch.cuda.synchronize();start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                        gpu=[];wall=[]
                        for _ in range(30):
                            torch.cuda.synchronize();t0=time.perf_counter_ns();start.record();op.forward();end.record();end.synchronize()
                            gpu.append(start.elapsed_time(end));wall.append((time.perf_counter_ns()-t0)/1e6)
                        def stats(values):return {'median_ms':statistics.median(values),'p10_ms':float(np.percentile(values,10)),
                                                  'p90_ms':float(np.percentile(values,90)),'samples_ms':values}
                        metadata.update(gpu=stats(gpu),synchronized_wall=stats(wall))
                        def split_forward():
                            error=split(op.handle,torch.cuda.current_stream().cuda_stream)
                            if error:raise RuntimeError(f'Split baseline CUDA error {error}')
                        for _ in range(5):split_forward()
                        torch.cuda.synchronize();gpu=[];wall=[]
                        for _ in range(30):
                            torch.cuda.synchronize();t0=time.perf_counter_ns();start.record();split_forward();end.record();end.synchronize()
                            gpu.append(start.elapsed_time(end));wall.append((time.perf_counter_ns()-t0)/1e6)
                        metadata.update(split_gpu=stats(gpu),split_synchronized_wall=stats(wall))
                    records.append(metadata)
                    print(f'{kind} {rows}x{k} -> {n}, {workers} CTAs: producer exact, projection NRMSE={maximum:g}, whole chain={chain_maximum:g}',flush=True)

        # Independent instances can execute concurrently even when their combined
        # CTA counts exceed the SM count: only resident consumers claim rows.
        x=torch.randn(256,192).bfloat16();weight=pack_nvfp4(torch.randn(8192,192).numpy()*.05)
        with ResidentNormProjection(library,x,weight,workers=148) as left, ResidentNormProjection(library,x+.25,weight,workers=148) as right:
            gold=[op.forward().clone() for op in (left,right)]
            torch.cuda.synchronize()
            streams=[torch.cuda.Stream(),torch.cuda.Stream()]
            for op,stream in zip((left,right),streams):
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    op.norm.output[:256].fill_(255)
                    op.forward()
            for stream in streams:stream.synchronize()
            for op,expected in zip((left,right),gold):torch.testing.assert_close(op.output,expected,rtol=0,atol=0)
            assert not torch.equal(gold[0],gold[1])
            print('Concurrent independent 148-CTA instances: exact outputs against separate references',flush=True)
            if args.profile:
                with LaunchTrace(library) as trace:
                    left.forward();right.forward();torch.cuda.synchronize()
                assert trace.count==2,trace.count
                print('Two-forward launch-count negative control: two launches',flush=True)
    if args.benchmark_output:
        sources=['native_pi0/patches/megakernels-dynamic-instructions.patch','third_party/Megakernels/include/util.cuh','third_party/Megakernels/include/controller/controller.cuh','third_party/Megakernels/include/controller/instruction_fetch.cuh','native_pi0/norm_projection.cu','native_pi0/norm_projection.py','native_pi0/norm.cu','native_pi0/normalization.py',
                 'native_pi0/nvfp4_probe.cu','native_pi0/tcgen_probe.cu','native_pi0/quantization.py','native_pi0/launch_trace.cpp',
                 'native_pi0/launch_trace.py','native_pi0/patches/megakernels-serial-barrier-init.patch',
                 'third_party/Megakernels/include/megakernel.cuh','scripts/validate_norm_projection.py']
        report={'timestamp_utc':datetime.datetime.now(datetime.UTC).isoformat(),'seed':97,'producer_rows':args.producer_rows,
                'scope':'Synthetic native normalization -> NVFP4 projection chain, not full pi0 or pretrained parity',
                'timed_boundary':'GPU-resident input/parameters to reused output and residual; excludes setup/reference/transfers; events include host submission gaps',
                'scheduling':'On-demand row-group CAS ownership, release/acquire epoch readiness; no grid barrier',
                'concurrent_independent_instances':'passed, 148 CTAs each','two_forward_negative_control':2,
                'upstreams':json.loads((ROOT/'upstreams.json').read_text()),
                'environment':{'gpu':torch.cuda.get_device_name(),'torch':torch.__version__,'cuda':torch.version.cuda},
                'source_sha256':{name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in sources},'results':records}
        args.benchmark_output.parent.mkdir(parents=True,exist_ok=True)
        args.benchmark_output.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
