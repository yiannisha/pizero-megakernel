"""Native VM norm/quant producers: OpenPI math and pinned FlashRT references."""
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
from native_pi0.normalization import ResidentNorm
from native_pi0.quantization import PackedNVFP4,pack_nvfp4,unswizzle_scales
from native_pi0.launch_trace import LaunchTrace


def decode(operation, rows, width):
    if operation.output.dtype==torch.uint8:
        data=operation.output.cpu().numpy()[:rows]
        scales=unswizzle_scales(operation.output_scales.cpu().numpy(),operation.output.shape[0],width//16)[:rows]
        return PackedNVFP4(data,scales,operation.global_scale,(rows,width)).dequantize()
    return operation.output[:rows].float().cpu().numpy()*operation.global_scale


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile',action='store_true')
    parser.add_argument('--row-tile',type=int,choices=(1,4,8),default=1)
    parser.add_argument('--bf16-workers',type=int,default=16,help='Resident CTA count for BF16 cases (1..148 on B200)')
    parser.add_argument('--benchmark-output',type=Path,help='30-sample resident timings; use without sanitizer')
    args=parser.parse_args()
    if not 1<=args.bf16_workers<=148:parser.error('--bf16-workers must be in 1..148')
    if args.benchmark_output and not args.profile:parser.error('Timing report requires --profile launch gates')
    torch.set_num_threads(8);torch.manual_seed(83);records=[]
    with tempfile.TemporaryDirectory() as directory:
        library=Path(directory)/'norm.so'
        subprocess.run(['nvcc','-O3','-std=c++20','-arch=sm_100a','-DKITTENS_HOPPER','-DKITTENS_BLACKWELL',
            '-DNATIVE_NORM_ROW_TILE='+str(args.row_tile),'-DCUTLASS_ARCH_MMA_SM100_SUPPORTED','--expt-relaxed-constexpr','--expt-extended-lambda',
            '-I'+str(ROOT/'third_party/Megakernels/include'),'-I'+str(ROOT/'third_party/Megakernels/ThunderKittens/include'),
            '-I'+str(ROOT/'third_party/FlashRT/csrc'),'-I'+str(ROOT/'third_party/cutlass/include'),
            '-shared','-Xcompiler','-fPIC',str(ROOT/'native_pi0/norm.cu'),str(ROOT/'native_pi0/norm_reference.cu'),
            str(ROOT/'third_party/FlashRT/csrc/fused_fp4/siglip_ln_vec.cu'),str(ROOT/'native_pi0/launch_trace.cpp'),
            '-lcupti','-Xlinker','-rpath','-Xlinker','/usr/local/cuda/lib64','-lcuda','-o',str(library)],check=True)
        reference=ctypes.CDLL(str(library)).reference_layer_norm
        reference.argtypes=[ctypes.c_void_p]*6+[ctypes.c_int]*3;reference.restype=ctypes.c_int
        quant_reference=ctypes.CDLL(str(library)).reference_norm_quant
        quant_reference.argtypes=[ctypes.c_void_p]*3+[ctypes.c_int]*2+[ctypes.c_float];quant_reference.restype=ctypes.c_int

        def record(operation,metadata):
            metadata['workers']=operation.workers
            if args.profile:
                with LaunchTrace(library) as trace:operation.forward();torch.cuda.synchronize()
                assert trace.count==1 and 'megakernel' in trace.symbol,(trace.count,trace.symbol)
                metadata.update(cuda_launch_count=trace.count,kernel_symbol=trace.symbol,trace_method='CUPTI driver callbacks')
            if args.benchmark_output:
                for _ in range(5):operation.forward()
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
                metadata.update(gpu=stats(gpu),synchronized_wall=stats(wall))
            records.append(metadata)

        for rows,width in ((1,1152),(256,1152),(768,2048)):
            x=torch.randn(rows,width).half()
            gamma=(1+torch.randn(width)*.05).half();beta=(torch.randn(width)*.05).half();awq=(1+torch.rand(width)*.25).half()
            for precision in ('fp4','fp8'):
                with ResidentNorm(library,x,kind='layer',precision=precision,gamma=gamma.float(),beta=beta.float(),
                                  inverse_scale=awq.float() if precision=='fp4' else None,workers=min(rows,148),
                                  round_normalized=precision=='fp4',legacy_ties=True) as operation:
                    native=operation.forward().cpu()
                    g,b,inv=[t.cuda() for t in (gamma,beta,awq)]
                    expected=torch.empty_like(operation.output);sf=None if precision=='fp8' else torch.zeros_like(operation.output_scales)
                    error=reference(operation.inputs.data_ptr(),g.data_ptr(),b.data_ptr(),inv.data_ptr() if precision=='fp4' else None,
                                    expected.data_ptr(),None if sf is None else sf.data_ptr(),rows,width,4 if precision=='fp4' else 8)
                    if error:raise RuntimeError(f'FlashRT reference error {error}')
                    torch.cuda.synchronize()
                    np.testing.assert_array_equal(native.view(torch.uint8).numpy(),expected.cpu().view(torch.uint8).numpy())
                    if sf is not None:np.testing.assert_array_equal(operation.output_scales.cpu().numpy(),sf.cpu().numpy())
                    record(operation,{'rows':rows,'width':width,'kind':'layer','dtype':'fp16','precision':precision,
                                      'global_scale':1.,'reference':'FlashRT siglip_ln_vec','packed_bytes':'exact','scale_bytes':'exact' if sf is not None else None})
                print(f'FP16 LayerNorm {rows}x{width} -> {precision}: FlashRT byte-exact',flush=True)

        def bf16_case(rows,width,kind,precision,global_scale,pattern=None,round_normalized=True,round_residual=True):
            x=torch.randn(rows,width).bfloat16();r=(torch.randn(rows,width)*.1).bfloat16()
            gamma=1+torch.randn(width)*.05;beta=torch.randn(width)*.05 if kind=='layer' else None
            if pattern=='folded':
                x.zero_();x[1].fill_(3);x[2,0]=1000;r=None;gamma=None;beta=None
            with ResidentNorm(library,x,kind=kind,precision=precision,gamma=gamma,beta=beta,residual=r,workers=args.bf16_workers,
                              round_normalized=round_normalized,round_residual=round_residual,global_scale=global_scale,row_alignment=128) as operation:
                operation.forward();actual=decode(operation,rows,width)
                value=x.float() if r is None else x.float()+r.float()
                if r is not None and round_residual:value=value.bfloat16().float()
                expected=value
                if kind=='layer':expected=expected-expected.mean(-1,keepdim=True)
                expected=expected*torch.rsqrt(expected.square().mean(-1,keepdim=True)+1e-6)
                if gamma is not None:expected*=gamma
                if beta is not None:expected+=beta
                if round_normalized:expected=expected.bfloat16().float()
                normalized=expected.cuda();q=torch.empty((rows,width//2 if precision=='fp4' else width),dtype=operation.output.dtype,device='cuda')
                sf=None if precision=='fp8' else torch.empty((rows,width//16),dtype=torch.uint8,device='cuda')
                code=quant_reference(normalized.data_ptr(),q.data_ptr(),None if sf is None else sf.data_ptr(),rows*width,
                                     4 if precision=='fp4' else 8,float(global_scale))
                if code:raise RuntimeError(f'Quantization reference CUDA error {code}')
                if precision=='fp4':
                    q_cpu,sf_cpu=q.cpu().numpy(),sf.cpu().numpy()
                    reference_values=PackedNVFP4(q_cpu,sf_cpu,np.float32(global_scale),(rows,width)).dequantize()
                    cpu_packed=pack_nvfp4(expected.numpy(),global_scale=global_scale)
                    np.testing.assert_array_equal(cpu_packed.data,q_cpu);np.testing.assert_array_equal(cpu_packed.scales,sf_cpu)
                else:reference_values=q.float().cpu().numpy()*np.float32(global_scale)
                error=float(np.linalg.norm(actual-reference_values)/max(np.linalg.norm(reference_values),1e-12))
                assert error<.005,(rows,width,kind,precision,error)
                if r is not None:torch.testing.assert_close(operation.residual_output.cpu(),(x.float()+r.float()).bfloat16(),rtol=0,atol=0)
                np.testing.assert_array_equal(operation.output[rows:].view(torch.uint8).cpu().numpy(),0)
                if operation.output_scales is not None:
                    padded_scales=unswizzle_scales(operation.output_scales.cpu().numpy(),operation.output.shape[0],width//16)
                    np.testing.assert_array_equal(padded_scales[rows:],0)
                if pattern=='folded':np.testing.assert_array_equal(actual[0],0)
                for _ in range(2):
                    operation.forward();np.testing.assert_array_equal(decode(operation,rows,width),actual)
                stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):operation.forward()
                stream.synchronize();np.testing.assert_array_equal(decode(operation,rows,width),actual)
                record(operation,{'rows':rows,'padded_rows':operation.output.shape[0],'width':width,'kind':kind,'dtype':'bf16',
                                  'precision':precision,'global_scale':global_scale,'pattern':pattern,'round_normalized':round_normalized,
                                  'round_residual':round_residual,'reference':'OpenPI normalization math + FlashRT device quantization',
                                  'nrmse':error,'nrmse_limit':.005,'residual_and_padding':'exact','cpu_packer_gpu_quantizer':'exact' if precision=='fp4' else None})
            print(f'BF16 {kind} {rows}x{width} -> {precision} gs={global_scale} {pattern or "residual"}: NRMSE={error:.6g}',flush=True)

        for rows,width in ((51,1024),(256,1152),(768,2048)):
            for kind in ('rms','layer'):
                for precision in ('fp4','fp8'):
                    for global_scale in (1.,.025):bf16_case(rows,width,kind,precision,global_scale)
        for kind in ('rms','layer'):
            for precision in ('fp4','fp8'):
                for global_scale in (1.,.025):bf16_case(3,1024,kind,precision,global_scale,pattern='folded')
        for kind in ('rms','layer'):bf16_case(3,16,kind,'fp4',1.,pattern='folded')
        bf16_case(51,1024,'rms','fp4',1.,round_normalized=False,round_residual=False)
    if args.benchmark_output:
        sources=['native_pi0/patches/megakernels-dynamic-instructions.patch','third_party/Megakernels/include/util.cuh','third_party/Megakernels/include/controller/controller.cuh','third_party/Megakernels/include/controller/instruction_fetch.cuh','native_pi0/patches/megakernels-serial-barrier-init.patch','third_party/Megakernels/include/megakernel.cuh','native_pi0/norm.cu','native_pi0/normalization.py','native_pi0/norm_reference.cu','native_pi0/quantization.py',
                 'native_pi0/launch_trace.cpp','native_pi0/launch_trace.py','scripts/validate_norm.py']
        report={'timestamp_utc':datetime.datetime.now(datetime.UTC).isoformat(),'seed':83,'row_tile':args.row_tile,
                'scope':'Synthetic standalone native normalization/quantization producer, not complete pi0 or pretrained parity',
                'timed_boundary':'GPU-resident input/parameters to reused quantized output, scales and optional residual; excludes setup/references/transfers; events include submission gaps',
                'upstreams':json.loads((ROOT/'upstreams.json').read_text()),
                'environment':{'gpu':torch.cuda.get_device_name(),'torch':torch.__version__,'cuda':torch.version.cuda},
                'source_sha256':{name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in sources},'results':records}
        args.benchmark_output.parent.mkdir(parents=True,exist_ok=True)
        args.benchmark_output.write_text(json.dumps(report,indent=2)+'\n')

if __name__=='__main__':main()
