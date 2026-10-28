"""Verify real tile-range loops on the original VM's static instruction path."""
import argparse
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import torch

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from native_pi0.dense import ResidentDense
from native_pi0.linear import NativeLinearProgram
from native_pi0.launch_trace import LaunchTrace


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--library',type=Path);p.add_argument('--quick',action='store_true')
    p.add_argument('--profile',action='store_true');p.add_argument('--report',type=Path);args=p.parse_args()
    torch.manual_seed(113);torch.set_num_threads(8);torch.backends.cuda.matmul.allow_tf32=False
    library=args.library or Path('/tmp/pi0_static_linear.so')
    paths=['native_pi0/native.cu','native_pi0/program.py','native_pi0/linear.py','native_pi0/dense.cu','native_pi0/dense.py','native_pi0/tcgen_probe.cu',
        'scripts/validate_native_linear.py','third_party/Megakernels/include/controller/instruction_fetch.cuh','third_party/Megakernels/include/megakernel.cuh']
    hashes={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in paths}
    if args.library is None:
        subprocess.run(['nvcc','-O3','-lineinfo','-std=c++20','-arch=sm_100a','-DKITTENS_HOPPER','-DKITTENS_BLACKWELL','-DNATIVE_COMPUTE_CONSUMER_WARPS=4',
            '--expt-relaxed-constexpr','--expt-extended-lambda','-I'+str(ROOT/'third_party/Megakernels/include'),
            '-I'+str(ROOT/'third_party/Megakernels/ThunderKittens/include'),'-shared','-Xcompiler','-fPIC',str(ROOT/'native_pi0/native.cu'),
            str(ROOT/'native_pi0/launch_trace.cpp'),'-lcupti','-Xlinker','-rpath','-Xlinker','/usr/local/cuda/lib64','-lcuda','-o',str(library)],check=True)
    records=[];cases=[('one-worker-odd-range',1,128,128,384,256),('cross-band',4,256,192,512,384)]
    if not args.quick:cases += [('expert-shaped',148,51,1024,8192,1024),('prefix-shaped',148,768,2048,4096,2048)]
    for label,workers,m,k,n,o in cases:
        with ExitStack() as owners:
            x=torch.randn((m,k),device='cuda',dtype=torch.bfloat16)
            w=(torch.randn((n,k),device='cuda')*.02).bfloat16();v=(torch.randn((o,n),device='cuda')*.02).bfloat16()
            first=owners.enter_context(ResidentDense(library,x,w,workers=workers))
            second=owners.enter_context(ResidentDense(library,torch.zeros((m,n),device='cuda',dtype=torch.bfloat16),v,workers=workers))
            program=owners.enter_context(NativeLinearProgram(library,[first,second],workers=workers))
            for replay in range(3):
                stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    first.inputs.copy_(torch.randn_like(first.inputs));first.output.fill_(float('nan'));second.output.fill_(float('nan'));program.forward()
                stream.synchronize();counters=program.read_counters()
                assert counters['epoch']==program.epoch and (counters['counts']==counters['expected']).all(),counters
                snapshots=[first.output.clone(),second.output.clone()]
                first.forward();second.forward();torch.cuda.synchronize()
                for stage,snapshot in zip((first,second),snapshots):assert torch.equal(stage.output,snapshot),(label,replay,'split mismatch')
                gold=torch.nn.functional.linear(torch.nn.functional.linear(first.inputs,first.weights),second.weights)
                error=float((program.output.float()-gold.float()).square().mean().sqrt()/gold.float().square().mean().sqrt())
                assert error<=.003,(label,error)
            record={'label':label,'plan':program.plan,'sources':[ -1,0 ],'replays':3,'same_native_stages':'bit-exact','torch_nrmse':error,
                'counter_check':'exact per row band','output_sha256':hashlib.sha256(program.output.cpu().view(torch.uint8).numpy().tobytes()).hexdigest()}
            if args.profile:
                with LaunchTrace(library) as trace:program.forward();torch.cuda.synchronize()
                assert trace.count==1;record['cuda_launch_count']=trace.count;record['kernel_symbol']=trace.symbol
            records.append(record);print(label,program.plan,'exact outputs and counters',flush=True)
    assert hashes=={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in paths},'Sources changed during validation'
    result={'scope':'Native static tile-range matrix programs; not yet complete Pi0','source_sha256':hashes,
        'library_sha256':hashlib.sha256(Path(library).read_bytes()).hexdigest(),'results':records}
    if args.report:args.report.write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
