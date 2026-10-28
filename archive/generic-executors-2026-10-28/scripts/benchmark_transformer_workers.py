"""Sweep VM worker counts on an unchanged full-depth native transformer stack.

The supplied build report pins the compiled library. This is a synthetic
component benchmark against serial native blocks, not a policy benchmark.
"""
import argparse
from contextlib import ExitStack
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from native_pi0.normalization import ResidentNorm
from native_pi0.transformer_stack import ResidentTransformerStack
from native_pi0.launch_trace import LaunchTrace
from scripts.validate_transformer_stack import make_block
from scripts.validate_transformer_block import timing
from scripts.validate_dag_matmul import check_diagnostics


class GpuMonitor:
    """Record competing CUDA contexts and clocks during timed samples."""
    def __enter__(self):
        self.active=False;self.others=set();self.process_samples=0;self.clocks=[]
        self.processes=[];self.threads=[]
        for query,callback in [('compute-apps=pid',self.process_line),('gpu=clocks.sm,clocks.mem,power.draw,utilization.gpu',self.clock_line)]:
            process=subprocess.Popen(['nvidia-smi','--query-'+query,'--format=csv,noheader,nounits','-lms','100'],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
            self.processes.append(process)
            def read(p=process,cb=callback):
                for line in p.stdout:
                    if self.active:cb(line.strip())
            thread=threading.Thread(target=read,daemon=True);thread.start();self.threads.append(thread)
        return self

    def process_line(self,line):
        if line.isdigit():
            self.process_samples+=1
            if int(line)!=os.getpid():self.others.add(int(line))

    def clock_line(self,line):
        try:self.clocks.append([float(x.strip()) for x in line.split(',')])
        except ValueError:pass

    def measure(self,fn):
        self.others=set();self.process_samples=0;self.clocks=[];self.active=True
        try:result=timing(fn)
        finally:self.active=False
        result['gpu_monitor']={'other_compute_pids':sorted(self.others),'process_samples':self.process_samples,
            'clock_columns':['sm_mhz','memory_mhz','power_watts','gpu_utilization_percent'],'clock_samples':self.clocks}
        result['gpu_isolation_verified']=self.process_samples>0 and not self.others
        return result

    def __exit__(self,*_):
        self.active=False
        for process in self.processes:process.terminate()
        for process in self.processes:process.wait(timeout=5)
        for thread in self.threads:thread.join(timeout=1)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--library',type=Path,required=True);parser.add_argument('--build-report',type=Path,required=True)
    parser.add_argument('--kind',choices=('expert','prefix'),default='expert');parser.add_argument('--report',type=Path,required=True)
    parser.add_argument('--workers',type=int,nargs='+',default=[8,16,32,48,64,96,148])
    args=parser.parse_args();torch.set_num_threads(8);torch.manual_seed(739);rng=np.random.default_rng(739)
    build=json.loads(args.build_report.read_text());library=args.library.resolve();libhash=hashlib.sha256(library.read_bytes()).hexdigest()
    assert libhash==build['library_sha256'],'Library does not match build report'
    for path,h in build['source_sha256'].items():
        if path.startswith('native_pi0/') and path.endswith('.py'):
            assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==h,('Python API changed since validation',path)
    if not args.workers or len(set(args.workers))!=len(args.workers) or any(not 1<=w<=148 for w in args.workers):raise ValueError('Unique worker counts in 1..148 required')
    rows,width,ff,cached_rows,splits=(51,1024,4096,768,8) if args.kind=='expert' else (768,2048,16384,0,1)
    with ExitStack() as owners,GpuMonitor() as monitor:
        blocks=[]
        for layer in range(18):
            blocks.append(make_block(owners,library,rng,batch=1,rows=rows,width=width,ff=ff,heads=8,hk=1,hd=256,
                fp8=True,splits=splits,workers=148,cached_rows=cached_rows,write_cache=True))
            if (layer+1)%3==0:print('prepared layers',layer+1,flush=True)
        blocks[0].inputs[:rows].copy_(torch.randn((rows,width),device='cuda').to(torch.bfloat16))
        final=owners.enter_context(ResidentNorm(library,torch.zeros_like(blocks[-1].down.output),precision='bf16',gamma=1+torch.randn(width)*.1,workers=148))
        def serial():
            for block in blocks:block.forward()
            final.forward()
        entries=[];snapshot=None
        # Parent worker counts can differ from the child contexts. Every
        # scheduled math operation uses explicit node task IDs, not CTA IDs.
        for workers in args.workers:
            with ResidentTransformerStack(library,blocks,final,workers=workers) as stack:
                stack.forward();torch.cuda.synchronize();check_diagnostics(stack)
                if snapshot is None:snapshot=stack.output.clone()
                else:assert torch.equal(stack.output,snapshot),('worker-dependent output',workers)
                with LaunchTrace(library) as trace:stack.forward();torch.cuda.synchronize()
                assert trace.count==1
                samples=monitor.measure(stack.forward)
                diagnostic=check_diagnostics(stack)
                # Consumer intervals include operand/MMA waits. They are
                # overlapping service intervals, not additive kernel times.
                services=[]
                names=('qkv','attention','o_residual','pre_norm','post_norm','gate_up_geglu','down_residual')
                for kind,name in enumerate(names):
                    selected=stack.node_kind==kind
                    durations=(diagnostic['ends'][selected]-diagnostic['starts'][selected]).astype(np.float64)/1000
                    services.append({'kind':name,'nodes':int(selected.sum()),'median_us':float(np.median(durations)),
                        'p90_us':float(np.percentile(durations,90)),'max_us':float(durations.max())})
                samples['consumer_service_intervals']=services
                samples['consumer_timeline_ms']=float((diagnostic['ends'].max()-diagnostic['starts'].min())/1e6)
                stack.forward();torch.cuda.synchronize();assert torch.equal(stack.output,snapshot)
                samples.update({'workers':workers,'p10_ms':float(np.percentile(samples['samples_ms'],10)),
                    'p90_ms':float(np.percentile(samples['samples_ms'],90)),'cuda_launch_count':trace.count})
                entries.append(samples);print(workers,samples['median_ms'],'ms',flush=True)
        serial();torch.cuda.synchronize();assert torch.equal(final.output[:rows],snapshot)
        baseline=monitor.measure(serial);serial();torch.cuda.synchronize();assert torch.equal(final.output[:rows],snapshot)
        baseline.update({'p10_ms':float(np.percentile(baseline['samples_ms'],10)),'p90_ms':float(np.percentile(baseline['samples_ms'],90)),
            'child_workers':148,'cuda_launch_count':19})
        with LaunchTrace(library) as trace:serial();torch.cuda.synchronize()
        assert trace.count==19
        assert hashlib.sha256(library.read_bytes()).hexdigest()==libhash
        report={'timestamp_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'kind':args.kind,'depth':18,'rows':rows,'width':width,'ffn_width':ff,
            'cached_prefix_rows':cached_rows,'library_sha256':libhash,'compiled_source_sha256':build['source_sha256'],
            'benchmark_script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'output_sha256':hashlib.sha256(snapshot.cpu().view(torch.uint8).numpy().tobytes()).hexdigest(),
            'outputs':'bit-exact across every worker count and serial native blocks','workers':entries,'serial_native':baseline,
            'scope':'Synthetic native stack with independently supplied expert caches; no end-to-end model or external speedup claim'}
        args.report.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
