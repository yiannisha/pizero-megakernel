"""CUDA-event and resident-VM instruction profile for the three native pi0 regions."""
import argparse
import json
from pathlib import Path
import statistics
import sys
from types import SimpleNamespace

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'third_party/openpi/src')]
from native_pi0.model import NativePi0
from native_pi0.program import program_summary
from openpi.models_pytorch.pi0_pytorch import PI0Pytorch

OPS=('norm1','qkv','attention','out_projection','norm2','ffn_up','ffn_down')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--library',type=Path,required=True)
    parser.add_argument('--report',type=Path,required=True)
    parser.add_argument('--iterations',type=int,default=7)
    args=parser.parse_args()
    torch.manual_seed(101);torch.set_num_threads(8)
    config=SimpleNamespace(pi05=False,paligemma_variant='gemma_2b',action_expert_variant='gemma_300m',
                           dtype='bfloat16',action_dim=32,action_horizon=50,pytorch_compile_mode=None)
    with torch.device('cuda'):model=PI0Pytorch(config).eval()
    pg=model.paligemma_with_expert
    modules={'vision':pg.paligemma.vision_tower.vision_model,
             'prefix':pg.paligemma.language_model,'expert':pg.gemma_expert.model}
    param_bytes={name:sum(p.numel()*p.element_size() for p in module.parameters()) for name,module in modules.items()}
    marks={};original=NativePi0.block
    def marked(self,*a,**kw):
        first=len(self.stages);result=original(self,*a,**kw)
        marks[kw['name']]=(first,len(self.stages));return result
    NativePi0.block=marked
    images=torch.randn((3,224,224,3),device='cuda')
    state=torch.randn((1,32),device='cuda')
    noise=torch.randn((1,50,32),device='cuda')
    tokens=torch.empty((1,0),device='cuda',dtype=torch.int64)
    report={'workload':{'views':3,'prompt_len':0,'chunk':50,'flow_steps':10,'seed':101},
            'gpu':torch.cuda.get_device_name(),'parameter_bytes':param_bytes,
            'timing_note':'VM consumer instruction cycles include dependency waits; active worker counts are exact CTA schedule counts.',
            'components':{}}
    with NativePi0.from_model(model,num_views=3,prompt_len=0,chunk_size=50,flow_steps=10,
                              library=args.library,partitioned=True,prefix_flash=True,fast_patch=True) as native:
        native.forward(images,state,noise,tokens);torch.cuda.synchronize()
        for name,(first,last) in native.component_ranges.items():
            fn=lambda:native.launch_component(name,images,state,noise,tokens)
            for _ in range(2):fn()
            torch.cuda.synchronize();samples=[]
            for _ in range(args.iterations):
                begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                begin.record();fn();end.record();end.synchronize();samples.append(begin.elapsed_time(end))
            timings=native.read_timings(name)
            code=native.component_codes[name]
            valid=(code[:,:,0]!=0)&(timings[:,:,11]>0)&(timings[:,:,12]>0)
            if not valid.any():raise RuntimeError('The library was built without ComputeConfig.TIMING_RECORD_ENABLED')
            elapsed=timings[:,:,12].astype(np.int64)-timings[:,:,11].astype(np.int64)
            stage_stats={}
            for local in range(last-first):
                mask=valid&(code[:,:,1]==local)
                if not mask.any():continue
                stage_stats[first+local]={'active_workers':int(mask.any(axis=1).sum()),
                                          'consumer_cycles_median':int(np.median(elapsed[mask])),
                                          'consumer_cycles_max':int(elapsed[mask].max()),
                                          'tasks':int(native.stage_tasks[first+local]),
                                          'opcode':int(native.opcodes[first+local])}
            grouped={}
            for block,(a,b) in marks.items():
                if not block.startswith(name+'.'):continue
                for offset,stage in enumerate(range(a,b)):
                    if offset>=len(OPS) or stage not in stage_stats:continue
                    grouped.setdefault(OPS[offset],[]).append(stage_stats[stage])
            operations={op:{'layers':len(items),
                            'active_workers_median':statistics.median(item['active_workers'] for item in items),
                            'consumer_cycles_median':statistics.median(item['consumer_cycles_median'] for item in items)}
                        for op,items in grouped.items()}
            worker_spans=[]
            for worker in range(native.workers):
                mask=valid[worker]
                if mask.any():worker_spans.append(int(timings[worker,mask,12].max()-timings[worker,mask,11].min()))
            report['components'][name]={'gpu_median_ms':statistics.median(samples),'gpu_samples_ms':samples,
                                        'program':program_summary(code),
                                        'worker_span_cycles_median':statistics.median(worker_spans),
                                        'operations':operations,
                                        'top_stages_by_consumer_cycles':sorted(
                                            ({'stage':stage,**stat} for stage,stat in stage_stats.items()),
                                            key=lambda item:item['consumer_cycles_median'],reverse=True)[:12]}
            print(name,report['components'][name]['gpu_median_ms'],operations,flush=True)
    args.report.parent.mkdir(parents=True,exist_ok=True)
    args.report.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
