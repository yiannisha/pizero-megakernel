"""Original native VM dynamic ready-queue and termination correctness probe."""
import argparse
import ctypes
import datetime
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from native_pi0.dag_probe import ResidentDagProbe,lcg_advance
from native_pi0.launch_trace import LaunchTrace


def permute(graph,rng):
    ids=rng.permutation(len(graph));result=[None]*len(graph)
    for old,parents in enumerate(graph):result[int(ids[old])]=[int(ids[p]) for p in parents]
    return result


def random_graph(nodes,rng):
    return [[] if not i or i%17==0 else list(map(int,rng.choice(i,size=min(i,int(rng.integers(1,5))),replace=False))) for i in range(nodes)]


def validate(probe,result,seed):
    np.testing.assert_array_equal(result['values'],probe.expected(seed))
    np.testing.assert_array_equal(result['visits'],np.full(probe.nodes,probe.epoch,dtype=np.uint64))
    np.testing.assert_array_equal(result['counts'],[probe.nodes,probe.nodes,probe.nodes,0,probe.epoch])
    assert np.all(result['owners']<probe.workers)
    assert np.all(result['ordinals']<probe.nodes)
    assert int(result['exits'].sum())==probe.nodes
    assert np.all(result['ends']>=result['starts'])
    for node,parents in enumerate(probe.parents):
        if parents:assert result['starts'][node]>=max(result['ends'][p] for p in parents),(node,parents)
    return {'seed':seed,'epoch':probe.epoch,'nodes_executed_once':probe.nodes,'queue_head_tail_completed':result['counts'][:3].tolist(),
            'terminal_ordinals_sum':int(result['exits'].sum()),'max_terminal_ordinal':int(result['exits'].max()),
            'active_ctas':int(len(np.unique(result['owners']))),'dependency_timestamps':'ordered',
            'value_sha256':hashlib.sha256(result['values'].tobytes()).hexdigest()}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile',action='store_true')
    parser.add_argument('--report',type=Path)
    args=parser.parse_args()
    if args.report and not args.profile:parser.error('Reports require --profile launch gates')
    torch.set_num_threads(8);rng=np.random.default_rng(101);records=[]
    with tempfile.TemporaryDirectory() as directory:
        library=Path(directory)/'dag.so'
        subprocess.run(['nvcc','-O3','-lineinfo','-std=c++20','-arch=sm_100a','-DKITTENS_HOPPER','-DKITTENS_BLACKWELL',
            '--expt-relaxed-constexpr','--expt-extended-lambda','-I'+str(ROOT/'third_party/Megakernels/include'),
            '-I'+str(ROOT/'third_party/Megakernels/ThunderKittens/include'),'-shared','-Xcompiler','-fPIC',
            str(ROOT/'native_pi0/dag_probe.cu'),str(ROOT/'native_pi0/launch_trace.cpp'),'-lcupti','-Xlinker','-rpath',
            '-Xlinker','/usr/local/cuda/lib64','-lcuda','-o',str(library)],check=True)
        query=ctypes.CDLL(str(library)).dag_probe_resources
        query.argtypes=[ctypes.c_void_p,ctypes.c_int];query.restype=ctypes.c_int
        values=np.empty(6,dtype=np.int32)
        assert query(values.ctypes.data,len(values))==0
        resources=dict(zip(('ctas_per_sm','sms','static_shared_bytes','dynamic_shared_bytes','registers','local_bytes'),map(int,values)))
        assert resources['ctas_per_sm']==1,resources
        for value in (0,17,0xffffffff):
            for steps in (0,1,2,31,1000):
                expected=value
                for _ in range(steps):expected=(expected*1664525+1013904223)&0xffffffff
                assert lcg_advance(value,steps)==expected
        for invalid in ([],[[0]],[[1],[0]],[[],[0,0]],[[],[2]]):
            try:ResidentDagProbe(library,invalid)
            except ValueError:pass
            else:raise AssertionError('Invalid graph accepted')
        cases=[]
        for nodes in (1,2,3,17,65,257,1024):
            cases.append((f'chain{nodes}',[[]]+[[i-1] for i in range(1,nodes)],1 if nodes<100 else 16))
        cases.append(('independent513',[[] for _ in range(513)],148))
        for width,workers in ((147,148),(511,16),(511,148)):
            graph=[[]]+[[0] for _ in range(width)]+[list(range(1,width+1))]
            cases.append((f'fanout_join{width}',permute(graph,rng),workers))
        layered=[[] for _ in range(16)]
        for layer in range(1,32):
            for col in range(16):layered.append([(layer-1)*16+col,(layer-1)*16+(col+1)%16])
        for workers in (1,16,148):cases.append(('layered512',permute(layered,rng),workers))
        graph=random_graph(1024,rng)
        for workers in (1,16,148):cases.append(('random1024',permute(graph,rng),workers))
        cases.append(('random4096',permute(random_graph(4096,rng),rng),148))
        for case_index,(name,graph,workers) in enumerate(cases):
            delays=(np.arange(len(graph),dtype=np.int64)%7)*128
            for node,parents in enumerate(graph):
                if not parents:delays[node]+=4096
            with ResidentDagProbe(library,graph,workers=workers,delays=delays) as probe:
                replays=[]
                for replay in range(3):
                    seed=1000+case_index*3+replay
                    stream=torch.cuda.Stream();probe.forward(seed,stream);stream.synchronize()
                    replays.append(validate(probe,probe.read(),seed))
                record={'name':name,'nodes':len(graph),'edges':sum(map(len,graph)),'roots':len(probe.roots),'workers':workers,'replays':replays}
                if args.profile:
                    with LaunchTrace(library) as trace:probe.forward(91);torch.cuda.synchronize()
                    assert trace.count==1 and 'megakernel' in trace.symbol,(trace.count,trace.symbol)
                    validate(probe,probe.read(),91)
                    record.update(cuda_launch_count=trace.count,kernel_symbol=trace.symbol)
                records.append(record)
                print(f'{name}: {len(graph)} nodes, {workers} CTAs, exact values/epochs/queue counts, three replays',flush=True)
        # Each grid uses 100 CTAs, so their combined 200 CTAs exceed 148 SMs.
        # Slow roots make compute overlap observable, rather than assuming that
        # submissions on two streams imply concurrent GPU execution.
        graph=[[]]+[[i-1] for i in range(1,257)];delays=np.full(257,1024,dtype=np.int32);delays[0]=1_000_000
        with ResidentDagProbe(library,graph,workers=100,delays=delays) as left, ResidentDagProbe(library,graph,workers=100,delays=delays) as right:
            streams=[torch.cuda.Stream(),torch.cuda.Stream()]
            for stream in streams:stream.wait_stream(torch.cuda.current_stream())
            left.forward(711,streams[0]);right.forward(927,streams[1]);torch.cuda.synchronize()
            results=[left.read(),right.read()]
            for probe,result,seed in zip((left,right),results,(711,927)):validate(probe,result,seed)
            root_overlap=max(0,min(int(r['ends'][0]) for r in results)-max(int(r['starts'][0]) for r in results))
            if args.profile:assert root_overlap>0,'No concurrent root work observed in unsanitized run'
            concurrent={'workers_per_instance':100,'total_submitted_ctas':200,'root_compute_overlap_ns':root_overlap,'values_and_epochs':'exact'}
            print(f'Independent 100-CTA grids: exact outputs, root overlap {root_overlap} ns',flush=True)
            if args.profile:
                with LaunchTrace(library) as trace:
                    left.forward(1);right.forward(2);torch.cuda.synchronize()
                assert trace.count==2,trace.count
                concurrent['two_forward_launch_count']=trace.count
    if args.report:
        sources=['native_pi0/dag_scheduler.cuh','native_pi0/dag_probe.cu','native_pi0/dag_probe.py','scripts/validate_dag.py',
                 'native_pi0/launch_trace.cpp','native_pi0/launch_trace.py','native_pi0/patches/megakernels-dynamic-instructions.patch',
                 'native_pi0/patches/megakernels-serial-barrier-init.patch','third_party/Megakernels/include/megakernel.cuh',
                 'third_party/Megakernels/include/util.cuh','third_party/Megakernels/include/controller/controller.cuh',
                 'third_party/Megakernels/include/controller/instruction_fetch.cuh']
        report={'timestamp_utc':datetime.datetime.now(datetime.UTC).isoformat(),'seed':101,
                'scope':'Scalar DAG scheduling and VM retirement probe, not GEMM/model or pretrained validation',
                'mechanism':'MPMC ready queue; lazy epoch dependency counters; resident initializer; dynamic VM fetch and terminal slot',
                'initialization':'Inside the single native invocation; no GPU reset kernel/copy in forward',
                'upstreams':json.loads((ROOT/'upstreams.json').read_text()),
                'environment':{'gpu':torch.cuda.get_device_name(),'torch':torch.__version__,'cuda':torch.version.cuda},
                'resources':resources,
                'source_sha256':{name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in sources},
                'results':records,'concurrent':concurrent}
        args.report.parent.mkdir(parents=True,exist_ok=True);args.report.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
