"""Native ready-queue matrix chains, including compact GeGLU/down FFN cores."""
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
from contextlib import ExitStack
from pathlib import Path

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from native_pi0.primitives import ResidentNVFP4
from native_pi0.dag_matmul import ResidentDagMatmul
from native_pi0.quantization import PackedNVFP4,pack_nvfp4,interleave_gate_up,swizzle_scales
from native_pi0.launch_trace import LaunchTrace


def physical(packed):
    return PackedNVFP4(packed.data,packed.scales,packed.global_scale,(packed.data.shape[0],packed.data.shape[1]*2)).dequantize()


def nrmse(actual,expected):
    d=actual.astype(np.float64)-expected.astype(np.float64)
    return float(np.linalg.norm(d)/max(np.linalg.norm(expected.astype(np.float64)),1e-30))


def check_diagnostics(graph):
    d=graph.read_diagnostics();n=graph._queue.nodes;epoch=graph._queue.epoch
    np.testing.assert_array_equal(d['counts'],[n,n,n,0,epoch])
    np.testing.assert_array_equal(d['values'],np.full(n,epoch,dtype=np.uint32))
    np.testing.assert_array_equal(d['visits'],np.full(n,epoch,dtype=np.uint64))
    assert int(d['exits'].sum())==n
    assert np.all(d['owners']<graph.workers)
    for node,parents in enumerate(graph._queue.parents):
        if parents:assert d['starts'][node]>=max(d['ends'][p] for p in parents)
    return d


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile',action='store_true')
    parser.add_argument('--quick',action='store_true')
    parser.add_argument('--baseline-queue',action='store_true',help='Compile the original atomic-poll/root-release queue')
    parser.add_argument('--per-task-tensormap-acquire',action='store_true')
    parser.add_argument('--idle-sleep-ns',type=int,choices=(32,128,512,1024),default=512)
    parser.add_argument('--cas-decrement',action='store_true')
    parser.add_argument('--global-compute-metadata',action='store_true')
    parser.add_argument('--uncoalesced-dependencies',action='store_true')
    parser.add_argument('--single-lane-bootstrap',action='store_true')
    parser.add_argument('--node-order',choices=('ordered','permuted'),default='ordered')
    parser.add_argument('--benchmark-output',type=Path)
    args=parser.parse_args()
    if args.benchmark_output and not args.profile:parser.error('Timing requires --profile launch gates')
    torch.set_num_threads(8);rng=np.random.default_rng(137);records=[]
    with tempfile.TemporaryDirectory() as directory:
        library=Path(directory)/'dag_matmul.so'
        subprocess.run(['nvcc','-O3','-lineinfo','-std=c++20','-arch=sm_100a','-DKITTENS_HOPPER','-DKITTENS_BLACKWELL',
            '-DNATIVE_COMPUTE_CONSUMER_WARPS=4','-DNATIVE_FP4_K_TILE=256','-DNATIVE_DAG_MATMUL_VALIDATION',
            '-DNATIVE_DAG_BASELINE_QUEUE='+str(int(args.baseline_queue)),
            '-DNATIVE_DAG_IDLE_SLEEP_NS='+str(args.idle_sleep_ns),
            '-DNATIVE_DAG_CAS_DECREMENT='+str(int(args.cas_decrement)),
            '-DNATIVE_DAG_CACHE_COMPUTE_METADATA='+str(int(not args.global_compute_metadata)),
            '-DNATIVE_DAG_COALESCE_DEPENDENCIES='+str(int(not args.uncoalesced_dependencies)),
            '-DNATIVE_DAG_WARP_BOOTSTRAP='+str(int(not args.single_lane_bootstrap)),
            '-DNATIVE_DAG_PER_TASK_TENSORMAP_ACQUIRE='+str(int(args.per_task_tensormap_acquire)),
            '--expt-relaxed-constexpr','--expt-extended-lambda','-I'+str(ROOT/'third_party/Megakernels/include'),
            '-I'+str(ROOT/'third_party/Megakernels/ThunderKittens/include'),'-I'+str(ROOT/'third_party/FlashRT/csrc'),
            '-I'+str(ROOT/'third_party/cutlass/include'),'-shared','-Xcompiler','-fPIC',str(ROOT/'native_pi0/dag_matmul.cu'),
            str(ROOT/'native_pi0/fp4_output_reference.cu'),str(ROOT/'native_pi0/launch_trace.cpp'),
            '-lcupti','-Xlinker','-rpath','-Xlinker','/usr/local/cuda/lib64','-lcuda','-o',str(library)],check=True)
        module=ctypes.CDLL(str(library));split=module.validation_split_dag_matmul
        split.argtypes=[ctypes.c_void_p]*2;split.restype=ctypes.c_int
        oracle=module.reference_fp4_output;oracle.argtypes=[ctypes.c_void_p]*3+[ctypes.c_int]*2+[ctypes.c_float,ctypes.c_int];oracle.restype=ctypes.c_int
        query=module.dag_matmul_resources;query.argtypes=[ctypes.c_void_p,ctypes.c_int];query.restype=ctypes.c_int
        values=np.empty(6,dtype=np.int32);assert query(values.ctypes.data,len(values))==0
        resources=dict(zip(('ctas_per_sm','sms','static_shared_bytes','dynamic_shared_bytes','registers','local_bytes'),map(int,values)))
        assert resources['ctas_per_sm']==1
        shapes=[('compact',1,64,256,128,1,1.),('compact',3,192,768,256,4,8.),('compact',129,192,768,256,148,1.)]
        if not args.quick:shapes.extend([('compact',51,1024,8192,1024,64,1.),('gelu',256,1152,4304,1152,148,1.),('compact',768,2048,32768,2048,148,8.)])
        shapes.append(('three-stage',3,192,768,256,4,1.))
        for mode,m,k,n,out_n,workers,gs in shapes:
            a=pack_nvfp4(rng.normal(size=(m,k)).astype(np.float32),global_scale=.125,row_alignment=128,column_alignment=64)
            if mode=='gelu':first=rng.normal(0,.1,size=(n,k)).astype(np.float32);hidden=(n+127)//128*128
            else:
                first=interleave_gate_up(rng.normal(0,.1,size=(n//2,k)).astype(np.float32),rng.normal(0,.1,size=(n//2,k)).astype(np.float32));hidden=n//2
            weights=[pack_nvfp4(first,global_scale=.25,scale_policy='ceil',row_alignment=128,column_alignment=64),
                     pack_nvfp4(rng.normal(0,.05,size=(out_n,n if mode=='gelu' else hidden)).astype(np.float32),global_scale=.5,row_alignment=128,column_alignment=64)]
            placeholders=[a,pack_nvfp4(np.zeros((m,hidden),np.float32),global_scale=gs,row_alignment=128,column_alignment=64)]
            if mode=='three-stage':
                weights.append(pack_nvfp4(rng.normal(0,.05,size=(128,out_n)).astype(np.float32),global_scale=.25,row_alignment=128,column_alignment=64))
                placeholders.append(pack_nvfp4(np.zeros((m,out_n),np.float32),global_scale=1.,row_alignment=128,column_alignment=64))
            with ExitStack() as stack:
                stages=[]
                for index,(activation,weight) in enumerate(zip(placeholders,weights)):
                    final=index==len(weights)-1
                    regular=not final and (mode=='gelu' or index>0)
                    bias=None;residual=None
                    if final or regular:
                        bias=rng.normal(0,.02,size=weight.data.shape[0]).astype(np.float32)
                        bias[weight.logical_shape[0]:]=0
                    if final:
                        residual=rng.normal(0,.1,size=(activation.data.shape[0],weight.data.shape[0])).astype(np.float32)
                        residual[m:]=0
                    stages.append(stack.enter_context(ResidentNVFP4(library,activation,weight,workers=workers,
                        output_dtype=torch.bfloat16 if final else torch.uint8,output_global_scale=1. if final or index else gs,
                        output_gelu=regular,output_geglu=not final and index==0 and mode!='gelu',bias=bias,residual=residual)))
                node_count=sum((s.a.shape[0]//128)*(s.b.shape[0]//128) for s in stages)
                permutation=rng.permutation(node_count)  # Keep input RNG identical between order variants.
                graph=stack.enter_context(ResidentDagMatmul(library,stages,workers=workers,
                    node_order=permutation if args.node_order=='permuted' else None))
                maximum=0.;last_diagnostic=None
                for replay in range(3):
                    current=pack_nvfp4(rng.normal(size=(m,k)).astype(np.float32),global_scale=.125,row_alignment=128,column_alignment=64)
                    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(stream):
                        stages[0].a.copy_(torch.from_numpy(current.data));stages[0].sa.copy_(torch.from_numpy(swizzle_scales(current.scales)))
                        for stage in stages:
                            stage.output.fill_(0 if stage.output.dtype==torch.uint8 else float('nan'))
                            if stage.output_scales is not None:stage.output_scales.fill_(255)
                        graph.forward()
                    stream.synchronize();last_diagnostic=check_diagnostics(graph)
                    snapshots=[s.output.clone() for s in stages];scale_snapshots=[None if s.output_scales is None else s.output_scales.clone() for s in stages]
                    reference_input=current
                    for index,(stage,weight) in enumerate(zip(stages,weights)):
                        # Independent CPU matmul gate, then pinned CUDA epilogue
                        # oracle driven by a separately launched FP32 projection.
                        with ResidentNVFP4(library,reference_input,weight,workers=workers) as plain:
                            projection=plain.forward().cpu().numpy()
                        cpu=physical(reference_input)@physical(weight).T
                        np.testing.assert_allclose(projection,cpu,atol=1e-5*max(1,float(np.abs(cpu).max())),rtol=3e-5)
                        for extra in (stage.residual,stage.bias):
                            if extra is not None:
                                extra=extra.float().cpu().numpy();projection+=extra;cpu+=extra
                        if stage.output.dtype==torch.uint8:
                            width=stage.output_width;packed=np.empty((projection.shape[0],width//2),np.uint8);sf=np.empty((projection.shape[0],width//16),np.uint8)
                            error=oracle(projection.ctypes.data,packed.ctypes.data,sf.ctypes.data,*projection.shape,float(stage.output_global_scale),2 if index==0 and mode!='gelu' else 1)
                            assert error==0,error
                            np.testing.assert_array_equal(snapshots[index].cpu().numpy(),packed)
                            np.testing.assert_array_equal(scale_snapshots[index].cpu().numpy(),swizzle_scales(sf))
                            reference_input=PackedNVFP4(packed,sf,stage.output_global_scale,(packed.shape[0],width))
                        else:
                            gold=torch.from_numpy(cpu).bfloat16().float().numpy();actual=snapshots[index].float().cpu().numpy()
                            error=nrmse(actual[:m],gold[:m]);maximum=max(maximum,error);assert error<=.005,error
                    error=split(graph.handle,torch.cuda.current_stream().cuda_stream);assert error==0,error;torch.cuda.synchronize()
                    for stage,expected,sf in zip(stages,snapshots,scale_snapshots):
                        torch.testing.assert_close(stage.output,expected,rtol=0,atol=0)
                        if sf is not None:torch.testing.assert_close(stage.output_scales,sf,rtol=0,atol=0)
                record={'mode':mode,'m':m,'padded_m':stages[0].a.shape[0],'k':k,'n_first_logical':n,'n_hidden':hidden,'n_out':stages[-1].output_width,
                        'workers':workers,'stages':len(stages),'nodes':node_count,'edges':sum(map(len,graph._queue.parents)),
                        'plan':graph._queue.plan,
                        'replays':3,'node_order':args.node_order,'producer_payloads_and_scales':'byte-exact pinned CUDA references',
                        'split_outputs':'bit-exact','final_nrmse':maximum,'final_nrmse_limit':.005,
                        'bias_stages':[i for i,s in enumerate(stages) if s.bias is not None],
                        'residual_stages':[i for i,s in enumerate(stages) if s.residual is not None],
                        'stage_scales':[{'activation':float(s.activation_global_scale),'weight':float(s.weight_global_scale),
                                         'output':float(s.output_global_scale),'output_dtype':str(s.output.dtype)} for s in stages],
                        'output_sha256':[hashlib.sha256(s.contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest() for s in snapshots],
                        'scale_sha256':[None if s is None else hashlib.sha256(s.cpu().numpy().tobytes()).hexdigest() for s in scale_snapshots],
                        'diagnostics':'every node once; exact queue/epoch counts; ordered dependency timestamps',
                        'max_terminal_ordinal':int(last_diagnostic['exits'].max())}
                activity=[]
                for index in range(len(stages)):
                    mask=graph.node_stage==index;owners=last_diagnostic['owners'][mask]
                    starts=last_diagnostic['starts'][mask];ends=last_diagnostic['ends'][mask]
                    counts=np.bincount(owners.astype(np.int64),minlength=workers)
                    activity.append({'stage':index,'active_ctas':int(np.count_nonzero(counts)),
                        'max_tasks_per_cta':int(counts.max()),'consumer_span_us':float((ends.max()-starts.min())/1000),
                        'median_consumer_interval_us':float(np.median((ends-starts).astype(np.float64))/1000)})
                record['stage_activity']=activity
                if args.profile:
                    with LaunchTrace(library) as trace:graph.forward();torch.cuda.synchronize()
                    assert trace.count==1 and 'DagMatmul' in trace.symbol,(trace.count,trace.symbol)
                    check_diagnostics(graph);record.update(cuda_launch_count=trace.count,kernel_symbol=trace.symbol)
                    with LaunchTrace(library) as trace:
                        error=split(graph.handle,torch.cuda.current_stream().cuda_stream);torch.cuda.synchronize()
                    assert error==0 and trace.count==len(stages),(error,trace.count)
                    record['split_cuda_launch_count']=trace.count
                if args.benchmark_output:
                    def split_forward():
                        error=split(graph.handle,torch.cuda.current_stream().cuda_stream)
                        if error:raise RuntimeError(f'Split baseline CUDA error {error}')
                    def measure(call):
                        for _ in range(5):call()
                        torch.cuda.synchronize();start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);gpu=[];wall=[]
                        for _ in range(30):
                            torch.cuda.synchronize();t0=time.perf_counter_ns();start.record();call();end.record();end.synchronize()
                            gpu.append(start.elapsed_time(end));wall.append((time.perf_counter_ns()-t0)/1e6)
                        def stats(v):return {'median_ms':statistics.median(v),'p10_ms':float(np.percentile(v,10)),'p90_ms':float(np.percentile(v,90)),'samples_ms':v}
                        return {'gpu':stats(gpu),'synchronized_wall':stats(wall)}
                    record['integrated']=measure(graph.forward);check_diagnostics(graph)
                    for stage,expected,sf in zip(stages,snapshots,scale_snapshots):
                        torch.testing.assert_close(stage.output,expected,rtol=0,atol=0)
                        if sf is not None:torch.testing.assert_close(stage.output_scales,sf,rtol=0,atol=0)
                    record['split']=measure(split_forward)
                    for stage,expected in zip(stages,snapshots):torch.testing.assert_close(stage.output,expected,rtol=0,atol=0)
                    record['post_timing_outputs']='bit-exact'
                records.append(record)
                print(f'{mode} M{m} K{k} N{n}: {len(stages)} native stages, {node_count} nodes, producer bytes exact, final NRMSE={maximum:g}',flush=True)
        # Real dependent matrix work, with more total CTAs than resident slots.
        with ExitStack() as stack:
            graphs=[]
            for instance in range(2):
                a=pack_nvfp4(rng.normal(size=(256,1024)).astype(np.float32),row_alignment=128)
                w=pack_nvfp4(rng.normal(0,.05,size=(8192,1024)).astype(np.float32))
                b=pack_nvfp4(np.zeros((256,4096),np.float32));v=pack_nvfp4(rng.normal(0,.05,size=(1024,4096)).astype(np.float32))
                gate=stack.enter_context(ResidentNVFP4(library,a,w,workers=100,output_dtype=torch.uint8,output_geglu=True))
                down=stack.enter_context(ResidentNVFP4(library,b,v,workers=100,output_dtype=torch.bfloat16))
                graphs.append(stack.enter_context(ResidentDagMatmul(library,[gate,down],workers=100)))
            gold=[]
            for graph in graphs:
                assert split(graph.handle,torch.cuda.current_stream().cuda_stream)==0
                gold.append(graph.output.clone())
            torch.cuda.synchronize();streams=[torch.cuda.Stream(),torch.cuda.Stream()]
            for graph,stream in zip(graphs,streams):
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    graph.stages[0].output_scales.fill_(255);graph.forward()
            torch.cuda.synchronize();diagnostics=[]
            for graph,expected in zip(graphs,gold):
                torch.testing.assert_close(graph.output,expected,rtol=0,atol=0);diagnostics.append(check_diagnostics(graph))
            overlap=max(0,min(int(d['ends'].max()) for d in diagnostics)-max(int(d['starts'].min()) for d in diagnostics))
            if args.profile:assert overlap>0,'No overlapping active node intervals observed'
            concurrent={'workers_per_instance':100,'total_submitted_ctas':200,'active_node_interval_overlap_ns':overlap,'outputs':'bit-exact split baselines'}
            if args.profile:
                with LaunchTrace(library) as trace:
                    for graph in graphs:graph.forward()
                    torch.cuda.synchronize()
                assert trace.count==2,trace.count
                concurrent['two_forward_negative_control']=trace.count
            print(f'Concurrent independent matrix graphs: exact outputs, active-node overlap {overlap} ns',flush=True)
    if args.benchmark_output:
        sources=['native_pi0/dag_matmul.cu','native_pi0/dag_matmul.py','native_pi0/dag_scheduler.cuh','native_pi0/dag_probe.cu','native_pi0/dag_probe.py',
                 'native_pi0/nvfp4_probe.cu','native_pi0/tcgen_probe.cu','native_pi0/primitives.py','native_pi0/quantization.py',
                 'native_pi0/fp4_output_reference.cu','native_pi0/launch_trace.cpp','native_pi0/launch_trace.py','scripts/validate_dag_matmul.py',
                 'native_pi0/patches/megakernels-dynamic-instructions.patch','native_pi0/patches/megakernels-serial-barrier-init.patch',
                 'third_party/Megakernels/include/megakernel.cuh','third_party/Megakernels/include/util.cuh',
                 'third_party/Megakernels/include/controller/controller.cuh','third_party/Megakernels/include/controller/instruction_fetch.cuh']
        report={'timestamp_utc':datetime.datetime.now(datetime.UTC).isoformat(),'seed':137,'quick':args.quick,'baseline_queue':args.baseline_queue,
                'idle_sleep_ns':args.idle_sleep_ns,'per_task_tensormap_acquire':args.per_task_tensormap_acquire,
                'cas_decrement':args.cas_decrement,
                'global_compute_metadata':args.global_compute_metadata,
                'uncoalesced_dependencies':args.uncoalesced_dependencies,
                'single_lane_bootstrap':args.single_lane_bootstrap,
                'node_order':args.node_order,
                'nvcc':subprocess.check_output(['nvcc','--version'],text=True).strip(),
                'scope':'Synthetic device-scheduled native matrix/FFN cores, not normalization, attention, pretrained or complete pi0 inference',
                'timed_boundary':'Resident packed root input/weights to reused outputs; excludes setup/reference/transfers; events include host submission gaps',
                'resources':resources,'upstreams':json.loads((ROOT/'upstreams.json').read_text()),
                'environment':{'gpu':torch.cuda.get_device_name(),'torch':torch.__version__,'cuda':torch.version.cuda},
                'source_sha256':{name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in sources},'results':records,'concurrent':concurrent}
        args.benchmark_output.parent.mkdir(parents=True,exist_ok=True);args.benchmark_output.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
