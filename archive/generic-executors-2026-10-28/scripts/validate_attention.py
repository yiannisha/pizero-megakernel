"""Validate FlashRT-derived native VM attention, GQA and pi0 suffix masks."""
import argparse
import contextlib
import ctypes
import datetime
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile
import time

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from native_pi0.attention import ResidentAttention
from native_pi0.launch_trace import LaunchTrace


def reference(operation):
    """Independent CPU FP32 attention math, including rotate-half and masks."""
    q,k,v=[x.float().cpu() for x in (operation.q,operation.k,operation.v)]
    for x,bias in zip((q,k,v),operation.bias):
        if bias is not None:x.add_(bias.float().cpu())
    for x,tables in ((q,operation.q_rope),(k,operation.k_rope)):
        if tables:
            half=x.shape[-1]//2;c,s=[t.float().cpu()[...,:half].unsqueeze(2) for t in tables]
            lo,hi=x[...,:half].clone(),x[...,half:].clone()
            x[...,:half]=lo*c-hi*s;x[...,half:]=hi*c+lo*s
    groups=q.shape[2]//k.shape[2]
    k=k.repeat_interleave(groups,2);v=v.repeat_interleave(groups,2)
    score=(q.transpose(1,2)@k.transpose(1,2).transpose(-1,-2))*operation.scale
    visible=torch.ones((q.shape[0],q.shape[1],k.shape[1]),dtype=torch.bool)
    if operation.prefix is not None:visible[:,0,operation.prefix+1:]=False
    if operation.mask is not None:visible &= operation.mask.cpu()
    score.masked_fill_(~visible.unsqueeze(1),float('-inf'))
    probability=torch.softmax(score,dim=-1).nan_to_num(0)
    return (probability@v.transpose(1,2)).transpose(1,2)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile',action='store_true')
    parser.add_argument('--quick',action='store_true')
    parser.add_argument('--kv-splits',type=int,default=1)
    parser.add_argument('--disable-fma',action='store_true',help='Diagnostic build requiring bit-exact FlashRT parity without FP32 FMA contraction')
    parser.add_argument('--uncached-pages',action='store_true',help='Benchmark the original per-access VM page lookup')
    parser.add_argument('--cached-merge',action='store_true',help='Experimental shared cache of per-row split-softmax coefficients')
    parser.add_argument('--report',type=Path)
    parser.add_argument('--build-directory',type=Path)
    args=parser.parse_args()
    if not 1<=args.kv_splits<=64:parser.error('KV splits must be in 1..64')
    if args.report and not args.profile:parser.error('Timing reports require --profile launch gates')
    torch.set_num_threads(8);torch.manual_seed(419);records=[]
    if args.build_directory:args.build_directory.mkdir(parents=True,exist_ok=True)
    directory_context=contextlib.nullcontext(args.build_directory) if args.build_directory else tempfile.TemporaryDirectory()
    with directory_context as directory:
        library=Path(directory).resolve()/'attention.so'
        command=['nvcc','-O3','-lineinfo','-std=c++20','-arch=sm_100a','-DKITTENS_HOPPER','-DKITTENS_BLACKWELL',
            '--expt-relaxed-constexpr','--expt-extended-lambda',
            '-I'+str(ROOT/'third_party/Megakernels/include'),'-I'+str(ROOT/'third_party/Megakernels/ThunderKittens/include'),
            '-I'+str(ROOT/'third_party/FlashRT/csrc'),'-shared','-Xcompiler','-fPIC',
            str(ROOT/'native_pi0/attention.cu'),str(ROOT/'native_pi0/attention_reference.cu'),
            str(ROOT/'third_party/FlashRT/csrc/kernels/fused_qkv_attention.cu'),str(ROOT/'native_pi0/launch_trace.cpp'),
            '-lcupti','-Xlinker','-rpath','-Xlinker','/usr/local/cuda/lib64','-lcuda','-o',str(library)]
        if args.disable_fma:command.insert(1,'--fmad=false')
        command.insert(1,'-DNATIVE_ATTENTION_CACHE_PAGES='+str(int(not args.uncached_pages)))
        command.insert(1,'-DNATIVE_ATTENTION_CACHE_MERGE='+str(int(args.cached_merge)))
        subprocess.run(command,check=True)
        actual=ctypes.CDLL(str(library)).reference_attention
        actual.argtypes=[ctypes.c_int]+[ctypes.c_void_p]*3+[ctypes.c_int]+[ctypes.c_void_p]*6+[ctypes.c_int]*5+[ctypes.c_float,ctypes.c_void_p]
        actual.restype=ctypes.c_int
        resource_values=(ctypes.c_int*6)()
        resources=ctypes.CDLL(str(library)).attention_resources
        resources.argtypes=[ctypes.c_void_p,ctypes.c_int];resources.restype=ctypes.c_int
        assert resources(resource_values,6)==0
        resource_record=dict(zip(('ctas_per_sm','sms','registers','local_bytes','static_shared_bytes','dynamic_shared_bytes'),resource_values))
        # (label, batch, queries, keys, Q heads, KV heads, dimension, dtype,
        #  projection view, bias, RoPE, mask, suffix prefix, workers)
        cases=[
            ('one',1,1,1,1,1,64,torch.bfloat16,False,False,False,False,None,1),
            ('tail',2,65,65,2,2,72,torch.float16,True,True,True,False,None,4),
            ('gqa-tail',1,3,67,4,1,256,torch.bfloat16,False,True,True,True,None,16),
            ('suffix-small',1,3,8,4,1,256,torch.bfloat16,False,False,False,False,5,4),
            ('all-masked',1,17,65,2,1,64,torch.float16,False,False,False,True,None,4),
        ]
        if not args.quick:cases += [
            ('siglip',3,256,256,16,16,72,torch.float16,True,True,False,False,None,148),
            ('siglip-bf16',3,256,256,16,16,72,torch.bfloat16,True,True,False,False,None,148),
            ('prefix',1,768,768,8,1,256,torch.bfloat16,False,False,True,False,None,148),
            ('expert',1,51,819,8,1,256,torch.bfloat16,False,False,False,False,768,148),
            ('expert-fp16',1,51,819,8,1,256,torch.float16,False,False,True,True,768,16),
            ('long-reference',1,513,513,2,2,64,torch.bfloat16,True,True,False,False,None,16),
            ('uniform-large',1,65,129,2,1,256,torch.bfloat16,False,False,False,False,None,4),
        ]
        def pointer(t):return None if t is None else t.data_ptr()
        def check_counts(op):
            if op.splits==1:return
            state=op.read_split_counts()
            assert state['epoch']==op.epoch,(state['epoch'],op.epoch)
            assert all(v==(op.epoch<<32)|op.splits for v in state['counts']),state
        for label,b,nq,nk,h,hk,d,dtype,fused,bias,rope,masked,prefix,workers in cases:
            if fused:
                storage=torch.randn((b,nq,3*h,d),device='cuda',dtype=dtype)
                q,k,v=storage[:,:,:h],storage[:,:,h:2*h],storage[:,:,2*h:]
            else:
                q=torch.randn((b,nq,h,d),device='cuda',dtype=dtype)
                k=torch.randn((b,nk,hk,d),device='cuda',dtype=dtype);v=torch.randn_like(k)
            if label=='uniform-large':q.fill_(32);k.fill_(32)
            biases=[torch.randn((heads,d),device='cuda',dtype=dtype)*.1 for heads in (h,hk,hk)] if bias else None
            def rotary(rows):
                angle=torch.randn((b,rows,d),device='cuda')
                return angle.cos().to(dtype),angle.sin().to(dtype)
            qrope=rotary(nq) if rope else None;krope=(qrope if nq==nk else rotary(nk)) if rope else None
            mask=None
            if masked:
                mask=torch.rand((b,nq,nk),device='cuda')>.2;mask[:,0]=False
                if nq>2:mask[:,2,:min(64,nk)]=False
            with ResidentAttention(library,q,k,v,bias=biases,q_rope=qrope,k_rope=krope,mask=mask,prefix=prefix,workers=workers,splits=args.kv_splits) as operation:
                errors=[];oracle_exact=[]
                for replay in range(3):
                    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(stream):
                        if replay and label!='uniform-large':operation.q.mul_(.9375).add_(.015625)
                        if operation.splits>1:operation.poison_partials()
                        operation.output.fill_(float('nan'));operation.forward()
                    stream.synchronize();output=operation.output.float().cpu();gold=reference(operation)
                    check_counts(operation)
                    assert torch.isfinite(output).all(),label
                    error=float((output-gold).square().mean().sqrt()/gold.square().mean().sqrt().clamp_min(1e-12))
                    assert error<=.01,(label,error)
                    errors.append(error)
                    if masked:assert torch.count_nonzero(output[:,0])==0,label
                    if prefix is not None:
                        # Hidden action K/V must not affect the state query.
                        original=operation.output[:,0].clone();saved_k=operation.k[:,prefix+1:].clone();saved_v=operation.v[:,prefix+1:].clone()
                        operation.k[:,prefix+1:].add_(2);operation.v[:,prefix+1:].mul_(3)
                        operation.forward();torch.cuda.synchronize()
                        check_counts(operation)
                        torch.testing.assert_close(operation.output[:,0],original,rtol=0,atol=0)
                        operation.k[:,prefix+1:].copy_(saved_k);operation.v[:,prefix+1:].copy_(saved_v)
                        operation.forward();torch.cuda.synchronize()
                    if h==hk and nq==nk and mask is None and prefix is None:
                        expected=torch.empty_like(operation.output)
                        tables=operation.q_rope or [None,None]
                        reference_args=(int(dtype==torch.bfloat16),*[pointer(t) for t in (operation.q,operation.k,operation.v)],operation.q.stride(1),
                            *[pointer(t) for t in operation.bias+tables+[expected]],expected.stride(1),b,nq,h,d,operation.scale,torch.cuda.current_stream().cuda_stream)
                        error=actual(*reference_args)
                        assert error==0,error;torch.cuda.synchronize()
                        def ordered_bits(x):
                            raw=x.contiguous().view(torch.int16).int()&65535
                            return torch.where(raw&32768!=0,32768-(raw&32767),32768+raw)
                        difference=(operation.output.float()-expected.float()).abs()
                        max_ulp=int((ordered_bits(operation.output)-ordered_bits(expected)).abs().max())
                        source_nrmse=float(difference.square().mean().sqrt()/expected.float().square().mean().sqrt().clamp_min(1e-12))
                        bitwise_mismatches=int((operation.output.contiguous().view(torch.int16)!=expected.contiguous().view(torch.int16)).sum())
                        # Different register/shared-memory schedules permit
                        # different FP32 FMA contractions at output rounding
                        # boundaries. Disabling contraction restores exact
                        # parity; the normal build permits one output ULP and
                        # independently requires tiny aggregate source error.
                        if args.kv_splits==1:
                            assert max_ulp<=(0 if args.disable_fma else 1),(label,max_ulp)
                            assert source_nrmse<=1e-4,(label,source_nrmse)
                            if args.disable_fma:assert bitwise_mismatches==0,(label,bitwise_mismatches)
                        else:
                            # Splitting changes the probability-rounding point;
                            # retain the independent FP32 gate and compare the
                            # resulting attention against the unsplit source.
                            assert source_nrmse<=.01,(label,source_nrmse)
                        oracle_exact.append({'max_ulp':max_ulp,'nrmse':source_nrmse,
                            'bitwise_mismatched_elements':bitwise_mismatches,
                            'mismatched_elements':int((difference!=0).sum()),'max_absolute_error':float(difference.max())})
                logical_tasks=b*h*((nq+63)//64)
                key_chunks=(nk+63)//64;chunks_per_split=(key_chunks+args.kv_splits-1)//args.kv_splits
                nonempty=(key_chunks+chunks_per_split-1)//chunks_per_split
                record={'label':label,'batch':b,'queries':nq,'keys':nk,'heads':h,'kv_heads':hk,'head_dim':d,'dtype':str(dtype),
                    'workers':workers,'fused_projection_view':fused,'bias':bias,'rope':rope,'mask':masked,'suffix_prefix':prefix,
                    'kv_splits':args.kv_splits,'query_tile_rows':64,'key_chunk_rows':64,'tasks':b*h*((nq+63)//64)*args.kv_splits,
                    'ctas_with_assigned_tasks':min(workers,b*h*((nq+63)//64)*args.kv_splits),
                    'nonempty_kv_partitions':nonempty,'nonempty_attention_tasks':logical_tasks*nonempty,
                    'split_scratch_bytes':logical_tasks*(args.kv_splits*(64*d+128)*4+8) if args.kv_splits>1 else 0,
                    'split_replay_checks':'poisoned partials; exact epoch/completion counts' if args.kv_splits>1 else None,
                    'replays':3,'nrmse':max(errors),'nrmse_limit':.01,
                    'pinned_flashrt_byte_exact':all(x['bitwise_mismatched_elements']==0 for x in oracle_exact) if oracle_exact else None,
                    'pinned_flashrt_comparisons':oracle_exact,'pinned_flashrt_max_ulp_limit':(0 if args.disable_fma else 1) if args.kv_splits==1 else None,
                    'pinned_flashrt_nrmse_limit':1e-4 if args.kv_splits==1 else .01}
                if args.profile:
                    with LaunchTrace(library) as trace:operation.forward();torch.cuda.synchronize()
                    assert trace.count==1 and 'megakernel' in trace.symbol,(trace.count,trace.symbol)
                    record.update(cuda_launch_count=trace.count,kernel_symbol=trace.symbol)
                    check_counts(operation)
                if args.report:
                    def measure(call):
                        for _ in range(5):call()
                        torch.cuda.synchronize();start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);gpu=[];wall=[]
                        for _ in range(30):
                            torch.cuda.synchronize();t=time.perf_counter_ns();start.record();call();end.record();end.synchronize()
                            gpu.append(start.elapsed_time(end));wall.append((time.perf_counter_ns()-t)/1e6)
                        def stats(a):return {'median_ms':statistics.median(a),'p10_ms':float(np.percentile(a,10)),'p90_ms':float(np.percentile(a,90)),'samples_ms':a}
                        return {'gpu':stats(gpu),'synchronized_wall':stats(wall)}
                    record['native']=measure(operation.forward)
                    check_counts(operation)
                    if operation.splits>1:
                        with ResidentAttention(library,operation.q,operation.k,operation.v,bias=operation.bias,
                             q_rope=operation.q_rope or None,k_rope=operation.k_rope or None,mask=operation.mask,
                             prefix=operation.prefix,scale=operation.scale,workers=workers) as unsplit:
                            record['unsplit_native']=measure(unsplit.forward)
                            split_error=float((operation.output.float()-unsplit.output.float()).square().mean().sqrt()/unsplit.output.float().square().mean().sqrt().clamp_min(1e-12))
                            assert split_error<=.01,(label,split_error)
                            record['unsplit_native_nrmse']=split_error
                    if oracle_exact:
                        def standalone():
                            error=actual(*reference_args)
                            if error:raise RuntimeError(f'FlashRT reference error {error}')
                        record['standalone_flashrt']=measure(standalone)
                    torch.testing.assert_close(operation.output.float().cpu(),output,rtol=0,atol=0)
                    record['post_timing_output']='bit-exact replay'
                records.append(record)
                print(f'{label}: NRMSE={max(errors):g}, FlashRT exact={record["pinned_flashrt_byte_exact"]}',flush=True)
        with contextlib.ExitStack() as stack:
            operations=[];expected=[]
            for _ in range(2):
                cq=torch.randn((1,51,8,256),device='cuda',dtype=torch.bfloat16)
                ck=torch.randn((1,819,1,256),device='cuda',dtype=torch.bfloat16);cv=torch.randn_like(ck)
                op=stack.enter_context(ResidentAttention(library,cq,ck,cv,prefix=768,workers=100,splits=args.kv_splits))
                op.forward();expected.append(op.output.clone());operations.append(op)
            torch.cuda.synchronize();streams=[torch.cuda.Stream(),torch.cuda.Stream()]
            for op,stream in zip(operations,streams):
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):op.forward()
            torch.cuda.synchronize()
            for op,gold in zip(operations,expected):
                check_counts(op)
                assert torch.equal(op.output.view(torch.int16),gold.view(torch.int16))
            concurrent={'instances':2,'workers_per_instance':100,'submitted_ctas':200,'outputs':'bit-exact serial baselines',
                        'scope':'independent stream submissions; no active-overlap claim'}
            print('Concurrent independent attention instances: bit-exact serial baselines',flush=True)
        if args.profile:
            with ResidentAttention(library,q,k,v,workers=workers,splits=args.kv_splits) as operation:
                with LaunchTrace(library) as trace:operation.forward();operation.forward();torch.cuda.synchronize()
                assert trace.count==2,trace.count
    if args.report:
        paths=['native_pi0/attention.cu','native_pi0/attention.py','native_pi0/attention_reference.cu','scripts/validate_attention.py',
               'native_pi0/launch_trace.cpp','native_pi0/launch_trace.py','third_party/FlashRT/csrc/kernels/fused_qkv_attention.cu',
               'third_party/Megakernels/include/megakernel.cuh','third_party/Megakernels/include/util.cuh',
               'third_party/Megakernels/include/config.cuh','third_party/Megakernels/include/noop.cuh',
               'third_party/Megakernels/include/controller/controller.cuh','third_party/Megakernels/include/controller/instruction_fetch.cuh']
        report={'timestamp_utc':datetime.datetime.now(datetime.UTC).isoformat(),'scope':'Standalone native VM attention operator; synthetic inputs; not an integrated transformer or pretrained model',
            'seed':419,'fma_disabled':args.disable_fma,
            'cached_page_addresses':not args.uncached_pages,
            'cached_merge_coefficients':args.cached_merge,
            'kv_splits':args.kv_splits,
            'timed_boundary':'Resident Q/K/V and optional parameters to reused output; excludes setup and references; event times include host submission gaps',
            'upstreams':json.loads((ROOT/'upstreams.json').read_text()),'build_command':command,'resources':resource_record,
            'source_sha256':{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths},'results':records,'concurrent':concurrent,'two_forward_negative_control':2}
        args.report.parent.mkdir(parents=True,exist_ok=True);args.report.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
