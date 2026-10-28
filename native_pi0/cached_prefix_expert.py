"""One native launch for a prefix attention sublayer and cached-prefix expert."""
import ctypes
from pathlib import Path

import numpy as np
import torch

from native_pi0.dag_probe import ResidentDagProbe


class ResidentCachedPrefixExpert:
    """Compose two caller-owned projection/attention[/O] chains in one VM.

    Prefix cache writers publish dependencies directly to expert key partitions.
    Both child chains and their resident buffers must outlive this wrapper.
    This is one layer pair, not the complete prefix or denoising model.
    """
    def __init__(self,library,prefix,expert,*,workers=148,node_order=None):
        self.handle=ctypes.c_void_p();self.prefix=prefix;self.expert=expert;self.device=prefix.device
        self.library=ctypes.CDLL(str(library));self._queue=None
        for chain in (prefix,expert):
            if not chain.handle.value or chain.device!=self.device or Path(chain.library._name).resolve()!=Path(library).resolve():raise ValueError('Child chains must be live in this library/device')
        pa,ea=prefix.attention,expert.attention
        if pa.kv_cache is None or ea.kv_cache is None or pa.cached_prefix_rows or ea.cached_prefix_rows!=pa.raw_key_rows:raise ValueError('Expected prefix producer and compatible cached-prefix expert')
        if any(a.data_ptr()!=b.data_ptr() for a,b in zip(pa.kv_cache,ea.kv_cache)):raise ValueError('Child chains must share the same KV cache')
        pcount=prefix._queue.nodes
        parents=[list(row) for row in prefix._queue.parents]+[[p+pcount for p in row] for row in expert._queue.parents]
        kinds=list(prefix.node_kind)+list(expert.node_kind);tasks=list(prefix.node_task)+list(expert.node_task)
        stages=[0]*pcount+[1]*expert._queue.nodes
        prefix_tiles=(pa.q.shape[1]+63)//64;expert_tiles=(ea.q.shape[1]+63)//64
        prefix_heads=pa.q.shape[2];expert_heads=ea.q.shape[2];kv_heads=pa.k.shape[2]
        writer_nodes={int(task):node for node,(kind,task) in enumerate(zip(prefix.node_kind,prefix.node_task)) if kind==1}
        pchunks=(pa.raw_key_rows+63)//64;pper=(pchunks+pa.splits-1)//pa.splits
        total_keys=ea.cached_prefix_rows+ea.raw_key_rows;echunks=(total_keys+63)//64;eper=(echunks+ea.splits-1)//ea.splits
        for node,(kind,task) in enumerate(zip(expert.node_kind,expert.node_task)):
            if kind!=1:continue
            logical=int(task)//ea.splits;part=int(task)%ea.splits
            batch=logical//(expert_heads*expert_tiles);head=(logical//expert_tiles)%expert_heads;kh=head//(expert_heads//kv_heads)
            begin=part*eper*64;end=min(pa.raw_key_rows,begin+eper*64)
            if begin>=end:continue
            extra=[]
            for pp in range(pa.splits):
                pbegin=pp*pper*64;pend=min(pa.raw_key_rows,pbegin+pper*64)
                if pbegin<end and begin<pend:
                    writer_head=kh*(prefix_heads//kv_heads)
                    task_id=((batch*prefix_heads+writer_head)*prefix_tiles)*pa.splits+pp
                    extra.append(writer_nodes[task_id])
            parents[pcount+node]=sorted(set(parents[pcount+node]+extra))
        if node_order is not None:
            order=np.asarray(node_order,dtype=np.int64)
            if order.shape!=(len(parents),) or not np.array_equal(np.sort(order),np.arange(len(parents))):raise ValueError('Invalid node permutation')
            inverse=np.empty(len(order),dtype=np.int64);inverse[order]=np.arange(len(order))
            parents=[[int(inverse[p]) for p in parents[old]] for old in order]
            kinds=[kinds[old] for old in order];tasks=[tasks[old] for old in order];stages=[stages[old] for old in order]
        with torch.cuda.device(self.device):self._queue=ResidentDagProbe(library,parents,workers=workers)
        self.node_kind=np.asarray(kinds,dtype=np.int32);self.node_task=np.asarray(tasks,dtype=np.int32);self.node_stage=np.asarray(stages,dtype=np.int32)
        create=self.library.create_cached_prefix_expert
        create.argtypes=[ctypes.POINTER(ctypes.c_void_p)]+[ctypes.c_void_p]*6;create.restype=ctypes.c_int
        self.launch=self.library.launch_cached_prefix_expert;self.launch.argtypes=[ctypes.c_void_p]*2;self.launch.restype=ctypes.c_int
        self.destroy=self.library.destroy_cached_prefix_expert;self.destroy.argtypes=[ctypes.c_void_p];self.destroy.restype=ctypes.c_int
        with torch.cuda.device(self.device):error=create(ctypes.byref(self.handle),self._queue.handle,prefix.handle,expert.handle,
            self.node_kind.ctypes.data,self.node_task.ctypes.data,self.node_stage.ctypes.data)
        if error:self._queue.close();raise RuntimeError(f'Cached prefix/expert setup CUDA error {error}')
        self.output=expert.output

    def forward(self):
        if not self.handle.value:raise RuntimeError('Cached prefix/expert instance is closed')
        for chain in (self.prefix,self.expert):
            if not chain.handle.value or not chain.projection.handle.value or not chain.attention.handle.value or (chain.output_projection is not None and not chain.output_projection.handle.value):raise RuntimeError('Child stage is closed')
        epoch=max(self._queue.epoch,self.prefix.attention.epoch,self.expert.attention.epoch)+1
        with torch.cuda.device(self.device):error=self.launch(self.handle,torch.cuda.current_stream(self.device).cuda_stream)
        if error:raise RuntimeError(f'Cached prefix/expert launch CUDA error {error}')
        self._queue.epoch=epoch;self.prefix.attention.epoch=epoch;self.expert.attention.epoch=epoch
        return self.output

    def read_diagnostics(self):return self._queue.read()

    def close(self):
        if self.handle.value:
            with torch.cuda.device(self.device):error=self.destroy(self.handle)
            if error:raise RuntimeError(f'Cached prefix/expert teardown CUDA error {error}')
            self.handle=ctypes.c_void_p()
        if self._queue is not None:self._queue.close();self._queue=None

    def __enter__(self):return self
    def __exit__(self,*_):self.close()
