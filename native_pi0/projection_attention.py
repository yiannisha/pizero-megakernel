"""One-launch native BF16/FP8 projection and dependent self-attention."""
import ctypes
from pathlib import Path

import numpy as np
import torch

from native_pi0.attention import ResidentAttention
from native_pi0.dag_probe import ResidentDagProbe


class ResidentProjectionAttention:
    """Caller-owned dense QKV projection -> owned attention in the native VM.

    Projection columns are Q, then K, then V. Each batch has independent
    bidirectional self-attention; optional masks can further restrict it.
    The projection must outlive this wrapper. Calls require stream ordering.
    """
    def __init__(self,library,projection,*,batches,rows,heads,kv_heads,head_dim,workers=148,splits=1,
                 bias=None,q_rope=None,k_rope=None,mask=None,prefix=None,node_order=None,output_projection=None,
                 kv_cache=None,cached_prefix_rows=0):
        self.handle=ctypes.c_void_p();self.projection=projection;self.device=projection.device
        self.library=ctypes.CDLL(str(library));self._queue=None;self.attention=None
        self.output_projection=output_projection
        if not projection.handle.value or Path(projection.library._name).resolve()!=Path(library).resolve():raise ValueError('Projection must be live in this library')
        if projection.output.dtype!=torch.bfloat16:raise ValueError('Attention requires BF16 projection output')
        if projection.logical_rows!=batches*rows or projection.logical_columns!=(heads+2*kv_heads)*head_dim:raise ValueError('Projection does not match QKV dimensions')
        if output_projection is not None:
            o=output_projection
            if not o.handle.value or Path(o.library._name).resolve()!=Path(library).resolve() or o.device!=self.device:raise ValueError('Output projection must be live in this library/device')
            if o.a.dtype!=torch.bfloat16 or o.logical_rows!=batches*rows or o.logical_k!=heads*head_dim:raise ValueError('Output projection must consume BF16 attention rows')
        source=projection.output[:batches*rows]
        q=source[:,:heads*head_dim].view(batches,rows,heads,head_dim)
        k=source[:,heads*head_dim:(heads+kv_heads)*head_dim].view(batches,rows,kv_heads,head_dim)
        v=source[:,(heads+kv_heads)*head_dim:(heads+2*kv_heads)*head_dim].view(batches,rows,kv_heads,head_dim)
        create=self.library.create_projection_attention if output_projection is None else self.library.create_projection_attention_output
        create.argtypes=[ctypes.POINTER(ctypes.c_void_p)]+[ctypes.c_void_p]*(5 if output_projection is None else 6);create.restype=ctypes.c_int
        self.launch=self.library.launch_projection_attention;self.launch.argtypes=[ctypes.c_void_p]*2;self.launch.restype=ctypes.c_int
        self.destroy=self.library.destroy_projection_attention;self.destroy.argtypes=[ctypes.c_void_p];self.destroy.restype=ctypes.c_int
        try:
            self.attention=ResidentAttention(library,q,k,v,bias=bias,q_rope=q_rope,k_rope=k_rope,mask=mask,prefix=prefix,workers=workers,splits=splits,
                kv_cache=kv_cache,cached_prefix_rows=cached_prefix_rows)
            mtiles=projection.output.shape[0]//128;ntiles=projection.output.shape[1]//128;projection_tasks=mtiles*ntiles
            parents=[[] for _ in range(projection_tasks)];kinds=[0]*projection_tasks;tasks=list(range(projection_tasks))
            query_tiles=(rows+63)//64;key_chunks=(rows+cached_prefix_rows+63)//64;per_split=(key_chunks+splits-1)//splits
            for index in range(batches*heads*query_tiles*splits):
                logical=index//splits;part=index%splits;tile=logical%query_tiles;batch=logical//(query_tiles*heads)
                bands=[(batch*rows+tile*64,batch*rows+min(rows,(tile+1)*64))]
                begin=max(0,part*per_split*64-cached_prefix_rows);end=min(rows,(part+1)*per_split*64-cached_prefix_rows)
                if begin<end:bands.append((batch*rows+begin,batch*rows+end))
                row_tiles=set()
                for begin,end in bands:row_tiles.update(range(begin//128,(end-1)//128+1))
                parents.append([r*ntiles+c for r in sorted(row_tiles) for c in range(ntiles)])
                kinds.append(1);tasks.append(index)
            if output_projection is not None:
                otiles=output_projection.output.shape[1]//128
                for index in range((output_projection.output.shape[0]//128)*otiles):
                    begin=(index//otiles)*128;end=min(batches*rows,begin+128);dependencies=[]
                    for batch in range(batches):
                        qbegin=max(0,begin-batch*rows);qend=min(rows,end-batch*rows)
                        if qbegin>=qend:continue
                        for head in range(heads):
                            for tile in range(qbegin//64,(qend-1)//64+1):
                                logical=(batch*heads+head)*query_tiles+tile
                                dependencies.extend(projection_tasks+logical*splits+part for part in range(splits))
                    parents.append(dependencies);kinds.append(2);tasks.append(index)
            if node_order is not None:
                order=np.asarray(node_order,dtype=np.int64)
                if order.shape!=(len(parents),) or not np.array_equal(np.sort(order),np.arange(len(parents))):raise ValueError('Invalid node permutation')
                inverse=np.empty(len(order),dtype=np.int64);inverse[order]=np.arange(len(order))
                parents=[[int(inverse[p]) for p in parents[old]] for old in order]
                kinds=[kinds[old] for old in order];tasks=[tasks[old] for old in order]
            with torch.cuda.device(self.device):self._queue=ResidentDagProbe(library,parents,workers=workers)
            self.node_kind=np.asarray(kinds,dtype=np.int32);self.node_task=np.asarray(tasks,dtype=np.int32)
            with torch.cuda.device(self.device):error=create(ctypes.byref(self.handle),self._queue.handle,projection.handle,self.attention.handle,
                self.node_kind.ctypes.data,self.node_task.ctypes.data,*([output_projection.handle] if output_projection is not None else []))
            if error:raise RuntimeError(f'Projection-attention setup CUDA error {error}')
            if output_projection is not None:
                self.attention.output=output_projection.a[:batches*rows,:heads*head_dim].view(batches,rows,heads,head_dim)
        except Exception:
            if self._queue is not None:self._queue.close()
            if self.attention is not None:self.attention.close()
            raise
        self.output=self.attention.output if output_projection is None else output_projection.logical_output

    def forward(self):
        if not self.handle.value or not self.projection.handle.value or not self.attention.handle.value:raise RuntimeError('Projection-attention stage is closed')
        if self.output_projection is not None and not self.output_projection.handle.value:raise RuntimeError('Output projection is closed')
        epoch=max(self._queue.epoch,self.attention.epoch)+1
        with torch.cuda.device(self.device):error=self.launch(self.handle,torch.cuda.current_stream(self.device).cuda_stream)
        if error:raise RuntimeError(f'Projection-attention launch CUDA error {error}')
        self._queue.epoch=epoch;self.attention.epoch=epoch
        return self.output

    def read_diagnostics(self):return self._queue.read()

    def close(self):
        if self.handle.value:
            with torch.cuda.device(self.device):error=self.destroy(self.handle)
            if error:raise RuntimeError(f'Projection-attention teardown CUDA error {error}')
            self.handle=ctypes.c_void_p()
        if self._queue is not None:self._queue.close();self._queue=None
        if self.attention is not None:self.attention.close()

    def __enter__(self):return self
    def __exit__(self,*_):self.close()
