"""One original VM launch for a resident transformer layer stack and final norm."""
import ctypes
from pathlib import Path

import numpy as np
import torch

from native_pi0.dag_probe import ResidentDagProbe


class ResidentTransformerStack:
    """Compose distinct caller-owned blocks with direct row-band handoffs.

    Setup binds each later block's input/residual to its predecessor's down
    output and binds the final RMSNorm to the last block. Calls reuse resident
    storage and require stream ordering. This is a transformer stack, without
    embeddings, action heads or a denoising loop.
    """
    def __init__(self,library,blocks,final_norm,*,workers=148,node_order=None):
        self.handle=ctypes.c_void_p();self._queue=None;self.blocks=tuple(blocks);self.final_norm=final_norm
        if not self.blocks or len(self.blocks)>256:raise ValueError('Expected 1..256 distinct transformer blocks')
        self.device=self.blocks[0].device;self.library=ctypes.CDLL(str(library));self.workers=workers
        handles=[]
        for block in self.blocks:
            children=(block,block.pre_norm,block.post_norm,block.attention,block.attention.attention,
                block.attention.projection,block.attention.output_projection,block.gate,block.down)
            for child in children:
                lib=getattr(child,'library',getattr(child,'_library',None))
                h=getattr(child,'handle',getattr(child,'_handle',None))
                if lib is None or h is None or not h.value or Path(lib._name).resolve()!=Path(library).resolve() or child.device!=self.device:
                    raise ValueError('Blocks and children must be live in this library/device')
                handles.append(h.value)
        if len(set(handles))!=len(handles):raise ValueError('Layers must have distinct child contexts and completion state')
        if not final_norm.handle.value or final_norm.device!=self.device or Path(final_norm.library._name).resolve()!=Path(library).resolve():
            raise ValueError('Final normalization must be live in this library/device')
        if final_norm.handle.value in handles:raise ValueError('Final normalization must have a distinct context')
        rows,width=self.blocks[0].inputs.shape
        if any(tuple(b.inputs.shape)!=(rows,width) or tuple(b.down.output.shape)!=(rows,width) for b in self.blocks):
            raise ValueError('Layer padded input/output dimensions must match')
        if tuple(final_norm.inputs.shape)!=(rows,width) or final_norm.output.dtype!=torch.bfloat16 or final_norm.residual is not None:
            raise ValueError('Final normalization must produce full-width padded BF16 rows without a residual')
        group_rows=self.blocks[0].norm_group_rows
        if any(b.norm_group_rows!=group_rows for b in self.blocks):raise ValueError('Normalization task geometry must match')
        parents=[];kinds=[];tasks=[];stages=[];previous_down=None
        for layer,block in enumerate(self.blocks):
            offset=len(parents)
            for node,(row,kind,task) in enumerate(zip(block._queue.parents,block.node_kind,block.node_task)):
                deps=[offset+p for p in row]
                if previous_down is not None and kind==3:
                    band=(int(task)*group_rows)//128
                    deps.extend(previous_down[band])
                parents.append(deps);kinds.append(int(kind));tasks.append(int(task));stages.append(layer)
            down_tiles=block.down.b.shape[0]//128
            previous_down=[[] for _ in range(rows//128)]
            for node,(kind,task) in enumerate(zip(block.node_kind,block.node_task)):
                if kind==6:previous_down[int(task)//down_tiles].append(offset+node)
        for task in range((rows+group_rows-1)//group_rows):
            parents.append(list(previous_down[task*group_rows//128]));kinds.append(3);tasks.append(task);stages.append(len(self.blocks))
        if node_order is not None:
            order=np.asarray(node_order,dtype=np.int64)
            if order.shape!=(len(parents),) or not np.array_equal(np.sort(order),np.arange(len(parents))):raise ValueError('Invalid node permutation')
            inverse=np.empty(len(order),dtype=np.int64);inverse[order]=np.arange(len(order))
            parents=[[int(inverse[p]) for p in parents[old]] for old in order]
            kinds=[kinds[old] for old in order];tasks=[tasks[old] for old in order];stages=[stages[old] for old in order]
        self.node_kind=np.asarray(kinds,dtype=np.int32);self.node_task=np.asarray(tasks,dtype=np.int32);self.node_stage=np.asarray(stages,dtype=np.int32)
        block_handles=np.asarray([b.handle.value for b in self.blocks],dtype=np.uintp)
        create=self.library.create_transformer_stack
        create.argtypes=[ctypes.POINTER(ctypes.c_void_p),ctypes.c_void_p,ctypes.c_int]+[ctypes.c_void_p]*5;create.restype=ctypes.c_int
        self.launch=self.library.launch_transformer_stack;self.launch.argtypes=[ctypes.c_void_p]*2;self.launch.restype=ctypes.c_int
        self.destroy=self.library.destroy_transformer_stack;self.destroy.argtypes=[ctypes.c_void_p];self.destroy.restype=ctypes.c_int
        try:
            with torch.cuda.device(self.device):
                self._queue=ResidentDagProbe(library,parents,workers=workers)
                error=create(ctypes.byref(self.handle),self._queue.handle,len(self.blocks),block_handles.ctypes.data,final_norm.handle,
                    self.node_kind.ctypes.data,self.node_task.ctypes.data,self.node_stage.ctypes.data)
            if error:raise RuntimeError(f'Transformer stack setup CUDA error {error}')
        except Exception:
            if self._queue is not None:self._queue.close()
            raise
        for parent,child in zip(self.blocks,self.blocks[1:]):
            child.inputs=parent.down.output;child.pre_norm.inputs=parent.down.output
            child.attention.output_projection.residual=parent.down.output
        final_norm.inputs=self.blocks[-1].down.output
        self.inputs=self.blocks[0].inputs
        self.output=final_norm.output[:self.blocks[-1].attention.output_projection.logical_rows,:width]

    def forward(self):
        if not self.handle.value or not self.final_norm.handle.value:raise RuntimeError('Transformer stack or final norm is closed')
        for b in self.blocks:
            handles=(b.handle,b.pre_norm.handle,b.post_norm.handle,b.attention.handle,b.attention.attention.handle,
                b.attention.projection.handle,b.attention.output_projection.handle,b.gate._handle,b.down._handle)
            if any(not h.value for h in handles):raise RuntimeError('Transformer stack child is closed')
        epoch=max([self._queue.epoch]+[b.attention.attention.epoch for b in self.blocks])+1
        with torch.cuda.device(self.device):error=self.launch(self.handle,torch.cuda.current_stream(self.device).cuda_stream)
        if error:raise RuntimeError(f'Transformer stack launch CUDA error {error}')
        self._queue.epoch=epoch
        for b in self.blocks:b.attention.attention.epoch=epoch
        return self.output

    def read_diagnostics(self):return self._queue.read()

    def close(self):
        if self.handle.value:
            with torch.cuda.device(self.device):error=self.destroy(self.handle)
            if error:raise RuntimeError(f'Transformer stack teardown CUDA error {error}')
            self.handle=ctypes.c_void_p()
        if self._queue is not None:self._queue.close();self._queue=None

    def __enter__(self):return self
    def __exit__(self,*_):self.close()
