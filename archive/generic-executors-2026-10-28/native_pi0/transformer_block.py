"""Pre-norm attention and compact NVFP4 GeGLU FFN in one native VM launch."""
import ctypes
from pathlib import Path

import numpy as np
import torch

from native_pi0.dag_probe import ResidentDagProbe


class ResidentTransformerBlock:
    """Compose caller-owned operators; setup binds their resident buffers.

    The block computes x + attention(norm(x)), followed by
    h + down(GeGLU(gate_up(norm(h)))). Both norms have padded row counts;
    attention retains the logical batch/sequence lengths. Child forwards
    remain useful as a split-launch reference after binding. Child handles
    and buffers must remain live, with calls ordered on a CUDA stream.
    """
    def __init__(self,library,attention,pre_norm,post_norm,gate,down,*,workers=148,node_order=None):
        self.handle=ctypes.c_void_p();self._queue=None
        self.attention=attention;self.pre_norm=pre_norm;self.post_norm=post_norm;self.gate=gate;self.down=down
        self.device=attention.device;self.library=ctypes.CDLL(str(library));self.workers=workers
        children=(attention,pre_norm,post_norm,gate,down)
        for child in children:
            lib=getattr(child,'library',getattr(child,'_library',None))
            handle=getattr(child,'handle',getattr(child,'_handle',None))
            if lib is None or Path(lib._name).resolve()!=Path(library).resolve() or not handle.value or child.device!=self.device:
                raise ValueError('Children must be live on this device and from this library')
        o=attention.output_projection
        if o is None or o.output.dtype!=torch.bfloat16:raise ValueError('Attention needs a BF16 output projection')
        rows,width=o.output.shape
        if width%128 or tuple(pre_norm.inputs.shape)!=(rows,width) or tuple(post_norm.inputs.shape)!=(rows,width):
            raise ValueError('Norm inputs must include the padded dense rows and full width')
        if pre_norm.residual is not None or post_norm.residual is not None:raise ValueError('Block binds residuals in projection epilogues')
        if pre_norm.output.dtype!=attention.projection.a.dtype or post_norm.output.dtype!=torch.uint8:
            raise ValueError('Pre-norm must match QKV precision; post-norm must produce FP4')
        if tuple(pre_norm.output.shape)!=tuple(attention.projection.a.shape) or tuple(post_norm.output.shape)!=tuple(gate.a.shape):
            raise ValueError('Norm output dimensions must match the projection inputs')
        if post_norm.global_scale!=gate.activation_global_scale or gate.output_global_scale!=down.activation_global_scale:
            raise ValueError('FFN activation descales must match producer outputs')
        if gate.output.dtype!=torch.uint8 or down.output.dtype!=torch.bfloat16 or tuple(down.output.shape)!=(rows,width):
            raise ValueError('Expected compact FP4 gate/up output and full-width BF16 down output')
        parents=[list(p) for p in attention._queue.parents]
        kinds=list(map(int,attention.node_kind));tasks=list(map(int,attention.node_task))
        group_rows=self.library.transformer_block_norm_rows;group_rows.argtypes=[];group_rows.restype=ctypes.c_int
        self.norm_group_rows=group_rows();groups=(rows+self.norm_group_rows-1)//self.norm_group_rows
        pre_base=len(parents)
        for group in range(groups):parents.append([]);kinds.append(3);tasks.append(group)
        qtiles=attention.projection.output.shape[1]//128
        otiles=o.output.shape[1]//128
        o_nodes={int(task):node for node,(kind,task) in enumerate(zip(kinds,tasks)) if kind==2}
        for node,(kind,task) in enumerate(zip(kinds,tasks)):
            if kind==0:
                first=(task//qtiles)*128
                parents[node].extend(range(pre_base+first//self.norm_group_rows,pre_base+(first+128)//self.norm_group_rows))
        post_base=len(parents)
        for group in range(groups):
            row=group*self.norm_group_rows
            parents.append([o_nodes[(row//128)*otiles+c] for c in range(otiles)]);kinds.append(4);tasks.append(group)
        gate_base=len(parents);gtiles=gate.b.shape[0]//128
        for task in range((rows//128)*gtiles):
            first=(task//gtiles)*128
            parents.append(list(range(post_base+first//self.norm_group_rows,post_base+(first+128)//self.norm_group_rows)));kinds.append(5);tasks.append(task)
        dtiles=down.b.shape[0]//128
        for task in range((rows//128)*dtiles):
            first=(task//dtiles)*gtiles
            parents.append(list(range(gate_base+first,gate_base+first+gtiles)));kinds.append(6);tasks.append(task)
        if node_order is not None:
            order=np.asarray(node_order,dtype=np.int64)
            if order.shape!=(len(parents),) or not np.array_equal(np.sort(order),np.arange(len(parents))):raise ValueError('Invalid node permutation')
            inverse=np.empty(len(order),dtype=np.int64);inverse[order]=np.arange(len(order))
            parents=[[int(inverse[p]) for p in parents[old]] for old in order]
            kinds=[kinds[old] for old in order];tasks=[tasks[old] for old in order]
        self.node_kind=np.asarray(kinds,dtype=np.int32);self.node_task=np.asarray(tasks,dtype=np.int32)
        create=self.library.create_transformer_block
        create.argtypes=[ctypes.POINTER(ctypes.c_void_p)]+[ctypes.c_void_p]*8;create.restype=ctypes.c_int
        self.launch=self.library.launch_transformer_block;self.launch.argtypes=[ctypes.c_void_p]*2;self.launch.restype=ctypes.c_int
        self.destroy=self.library.destroy_transformer_block;self.destroy.argtypes=[ctypes.c_void_p];self.destroy.restype=ctypes.c_int
        try:
            with torch.cuda.device(self.device):
                self._queue=ResidentDagProbe(library,parents,workers=workers)
                error=create(ctypes.byref(self.handle),self._queue.handle,attention.handle,pre_norm.handle,post_norm.handle,
                    gate._handle,down._handle,self.node_kind.ctypes.data,self.node_task.ctypes.data)
            if error:raise RuntimeError(f'Transformer block setup CUDA error {error}')
        except Exception:
            if self._queue is not None:self._queue.close()
            raise
        # Mirror the C++ metadata bindings while retaining all tensor owners.
        attention.projection.a=pre_norm.output
        attention.projection.inputs=pre_norm.output[:attention.projection.logical_rows,:width]
        o.residual=pre_norm.inputs
        post_norm.inputs=o.output
        gate.a=post_norm.output;gate.sa=post_norm.output_scales
        down.a=gate.output;down.sa=gate.output_scales;down.residual=o.output
        self.inputs=pre_norm.inputs;self.output=down.output[:o.logical_rows,:o.logical_columns]

    def forward(self):
        handles=(self.handle,self.attention.handle,self.attention.attention.handle,self.attention.projection.handle,
                 self.attention.output_projection.handle,self.pre_norm.handle,self.post_norm.handle,self.gate._handle,self.down._handle)
        if any(not h.value for h in handles):raise RuntimeError('Transformer block or child is closed')
        epoch=max(self._queue.epoch,self.attention.attention.epoch)+1
        with torch.cuda.device(self.device):error=self.launch(self.handle,torch.cuda.current_stream(self.device).cuda_stream)
        if error:raise RuntimeError(f'Transformer block launch CUDA error {error}')
        self._queue.epoch=epoch;self.attention.attention.epoch=epoch
        return self.output

    def read_diagnostics(self):return self._queue.read()

    def close(self):
        if self.handle.value:
            with torch.cuda.device(self.device):error=self.destroy(self.handle)
            if error:raise RuntimeError(f'Transformer block teardown CUDA error {error}')
            self.handle=ctypes.c_void_p()
        if self._queue is not None:self._queue.close();self._queue=None

    def __enter__(self):return self
    def __exit__(self,*_):self.close()
