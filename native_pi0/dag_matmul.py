"""Device-scheduled native NVFP4 stages; a development runtime, not NativePi0."""
import ctypes
from pathlib import Path

import numpy as np
import torch

from native_pi0.dag_probe import ResidentDagProbe


class ResidentDagMatmul:
    """Schedule packed matrix tiles and their dependencies in one native launch.

    Stages are live ResidentNVFP4 instances from the same library. A source of
    -1 reads that stage's resident activation; a prior stage index binds its
    FP4 output/scales as this stage's input. Source -2 reads normalization.
    All stages, including optional normalization, remain caller-owned and must
    outlive this wrapper. Calls on an instance require stream ordering.
    """
    def __init__(self,library,stages,*,sources=None,workers=16,node_order=None,normalization=None):
        self.stages=tuple(stages);self.handle=ctypes.c_void_p()
        if not self.stages:raise ValueError('At least one matrix stage is required')
        if len({s._handle.value for s in self.stages})!=len(self.stages) or any(not s._handle.value for s in self.stages):
            raise ValueError('Stages must be distinct live instances')
        if any(Path(s._library._name).resolve()!=Path(library).resolve() for s in self.stages):
            raise ValueError('All stages must come from this library')
        self.normalization=normalization
        self.library=ctypes.CDLL(str(library))
        self.device=self.stages[0].device
        if any(s.device!=self.device for s in self.stages):raise ValueError('Stage devices must match')
        self.norm_group_rows=0
        if normalization is not None:
            if not normalization.handle.value or Path(normalization.library._name).resolve()!=Path(library).resolve():raise ValueError('Normalization must be live in this library')
            if normalization.device!=self.device or normalization.output.dtype!=torch.uint8:raise ValueError('Normalization must produce FP4 on the stage device')
            group_rows=self.library.dag_norm_group_rows;group_rows.argtypes=[];group_rows.restype=ctypes.c_int
            self.norm_group_rows=group_rows()
        sources=[-2 if normalization is not None else -1]+list(range(len(stages)-1)) if sources is None else list(sources)
        if len(sources)!=len(stages) or any(not isinstance(src,int) or src<-2 or src>=i for i,src in enumerate(sources)):
            raise ValueError('Sources must reference a prior stage, -1 for packed input, or -2 for normalization')
        if normalization is not None and -2 not in sources:raise ValueError('Normalization must have a consumer')
        for i,source in enumerate(sources):
            if source==-2:
                if normalization is None:raise ValueError('Source -2 requires normalization')
                child=self.stages[i]
                if tuple(normalization.output.shape)!=tuple(child.a.shape):raise ValueError('Normalization dimensions do not match packed input')
                if normalization.global_scale!=child.activation_global_scale:raise ValueError('Normalization descale must match activation descale')
                continue
            if source<0:continue
            parent,child=self.stages[source],self.stages[i]
            if parent.output.dtype!=torch.uint8 or parent.output_scales is None:raise ValueError('Source must produce packed FP4')
            if parent.output.shape[0]!=child.a.shape[0] or parent.output_width!=child.a.shape[1]*2:
                raise ValueError('Source and consumer dimensions do not match')
            if parent.output_global_scale!=child.activation_global_scale:raise ValueError('Activation descale must match source output descale')
        groups=0 if normalization is None else (normalization.logical_rows+self.norm_group_rows-1)//self.norm_group_rows
        node_stage=[-1]*groups;node_task=list(range(groups));parents=[[] for _ in range(groups)];base=[]
        for i,stage in enumerate(self.stages):
            base.append(len(parents));m_tiles=stage.a.shape[0]//128;n_tiles=stage.b.shape[0]//128
            for task in range(m_tiles*n_tiles):
                row=task//n_tiles;source=sources[i]
                if source==-2:
                    begin=row*128//self.norm_group_rows
                    end=(min((row+1)*128,normalization.logical_rows)+self.norm_group_rows-1)//self.norm_group_rows
                    dependencies=list(range(begin,end))
                elif source<0:dependencies=[]
                else:
                    source_tiles=self.stages[source].b.shape[0]//128
                    dependencies=list(range(base[source]+row*source_tiles,base[source]+(row+1)*source_tiles))
                parents.append(dependencies);node_stage.append(i);node_task.append(task)
        if node_order is not None:
            order=np.asarray(node_order,dtype=np.int64)
            if order.shape!=(len(parents),) or not np.array_equal(np.sort(order),np.arange(len(parents))):raise ValueError('Invalid node permutation')
            inverse=np.empty(len(order),dtype=np.int64);inverse[order]=np.arange(len(order))
            parents=[[int(inverse[p]) for p in parents[old]] for old in order]
            node_stage=[node_stage[old] for old in order];node_task=[node_task[old] for old in order]
        with torch.cuda.device(self.device):self._queue=ResidentDagProbe(library,parents,workers=workers)
        self.node_stage=np.asarray(node_stage,dtype=np.int32);self.node_task=np.asarray(node_task,dtype=np.int32)
        self.sources=sources;self.workers=workers
        handles=np.array([s._handle.value for s in self.stages],dtype=np.uintp);source_array=np.array(sources,dtype=np.int32)
        create=self.library.create_dag_matmul if normalization is None else self.library.create_dag_norm_matmul
        create.argtypes=[ctypes.POINTER(ctypes.c_void_p),ctypes.c_void_p,ctypes.c_int]+[ctypes.c_void_p]*(4 if normalization is None else 5);create.restype=ctypes.c_int
        self.launch=self.library.launch_dag_matmul;self.launch.argtypes=[ctypes.c_void_p]*2;self.launch.restype=ctypes.c_int
        self.destroy=self.library.destroy_dag_matmul;self.destroy.argtypes=[ctypes.c_void_p];self.destroy.restype=ctypes.c_int
        with torch.cuda.device(self.device):
            error=create(ctypes.byref(self.handle),self._queue.handle,len(stages),handles.ctypes.data,
                         self.node_stage.ctypes.data,self.node_task.ctypes.data,source_array.ctypes.data,*([] if normalization is None else [normalization.handle]))
        if error:
            self._queue.close();raise RuntimeError(f'DAG matrix setup CUDA error {error}')
        self.output=self.stages[-1].output

    def forward(self):
        if not self.handle.value:raise RuntimeError('DAG matrix instance is closed')
        with torch.cuda.device(self.device):error=self.launch(self.handle,torch.cuda.current_stream(self.device).cuda_stream)
        if error:raise RuntimeError(f'DAG matrix launch CUDA error {error}')
        self._queue.epoch+=1
        return self.output

    def read_diagnostics(self):return self._queue.read()

    def close(self):
        if self.handle.value:
            with torch.cuda.device(self.device):error=self.destroy(self.handle)
            if error:raise RuntimeError(f'DAG matrix destroy CUDA error {error}')
            self.handle=ctypes.c_void_p()
        self._queue.close()
    def __enter__(self):return self
    def __exit__(self,*_):self.close()
