"""Batched native matrix ranges with static streams and row-band counters."""
import ctypes as C
from pathlib import Path

import numpy as np
import torch

from native_pi0.program import LinearWork,linear_streams,program_summary


class NativeLinearProgram:
    """Execute caller-owned BF16 projections in one original static VM launch.

    A source of -1 reads that stage's resident activation. Other sources bind
    an earlier stage's BF16 output directly as input. A loader waits only for
    its input row band. Calls require stream ordering; child handles and buffers
    must remain live. There is no device task allocator or runtime graph walk.
    """
    def __init__(self,library,stages,*,sources=None,workers=148):
        self.handle=C.c_void_p();self.stages=tuple(stages);self.library=C.CDLL(str(library));self.epoch=0
        if not self.stages:raise ValueError('At least one projection is required')
        self.device=self.stages[0].device
        if len({s.handle.value for s in self.stages})!=len(self.stages):raise ValueError('Stages must have distinct handles')
        for stage in self.stages:
            if not stage.handle.value or stage.device!=self.device or Path(stage.library._name).resolve()!=Path(library).resolve():
                raise ValueError('Projection handles must be live in this library/device')
            if stage.a.dtype!=torch.bfloat16 or stage.output.dtype!=torch.bfloat16:raise ValueError('This native range operator requires BF16 matrices')
        sources=[-1]+list(range(len(self.stages)-1)) if sources is None else list(sources)
        if len(sources)!=len(self.stages) or any(not isinstance(p,int) or p<-1 or p>=i for i,p in enumerate(sources)):
            raise ValueError('Sources must be -1 or an earlier stage')
        for index,parent in enumerate(sources):
            if parent>=0 and tuple(self.stages[parent].output.shape)!=tuple(self.stages[index].a.shape):raise ValueError('Padded producer output and input dimensions must match')
        self.sources=sources
        self.code=linear_streams([LinearWork((s.output.shape[0]//128)*(s.output.shape[1]//128),s.a.shape[1]) for s in self.stages],workers)
        self.plan=program_summary(self.code)
        self.counter_tiles=np.concatenate([np.full(s.output.shape[0]//128,s.output.shape[1]//128,dtype=np.uint64) for s in self.stages])
        handles=np.asarray([s.handle.value for s in self.stages],dtype=np.uintp);parents=np.asarray(sources,dtype=np.int32)
        create=self.library.create_native_linear
        create.argtypes=[C.POINTER(C.c_void_p),C.c_int,C.c_void_p,C.c_void_p,C.c_void_p,C.c_int,C.c_int];create.restype=C.c_int
        self.launch=self.library.launch_native_linear;self.launch.argtypes=[C.c_void_p]*2;self.launch.restype=C.c_int
        self.read=self.library.read_native_linear;self.read.argtypes=[C.c_void_p,C.c_void_p,C.c_int,C.c_void_p];self.read.restype=C.c_int
        self.destroy=self.library.destroy_native_linear;self.destroy.argtypes=[C.c_void_p];self.destroy.restype=C.c_int
        with torch.cuda.device(self.device):error=create(C.byref(self.handle),len(self.stages),handles.ctypes.data,parents.ctypes.data,self.code.ctypes.data,workers,self.code.shape[1])
        if error:raise RuntimeError(f'Native linear setup CUDA error {error}')
        for index,parent in enumerate(sources):
            if parent>=0:
                stage=self.stages[index];stage.a=self.stages[parent].output;stage.inputs=stage.a[:stage.logical_rows,:stage.logical_k]
        self.output=self.stages[-1].logical_output

    def forward(self):
        if not self.handle.value or any(not s.handle.value for s in self.stages):raise RuntimeError('Native program or child is closed')
        with torch.cuda.device(self.device):error=self.launch(self.handle,torch.cuda.current_stream(self.device).cuda_stream)
        if error:raise RuntimeError(f'Native linear launch CUDA error {error}')
        self.epoch+=1;return self.output

    def read_counters(self):
        counters=np.empty(len(self.counter_tiles),dtype=np.uint64);epoch=np.zeros(1,dtype=np.uint64)
        with torch.cuda.device(self.device):
            torch.cuda.synchronize(self.device);error=self.read(self.handle,counters.ctypes.data,len(counters),epoch.ctypes.data)
        if error:raise RuntimeError(f'Native counter read CUDA error {error}')
        return {'counts':counters,'expected':self.counter_tiles*epoch[0],'epoch':int(epoch[0])}

    def close(self):
        if self.handle.value:
            with torch.cuda.device(self.device):error=self.destroy(self.handle)
            if error:raise RuntimeError(f'Native linear teardown CUDA error {error}')
            self.handle=C.c_void_p()

    def __enter__(self):return self
    def __exit__(self,*_):self.close()
