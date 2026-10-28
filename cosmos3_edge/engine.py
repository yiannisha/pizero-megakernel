"""Offline model DAG setup; forward launches the shared native VM once."""
import ctypes
from pathlib import Path

import numpy as np
import torch
from native_pi0.dag_probe import ResidentDagProbe


class NativeEngine:
    def __init__(self, library, stages, parents, node_stage, node_task, *, workers=148, permutation=None):
        self.stages = tuple(stages); self.handle = ctypes.c_void_p()
        objects = [obj for kind, obj in stages if kind != 7]
        if not objects:
            raise ValueError('Model operators are required')
        self.device = objects[0].device
        if any(not obj.handle.value or obj.device!=self.device or Path(obj.library._name).resolve()!=Path(library).resolve() for obj in objects):
            raise ValueError('Model operators must be live in one library and device')
        if permutation is not None:
            order = np.asarray(permutation, np.int64)
            if order.shape!=(len(parents),) or not np.array_equal(np.sort(order),np.arange(len(parents))):
                raise ValueError('Invalid node permutation')
            inverse = np.empty_like(order); inverse[order] = np.arange(len(order))
            parents = [[int(inverse[p]) for p in parents[old]] for old in order]
            node_stage, node_task = [node_stage[i] for i in order], [node_task[i] for i in order]
        self.node_stage, self.node_task = np.array(node_stage,np.int32), np.array(node_task,np.int32)
        self.workers = workers; self.library = ctypes.CDLL(str(library))
        with torch.cuda.device(self.device):
            self._queue = ResidentDagProbe(library, parents, workers=workers)
        create = self.library.create_edge_engine
        create.argtypes = [ctypes.POINTER(ctypes.c_void_p),ctypes.c_void_p,ctypes.c_int]+[ctypes.c_void_p]*4
        create.restype = ctypes.c_int
        kinds = np.array([kind for kind, _ in stages],np.int32)
        handles = np.array([0 if obj is None else obj.handle.value for _, obj in stages],np.uintp)
        self.launch = self.library.launch_edge_engine; self.launch.argtypes=[ctypes.c_void_p]*2; self.launch.restype=ctypes.c_int
        self.epoch = self.library.edge_engine_epoch; self.epoch.argtypes=[ctypes.c_void_p]; self.epoch.restype=ctypes.c_uint
        self.destroy=self.library.destroy_edge_engine; self.destroy.argtypes=[ctypes.c_void_p]; self.destroy.restype=ctypes.c_int
        with torch.cuda.device(self.device):
            error=create(ctypes.byref(self.handle),self._queue.handle,len(stages),kinds.ctypes.data,handles.ctypes.data,
                         self.node_stage.ctypes.data,self.node_task.ctypes.data)
        if error:
            self._queue.close();raise RuntimeError(f'Native model schedule setup CUDA error {error}')

    def forward(self):
        if not self.handle.value:raise RuntimeError('Native model schedule is closed')
        with torch.cuda.device(self.device):
            error=self.launch(self.handle,torch.cuda.current_stream(self.device).cuda_stream)
        if error:raise RuntimeError(f'Native model launch CUDA error {error}')
        self._queue.epoch=int(self.epoch(self.handle))
        for kind,obj in self.stages:
            if kind==6:obj.epoch=self._queue.epoch

    def read_diagnostics(self):return self._queue.read()
    def close(self):
        if self.handle.value:
            with torch.cuda.device(self.device):error=self.destroy(self.handle)
            if error:raise RuntimeError(f'Native model teardown CUDA error {error}')
            self.handle=ctypes.c_void_p()
        self._queue.close()
    def __enter__(self):return self
    def __exit__(self,*_):self.close()
