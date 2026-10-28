"""Setup of BF16/FP8 projection tiles in the shared device-side DAG scheduler."""
import ctypes
from pathlib import Path

import numpy as np
import torch

from native_pi0.dag_probe import ResidentDagProbe


class DenseSchedule:
    """One native VM launch for a tile DAG of resident projections.

    Sources refer to prior BF16 outputs; a BF16 output can feed a BF16 input.
    FP8 input quantization requires a future producer operator, so FP8 consumers
    currently accept prepared inputs (source -1). Stages remain caller-owned.
    """
    def __init__(self, library, stages, *, sources=None, workers=148, node_order=None):
        self.stages = tuple(stages); self.handle = ctypes.c_void_p()
        if not self.stages or any(not s.handle.value for s in stages):
            raise ValueError('Live projection stages required')
        if len({s.handle.value for s in stages}) != len(stages):
            raise ValueError('Stages must be distinct instances')
        if any(Path(s.library._name).resolve() != Path(library).resolve() for s in stages):
            raise ValueError('Stages must use the schedule library')
        self.device = stages[0].device
        if any(s.device != self.device for s in stages):
            raise ValueError('Stage devices must match')
        sources = [-1]+list(range(len(stages)-1)) if sources is None else list(sources)
        if len(sources) != len(stages) or any(not isinstance(p, int) or p < -1 or p >= i for i, p in enumerate(sources)):
            raise ValueError('Sources must reference prior stages or -1')
        parents, node_stage, node_task, bases = [], [], [], []
        for i, stage in enumerate(stages):
            source = sources[i]
            if source >= 0:
                parent = stages[source]
                if stage.precision != 'bf16' or stage.a.shape != parent.physical_output.shape:
                    raise ValueError('Direct sources require matching physical BF16 dimensions')
                if stage.rows != parent.rows or stage.a.shape[1] != parent.columns:
                    raise ValueError('Logical dimensions must match across stages')
            bases.append(len(parents))
            mt, nt = stage.a.shape[0]//128, stage.b.shape[0]//128
            for task in range(mt*nt):
                row = task//nt
                source_tiles = 0 if source < 0 else stages[source].b.shape[0]//128
                dependencies = [] if source < 0 else list(range(bases[source]+row*source_tiles, bases[source]+(row+1)*source_tiles))
                parents.append(dependencies); node_stage.append(i); node_task.append(task)
        if node_order is not None:
            order = np.asarray(node_order, dtype=np.int64)
            if order.shape != (len(parents),) or not np.array_equal(np.sort(order), np.arange(len(parents))):
                raise ValueError('Invalid node permutation')
            inverse = np.empty_like(order); inverse[order] = np.arange(len(order))
            parents = [[int(inverse[p]) for p in parents[old]] for old in order]
            node_stage, node_task = [node_stage[i] for i in order], [node_task[i] for i in order]
        self.node_stage, self.node_task = np.array(node_stage, np.int32), np.array(node_task, np.int32)
        self.library = ctypes.CDLL(str(library)); self.workers = workers
        with torch.cuda.device(self.device):
            self._queue = ResidentDagProbe(library, parents, workers=workers)
        create = self.library.create_edge_dense_dag
        create.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p, ctypes.c_int]+[ctypes.c_void_p]*4
        create.restype = ctypes.c_int
        handles = np.array([s.handle.value for s in stages], np.uintp)
        sources = np.array(sources, np.int32)
        self.launch = self.library.launch_edge_dense_dag
        self.launch.argtypes = [ctypes.c_void_p]*2; self.launch.restype = ctypes.c_int
        self.destroy = self.library.destroy_edge_dense_dag
        self.destroy.argtypes = [ctypes.c_void_p]; self.destroy.restype = ctypes.c_int
        with torch.cuda.device(self.device):
            error = create(ctypes.byref(self.handle), self._queue.handle, len(stages), handles.ctypes.data,
                           self.node_stage.ctypes.data, self.node_task.ctypes.data, sources.ctypes.data)
        if error:
            self._queue.close()
            raise RuntimeError(f'Dense schedule setup CUDA error {error}')
        self.output = stages[-1].output

    def forward(self):
        if not self.handle.value:
            raise RuntimeError('Dense schedule is closed')
        with torch.cuda.device(self.device):
            error = self.launch(self.handle, torch.cuda.current_stream(self.device).cuda_stream)
        if error:
            raise RuntimeError(f'Dense schedule launch CUDA error {error}')
        self._queue.epoch += 1
        return self.output

    def read_diagnostics(self):
        return self._queue.read()

    def close(self):
        if self.handle.value:
            with torch.cuda.device(self.device):
                error = self.destroy(self.handle)
            if error:
                raise RuntimeError(f'Dense schedule destruction CUDA error {error}')
            self.handle = ctypes.c_void_p()
        self._queue.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
