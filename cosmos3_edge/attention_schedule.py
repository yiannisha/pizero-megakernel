"""Device-side QKV → Q/K normalization/RoPE → joint GQA schedule."""
import ctypes
from pathlib import Path

import numpy as np
import torch

from native_pi0.dag_probe import ResidentDagProbe


class GenerationAttentionSchedule:
    """One VM launch for caller-owned projections, preparation, and attention.

    The understanding prefix is prepared once before this boundary. Instances
    and their component buffers require ordered execution, not concurrent use.
    Components can also execute separately in stream order for validation.
    """
    def __init__(self, library, projections, preparation, attention, *, workers=148, node_order=None):
        self.projections = tuple(projections)
        self.preparation, self.attention = preparation, attention
        self.handle = ctypes.c_void_p()
        if len(self.projections) != 3:
            raise ValueError('Expected Q, K, V projection instances')
        components = self.projections+(preparation, attention)
        self.device = projections[0].device
        if any(not c.handle.value or c.device != self.device or Path(c.library._name).resolve() != Path(library).resolve() for c in components):
            raise ValueError('Components must be live in the same library and device')
        if len({p.handle.value for p in self.projections}) != 3:
            raise ValueError('Projection instances must be distinct')
        rows = preparation.k.shape[0]
        if preparation.q is None or preparation.v is None or preparation.q.shape[1:] != (16, 128) or preparation.k.shape[1:] != (8, 128):
            raise ValueError('Cosmos Q16/KV8 preparation with V publication is required')
        for stage, inp in zip(self.projections, (preparation.q, preparation.k, preparation.v)):
            if inp.data_ptr() != stage.output.data_ptr() or stage.rows != rows:
                raise ValueError('Preparation must read the corresponding full projection output')
        self.library = ctypes.CDLL(str(library)); self.workers = workers
        query = self.library.edge_rope_vectors_per_task
        query.argtypes = []; query.restype = ctypes.c_int
        batch = query()
        parents, node_stage, node_task, bases = [], [], [], []
        for stage, projection in enumerate(self.projections):
            bases.append(len(parents))
            count = (projection.a.shape[0]//128)*(projection.b.shape[0]//128)
            for task in range(count):
                parents.append([]); node_stage.append(stage); node_task.append(task)
        prep_base = len(parents)
        qvec, total = rows*16, rows*24
        prep_tasks = (total+batch-1)//batch
        for task in range(prep_tasks):
            deps = set()
            for vec in range(task*batch, min((task+1)*batch, total)):
                if vec < qvec:
                    row, head = divmod(vec, 16)
                    deps.add(bases[0]+(row//128)*16+head)
                else:
                    row, head = divmod(vec-qvec, 8)
                    for stage in (1, 2):
                        deps.add(bases[stage]+(row//128)*8+head)
            parents.append(sorted(deps)); node_stage.append(-1); node_task.append(task)
        # A single join avoids repeating the complete key prerequisite set for
        # every query tile. Attention retains tile-specific Q prerequisites.
        ready_node = len(parents)
        kv_tasks = range(qvec//batch, prep_tasks)
        parents.append([prep_base+i for i in kv_tasks]); node_stage.append(-3); node_task.append(0)
        offset = attention.q.data_ptr()-preparation.q_output.data_ptr()
        if offset < 0 or offset%(2048*2):
            raise ValueError('Attention queries must be complete rows from prepared Q')
        offset //= 2048*2
        queries = attention.q.shape[1]
        if offset+queries > rows:
            raise ValueError('Attention queries exceed prepared Q')
        tiles = (queries+63)//64
        for logical in range(16*tiles):
            head, tile = divmod(logical, tiles)
            deps = {ready_node}
            for row in range(offset+tile*64, offset+min((tile+1)*64, queries)):
                deps.add(prep_base+(row*16+head)//batch)
            for split in range(attention.splits):
                parents.append(sorted(deps)); node_stage.append(-2); node_task.append(logical*attention.splits+split)
        if node_order is not None:
            order = np.asarray(node_order, dtype=np.int64)
            if order.shape != (len(parents),) or not np.array_equal(np.sort(order), np.arange(len(parents))):
                raise ValueError('Invalid node permutation')
            inverse = np.empty_like(order); inverse[order] = np.arange(len(order))
            parents = [[int(inverse[p]) for p in parents[old]] for old in order]
            node_stage, node_task = [node_stage[i] for i in order], [node_task[i] for i in order]
        self.node_stage, self.node_task = np.array(node_stage, np.int32), np.array(node_task, np.int32)
        with torch.cuda.device(self.device):
            self._queue = ResidentDagProbe(library, parents, workers=workers)
        create = self.library.create_generation_attention
        create.argtypes = [ctypes.POINTER(ctypes.c_void_p)]+[ctypes.c_void_p]*6; create.restype = ctypes.c_int
        handles = np.array([p.handle.value for p in self.projections], np.uintp)
        self.launch = self.library.launch_generation_attention
        self.launch.argtypes = [ctypes.c_void_p]*2; self.launch.restype = ctypes.c_int
        self.epoch = self.library.generation_attention_epoch
        self.epoch.argtypes = [ctypes.c_void_p]; self.epoch.restype = ctypes.c_uint
        self.destroy = self.library.destroy_generation_attention
        self.destroy.argtypes = [ctypes.c_void_p]; self.destroy.restype = ctypes.c_int
        with torch.cuda.device(self.device):
            error = create(ctypes.byref(self.handle), self._queue.handle, handles.ctypes.data,
                           preparation.handle, attention.handle, self.node_stage.ctypes.data, self.node_task.ctypes.data)
        if error:
            self._queue.close()
            raise RuntimeError(f'Generation attention setup CUDA error {error}')
        self.output = attention.output

    def forward(self):
        if not self.handle.value:
            raise RuntimeError('Generation attention schedule is closed')
        with torch.cuda.device(self.device):
            error = self.launch(self.handle, torch.cuda.current_stream(self.device).cuda_stream)
        if error:
            raise RuntimeError(f'Generation attention launch CUDA error {error}')
        self._queue.epoch = int(self.epoch(self.handle)); self.attention.epoch = self._queue.epoch
        return self.output

    def read_diagnostics(self):
        return self._queue.read()

    def close(self):
        if self.handle.value:
            with torch.cuda.device(self.device):
                error = self.destroy(self.handle)
            if error:
                raise RuntimeError(f'Generation attention teardown CUDA error {error}')
            self.handle = ctypes.c_void_p()
        self._queue.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
