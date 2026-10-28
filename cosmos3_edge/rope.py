"""Checkpoint-derived interleaved 3D RoPE tables and native Q/K preparation."""
import ctypes
import json
import math

import torch

from .spec import EdgeSpec, SOURCES


def mrope_tables(position_ids, *, dtype=torch.bfloat16, device='cpu'):
    """Setup-only tables for batch 1: integer [3,rows] or text-only [rows].

    Matches official MultiModalRotaryEmbedding: temporal frequencies are the
    base; spatial axes replace slots 1/2 modulo 3 up to their section lengths.
    """
    s = EdgeSpec.from_checkpoint()
    config = json.loads((SOURCES/'transformer__config.json').read_text())
    ids = torch.as_tensor(position_ids, device='cpu')
    if ids.dtype not in (torch.int32, torch.int64):
        raise ValueError('Position IDs must be integer coordinates')
    if ids.ndim == 1:
        ids = ids.unsqueeze(0).expand(3, -1)
    if ids.ndim != 2 or ids.shape[0] != 3 or ids.shape[1] < 1:
        raise ValueError('Expected positions [3,rows] or [rows]')
    inv = 1. / (config['rope_theta'] ** (torch.arange(0, s.head_dim, 2).float()/s.head_dim))
    freqs = ids.float()[:, :, None]*inv[None, None, :]
    temporal = freqs[0].clone()
    for axis, offset in ((1, 1), (2, 2)):
        temporal[:, offset:s.rope_axes[axis]*3:3] = freqs[axis, :, offset:s.rope_axes[axis]*3:3]
    embedding = torch.cat((temporal, temporal), dim=-1)
    return embedding.cos().to(device=device, dtype=dtype), embedding.sin().to(device=device, dtype=dtype)


class NativeQKNormRope:
    """One original VM launch: Q/K RMSNorm + RoPE, optional V cache copy.

    Inputs are BF16 [rows, heads, 128], allowing row-strided projection views.
    A caller-owned cache output may be a contiguous head slice of a larger
    sequence, so static understanding K/V need no copying. K-only mode creates
    generation-normalized understanding keys from pre-RoPE K projections,
    writing a separate cache and leaving the input unchanged.
    Instances require ordered calls and caller-owned outputs must outlive them.
    """
    def __init__(self, library, q, k, q_weight, k_weight, cos, sin, *, v=None,
                 k_output=None, v_output=None, rounding='fused', epsilon=None,
                 workers=148, device='cuda'):
        self.handle = ctypes.c_void_p()
        if rounding not in ('fused', 'reference'):
            raise ValueError('Rounding must be fused or reference')
        self.device = torch.device(device)
        if self.device.type != 'cuda':
            raise ValueError('CUDA device required')
        if self.device.index is None:
            self.device = torch.device('cuda', torch.cuda.current_device())
        s = EdgeSpec.from_checkpoint()
        epsilon = s.rms_eps if epsilon is None else float(epsilon)
        if not math.isfinite(epsilon) or epsilon <= 0:
            raise ValueError('Epsilon must be finite and positive')
        if not isinstance(workers, int) or workers < 1:
            raise ValueError('Positive integer worker count required')
        def matrix(x, name):
            if x is None:
                return None
            if x.ndim != 3 or x.shape[0] < 1 or x.shape[1] < 1 or x.shape[2] != 128 or x.dtype != torch.bfloat16:
                raise ValueError(f'{name} must be BF16 [rows,heads,128]')
            y = x.to(self.device)
            if y.stride(2) != 1 or y.stride(1) != 128 or y.stride(0) < y.shape[1]*128 or y.stride(0)%4 or y.data_ptr()%8:
                raise ValueError(f'{name} has incompatible vector strides/alignment')
            return y
        with torch.cuda.device(self.device):
            self.q, self.k, self.v = matrix(q, 'Q'), matrix(k, 'K'), matrix(v, 'V')
            if self.k is None:
                raise ValueError('K is required')
            rows, kh, _ = self.k.shape
            qh = 0 if self.q is None else self.q.shape[1]
            if (self.q is not None and self.q.shape[0] != rows) or (self.v is not None and self.v.shape != self.k.shape):
                raise ValueError('Q/K/V rows and K/V heads must match')
            def parameter(x, shape, name):
                y = torch.as_tensor(x, dtype=torch.bfloat16, device=self.device).contiguous()
                if tuple(y.shape) != shape:
                    raise ValueError(f'{name} shape mismatch')
                return y
            if self.q is None and q_weight is not None:
                raise ValueError('K-only mode requires q_weight=None')
            if k_weight is None or (self.q is not None and q_weight is None):
                raise ValueError('Normalization weights are required')
            self.qw = None if self.q is None else parameter(q_weight, (128,), 'Q norm')
            self.kw = parameter(k_weight, (128,), 'K norm')
            self.cos, self.sin = parameter(cos, (rows, 128), 'cos'), parameter(sin, (rows, 128), 'sin')
            self.q_output = None if self.q is None else torch.empty(self.q.shape, dtype=torch.bfloat16, device=self.device)
            def output(x, name):
                if x is not None and x.device != self.device:
                    raise ValueError('Caller-owned outputs must already be on the target device')
                y = torch.empty(self.k.shape, dtype=torch.bfloat16, device=self.device) if x is None else matrix(x, name)
                if y.shape != self.k.shape:
                    raise ValueError('Cache output must match K/V logical shape')
                return y
            self.k_output = output(k_output, 'K output')
            if self.v is None and v_output is not None:
                raise ValueError('V output requires V input')
            self.v_output = None if self.v is None else output(v_output, 'V output')
            # Global heads from different warps must not overwrite unread inputs.
            reads = [x for x in (self.q, self.k, self.v) if x is not None]
            writes = [x for x in (self.q_output, self.k_output, self.v_output) if x is not None]
            def bounds(x):
                return x.data_ptr(), x.data_ptr()+((x.shape[0]-1)*x.stride(0)+x.shape[1]*128)*2
            for i, out in enumerate(writes):
                lo, hi = bounds(out)
                for other in reads+writes[:i]:
                    a, b = bounds(other)
                    if lo < b and a < hi:
                        raise ValueError('Q/K/V output ranges must not alias inputs or each other')
            self.library = ctypes.CDLL(str(library))
            create = self.library.create_edge_rope
            create.argtypes = [ctypes.POINTER(ctypes.c_void_p)]+[ctypes.c_void_p]*10+[ctypes.c_int]*10+[ctypes.c_float, ctypes.c_int]
            create.restype = ctypes.c_int
            pointers = [self.q, self.k, self.v, self.qw, self.kw, self.cos, self.sin, self.q_output, self.k_output, self.v_output]
            strides = [0 if x is None else x.stride(0) for x in (self.q, self.k, self.v, self.q_output, self.k_output, self.v_output)]
            error = create(ctypes.byref(self.handle), *[None if x is None else x.data_ptr() for x in pointers],
                           rows, qh, kh, *strides, workers, epsilon, int(rounding == 'reference'))
        if error:
            raise RuntimeError(f'Native QK preparation setup CUDA error {error}')
        self.launch = self.library.launch_edge_rope
        self.launch.argtypes = [ctypes.c_void_p]*2; self.launch.restype = ctypes.c_int
        self.destroy = self.library.destroy_edge_rope
        self.destroy.argtypes = [ctypes.c_void_p]; self.destroy.restype = ctypes.c_int

    def forward(self):
        if not self.handle.value:
            raise RuntimeError('QK preparation is closed')
        with torch.cuda.device(self.device):
            error = self.launch(self.handle, torch.cuda.current_stream(self.device).cuda_stream)
        if error:
            raise RuntimeError(f'Native QK preparation launch CUDA error {error}')
        return self.q_output, self.k_output, self.v_output

    def close(self):
        if self.handle.value:
            with torch.cuda.device(self.device):
                error = self.destroy(self.handle)
            if error:
                raise RuntimeError(f'Native QK preparation teardown CUDA error {error}')
            self.handle = ctypes.c_void_p()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
