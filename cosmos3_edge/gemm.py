"""Resident native BF16/FP8 TMA tensor-core projections.

Conversion, padding, and quantization are setup-only. For FP8, pass prequantized
E4M3 tensors and calibrated descales; no activation quantization runs in forward.
"""
import ctypes
import math

import torch


class NativeProjection:
    def __init__(self, library, activation, weight, *, precision='bf16',
                 activation_scale=1., weight_scale=1., bias=None, residual=None,
                 relu2=False, workers=148, device='cuda', share_activation=False, share_residual=False, relu2_round_input=False, round_before_residual=False):
        if precision not in ('bf16', 'fp8'):
            raise ValueError('Precision must be bf16 or fp8')
        dtype = torch.bfloat16 if precision == 'bf16' else torch.float8_e4m3fn
        activation, weight = torch.as_tensor(activation), torch.as_tensor(weight)
        if activation.ndim != 2 or weight.ndim != 2 or not all(activation.shape) or not all(weight.shape):
            raise ValueError('Activation and weight must be nonempty matrices')
        if activation.shape[1] != weight.shape[1] or activation.shape[1] % 64:
            raise ValueError('Projection K must match and be divisible by 64')
        if precision == 'fp8' and (activation.dtype != dtype or weight.dtype != dtype):
            raise ValueError('FP8 input and weights must be prequantized E4M3 tensors')
        if precision == 'bf16' and (activation_scale != 1. or weight_scale != 1.):
            raise ValueError('BF16 uses unit descales')
        alpha = float(activation_scale)*float(weight_scale)
        if not all(math.isfinite(float(s)) and float(s)>0 for s in (activation_scale, weight_scale, alpha)):
            raise ValueError('Descales must be finite and positive')
        self.device = torch.device(device)
        if self.device.type != 'cuda':
            raise ValueError('CUDA device required')
        if self.device.index is None:
            self.device = torch.device('cuda', torch.cuda.current_device())
        if self.device.type != 'cuda':
            raise ValueError('CUDA device required')
        if not isinstance(workers, int) or workers < 1:
            raise ValueError('Positive integer worker count required')
        self.precision = precision
        self.workers = workers
        self.rows, k = activation.shape
        self.columns = weight.shape[0]
        m, n = ((d+127)//128*128 for d in (self.rows, self.columns))
        with torch.cuda.device(self.device):
            if share_activation:
                if activation.device!=self.device or activation.dtype!=dtype or activation.shape!=(m,k) or not activation.is_contiguous():
                    raise ValueError('Shared activation must be a padded contiguous device matrix of the expected dtype')
                self.a = activation
            else:
                self.a = torch.zeros((m, k), dtype=dtype, device=self.device)
            self.b = torch.zeros((n, k), dtype=dtype, device=self.device)
            if not share_activation:
                self.a[:self.rows].copy_(activation)
            self.b[:self.columns].copy_(weight)
            self.bias, self.residual = None, None
            if bias is not None:
                bias = torch.as_tensor(bias)
                if tuple(bias.shape) != (self.columns,):
                    raise ValueError('Bias must match logical output width')
                self.bias = torch.zeros(n, dtype=torch.bfloat16, device=self.device)
                self.bias[:self.columns].copy_(bias)
            if residual is not None:
                residual = torch.as_tensor(residual)
                if tuple(residual.shape) != (self.rows, self.columns):
                    raise ValueError('Residual must match logical output shape')
                if share_residual:
                    if residual.device!=self.device or residual.dtype!=torch.bfloat16 or residual.shape!=(m,n) or not residual.is_contiguous():
                        raise ValueError('Shared residual must be a padded contiguous BF16 device matrix')
                    self.residual = residual
                else:
                    self.residual = torch.zeros((m, n), dtype=torch.bfloat16, device=self.device)
                    self.residual[:self.rows, :self.columns].copy_(residual)
            self.physical_output = torch.empty((m, n), dtype=torch.bfloat16, device=self.device)
            self.output = self.physical_output[:self.rows, :self.columns]
            self.library = ctypes.CDLL(str(library)); self.handle = ctypes.c_void_p()
            create = self.library.create_edge_projection
            create.argtypes = [ctypes.POINTER(ctypes.c_void_p)]+[ctypes.c_void_p]*5+[ctypes.c_int]*4+[ctypes.c_float]+[ctypes.c_int]*2
            create.restype = ctypes.c_int
            self.launch = self.library.launch_edge_projection
            self.launch.argtypes = [ctypes.c_void_p]*2; self.launch.restype = ctypes.c_int
            self.destroy = self.library.destroy_edge_projection
            self.destroy.argtypes = [ctypes.c_void_p]; self.destroy.restype = ctypes.c_int
            error = create(ctypes.byref(self.handle), self.a.data_ptr(), self.b.data_ptr(),
                self.physical_output.data_ptr(), None if self.bias is None else self.bias.data_ptr(),
                None if self.residual is None else self.residual.data_ptr(), m, n, k, workers,
                alpha, 2 if relu2 and relu2_round_input else int(relu2), int(precision == 'fp8'))
        if error:
            raise RuntimeError(f'Native projection setup CUDA error {error}')
        if round_before_residual:
            configure = self.library.configure_edge_projection_reference
            configure.argtypes = [ctypes.c_void_p]; configure.restype = ctypes.c_int
            error = configure(self.handle)
            if error:
                self.close()
                raise RuntimeError(f'Reference residual configuration CUDA error {error}')

    def forward(self):
        if not self.handle.value:
            raise RuntimeError('Native projection is closed')
        with torch.cuda.device(self.device):
            error = self.launch(self.handle, torch.cuda.current_stream(self.device).cuda_stream)
        if error:
            raise RuntimeError(f'Native projection launch CUDA error {error}')
        return self.output

    def close(self):
        if self.handle.value:
            with torch.cuda.device(self.device):
                error = self.destroy(self.handle)
            if error:
                raise RuntimeError(f'Native projection destruction CUDA error {error}')
            self.handle = ctypes.c_void_p()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
