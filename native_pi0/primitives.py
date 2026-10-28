"""Development native VM primitives; not the complete NativePi0 interface."""

import ctypes

import numpy as np
import torch
from native_pi0.storage import packed_tma_buffer

from native_pi0.quantization import swizzle_scales


class ResidentNVFP4:
    """One device-resident FP4 GEMM per VM launch, with offline packed inputs.

    Constructor performs setup and transfers. `forward` launches only the
    original native VM on the current PyTorch stream and returns reused output.
    Concurrent calls on different streams must use independent instances.
    With uint8 output, `output` holds packed E2M1 data and `output_scales`
    holds swizzled UE4M3 bytes; `output_global_scale` is its descale factor.
    """

    def __init__(self, library, activation, weight, *, workers=16, device="cuda",
                 output_dtype=torch.float32, bias=None, residual=None,
                 output_global_scale=1.0, output_gelu=False, output_geglu=False, output_relu2=False):
        self._handle = ctypes.c_void_p()
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise ValueError("CUDA device required")
        if self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        for operand in (activation, weight):
            if (operand.data.dtype != np.uint8 or operand.data.ndim != 2
                    or not all(operand.data.shape) or operand.scales.dtype != np.uint8
                    or operand.scales.ndim != 2 or np.any(operand.scales > 126)):
                raise ValueError("Expected packed uint8 E2M1 data and finite unsigned E4M3 scales")
        m, half_k = activation.data.shape
        n, weight_half_k = weight.data.shape
        k = half_k * 2
        if weight_half_k != half_k or m % 128 or n % 128 or k % 64:
            raise ValueError("Packed M/N must be multiples of 128 and K a multiple of 64")
        if activation.scales.shape != (m, k // 16) or weight.scales.shape != (n, k // 16):
            raise ValueError("Scale dimensions do not match operands")
        if output_dtype not in (torch.float32, torch.bfloat16, torch.uint8):
            raise ValueError("Output must be FP32, BF16, or packed FP4 (uint8)")
        if output_dtype == torch.float32 and (bias is not None or residual is not None):
            raise ValueError("Bias/residual fusion requires BF16 or FP4 output")
        self.output_global_scale = np.float32(output_global_scale)
        self.activation_global_scale = np.float32(activation.global_scale)
        self.weight_global_scale = np.float32(weight.global_scale)
        if not np.isfinite(self.output_global_scale) or self.output_global_scale <= 0:
            raise ValueError("Output global scale must be finite and positive")
        if sum(map(bool, (output_gelu, output_geglu, output_relu2))) > 1:
            raise ValueError("Select only one output activation")
        if output_geglu and (output_gelu or bias is not None or residual is not None):
            raise ValueError("Compact GeGLU requires interleaved gate/up weights without bias/residual")
        if output_dtype != torch.uint8 and (output_gelu or output_geglu or output_relu2 or self.output_global_scale != 1):
            raise ValueError("Output activation/global scaling requires packed FP4 output")
        self._library = ctypes.CDLL(str(library))
        self._create = self._library.create_resident_fp4
        self._create.argtypes = [ctypes.POINTER(ctypes.c_void_p)] + [ctypes.c_void_p]*5 + [ctypes.c_int]*4 + [ctypes.c_float]
        self._create.restype = ctypes.c_int
        self._launch = self._library.launch_resident_fp4
        self._launch.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        self._launch.restype = ctypes.c_int
        self._destroy = self._library.destroy_resident_fp4
        self._destroy.argtypes = [ctypes.c_void_p]
        self._destroy.restype = ctypes.c_int
        alpha = np.float32(activation.global_scale * weight.global_scale)
        with torch.cuda.device(self.device):
            self.a,self.b=[packed_tma_buffer(array.shape,device=self.device).copy_(torch.from_numpy(np.ascontiguousarray(array)))
                           for array in (activation.data,weight.data)]
            self.sa, self.sb = [
                torch.from_numpy(np.ascontiguousarray(array)).to(self.device)
                for array in (swizzle_scales(activation.scales), swizzle_scales(weight.scales))
            ]
            self.bias = None if bias is None else torch.as_tensor(bias, dtype=torch.bfloat16, device=self.device).contiguous()
            self.residual = None if residual is None else torch.as_tensor(residual, dtype=torch.bfloat16, device=self.device).contiguous()
            if self.bias is not None and tuple(self.bias.shape) != (n,):
                raise ValueError("Bias must have shape (N,)")
            if self.residual is not None and tuple(self.residual.shape) != (m, n):
                raise ValueError("Residual must have padded shape (M,N)")
            self.output_width = n//2 if output_geglu else n
            self.output = (packed_tma_buffer((m,self.output_width//2),device=self.device) if output_dtype==torch.uint8
                           else torch.empty((m,n),dtype=output_dtype,device=self.device))
            self.output_scales = None
            if output_dtype == torch.uint8:
                self.output_scales = torch.zeros(((m+127)//128)*((self.output_width//16+3)//4)*512, dtype=torch.uint8, device=self.device)
            error = self._create(ctypes.byref(self._handle), self.a.data_ptr(), self.b.data_ptr(),
                                 self.sa.data_ptr(), self.sb.data_ptr(), self.output.data_ptr(), m, n, k, workers, float(alpha))
        if error:
            raise RuntimeError(f"Native FP4 setup CUDA error {error}")
        if output_dtype == torch.uint8:
            configure = self._library.configure_resident_fp4_quantized
            configure.argtypes = [ctypes.c_void_p]*5+[ctypes.c_float,ctypes.c_int]
            configure.restype = ctypes.c_int
            error = configure(self._handle, None if self.bias is None else self.bias.data_ptr(),
                              None if self.residual is None else self.residual.data_ptr(), self.output.data_ptr(),
                              self.output_scales.data_ptr(), float(self.output_global_scale), 3 if output_relu2 else (2 if output_geglu else int(output_gelu)))
            if error:
                self.close()
                raise RuntimeError(f"Native FP4 quantized-output setup CUDA error {error}")
        if output_dtype == torch.bfloat16:
            configure = self._library.configure_resident_fp4_bf16
            configure.argtypes = [ctypes.c_void_p]*4
            configure.restype = ctypes.c_int
            error = configure(self._handle, None if self.bias is None else self.bias.data_ptr(),
                              None if self.residual is None else self.residual.data_ptr(), self.output.data_ptr())
            if error:
                self.close()
                raise RuntimeError(f"Native FP4 epilogue setup CUDA error {error}")
        query = self._library.resident_fp4_resources
        query.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int), ctypes.c_int]
        query.restype = ctypes.c_int
        values = (ctypes.c_int * 8)()
        with torch.cuda.device(self.device):
            error = query(self._handle, values, len(values))
        if error:
            self.close()
            raise RuntimeError(f"Native FP4 resource query CUDA error {error}")
        self.resources = dict(zip(("registers_per_thread", "static_shared_bytes", "dynamic_shared_bytes",
                                   "local_bytes_per_thread", "predicted_max_ctas_per_sm", "threads_per_cta",
                                   "max_threads_per_sm", "sm_count"), values))

    def forward(self):
        if not self._handle.value:
            raise RuntimeError("Native FP4 instance is closed")
        with torch.cuda.device(self.device):
            stream = torch.cuda.current_stream(self.device)
            error = self._launch(self._handle, stream.cuda_stream)
        if error:
            raise RuntimeError(f"Native FP4 launch CUDA error {error}")
        return self.output

    def close(self):
        if self._handle.value:
            with torch.cuda.device(self.device):
                error = self._destroy(self._handle)
            if error:
                raise RuntimeError(f"Native FP4 teardown CUDA error {error}")
            self._handle = ctypes.c_void_p()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
