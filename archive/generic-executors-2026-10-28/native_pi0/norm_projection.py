"""Native normalization → NVFP4 projection in one persistent invocation."""
import ctypes

import numpy as np
import torch

from native_pi0.normalization import ResidentNorm
from native_pi0.quantization import swizzle_scales


class ResidentNormProjection:
    """Development producer/consumer chain; concurrent calls need separate instances.

    Setup packs/uploads weights and allocates buffers. Forward reads mutable
    `inputs`/`residual` on the current stream and reuses output. Graph capture is
    unsupported because each invocation passes a fresh host-generated row epoch.
    A single instance must be ordered on one stream or by explicit stream events.
    """
    def __init__(self, library, inputs, weight, *, workers=16, output_dtype=torch.float32, **norm_options):
        if output_dtype not in (torch.float32,torch.bfloat16):
            raise ValueError("Projection output must be FP32 or BF16")
        if weight.data.dtype!=np.uint8 or weight.data.ndim!=2 or weight.scales.dtype!=np.uint8:
            raise ValueError("Expected packed NVFP4 weights")
        n,half_k=weight.data.shape
        if n<128 or n%128 or half_k*2!=inputs.shape[1] or half_k*2%64:
            raise ValueError("Projection N must be a multiple of 128 and K a multiple of 64")
        if weight.scales.shape!=(n,half_k*2//16) or np.any(weight.scales>126):
            raise ValueError("Invalid unsigned UE4M3 weight scales")
        if not np.isfinite(weight.global_scale) or weight.global_scale<=0:
            raise ValueError("Weight global scale must be positive and finite")
        if 'precision' in norm_options or 'row_alignment' in norm_options:
            raise ValueError("The projection requires FP4 normalization and 128-row padding")
        self.handle=ctypes.c_void_p()
        self.norm=ResidentNorm(library,inputs,precision='fp4',row_alignment=128,workers=min(workers,inputs.shape[0]),**norm_options)
        self.device=self.norm.device
        self.library=ctypes.CDLL(str(library))
        self.create=self.library.create_norm_projection
        self.create.argtypes=[ctypes.POINTER(ctypes.c_void_p)]+[ctypes.c_void_p]*4+[ctypes.c_int]*2+[ctypes.c_float,ctypes.c_int]
        self.create.restype=ctypes.c_int
        self.launch=self.library.launch_norm_projection
        self.launch.argtypes=[ctypes.c_void_p]*2;self.launch.restype=ctypes.c_int
        self.destroy=self.library.destroy_norm_projection
        self.destroy.argtypes=[ctypes.c_void_p];self.destroy.restype=ctypes.c_int
        with torch.cuda.device(self.device):
            self.weight=torch.from_numpy(np.ascontiguousarray(weight.data)).to(self.device)
            self.weight_scales=torch.from_numpy(swizzle_scales(weight.scales)).to(self.device)
            self.output=torch.empty((self.norm.output.shape[0],n),dtype=output_dtype,device=self.device)
            error=self.create(ctypes.byref(self.handle),self.norm.handle,self.weight.data_ptr(),self.weight_scales.data_ptr(),
                              self.output.data_ptr(),n,workers,float(weight.global_scale),int(output_dtype==torch.bfloat16))
        if error:
            self.norm.close()
            raise RuntimeError(f"Native normalization/projection setup CUDA error {error}")
        self.inputs=self.norm.inputs
        self.residual=self.norm.residual
        self.residual_output=self.norm.residual_output
        self.workers=workers
        self.weight_global_scale=np.float32(weight.global_scale)

    def forward(self):
        if not self.handle.value:raise RuntimeError("Normalization/projection is closed")
        with torch.cuda.device(self.device):
            error=self.launch(self.handle,torch.cuda.current_stream(self.device).cuda_stream)
        if error:raise RuntimeError(f"Native normalization/projection launch CUDA error {error}")
        return self.output

    def close(self):
        if self.handle.value:
            with torch.cuda.device(self.device):error=self.destroy(self.handle)
            if error:raise RuntimeError(f"Native normalization/projection teardown CUDA error {error}")
            self.handle=ctypes.c_void_p()
        self.norm.close()

    def __enter__(self):return self
    def __exit__(self,*_):self.close()
