"""Resident development normalization producers in the original native VM."""
import ctypes

import numpy as np
import torch


class ResidentNorm:
    """LayerNorm/RMSNorm → FP8 or FP4, with FP32 affine parameters.

    RMSNorm gamma must already contain (1+w), or be omitted for folded weights.
    Residual and normalized-value rounding policies are explicit setup choices.
    Forward reuses device buffers and launches one native VM kernel.
    """
    def __init__(self, library, inputs, *, kind="rms", precision="fp4", gamma=None,
                 beta=None, inverse_scale=None, residual=None, workers=16, epsilon=1e-6,
                 global_scale=1., round_normalized=True, round_residual=True, legacy_ties=False, row_alignment=1):
        if inputs.dtype not in (torch.bfloat16, torch.float16) or inputs.ndim!=2:
            raise ValueError("Inputs must be a BF16/FP16 matrix")
        rows,width=inputs.shape
        if rows<1 or width<16 or width>2048 or width%16:
            raise ValueError("Width must be a multiple of 16 up to 2048")
        if kind not in ("rms","layer") or precision not in ("fp4","fp8"):
            raise ValueError("Unknown normalization or precision")
        if not isinstance(row_alignment,int) or row_alignment<1:raise ValueError("Invalid row alignment")
        self.logical_rows=rows
        self.workers=workers
        output_rows=(rows+row_alignment-1)//row_alignment*row_alignment
        if kind=="rms" and beta is not None:
            raise ValueError("Standard pi0 RMSNorm has no beta")
        if not np.isfinite(epsilon) or epsilon<=0 or not np.isfinite(global_scale) or global_scale<=0:
            raise ValueError("Epsilon and global scale must be finite and positive")
        self.device=inputs.device if inputs.is_cuda else torch.device("cuda",torch.cuda.current_device())
        self.global_scale=np.float32(global_scale)
        self.library=ctypes.CDLL(str(library));self.handle=ctypes.c_void_p()
        self.create=self.library.create_norm
        self.create.argtypes=[ctypes.POINTER(ctypes.c_void_p)]+[ctypes.c_void_p]*8+[ctypes.c_int]*9+[ctypes.c_float]*2
        self.create.restype=ctypes.c_int
        self.launch=self.library.launch_norm;self.launch.argtypes=[ctypes.c_void_p]*2;self.launch.restype=ctypes.c_int
        self.destroy=self.library.destroy_norm;self.destroy.argtypes=[ctypes.c_void_p];self.destroy.restype=ctypes.c_int
        with torch.cuda.device(self.device):
            self.inputs=inputs.to(self.device).contiguous()
            self.residual=None if residual is None else torch.as_tensor(residual,dtype=inputs.dtype,device=self.device).contiguous()
            if self.residual is not None and self.residual.shape!=inputs.shape:raise ValueError("Residual shape mismatch")
            self.gamma,self.beta,self.inverse_scale=[None if x is None else torch.as_tensor(x,dtype=torch.float32,device=self.device).contiguous()
                                                      for x in (gamma,beta,inverse_scale)]
            for vector in (self.gamma,self.beta,self.inverse_scale):
                if vector is not None and tuple(vector.shape)!=(width,):raise ValueError("Affine/AWQ vector shape mismatch")
            self.residual_output=None if residual is None else torch.empty_like(self.inputs)
            allocate=torch.zeros if output_rows!=rows else torch.empty
            self.output=allocate((output_rows,width//2 if precision=="fp4" else width),device=self.device,
                                 dtype=torch.uint8 if precision=="fp4" else torch.float8_e4m3fn)
            self.output_scales=None if precision=="fp8" else torch.zeros(((output_rows+127)//128)*((width//16+3)//4)*512,
                                                                        dtype=torch.uint8,device=self.device)
            def pointer(t):return None if t is None else t.data_ptr()
            error=self.create(ctypes.byref(self.handle),*[pointer(t) for t in (self.inputs,self.residual,self.gamma,self.beta,self.inverse_scale,
                                                                                 self.residual_output,self.output,self.output_scales)],
                              rows,width,workers,int(inputs.dtype==torch.float16),4 if precision=="fp4" else 8,int(kind=="layer"),
                              int(round_normalized),int(round_residual),int(legacy_ties),float(epsilon),float(self.global_scale))
        if error:raise RuntimeError(f"Native norm setup CUDA error {error}")

    def forward(self):
        if not self.handle.value:raise RuntimeError("Normalization instance is closed")
        with torch.cuda.device(self.device):
            error=self.launch(self.handle,torch.cuda.current_stream(self.device).cuda_stream)
        if error:raise RuntimeError(f"Native norm launch CUDA error {error}")
        return self.output

    def close(self):
        if self.handle.value:
            with torch.cuda.device(self.device):error=self.destroy(self.handle)
            if error:raise RuntimeError(f"Native norm teardown CUDA error {error}")
            self.handle=ctypes.c_void_p()

    def __enter__(self):return self
    def __exit__(self,*_):self.close()
