"""Resident native BF16/FP8 projections with FP32 or fused BF16 output."""
import ctypes
import math

import torch


class ResidentDense:
    """Offline-padded [M,K] × [N,K] projection with one native VM launch.

    Alpha is the product of static activation/weight descales for FP8. Bias
    and residual are BF16, added to FP32 accumulators before final rounding.
    Buffers remain resident and calls on an instance require stream ordering.
    """
    def __init__(self,library,inputs,weights,*,alpha=1.,bias=None,residual=None,output_dtype=torch.bfloat16,workers=148,
                 input_buffer=None,weight_buffer=None,output_buffer=None,residual_buffer=None,round_before_residual=False,geglu_output=None):
        self.handle=ctypes.c_void_p();self.library=ctypes.CDLL(str(library))
        if inputs.ndim!=2 or weights.ndim!=2 or inputs.shape[1]!=weights.shape[1]:raise ValueError('Inputs and weights must be [M,K] and [N,K]')
        if inputs.dtype not in (torch.bfloat16,torch.float8_e4m3fn) or weights.dtype!=inputs.dtype:raise ValueError('Inputs and weights must share BF16 or FP8 E4M3 dtype')
        m,k=inputs.shape;n=weights.shape[0]
        if min(m,n,k)<1 or not math.isfinite(alpha) or alpha<=0:raise ValueError('Dimensions and descale must be positive')
        if output_dtype not in (torch.bfloat16,torch.float32):raise ValueError('Output must be BF16 or FP32')
        self.logical_rows=m;self.logical_columns=n;self.logical_k=k;self.workers=workers;self.alpha=ctypes.c_float(alpha).value
        if not math.isfinite(self.alpha) or self.alpha<=0:raise ValueError('Descale must be representable as positive finite FP32')
        self.device=inputs.device if inputs.is_cuda else torch.device('cuda',torch.cuda.current_device())
        mp=(m+127)//128*128;npad=(n+127)//128*128;kp=(k+63)//64*64
        with torch.cuda.device(self.device):
            def buffer(value,shape,dtype):
                if value is None:return torch.zeros(shape,dtype=dtype,device=self.device)
                if value.device!=self.device or value.dtype!=dtype or tuple(value.shape)!=shape or not value.is_contiguous() or value.data_ptr()%128:
                    raise ValueError('Bound dense buffers must have padded dimensions, matching dtype/device and 128-byte alignment')
                return value
            self.a=buffer(input_buffer,(mp,kp),inputs.dtype)
            self.b=buffer(weight_buffer,(npad,kp),inputs.dtype)
            if input_buffer is None:self.a[:m,:k].copy_(inputs)
            if weight_buffer is None:self.b[:n,:k].copy_(weights)
            self.inputs=self.a[:m,:k];self.weights=self.b[:n,:k]
            self.bias=None;self.residual=None
            if bias is not None:
                value=torch.as_tensor(bias,dtype=torch.bfloat16,device=self.device)
                if tuple(value.shape)!=(n,):raise ValueError('Bias shape must be [N]')
                self.bias=torch.zeros(npad,dtype=torch.bfloat16,device=self.device);self.bias[:n].copy_(value)
            if residual_buffer is not None and residual is None:raise ValueError('Residual buffer requires residual')
            if residual is not None and residual_buffer is None:
                value=torch.as_tensor(residual,dtype=torch.bfloat16,device=self.device)
                if tuple(value.shape)!=(m,n):raise ValueError('Residual shape must be [M,N]')
                self.residual=torch.zeros((mp,npad),dtype=torch.bfloat16,device=self.device);self.residual[:m,:n].copy_(value)
            if residual_buffer is not None:self.residual=buffer(residual_buffer,(mp,npad),torch.bfloat16)
            self.output=(torch.empty((mp,npad),dtype=output_dtype,device=self.device) if output_buffer is None else
                         buffer(output_buffer,(mp,npad),output_dtype))
            self.logical_output=self.output[:m,:n]
            create=self.library.create_dense
            create.argtypes=[ctypes.POINTER(ctypes.c_void_p)]+[ctypes.c_void_p]*5+[ctypes.c_int]*6+[ctypes.c_float]
            create.restype=ctypes.c_int
            error=create(ctypes.byref(self.handle),*[None if x is None else x.data_ptr() for x in (self.a,self.b,self.output,self.bias,self.residual)],
                mp,npad,kp,workers,int(inputs.dtype==torch.float8_e4m3fn),int(output_dtype==torch.bfloat16),self.alpha)
        if error:raise RuntimeError(f'Native dense setup CUDA error {error}')
        if round_before_residual:
            if self.residual is None or inputs.dtype!=torch.bfloat16:raise ValueError('BF16 residual required')
            configure=self.library.configure_dense_round_before_residual
            configure.argtypes=[ctypes.c_void_p];configure.restype=ctypes.c_int
            error=configure(self.handle)
            if error:raise RuntimeError(f'Native dense residual rounding error {error}')
        if geglu_output is not None:
            if inputs.dtype!=torch.bfloat16 or output_dtype!=torch.bfloat16 or bias is not None or residual is not None or n%256:
                raise ValueError('GeGLU epilogue requires unbiased, unresidualed BF16 gate/up pairs')
            if geglu_output.device!=self.device or geglu_output.dtype!=torch.bfloat16 or tuple(geglu_output.shape)!=(mp,npad//2) or not geglu_output.is_contiguous() or geglu_output.data_ptr()%128:
                raise ValueError('GeGLU output must be a contiguous aligned compact BF16 buffer')
            configure=self.library.configure_dense_geglu
            configure.argtypes=[ctypes.c_void_p,ctypes.c_void_p];configure.restype=ctypes.c_int
            error=configure(self.handle,geglu_output.data_ptr())
            if error:raise RuntimeError(f'Native dense GeGLU epilogue error {error}')
            self.logical_output=geglu_output[:m,:n//2]
        self.launch=self.library.launch_dense;self.launch.argtypes=[ctypes.c_void_p]*2;self.launch.restype=ctypes.c_int
        self.destroy=self.library.destroy_dense;self.destroy.argtypes=[ctypes.c_void_p];self.destroy.restype=ctypes.c_int

    def forward(self):
        if not self.handle.value:raise RuntimeError('Dense instance is closed')
        with torch.cuda.device(self.device):error=self.launch(self.handle,torch.cuda.current_stream(self.device).cuda_stream)
        if error:raise RuntimeError(f'Native dense launch CUDA error {error}')
        return self.logical_output

    def close(self):
        if self.handle.value:
            with torch.cuda.device(self.device):error=self.destroy(self.handle)
            if error:raise RuntimeError(f'Native dense teardown CUDA error {error}')
            self.handle=ctypes.c_void_p()

    def __enter__(self):return self
    def __exit__(self,*_):self.close()
