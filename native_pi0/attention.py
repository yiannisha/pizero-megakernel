"""Resident FlashRT-derived native VM attention development operator."""
import ctypes
import math

import torch


class ResidentAttention:
    """Read [batch, rows, heads, dimension] BF16/FP16 Q/K/V in one VM launch.

    Row-strided projection views are supported. Optional bias and rotate-half
    RoPE are applied during shared-memory staging. A suffix prefix length makes
    query zero (state) see prefix+state, while action queries see all keys.
    An optional boolean [batch, query, key] mask further restricts visibility.
    Calls require stream ordering; concurrent calls need independent instances.
    KV splitting publishes FP32 partials and merges them inside the same launch.
    Split instances use host-generated epochs and reject CUDA graph capture.
    """
    def __init__(self,library,q,k,v,*,bias=None,q_rope=None,k_rope=None,mask=None,prefix=None,scale=None,workers=148,splits=1,output=None,
                 kv_cache=None,cached_prefix_rows=0):
        self.handle=ctypes.c_void_p();self.library=ctypes.CDLL(str(library))
        if q.ndim!=4 or k.ndim!=4 or v.shape!=k.shape:raise ValueError('Q/K/V must be four-dimensional matrices')
        if q.dtype not in (torch.bfloat16,torch.float16) or k.dtype!=q.dtype or v.dtype!=q.dtype:raise ValueError('Q/K/V must share BF16 or FP16 dtype')
        b,nq,h,d=q.shape;bk,nk,hk,dk=k.shape
        if b<1 or nq<1 or nk<1 or h<1 or hk<1 or b!=bk or d!=dk or d not in (64,72,128,256) or h%hk:raise ValueError('Unsupported attention dimensions')
        if not isinstance(cached_prefix_rows,int) or cached_prefix_rows<0:raise ValueError('Invalid cached prefix length')
        self.raw_key_rows=nk;self.cached_prefix_rows=cached_prefix_rows
        if cached_prefix_rows:
            if kv_cache is None or nk!=nq:raise ValueError('Cached-prefix mode requires cache storage and matching suffix Q/K rows')
            if prefix is None:prefix=cached_prefix_rows
            if prefix!=cached_prefix_rows:raise ValueError('Cached-prefix mode uses the pi0 suffix visibility mask')
            nk+=cached_prefix_rows
        if prefix is not None and (not isinstance(prefix,int) or prefix<0 or prefix+nq!=nk):raise ValueError('Suffix keys must contain prefix followed by all suffix tokens')
        if not isinstance(splits,int) or not 1<=splits<=64:raise ValueError('KV splits must be an integer in 1..64')
        self.splits=splits
        self.epoch=0
        self.device=q.device if q.is_cuda else torch.device('cuda',torch.cuda.current_device())
        self.scale=float(d**-.5 if scale is None else scale)
        if not math.isfinite(self.scale) or self.scale<=0:raise ValueError('Scale must be finite and positive')
        self.workers=workers;self.prefix=prefix
        with torch.cuda.device(self.device):
            self.q,self.k,self.v=[x.to(self.device) for x in (q,k,v)]
            for x in (self.q,self.k,self.v):
                if x.stride(-1)!=1 or x.stride(-2)!=d or x.stride(1)%8 or x.stride(1)<x.shape[2]*d or x.stride(0)!=x.shape[1]*x.stride(1):raise ValueError('Unsupported projection strides')
                if x.data_ptr()%16 and d!=72:raise ValueError('Projection pointer must be vector aligned')
                if x.data_ptr()%8:raise ValueError('Projection pointer must be vector aligned')
            if self.k.stride()!=self.v.stride():raise ValueError('K/V strides must match')
            def convert(x,shape,dtype=q.dtype):
                if x is None:return None
                y=torch.as_tensor(x,dtype=dtype,device=self.device).contiguous()
                if tuple(y.shape)!=tuple(shape):raise ValueError('Optional attention parameter shape mismatch')
                return y
            if bias is None:bias=(None,None,None)
            if len(bias)!=3:raise ValueError('Bias must contain query, key and value vectors')
            self.bias=[convert(x,(heads,d)) for x,heads in zip(bias,(h,hk,hk))]
            self.q_rope=[] if q_rope is None else [convert(x,(b,nq,d)) for x in q_rope]
            self.k_rope=[] if k_rope is None else [convert(x,(b,self.raw_key_rows,d)) for x in k_rope]
            for tables in (self.q_rope,self.k_rope):
                if tables and (len(tables)!=2 or any(x is None for x in tables)):raise ValueError('RoPE needs cosine and sine tensors')
            self.kv_cache=None
            if kv_cache is not None:
                if len(kv_cache)!=2:raise ValueError('KV cache needs key and value tensors')
                ck,cv=kv_cache
                if ck.shape!=cv.shape or ck.ndim!=4 or ck.shape[0]!=b or ck.shape[1]<nk or tuple(ck.shape[2:])!=(hk,d):raise ValueError('KV cache shape mismatch')
                if any(t.device!=self.device or t.dtype!=q.dtype or not t.is_contiguous() for t in (ck,cv)):raise ValueError('KV cache must be contiguous on the attention device and dtype')
                if ck.data_ptr()==cv.data_ptr():raise ValueError('K/V cache tensors must be distinct')
                def span(t):
                    end=1+sum((size-1)*stride for size,stride in zip(t.shape,t.stride()))
                    return t.data_ptr(),t.data_ptr()+end*t.element_size()
                def overlaps(a,b):
                    alo,ahi=span(a);blo,bhi=span(b)
                    return alo<bhi and blo<ahi
                if overlaps(ck,cv) or any(overlaps(cache,source) for cache in (ck,cv) for source in (self.q,self.k,self.v)):
                    raise ValueError('KV caches must not overlap each other or raw Q/K/V inputs')
                if output is not None and any(overlaps(cache,output) for cache in (ck,cv)):
                    raise ValueError('KV caches must not overlap attention output')
                self.kv_cache=(ck,cv)
            self.mask=convert(mask,(b,nq,nk),torch.bool)
            if output is not None:
                if output.device!=self.device or output.dtype!=q.dtype or output.shape!=self.q.shape or not output.is_contiguous():
                    raise ValueError('Caller-owned output must match contiguous device Q shape and dtype')
                self.output=output
            else:self.output=torch.empty_like(self.q,memory_format=torch.contiguous_format)
            create=self.library.create_attention
            create.argtypes=[ctypes.POINTER(ctypes.c_void_p)]+[ctypes.c_void_p]*12+[ctypes.c_int]*12+[ctypes.c_float]
            create.restype=ctypes.c_int
            pointers=[self.q,self.k,self.v]+self.bias+(self.q_rope or [None]*2)+(self.k_rope or [None]*2)+[self.output,self.mask]
            error=create(ctypes.byref(self.handle),*[None if x is None else x.data_ptr() for x in pointers],
                b,nq,nk,h,hk,d,self.q.stride(1),self.k.stride(1),self.output.stride(1),int(q.dtype==torch.float16),workers,-1 if prefix is None else prefix,self.scale)
        if error:raise RuntimeError(f'Native attention setup CUDA error {error}')
        self.launch=self.library.launch_attention;self.launch.argtypes=[ctypes.c_void_p]*2;self.launch.restype=ctypes.c_int
        self.destroy=self.library.destroy_attention;self.destroy.argtypes=[ctypes.c_void_p];self.destroy.restype=ctypes.c_int
        if self.kv_cache is not None:
            configure=self.library.configure_attention_cache
            configure.argtypes=[ctypes.c_void_p]*3+[ctypes.c_int]*2;configure.restype=ctypes.c_int
            with torch.cuda.device(self.device):error=configure(self.handle,*[x.data_ptr() for x in self.kv_cache],self.kv_cache[0].shape[1],cached_prefix_rows)
            if error:
                self.close();raise RuntimeError(f'Native attention cache setup CUDA error {error}')
        if splits>1:
            configure=self.library.configure_attention_splits
            configure.argtypes=[ctypes.c_void_p,ctypes.c_int];configure.restype=ctypes.c_int
            with torch.cuda.device(self.device):error=configure(self.handle,splits)
            if error:
                self.close();raise RuntimeError(f'Native attention split setup CUDA error {error}')

    def forward(self):
        if not self.handle.value:raise RuntimeError('Attention instance is closed')
        with torch.cuda.device(self.device):error=self.launch(self.handle,torch.cuda.current_stream(self.device).cuda_stream)
        if error:raise RuntimeError(f'Native attention launch CUDA error {error}')
        if self.splits>1:self.epoch+=1
        return self.output

    def close(self):
        if self.handle.value:
            with torch.cuda.device(self.device):error=self.destroy(self.handle)
            if error:raise RuntimeError(f'Native attention teardown CUDA error {error}')
            self.handle=ctypes.c_void_p()

    def read_split_counts(self):
        """Development-only synchronous diagnostics outside the forward path."""
        if not self.handle.value or self.splits==1:raise RuntimeError('A live split attention instance is required')
        count=self.q.shape[0]*self.q.shape[2]*((self.q.shape[1]+63)//64)
        values=(ctypes.c_ulonglong*count)();epoch=ctypes.c_uint()
        read=self.library.attention_split_diagnostics
        read.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_int,ctypes.c_void_p];read.restype=ctypes.c_int
        with torch.cuda.device(self.device):error=read(self.handle,values,count,ctypes.byref(epoch))
        if error:raise RuntimeError(f'Attention diagnostic CUDA error {error}')
        return {'epoch':epoch.value,'counts':list(values)}

    def poison_partials(self):
        """Development-only stale-scratch check, outside launch tracing/timing."""
        if not self.handle.value or self.splits==1:raise RuntimeError('A live split attention instance is required')
        poison=self.library.attention_poison_partials
        poison.argtypes=[ctypes.c_void_p]*2;poison.restype=ctypes.c_int
        with torch.cuda.device(self.device):error=poison(self.handle,torch.cuda.current_stream(self.device).cuda_stream)
        if error:raise RuntimeError(f'Attention scratch poisoning CUDA error {error}')

    def __enter__(self):return self
    def __exit__(self,*_):self.close()
