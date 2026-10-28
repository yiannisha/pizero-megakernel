"""Resident scalar DAG probe for the native VM; not a model execution API."""
import ctypes
from collections import deque

import numpy as np
import torch


def lcg_advance(value,steps):
    mask=0xffffffff;mul,add=1,0;a,c=1664525,1013904223
    while steps:
        if steps&1:mul,add=(a*mul)&mask,(a*add+c)&mask
        a,c=(a*a)&mask,(a*c+c)&mask;steps>>=1
    return (mul*value+add)&mask


class ResidentDagProbe:
    """One native launch per epoch, with device initialization and ready queue.

    Each instance requires ordered calls. Independent instances can run on
    different streams. Graph capture is unsupported by host-generated epochs.
    `read` synchronizes and copies validation diagnostics, outside forward.
    """
    def __init__(self,library,parents,*,workers=16,delays=None):
        self.parents=tuple(tuple(int(p) for p in row) for row in parents)
        n=len(self.parents)
        if n<1 or n>=2**31-1:raise ValueError('Expected a nonempty graph with int32 node indices')
        children=[[] for _ in range(n)];indegree=[]
        for node,row in enumerate(self.parents):
            if len(set(row))!=len(row) or any(p<0 or p>=n or p==node for p in row):
                raise ValueError('Invalid or duplicate dependency')
            indegree.append(len(row))
            for p in row:children[p].append(node)
        self.roots=[i for i,d in enumerate(indegree) if not d]
        ready=deque(self.roots);remaining=indegree.copy();self.order=[]
        while ready:
            node=ready.popleft();self.order.append(node)
            for child in children[node]:
                remaining[child]-=1
                if not remaining[child]:ready.append(child)
        if len(self.order)!=n:raise ValueError('Graph contains a cycle')
        if not isinstance(workers,int) or not 1<=workers<=148:raise ValueError('B200 worker count must be in 1..148')
        delay=np.zeros(n,dtype=np.int64) if delays is None else np.asarray(delays,dtype=np.int64)
        if delay.shape!=(n,) or np.any(delay<0) or np.any(delay>1_000_000):raise ValueError('Invalid scalar work delays')
        self.delays=delay.astype(np.int32);self.workers=workers;self.nodes=n
        def csr(rows):
            offsets=np.array([0]+list(np.cumsum([len(row) for row in rows],dtype=np.int64)),dtype=np.int64)
            if offsets[-1]>=2**31:raise ValueError('Too many edges')
            return offsets.astype(np.int32),np.array([x for row in rows for x in row],dtype=np.int32)
        po,pp=csr(self.parents);co,cc=csr(children);roots=np.array(self.roots,dtype=np.int32)
        self.device=torch.cuda.current_device();self.handle=ctypes.c_void_p();self.epoch=0
        self.library=ctypes.CDLL(str(library))
        create=self.library.create_dag_probe
        create.argtypes=[ctypes.POINTER(ctypes.c_void_p)]+[ctypes.c_int]*4+[ctypes.c_void_p]*6;create.restype=ctypes.c_int
        self.launch=self.library.launch_dag_probe
        self.launch.argtypes=[ctypes.c_void_p]*2+[ctypes.c_uint];self.launch.restype=ctypes.c_int
        self.copy=self.library.read_dag_probe
        self.copy.argtypes=[ctypes.c_void_p]*9;self.copy.restype=ctypes.c_int
        self.destroy=self.library.destroy_dag_probe
        self.destroy.argtypes=[ctypes.c_void_p];self.destroy.restype=ctypes.c_int
        with torch.cuda.device(self.device):
            error=create(ctypes.byref(self.handle),n,len(pp),len(roots),workers,*[a.ctypes.data for a in (po,pp,co,cc,roots,self.delays)])
        if error:raise RuntimeError(f'DAG probe setup CUDA error {error}')
        plan=self.library.dag_probe_plan;plan.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_int];plan.restype=ctypes.c_int
        values=np.empty(3,dtype=np.int32);error=plan(self.handle,values.ctypes.data,len(values))
        if error:
            self.close();raise RuntimeError(f'DAG probe plan CUDA error {error}')
        self.plan={'dependency_groups':int(values[0]),'dependency_updates':int(values[1]),'nodes':int(values[2]),
                   'logical_edges':int(sum(map(len,self.parents)))}

    def forward(self,seed=0,stream=None):
        with torch.cuda.device(self.device):
            stream=torch.cuda.current_stream() if stream is None else stream
            error=self.launch(self.handle,stream.cuda_stream,int(seed)&0xffffffff)
        if error:raise RuntimeError(f'DAG probe launch CUDA error {error}')
        self.epoch+=1
        return self

    def expected(self,seed):
        values=np.zeros(self.nodes,dtype=np.uint32)
        for node in self.order:
            v=(int(seed)^(node*0x9e3779b9))&0xffffffff
            for p in self.parents[node]:v=(v*1664525+int(values[p])+1013904223)&0xffffffff
            values[node]=lcg_advance(v,int(self.delays[node]))
        return values

    def read(self):
        result={'values':np.empty(self.nodes,dtype=np.uint32),'owners':np.empty(self.nodes,dtype=np.uint32),
                'ordinals':np.empty(self.nodes,dtype=np.uint32),'visits':np.empty(self.nodes,dtype=np.uint64),
                'exits':np.empty(self.workers,dtype=np.uint32),'starts':np.empty(self.nodes,dtype=np.uint64),
                'ends':np.empty(self.nodes,dtype=np.uint64),'counts':np.empty(5,dtype=np.uint32)}
        with torch.cuda.device(self.device):error=self.copy(self.handle,*[a.ctypes.data for a in result.values()])
        if error:raise RuntimeError(f'DAG probe read CUDA error {error}')
        return result

    def close(self):
        if self.handle.value:
            with torch.cuda.device(self.device):error=self.destroy(self.handle)
            if error:raise RuntimeError(f'DAG probe destroy CUDA error {error}')
            self.handle=ctypes.c_void_p()
    def __enter__(self):return self
    def __exit__(self,*_):self.close()
