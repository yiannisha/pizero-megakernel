"""Complete BF16 pi0 with static native tile-range instruction streams."""
import ctypes as C
from contextlib import ExitStack
import math
from pathlib import Path
from types import SimpleNamespace
import subprocess
import sys

import numpy as np
import torch

from native_pi0.attention import ResidentAttention
from native_pi0.dense import ResidentDense
from native_pi0.normalization import ResidentNorm
from native_pi0.program import LinearWork,linear_streams,program_summary

ROOT=Path(__file__).resolve().parents[1]


class Scalar(C.Structure):
    _fields_=[(n,C.c_void_p) for n in ('a','b','c','d','output')]+[(n,C.c_int) for n in
        ('mode','m','n','k','lda','ldb','ldo','p0','p1','input_slot')]+[('alpha',C.c_float)]


class Stage(C.Structure):
    _fields_=[('kind',C.c_int),('handle',C.c_void_p),('scalar',Scalar),('sources',C.c_int*3)]


def pointer(x):return None if x is None else x.data_ptr()
def padded(n,a):return (n+a-1)//a*a


class NativePi0:
    """Experimental reference π0; full-model numerical acceptance is still open.

    Batch one, resident NHWC images and fixed input shapes.

    Setup owns layouts, instruction streams, scratch and all weight transforms. Forward
    accepts CUDA FP32 images [views,224,224,3], state [1,32], noise
    [1,chunk,32], and int64 tokens [1,prompt_len]. It invokes one original
    Megakernels VM kernel and returns reused FP32 actions. This initial path
    uses BF16 transformer GEMMs and FP32 patch embedding/action heads.
    """
    @classmethod
    def from_openpi(cls,checkpoint_path,*,num_views=3,prompt_len=0,chunk_size=50,flow_steps=10,precision='auto',library=None):
        sys.path.insert(0,str(ROOT/'third_party/openpi/src'))
        from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
        from safetensors.torch import load_model
        config=SimpleNamespace(pi05=False,paligemma_variant='gemma_2b',action_expert_variant='gemma_300m',dtype='bfloat16',
            action_dim=32,action_horizon=chunk_size,pytorch_compile_mode=None)
        with torch.device('cuda'):model=PI0Pytorch(config).eval()
        load_model(model,str(checkpoint_path),strict=True)
        return cls.from_model(model,num_views=num_views,prompt_len=prompt_len,chunk_size=chunk_size,flow_steps=flow_steps,precision=precision,library=library)

    @classmethod
    def from_model(cls,model,**kwargs):return cls(model,**kwargs)

    def __init__(self,model,*,num_views=3,prompt_len=0,chunk_size=50,flow_steps=10,precision='auto',library=None,workers=148):
        if model.config.pi05 or precision not in ('auto','bf16'):raise ValueError('Reference path supports ordinary π0 with BF16 transformers')
        if num_views not in (1,2,3) or prompt_len<0 or chunk_size<1 or flow_steps<1:raise ValueError('Invalid fixed inference dimensions')
        self.model=model;self.num_views=num_views;self.prompt_len=prompt_len;self.chunk_size=chunk_size;self.flow_steps=flow_steps
        self.precision_policy='bf16_reference';self.workers=workers;self.device=next(model.parameters()).device
        if self.device.type!='cuda':raise ValueError('Supply a CUDA OpenPI model')
        self.owners=ExitStack();self.buffers=[];self.weight_buffers={};self.stages=[];self.stage_tasks=[];self.landmarks={};self.handle=C.c_void_p();self.producers={};self.cache_producers={};self.opcodes=[];self.stage_cost=[]
        if library is None:
            library=Path('/tmp/pi0_native_model.so')
            subprocess.run(['nvcc','-O3','-lineinfo','-std=c++20','-arch=sm_100a','-DKITTENS_HOPPER','-DKITTENS_BLACKWELL',
                '-DNATIVE_COMPUTE_CONSUMER_WARPS=4','--expt-relaxed-constexpr','--expt-extended-lambda',
                '-I'+str(ROOT/'third_party/Megakernels/include'),'-I'+str(ROOT/'third_party/Megakernels/ThunderKittens/include'),
                '-shared','-Xcompiler','-fPIC',str(ROOT/'native_pi0/model.cu'),str(ROOT/'native_pi0/launch_trace.cpp'),
                '-lcupti','-Xlinker','-rpath','-Xlinker','/usr/local/cuda/lib64','-lcuda','-o',str(library)],check=True)
        self.library_path=Path(library).resolve();self.library=C.CDLL(str(self.library_path))
        try:
            with torch.cuda.device(self.device),torch.no_grad():self._build()
        except Exception:self.owners.close();raise

    def tensor(self,shape,dtype=torch.bfloat16):
        value=torch.zeros(shape,device=self.device,dtype=dtype);self.buffers.append(value);return value

    def add(self,kind,handle,tasks,scalar=None,*,output,inputs=(),cache_source=-1,cost=64):
        parents=[]
        for x in inputs:
            if x is not None:
                source=self.producers.get(x.untyped_storage().data_ptr(),-1)
                if source>=0 and source not in parents:parents.append(source)
        if cache_source>=0 and cache_source not in parents:parents.append(cache_source)
        if len(parents)>3:raise ValueError('Too many native tensor sources')
        index=len(self.stages);spec=scalar or Scalar()
        self.stages.append(Stage(kind,handle,spec,(C.c_int*3)(*(parents+[-1]*(3-len(parents))))));self.stage_tasks.append(tasks)
        self.opcodes.append(1 if kind==0 else (16 if spec.p1 else 15 if spec.p0 else 2) if kind==1 else 3 if kind==2 else spec.mode+3)
        self.stage_cost.append(padded(max(cost,64),64))
        self.producers[output.untyped_storage().data_ptr()]=index
        return index

    def scalar(self,mode,shape,*,a=None,b=None,c=None,d=None,dtype=torch.bfloat16,k=0,lda=None,ldb=None,p0=0,p1=0,slot=0,alpha=1.):
        m,n=shape;out=self.tensor(shape,dtype)
        spec=Scalar(*[pointer(t) for t in (a,b,c,d,out)],mode,m,n,k,lda or (a.shape[-1] if a is not None else n),
                    ldb or (b.shape[-1] if b is not None else n),n,p0,p1,slot,float(alpha))
        tile=128 if mode in (5,6) else 16384;self.add(3,None,(m*n+tile-1)//tile,spec,output=out,inputs=(a,b,c),cost=k if mode==5 else 588 if mode==6 else 64);return out

    def dense(self,x,weight,bias=None,*,rows=None,residual=None):
        rows=rows or x.shape[0];n,k=weight.shape;mp=padded(rows,128);kp=padded(k,64);npad=padded(n,128)
        if x.shape!=(mp,kp) or x.dtype!=torch.bfloat16:raise ValueError(('Dense layout',x.shape,(mp,kp),x.dtype))
        key=(weight.data_ptr(),n,k)
        if key not in self.weight_buffers:
            w=self.tensor((npad,kp));w[:n,:k].copy_(weight);self.weight_buffers[key]=w
        op=self.owners.enter_context(ResidentDense(self.library_path,x[:rows,:k],weight,bias=bias,workers=self.workers,
            input_buffer=x,weight_buffer=self.weight_buffers[key],residual=residual,
            residual_buffer=residual,round_before_residual=residual is not None))
        self.add(0,op.handle.value,(mp//128)*(npad//128),output=op.output,inputs=(x,residual),cost=kp);return op.output

    def norm(self,x,module,*,rms):
        gamma=module.weight.float()+(1. if rms else 0.);beta=None if rms else module.bias.float()
        op=self.owners.enter_context(ResidentNorm(self.library_path,x,kind='rms' if rms else 'layer',precision='bf16',gamma=gamma,beta=beta,
            epsilon=getattr(module,'eps',getattr(module,'variance_epsilon',1e-6)),workers=self.workers))
        self.add(2,op.handle.value,(x.shape[0]+7)//8,output=op.output,inputs=(x,));return op.output

    def linear32(self,x,module,rows,*,slot=0,bf16=False):
        weight=module.weight;key=('f32',weight.data_ptr())
        if key not in self.weight_buffers:
            self.weight_buffers[key]=weight.float().T.contiguous();self.buffers.append(self.weight_buffers[key])
        return self.scalar(5,(rows,weight.shape[0]),a=x,b=self.weight_buffers[key],c=module.bias,dtype=torch.float32,
            k=weight.shape[1],lda=weight.shape[1] if x is None else x.shape[-1],p0=int(bf16),slot=slot)

    def rope(self,rows,offset,dim):
        pos=torch.arange(offset,offset+rows,device=self.device,dtype=torch.float32)
        inverse=1./(10000.**(torch.arange(0,dim,2,device=self.device,dtype=torch.float32)/dim))
        angles=pos[:,None]*inverse[None,:];angles=torch.cat((angles,angles),-1)[None]
        out=(angles.cos().to(torch.bfloat16),angles.sin().to(torch.bfloat16));self.buffers.extend(out);return out

    def block(self,x,layer,*,rows,batches,heads,hk,hd,rms,cache=None,prefix=0,rope=None,name,cache_only=False):
        debug=name in ('prefix.0','vision.0');tag='debug.' if rms else 'visiondebug.';logical=batches*rows
        if debug:self.landmarks[tag+'input']=x[:logical]
        residual=x;y=self.norm(x,layer.input_layernorm if rms else layer.layer_norm1,rms=rms)
        if debug:self.landmarks[tag+'norm1']=y[:logical]
        att=layer.self_attn
        projections=([att.k_proj,att.v_proj] if cache_only else [att.q_proj,att.k_proj,att.v_proj])
        qkv_weight=torch.cat([projection.weight for projection in projections]);self.buffers.append(qkv_weight)
        qkv_bias=None if projections[0].bias is None else torch.cat([projection.bias for projection in projections])
        if qkv_bias is not None:self.buffers.append(qkv_bias)
        qkv=self.dense(y,qkv_weight,qkv_bias,rows=batches*rows)
        key_offset=0 if cache_only else heads*hd
        k=qkv[:batches*rows,key_offset:key_offset+hk*hd].view(batches,rows,hk,hd)
        v=qkv[:batches*rows,key_offset+hk*hd:key_offset+2*hk*hd].view(batches,rows,hk,hd)
        q=k if cache_only else qkv[:batches*rows,:heads*hd].view(batches,rows,heads,hd)
        if debug:
            self.landmarks[tag+'q']=q.reshape(logical,-1);self.landmarks[tag+'k']=k.reshape(logical,-1);self.landmarks[tag+'v']=v.reshape(logical,-1)
        if rope is not None:
            qkv=self.scalar(11,qkv.shape,a=qkv,b=rope[0],c=rope[1],k=batches*rows,p0=key_offset+hk*hd,p1=hd)
            k=qkv[:batches*rows,key_offset:key_offset+hk*hd].view(batches,rows,hk,hd)
            v=qkv[:batches*rows,key_offset+hk*hd:key_offset+2*hk*hd].view(batches,rows,hk,hd)
            q=k if cache_only else qkv[:batches*rows,:heads*hd].view(batches,rows,heads,hd)
        output=self.tensor((x.shape[0],(hk if cache_only else heads)*hd))
        op=self.owners.enter_context(ResidentAttention(self.library_path,q,k,v,workers=self.workers,
            splits=1,kv_cache=cache,cached_prefix_rows=prefix,
            output=output[:batches*rows].view(batches,rows,hk if cache_only else heads,hd)))
        cache_key=None if cache is None else cache[0].untyped_storage().data_ptr()
        index=self.add(1,op.handle.value,((batches*rows+63)//64 if cache_only else batches*heads*((rows+63)//64)),
            Scalar(alpha=op.scale,p0=int(rms),p1=int(cache_only)),
            output=output,inputs=(qkv,),cache_source=self.cache_producers.get(cache_key,-1) if prefix else -1,cost=(rows+prefix)*hd)
        if cache_key is not None and not prefix:self.cache_producers[cache_key]=index
        if cache_only:return x
        if debug:self.landmarks[tag+'attention']=output[:logical]
        y=self.dense(output,att.o_proj.weight if rms else att.out_proj.weight,att.o_proj.bias if rms else att.out_proj.bias,rows=batches*rows,residual=residual)
        x=y
        if debug:self.landmarks[tag+'residual1']=x[:logical]
        y=self.norm(x,layer.post_attention_layernorm if rms else layer.layer_norm2,rms=rms)
        if debug:self.landmarks[tag+'norm2']=y[:logical]
        if rms:
            weights=torch.cat([layer.mlp.gate_proj.weight,layer.mlp.up_proj.weight]);self.buffers.append(weights)
            y=self.dense(y,weights,rows=batches*rows)
            y=self.scalar(3,(x.shape[0],weights.shape[0]//2),a=y)
            if debug:self.landmarks[tag+'activation']=y[:logical]
            y=self.dense(y,layer.mlp.down_proj.weight,rows=batches*rows,residual=x)
        else:
            y=self.dense(y,layer.mlp.fc1.weight,layer.mlp.fc1.bias,rows=batches*rows)
            if debug:self.landmarks[tag+'fc1']=y[:logical,:layer.mlp.fc1.out_features]
            y=self.scalar(2,y.shape,a=y,p0=layer.mlp.fc1.out_features)
            if debug:self.landmarks[tag+'activation']=y[:logical,:layer.mlp.fc1.out_features]
            y=self.dense(y,layer.mlp.fc2.weight,layer.mlp.fc2.bias,rows=batches*rows,residual=x)
        x=y;self.landmarks[name]=x[:batches*rows];return x

    def _build(self):
        pg=self.model.paligemma_with_expert;vision=pg.paligemma.vision_tower.vision_model
        embedding=vision.embeddings;patch_weight=embedding.patch_embedding.weight.float().reshape(1152,-1).T.contiguous()
        self.buffers.append(patch_weight)
        x=self.scalar(6,(self.num_views*256,1152),b=patch_weight,c=embedding.patch_embedding.bias,d=embedding.position_embedding.weight,slot=1)
        self.landmarks['patch']=x
        for i,layer in enumerate(vision.encoder.layers):
            x=self.block(x,layer,rows=256,batches=self.num_views,heads=16,hk=16,hd=72,rms=False,name=f'vision.{i}')
            if (i+1)%9==0:print('Native reference: vision layers',i+1,flush=True)
        x=self.norm(x,vision.post_layernorm,rms=False)
        projector=pg.paligemma.multi_modal_projector.linear;x=self.dense(x,projector.weight,projector.bias,rows=self.num_views*256)
        prefix_rows=self.num_views*256+self.prompt_len
        x=self.scalar(10,(padded(prefix_rows,128),2048),a=x,b=pg.paligemma.language_model.embed_tokens.weight,
            p0=self.num_views*256,p1=self.prompt_len,alpha=math.sqrt(2048.))
        prefix_rope=self.rope(prefix_rows,0,256);self.caches=[]
        for i,layer in enumerate(pg.paligemma.language_model.layers):
            cache=tuple(self.tensor((1,prefix_rows+self.chunk_size+1,1,256)) for _ in range(2));self.caches.append(cache)
            x=self.block(x,layer,rows=prefix_rows,batches=1,heads=8,hk=1,hd=256,rms=True,cache=cache,rope=prefix_rope,name=f'prefix.{i}',cache_only=i==17)
            if (i+1)%6==0:print('Native reference: prefix layers',i+1,flush=True)
        state=self.linear32(None,self.model.state_proj,1,slot=2)
        actions=None;suffix_rows=self.chunk_size+1;suffix_rope=self.rope(suffix_rows,prefix_rows,256)
        # Immutable action half of the first time MLP: W_a(W_in noise + b_in)
        # becomes (W_a W_in) noise + W_a b_in. The time half is a per-step bias.
        first=self.model.action_time_mlp_in;action_in=self.model.action_in_proj
        half=action_in.out_features
        fused_weight=first.weight[:,:half].float()@action_in.weight.float()
        action_bias=first.weight[:,:half].float()@action_in.bias.float()
        self.buffers.extend((fused_weight,action_bias))
        time=np.float32(1.);dt=np.float32(-1./self.flow_steps)
        for step in range(self.flow_steps):
            fraction=torch.linspace(0.,1.,512,device=self.device,dtype=torch.float64)
            period=.004*(4./.004)**fraction;angle=float(time)/period*(2.*math.pi)
            time_embedding=torch.cat((angle.sin(),angle.cos())).float();self.buffers.append(time_embedding)
            step_bias=action_bias+(first.weight[:,half:].float()@time_embedding)+first.bias.float()
            self.buffers.append(step_bias)
            folded=SimpleNamespace(weight=fused_weight,bias=step_bias)
            y=self.linear32(actions,folded,self.chunk_size,slot=3 if step==0 else 0)
            y=self.scalar(4,y.shape,a=y,dtype=torch.float32)
            y=self.linear32(y,self.model.action_time_mlp_out,self.chunk_size)
            x=self.scalar(8,(padded(suffix_rows,128),1024),a=state,b=y,p0=self.chunk_size)
            for i,layer in enumerate(pg.gemma_expert.model.layers):
                x=self.block(x,layer,rows=suffix_rows,batches=1,heads=8,hk=1,hd=256,rms=True,cache=self.caches[i],prefix=prefix_rows,rope=suffix_rope,name=f'expert.{step}.{i}')
            x=self.norm(x,pg.gemma_expert.model.norm,rms=True)
            velocity=self.linear32(x[1:],self.model.action_out_proj,self.chunk_size,bf16=True)
            actions=self.scalar(9,(self.chunk_size,32),a=actions,b=velocity,dtype=torch.float32,slot=3 if step==0 else 0,alpha=dt)
            self.landmarks[f'actions.{step}']=actions;time=np.float32(time+dt)
            print('Native reference: expert denoising step',step+1,'wired',flush=True)
        self.output=actions.view(1,self.chunk_size,32)
        self.epoch=0
        self.code=linear_streams([LinearWork(t,c) for t,c in zip(self.stage_tasks,self.stage_cost)],self.workers)
        for worker in self.code:
            for instruction in worker:
                if instruction[0]:instruction[0]=self.opcodes[instruction[1]]
        self.plan=program_summary(self.code)
        print('Native pi0 static program:',len(self.stages),'operations,',self.plan,flush=True)
        specs=(Stage*len(self.stages))(*self.stages)
        create=self.library.create_native_model;create.argtypes=[C.POINTER(C.c_void_p),C.c_int,C.c_void_p,C.c_void_p,C.c_int,C.c_int];create.restype=C.c_int
        error=create(C.byref(self.handle),len(specs),C.cast(specs,C.c_void_p),self.code.ctypes.data,self.workers,self.code.shape[1])
        if error:raise RuntimeError(f'Native pi0 setup CUDA error {error}')
        self.launch=self.library.launch_native_model;self.launch.argtypes=[C.c_void_p]*6;self.launch.restype=C.c_int
        self.destroy=self.library.destroy_native_model;self.destroy.argtypes=[C.c_void_p];self.destroy.restype=C.c_int

    def forward(self,images,state,noise,tokens):
        if not self.handle.value:raise RuntimeError('Native π0 reference is closed')
        for tensor,shape,dtype in ((images,(self.num_views,224,224,3),torch.float32),(state,(1,32),torch.float32),
            (noise,(1,self.chunk_size,32),torch.float32),(tokens,(1,self.prompt_len),torch.int64)):
            if tensor.device!=self.device or tensor.dtype!=dtype or tuple(tensor.shape)!=shape or not tensor.is_contiguous():
                raise ValueError(('Resident π0 input layout',shape,dtype))
        with torch.cuda.device(self.device):error=self.launch(self.handle,torch.cuda.current_stream(self.device).cuda_stream,
            images.data_ptr(),state.data_ptr(),noise.data_ptr(),tokens.data_ptr())
        if error:raise RuntimeError(f'Native π0 reference launch error {error}')
        self.epoch+=1;return self.output

    def close(self):
        if self.handle.value:
            with torch.cuda.device(self.device):error=self.destroy(self.handle)
            if error:raise RuntimeError(f'Native π0 reference teardown error {error}')
            self.handle=C.c_void_p()
        self.owners.close()

    def __enter__(self):return self
    def __exit__(self,*_):self.close()
