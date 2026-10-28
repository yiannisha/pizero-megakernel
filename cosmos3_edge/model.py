"""Native generation transformer boundary with prepared hidden states and KV.

All layer operators execute in one persistent VM launch. This is the BF16
correctness reference, not yet the complete latent/action denoising pipeline.
"""
from contextlib import ExitStack
import ctypes

import numpy as np
import torch

from native_pi0.normalization import ResidentNorm
from .attention import NativeJointAttention
from .engine import NativeEngine
from .gemm import NativeProjection
from .rope import NativeQKNormRope, mrope_tables
from .spec import EdgeSpec


class NativeGeneratorTransformer:
    def __init__(self, library, weights, hidden, understanding_kv, position_ids, *,
                 layer_indices=None, workers=148, attention_splits=1, permute_nodes=False):
        spec = EdgeSpec.from_checkpoint()
        self.layer_indices = tuple(range(spec.layers)) if layer_indices is None else tuple(layer_indices)
        if not self.layer_indices or len(set(self.layer_indices))!=len(self.layer_indices) or any(not isinstance(i,int) or not 0<=i<spec.layers for i in self.layer_indices):
            raise ValueError('Invalid generator layer selection')
        if len(understanding_kv)!=len(self.layer_indices):
            raise ValueError('One prepared understanding K/V pair per selected layer is required')
        if hidden.ndim!=2 or hidden.shape[0]<1 or hidden.shape[1]!=spec.hidden:
            raise ValueError('Generation hidden states must have shape [rows,2048]')
        self.rows=rows=hidden.shape[0]; padded=(rows+127)//128*128
        self.device=hidden.device if hidden.is_cuda else torch.device('cuda',torch.cuda.current_device())
        self._stack=ExitStack();self.layers=[]
        try:
            self.physical_input=torch.zeros((padded,spec.hidden),dtype=torch.bfloat16,device=self.device)
            self.physical_input[:rows].copy_(hidden);self.inputs=self.physical_input[:rows]
            physical=self.physical_input
            cos,sin=mrope_tables(position_ids,device=self.device)
            if cos.shape[0]!=rows:raise ValueError('Position count must match generated tokens')
            module=ctypes.CDLL(str(library));query=module.edge_engine_norm_rows;query.argtypes=[];query.restype=ctypes.c_int
            norm_rows=query();query=module.edge_rope_vectors_per_task;query.argtypes=[];query.restype=ctypes.c_int;rope_batch=query()
            if 128%norm_rows:raise ValueError('Normalization task rows must divide matrix tile rows')
            stages=[];parents=[];node_stage=[];node_task=[]
            def nodes(kind,obj,dependencies):
                stage=len(stages);stages.append((kind,obj));result=[]
                for task,deps in enumerate(dependencies):
                    result.append(len(parents));parents.append(sorted(set(deps)));node_stage.append(stage);node_task.append(task)
                return result
            def norm_deps(previous):
                return [[] if previous is None else previous[(task*norm_rows//128)*16:(task*norm_rows//128+1)*16]
                        for task in range((rows+norm_rows-1)//norm_rows)]
            def projection_deps(norm_nodes,columns):
                result=[]
                for task in range((padded//128)*(columns//128)):
                    row=task//(columns//128)
                    begin=row*128//norm_rows;end=(min((row+1)*128,rows)+norm_rows-1)//norm_rows
                    result.append(norm_nodes[begin:end])
                return result
            own=self._stack.enter_context
            previous=None
            for layer,(und_k,und_v) in zip(self.layer_indices,understanding_kv):
                prefix=f'layers.{layer}.'
                w=lambda name:weights.load(prefix+name)
                if und_k.ndim!=3 or und_k.shape[1:]!=(8,128) or und_v.shape!=und_k.shape:
                    raise ValueError('Understanding KV must be [prefix_rows,8,128]')
                count=und_k.shape[0]
                norm1=own(ResidentNorm(library,physical[:rows],precision='bf16',gamma=w('input_layernorm_moe_gen.weight'),
                    epsilon=spec.rms_eps,row_alignment=128,workers=workers))
                n1=nodes(8,norm1,norm_deps(previous))
                projections=[];project_nodes=[]
                for name,width in (('add_q_proj',2048),('add_k_proj',1024),('add_v_proj',1024)):
                    op=own(NativeProjection(library,norm1.output,w(f'self_attn.{name}.weight'),workers=workers,device=self.device,share_activation=True))
                    projections.append(op);project_nodes.append(nodes(3,op,projection_deps(n1,width)))
                cache_k=torch.zeros((count+rows,8,128),dtype=torch.bfloat16,device=self.device);cache_v=torch.zeros_like(cache_k)
                cache_k[:count].copy_(und_k);cache_v[:count].copy_(und_v)
                raw=[op.output[:rows].view(rows,heads,128) for op,heads in zip(projections,(16,8,8))]
                prep=own(NativeQKNormRope(library,raw[0],raw[1],w('self_attn.norm_added_q.weight'),w('self_attn.norm_added_k.weight'),cos,sin,
                    v=raw[2],k_output=cache_k[count:],v_output=cache_v[count:],rounding='reference',workers=workers,device=self.device))
                qvec=rows*16;total=rows*24;prep_deps=[]
                for task in range((total+rope_batch-1)//rope_batch):
                    deps=set()
                    for vec in range(task*rope_batch,min((task+1)*rope_batch,total)):
                        if vec<qvec:
                            row,head=divmod(vec,16);deps.add(project_nodes[0][(row//128)*16+head])
                        else:
                            row,head=divmod(vec-qvec,8)
                            for stage in (1,2):deps.add(project_nodes[stage][(row//128)*8+head])
                    prep_deps.append(deps)
                pn=nodes(5,prep,prep_deps);ready=nodes(7,None,[pn[qvec//rope_batch:]])[0]
                attention_buffer=torch.zeros((padded,2048),dtype=torch.bfloat16,device=self.device)
                attention=own(NativeJointAttention(library,prep.q_output,cache_k,cache_v,workers=workers,splits=attention_splits,
                    output=attention_buffer[:rows].view(rows,16,128)))
                tiles=(rows+63)//64;attention_deps=[]
                for logical in range(16*tiles):
                    head,tile=divmod(logical,tiles);deps={ready}
                    for row in range(tile*64,min((tile+1)*64,rows)):deps.add(pn[(row*16+head)//rope_batch])
                    for part in range(attention_splits):attention_deps.append(deps)
                an=nodes(6,attention,attention_deps)
                output=own(NativeProjection(library,attention_buffer,w('self_attn.to_add_out.weight'),residual=physical,workers=workers,
                    device=self.device,share_activation=True,share_residual=True,round_before_residual=True))
                o_deps=[]
                for task in range((padded//128)*16):
                    block=task//16;deps=[]
                    for head in range(16):
                        for tile in range(block*2,min(block*2+2,tiles)):
                            deps.extend(an[(head*tiles+tile)*attention_splits:(head*tiles+tile+1)*attention_splits])
                    o_deps.append(deps)
                on=nodes(3,output,o_deps)
                norm2=own(ResidentNorm(library,output.output[:rows],precision='bf16',gamma=w('post_attention_layernorm_moe_gen.weight'),
                    epsilon=spec.rms_eps,row_alignment=128,workers=workers))
                n2=nodes(8,norm2,norm_deps(on))
                up=own(NativeProjection(library,norm2.output,w('mlp_moe_gen.up_proj.weight'),relu2=True,relu2_round_input=True,
                    workers=workers,device=self.device,share_activation=True))
                un=nodes(3,up,projection_deps(n2,9216))
                down=own(NativeProjection(library,up.output,w('mlp_moe_gen.down_proj.weight'),residual=output.output,
                    workers=workers,device=self.device,share_activation=True,share_residual=True,round_before_residual=True))
                dn=nodes(3,down,[un[(task//16)*72:(task//16+1)*72] for task in range((padded//128)*16)])
                self.layers.append(dict(index=layer,norm1=norm1,projections=projections,preparation=prep,attention=attention,
                    cache_k=cache_k,cache_v=cache_v,prefix_rows=count,attention_buffer=attention_buffer,
                    output_projection=output,norm2=norm2,up=up,down=down))
                physical=down.output;previous=dn
            self.final_norm=own(ResidentNorm(library,physical[:rows],precision='bf16',gamma=weights.load('norm_moe_gen.weight'),
                epsilon=spec.rms_eps,row_alignment=128,workers=workers))
            nodes(8,self.final_norm,norm_deps(previous))
            permutation=np.random.default_rng(251).permutation(len(parents)) if permute_nodes else None
            self.engine=own(NativeEngine(library,stages,parents,node_stage,node_task,workers=workers,permutation=permutation))
            self.output=self.final_norm.output[:rows]
        except BaseException:
            self._stack.close();raise

    def forward(self):
        self.engine.forward();return self.output
    def close(self):self._stack.close()
    def __enter__(self):return self
    def __exit__(self,*_):self.close()
