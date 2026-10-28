// Pre-norm attention + compact NVFP4 GeGLU FFN in one original VM launch.
#include "projection_attention.cu"
#include "nvfp4_probe.cu"
#include "norm.cu"
#ifndef NATIVE_BLOCK_NORM_ROWS
#define NATIVE_BLOCK_NORM_ROWS 8
#endif
static_assert(NATIVE_BLOCK_NORM_ROWS==1||NATIVE_BLOCK_NORM_ROWS==4||NATIVE_BLOCK_NORM_ROWS==8);
extern "C" int transformer_block_norm_rows(){return NATIVE_BLOCK_NORM_ROWS;}

struct TransformerBlockGlobals:ProjectionAttentionGlobals {
    const NormGlobals* norms;
    const Fp4Globals* ffn;
    __device__ int stage(State& s)const{return node_stage?node_stage[s.instruction()[1]]:0;}
    __device__ const NormGlobals& norm(State& s,int index)const{return norms[stage(s)*2+index];}
    __device__ const Fp4Globals& matrix(State& s,int index)const{return ffn[stage(s)*2+index];}
    __device__ void fetch_instruction(int* instruction,int index)const{
        if(index==0)queue.initialize_warp();
        if(kittens::laneid()==0){
            int node=queue.claim(false);instruction[0]=node<0?-1:node_kind[node]+1;
            if(node<0)exits[blockIdx.x]=index;
            else{instruction[1]=node;instruction[2]=node_task[node];}
        }
        __syncwarp();
    }
};
template<int Index> struct BlockNormOp {
    static constexpr int opcode=4+Index;
    using controller=megakernel::NoOp<Cfg>::controller;
    using loader=megakernel::NoOp<Cfg>::loader;
    using launcher=megakernel::NoOp<Cfg>::launcher;
    using storer=megakernel::NoOp<Cfg>::storer;
    struct consumer {static __device__ void run(const TransformerBlockGlobals& g,State& s){
        ProjectionAttentionDiagnostics::start(g,s);
        int first=s.instruction()[2]*NATIVE_BLOCK_NORM_ROWS;
        const auto& norm=g.norm(s,Index);
        for(int row=first;row<min(first+NATIVE_BLOCK_NORM_ROWS,norm.rows);++row)NormOp::run_row(norm,s,row);
        ProjectionAttentionDiagnostics::finish(g,s);
    }};
};
template<int Index> struct BlockFfnOp {
    static constexpr int opcode=6+Index;
    struct controller {
        static __device__ int release_lid(const TransformerBlockGlobals&,Cfg::instruction_t&,int& q){return q;}
        static __device__ int init_semaphores(const TransformerBlockGlobals& g,State& s){return Fp4Matmul::controller::init_semaphores(g.matrix(s,Index),s);}
    };
    struct loader {static __device__ void run(const TransformerBlockGlobals& g,State& s){
        const auto& f=g.matrix(s,Index);
        if(kittens::laneid()==0){
            const auto *a=f.a.template get_tma<PackedTile,2>(),*b=f.b.template get_tma<PackedTile,2>();
            asm volatile("fence.proxy.tensormap::generic.acquire.gpu [%0], 128;"::"l"(a):"memory");
            asm volatile("fence.proxy.tensormap::generic.acquire.gpu [%0], 128;"::"l"(b):"memory");
        }
        __syncwarp();Fp4Matmul::loader::run(f,s);
    }};
    struct launcher {static __device__ void run(const TransformerBlockGlobals& g,State& s){Fp4Matmul::launcher::run(g.matrix(s,Index),s);}};
    using storer=megakernel::NoOp<Cfg>::storer;
    struct consumer {static __device__ void run(const TransformerBlockGlobals& g,State& s){
        ProjectionAttentionDiagnostics::start(g,s);
        Fp4Matmul::consumer::run(g.matrix(s,Index),s);
        ProjectionAttentionDiagnostics::finish(g,s);
    }};
};
#define BLOCK_KERNEL megakernel::mk<Cfg,TransformerBlockGlobals,ScheduledProjectionOp,ScheduledAttentionOp,ScheduledOutputProjectionOp,BlockNormOp<0>,BlockNormOp<1>,BlockFfnOp<0>,BlockFfnOp<1>>
struct TransformerBlockContext {
    ProjectionAttentionContext* attention;DagProbeContext* queue;
    NormGlobals* norms;Fp4Globals* ffn;int *kind,*task;int device;
    NormContext* pre_norm;ResidentFp4* down;
};
// Setup binds existing child buffers. Owners and child handles must outlive us.
extern "C" int create_transformer_block(void** handle,void* queue_handle,void* attention_handle,void* pre_handle,void* post_handle,
    void* gate_handle,void* down_handle,const int* kinds,const int* tasks){
    if(!handle)return cudaErrorInvalidValue;*handle=nullptr;
    if(!queue_handle||!attention_handle||!pre_handle||!post_handle||!gate_handle||!down_handle||!kinds||!tasks)return cudaErrorInvalidValue;
    auto* a=static_cast<ProjectionAttentionContext*>(attention_handle);auto* q=static_cast<DagProbeContext*>(queue_handle);
    auto* pre=static_cast<NormContext*>(pre_handle);auto* post=static_cast<NormContext*>(post_handle);
    auto* gate=static_cast<ResidentFp4*>(gate_handle);auto* down=static_cast<ResidentFp4*>(down_handle);
    int device;auto error=cudaGetDevice(&device);if(error!=cudaSuccess)return error;
    if(device!=a->device||device!=q->device||device!=pre->device||device!=post->device||device!=gate->device||device!=down->device)return cudaErrorInvalidDevice;
    if(!a->output_projection||a->output_projection->fp8)return cudaErrorInvalidValue;
    auto& o=static_cast<DenseContext<kittens::bf16>*>(a->output_projection)->globals;
    auto& p=pre->globals;auto& n=post->globals;auto& f=gate->globals;auto& d=down->globals;
    int rows=o.out.rows(),width=o.out.cols();
    if(p.rows!=rows||n.rows!=rows||p.width!=width||n.width!=width||width%128||p.fp16||n.fp16||p.residual||n.residual
       ||p.format!=(a->projection->fp8?8:16)||n.format!=4||p.global_scale!=1.f||!f.fp4_output||f.output_gelu!=2||!d.bf16_output
       ||f.a.rows()!=rows||f.a.cols()*2!=width||d.a.rows()!=rows||d.out.cols()!=width||d.a.cols()*2!=f.b.rows()/2
       ||f.output_global_scale<=0)return cudaErrorInvalidValue;
    int groups=(rows+NATIVE_BLOCK_NORM_ROWS-1)/NATIVE_BLOCK_NORM_ROWS;
    int counts[]={dense_tasks(a->projection),a->attention->globals.tasks,dense_tasks(a->output_projection),groups,groups,f.tasks,d.tasks};
    std::vector<unsigned char> assigned[7];for(int i=0;i<7;++i)assigned[i].resize(counts[i],0);
    for(int node=0;node<q->globals.queue.nodes;++node){int k=kinds[node],t=tasks[node];
        if(k<0||k>=7||t<0||t>=counts[k]||assigned[k][t])return cudaErrorInvalidValue;assigned[k][t]=1;
    }
    for(int k=0;k<7;++k)if(std::count(assigned[k].begin(),assigned[k].end(),1)!=counts[k])return cudaErrorInvalidValue;
    // The input and residual matrices include padded rows. Norm output padding
    // is produced by real norm tasks, so every TMA read has a producer.
    if(a->projection->fp8){
        auto& v=static_cast<DenseContext<kittens::fp8e4m3>*>(a->projection)->globals;
        if(v.a.rows()!=rows||v.a.cols()!=width)return cudaErrorInvalidValue;
        v.a=decltype(v.a)(reinterpret_cast<kittens::fp8e4m3*>(p.output),nullptr,nullptr,rows,width);
        error=cudaMemcpy(a->device_projection,&v,sizeof(v),cudaMemcpyHostToDevice);
    }else{
        auto& v=static_cast<DenseContext<kittens::bf16>*>(a->projection)->globals;
        if(v.a.rows()!=rows||v.a.cols()!=width)return cudaErrorInvalidValue;
        v.a=decltype(v.a)(reinterpret_cast<kittens::bf16*>(p.output),nullptr,nullptr,rows,width);
        error=cudaMemcpy(a->device_projection,&v,sizeof(v),cudaMemcpyHostToDevice);
    }
    // OpenPI rounds projection results to BF16 before the residual addition.
    o.residual=static_cast<const kittens::bf16*>(p.input);o.round_before_residual=1;
    if(error==cudaSuccess)error=cudaMemcpy(a->device_output_projection,&o,sizeof(o),cudaMemcpyHostToDevice);
    n.input=o.bf16_output;
    f.a=PackedInput(reinterpret_cast<kittens::fp8e4m3*>(n.output),nullptr,nullptr,rows,width/2);f.sfa=n.scales;
    d.a=PackedInput(reinterpret_cast<kittens::fp8e4m3*>(f.fp4_output),nullptr,nullptr,rows,d.a.cols());d.sfa=f.output_scales;d.residual=o.bf16_output;d.round_before_residual=1;
    NormGlobals hn[]={p,n};Fp4Globals hf[]={f,d};hf[0].task_from_instruction=hf[1].task_from_instruction=true;
    NormGlobals* norms=nullptr;Fp4Globals* ffn=nullptr;int *dk=nullptr,*dt=nullptr;size_t bytes=size_t(q->globals.queue.nodes)*4;
    if(error==cudaSuccess)error=cudaMalloc(&norms,sizeof(hn));if(error==cudaSuccess)error=cudaMemcpy(norms,hn,sizeof(hn),cudaMemcpyHostToDevice);
    if(error==cudaSuccess)error=cudaMalloc(&ffn,sizeof(hf));if(error==cudaSuccess)error=cudaMemcpy(ffn,hf,sizeof(hf),cudaMemcpyHostToDevice);
    if(error==cudaSuccess)error=cudaMalloc(&dk,bytes);if(error==cudaSuccess)error=cudaMemcpy(dk,kinds,bytes,cudaMemcpyHostToDevice);
    if(error==cudaSuccess)error=cudaMalloc(&dt,bytes);if(error==cudaSuccess)error=cudaMemcpy(dt,tasks,bytes,cudaMemcpyHostToDevice);
    if(error==cudaSuccess)error=cudaFuncSetAttribute(BLOCK_KERNEL,cudaFuncAttributeMaxDynamicSharedMemorySize,Cfg::DYNAMIC_SHARED_MEMORY);
    if(error==cudaSuccess)*handle=new TransformerBlockContext{a,q,norms,ffn,dk,dt,device,pre,down};
    else{cudaFree(norms);cudaFree(ffn);cudaFree(dk);cudaFree(dt);}return error;
}
extern "C" int launch_transformer_block(void* handle,void* stream){
    if(!handle)return cudaErrorInvalidValue;auto& c=*static_cast<TransformerBlockContext*>(handle);int device;auto error=cudaGetDevice(&device);
    if(error!=cudaSuccess)return error;if(device!=c.device)return cudaErrorInvalidDevice;
    cudaStreamCaptureStatus capture;error=cudaStreamIsCapturing(static_cast<cudaStream_t>(stream),&capture);
    if(error!=cudaSuccess)return error;if(capture!=cudaStreamCaptureStatusNone)return cudaErrorStreamCaptureUnsupported;
    unsigned epoch=std::max(c.queue->epoch,c.attention->attention->globals.epoch);if(epoch==UINT_MAX)return cudaErrorInvalidValue;
    c.queue->epoch=++epoch;c.queue->globals.queue.epoch=epoch;c.attention->attention->globals.epoch=epoch;c.attention->attention->started=true;
    const auto& q=c.queue->globals;
    ProjectionAttentionGlobals base{q.instructions,q.timings,q.queue,c.attention->device_projection,c.attention->device_attention,c.attention->projection->fp8,
        c.attention->device_output_projection,c.kind,c.task,q.values,q.owners,q.ordinals,q.exits,q.visits,q.starts,q.ends};
    TransformerBlockGlobals g{base,c.norms,c.ffn};
    BLOCK_KERNEL<<<c.queue->workers,Cfg::NUM_THREADS,Cfg::DYNAMIC_SHARED_MEMORY,static_cast<cudaStream_t>(stream)>>>(g);return cudaGetLastError();
}
extern "C" int destroy_transformer_block(void* handle){
    if(!handle)return cudaSuccess;auto* c=static_cast<TransformerBlockContext*>(handle);int device;auto error=cudaGetDevice(&device);
    if(error!=cudaSuccess)return error;if(device!=c->device)return cudaErrorInvalidDevice;
    for(void* p:{static_cast<void*>(c->norms),static_cast<void*>(c->ffn),static_cast<void*>(c->kind),static_cast<void*>(c->task)}){auto e=cudaFree(p);if(error==cudaSuccess)error=e;}
    delete c;return error;
}

struct TransformerStackContext {
    std::vector<TransformerBlockContext*> blocks;DagProbeContext* queue;
    ProjectionAttentionStage* stages;NormGlobals* norms;Fp4Globals* ffn;
    int *kind,*task,*stage;int device;
};
extern "C" int create_transformer_stack(void** handle,void* queue_handle,int layers,void* const* block_handles,void* final_norm_handle,
    const int* kinds,const int* tasks,const int* stage_ids){
    if(!handle)return cudaErrorInvalidValue;*handle=nullptr;
    if(!queue_handle||layers<1||layers>256||!block_handles||!final_norm_handle||!kinds||!tasks||!stage_ids)return cudaErrorInvalidValue;
    auto* q=static_cast<DagProbeContext*>(queue_handle);auto* last=static_cast<NormContext*>(final_norm_handle);
    int device;auto error=cudaGetDevice(&device);if(error!=cudaSuccess)return error;
    if(q->device!=device||last->device!=device)return cudaErrorInvalidDevice;
    std::vector<TransformerBlockContext*> blocks;
    std::vector<ProjectionAttentionStage> hs(layers+1);
    std::vector<NormGlobals> hn(2*(layers+1),last->globals);std::vector<Fp4Globals> hf;hf.reserve(2*layers);
    std::vector<std::vector<unsigned char>> assigned((layers+1)*7);
    int rows=last->globals.rows,width=last->globals.width,groups=(rows+NATIVE_BLOCK_NORM_ROWS-1)/NATIVE_BLOCK_NORM_ROWS;
    if(last->globals.fp16||last->globals.format!=16||last->globals.residual||last->globals.global_scale!=1.f)return cudaErrorInvalidValue;
    for(int layer=0;layer<layers;++layer){
        auto* block=static_cast<TransformerBlockContext*>(block_handles[layer]);if(!block)return cudaErrorInvalidValue;
        if(block->device!=device)return cudaErrorInvalidDevice;
        if(std::find(blocks.begin(),blocks.end(),block)!=blocks.end())return cudaErrorInvalidValue;
        auto* a=block->attention;const auto& d=block->down->globals;
        if(d.out.rows()!=rows||d.out.cols()!=width||block->pre_norm->globals.rows!=rows||block->pre_norm->globals.width!=width
           ||(layer && (a->attention->globals.batches!=blocks[0]->attention->attention->globals.batches||a->attention->globals.nq!=blocks[0]->attention->attention->globals.nq)))return cudaErrorInvalidValue;
        blocks.push_back(block);hs[layer]={a->device_projection,a->device_attention,a->projection->fp8,a->device_output_projection};
        hf.push_back(block->down->globals);hf.push_back(block->down->globals);
        error=cudaMemcpy(hn.data()+layer*2,block->norms,2*sizeof(NormGlobals),cudaMemcpyDeviceToHost);if(error!=cudaSuccess)return error;
        error=cudaMemcpy(hf.data()+layer*2,block->ffn,2*sizeof(Fp4Globals),cudaMemcpyDeviceToHost);if(error!=cudaSuccess)return error;
        int counts[]={dense_tasks(a->projection),a->attention->globals.tasks,dense_tasks(a->output_projection),groups,groups,hf[layer*2].tasks,hf[layer*2+1].tasks};
        for(int kind=0;kind<7;++kind)assigned[layer*7+kind].resize(counts[kind],0);
    }
    assigned[layers*7+3].resize(groups,0);
    for(int node=0;node<q->globals.queue.nodes;++node){
        int stage=stage_ids[node],kind=kinds[node],task=tasks[node];
        if(stage<0||stage>layers||kind<0||kind>=7||task<0)return cudaErrorInvalidValue;
        auto& seen=assigned[stage*7+kind];if(task>=int(seen.size())||seen[task])return cudaErrorInvalidValue;seen[task]=1;
    }
    for(const auto& seen:assigned)if(std::count(seen.begin(),seen.end(),1)!=seen.size())return cudaErrorInvalidValue;
    // Cross-layer handoffs are pointer bindings, with row-band DAG edges.
    // Updating each child also makes serial child launches a faithful reference.
    for(int layer=1;layer<layers;++layer){
        auto* block=blocks[layer];auto* input=blocks[layer-1]->down->globals.bf16_output;
        hn[layer*2].input=input;block->pre_norm->globals.input=input;
        error=cudaMemcpy(block->norms,hn.data()+layer*2,2*sizeof(NormGlobals),cudaMemcpyHostToDevice);if(error!=cudaSuccess)return error;
        auto& o=static_cast<DenseContext<kittens::bf16>*>(block->attention->output_projection)->globals;o.residual=input;
        error=cudaMemcpy(block->attention->device_output_projection,&o,sizeof(o),cudaMemcpyHostToDevice);if(error!=cudaSuccess)return error;
    }
    last->globals.input=blocks.back()->down->globals.bf16_output;hn[layers*2]=hn[layers*2+1]=last->globals;
    ProjectionAttentionStage* stages=nullptr;NormGlobals* norms=nullptr;Fp4Globals* ffn=nullptr;int *dk=nullptr,*dt=nullptr,*ds=nullptr;
    size_t bytes=size_t(q->globals.queue.nodes)*sizeof(int);
    error=cudaMalloc(&stages,hs.size()*sizeof(hs[0]));if(error==cudaSuccess)error=cudaMemcpy(stages,hs.data(),hs.size()*sizeof(hs[0]),cudaMemcpyHostToDevice);
    if(error==cudaSuccess)error=cudaMalloc(&norms,hn.size()*sizeof(hn[0]));if(error==cudaSuccess)error=cudaMemcpy(norms,hn.data(),hn.size()*sizeof(hn[0]),cudaMemcpyHostToDevice);
    if(error==cudaSuccess)error=cudaMalloc(&ffn,hf.size()*sizeof(hf[0]));if(error==cudaSuccess)error=cudaMemcpy(ffn,hf.data(),hf.size()*sizeof(hf[0]),cudaMemcpyHostToDevice);
    if(error==cudaSuccess)error=cudaMalloc(&dk,bytes);if(error==cudaSuccess)error=cudaMemcpy(dk,kinds,bytes,cudaMemcpyHostToDevice);
    if(error==cudaSuccess)error=cudaMalloc(&dt,bytes);if(error==cudaSuccess)error=cudaMemcpy(dt,tasks,bytes,cudaMemcpyHostToDevice);
    if(error==cudaSuccess)error=cudaMalloc(&ds,bytes);if(error==cudaSuccess)error=cudaMemcpy(ds,stage_ids,bytes,cudaMemcpyHostToDevice);
    if(error==cudaSuccess)error=cudaFuncSetAttribute(BLOCK_KERNEL,cudaFuncAttributeMaxDynamicSharedMemorySize,Cfg::DYNAMIC_SHARED_MEMORY);
    if(error==cudaSuccess)*handle=new TransformerStackContext{blocks,q,stages,norms,ffn,dk,dt,ds,device};
    else{cudaFree(stages);cudaFree(norms);cudaFree(ffn);cudaFree(dk);cudaFree(dt);cudaFree(ds);}return error;
}
extern "C" int launch_transformer_stack(void* handle,void* stream){
    if(!handle)return cudaErrorInvalidValue;auto& c=*static_cast<TransformerStackContext*>(handle);int device;auto error=cudaGetDevice(&device);
    if(error!=cudaSuccess)return error;if(device!=c.device)return cudaErrorInvalidDevice;
    cudaStreamCaptureStatus capture;error=cudaStreamIsCapturing(static_cast<cudaStream_t>(stream),&capture);
    if(error!=cudaSuccess)return error;if(capture!=cudaStreamCaptureStatusNone)return cudaErrorStreamCaptureUnsupported;
    unsigned epoch=c.queue->epoch;for(auto* block:c.blocks)epoch=std::max(epoch,block->attention->attention->globals.epoch);
    if(epoch==UINT_MAX)return cudaErrorInvalidValue;c.queue->epoch=++epoch;c.queue->globals.queue.epoch=epoch;
    for(auto* block:c.blocks){block->attention->attention->globals.epoch=epoch;block->attention->attention->started=true;}
    const auto& q=c.queue->globals;
    ProjectionAttentionGlobals base{q.instructions,q.timings,q.queue,nullptr,nullptr,0,nullptr,c.kind,c.task,
        q.values,q.owners,q.ordinals,q.exits,q.visits,q.starts,q.ends,c.stages,c.stage};
    TransformerBlockGlobals g{base,c.norms,c.ffn};
    BLOCK_KERNEL<<<c.queue->workers,Cfg::NUM_THREADS,Cfg::DYNAMIC_SHARED_MEMORY,static_cast<cudaStream_t>(stream)>>>(g);return cudaGetLastError();
}
extern "C" int destroy_transformer_stack(void* handle){
    if(!handle)return cudaSuccess;auto* c=static_cast<TransformerStackContext*>(handle);int device;auto error=cudaGetDevice(&device);
    if(error!=cudaSuccess)return error;if(device!=c->device)return cudaErrorInvalidDevice;
    for(void* p:{static_cast<void*>(c->stages),static_cast<void*>(c->norms),static_cast<void*>(c->ffn),static_cast<void*>(c->kind),static_cast<void*>(c->task),static_cast<void*>(c->stage)}){auto e=cudaFree(p);if(error==cudaSuccess)error=e;}
    delete c;return error;
}
