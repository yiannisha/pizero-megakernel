// Device-scheduled BF16/FP8 QKV projection -> native attention in one original VM.
#include "dense.cu"
#include "attention.cu"
#include "dag_probe.cu"
static_assert(Cfg::NUM_CONSUMER_WARPS==4 && Cfg::NUM_PAGES==AC::NUM_PAGES && Cfg::PAGE_SIZE==AC::PAGE_SIZE);
struct ProjectionAttentionStage {
    const void* projection;const AttentionGlobals* attention;int fp8;
    const DenseGlobals<kittens::bf16>* output_projection;
};
struct ProjectionAttentionGlobals {
    static constexpr bool DYNAMIC_INSTRUCTIONS=true;
    megakernel::instruction_layout<Cfg> instructions;RingTimings timings;
    native_pi0::DagScheduler queue;
    const void* projection;const AttentionGlobals* attention;int fp8;
    const DenseGlobals<kittens::bf16>* output_projection;
    const int *node_kind,*node_task;
    unsigned *values,*owners,*ordinals,*exits;unsigned long long *visits,*starts,*ends;
    const ProjectionAttentionStage* stages=nullptr;const int* node_stage=nullptr;
    __device__ ProjectionAttentionGlobals select_stage(int node)const{
        auto view=*this;
        if(stages){
            const auto& stage=stages[node_stage[node]];
            view.projection=stage.projection;view.attention=stage.attention;view.fp8=stage.fp8;view.output_projection=stage.output_projection;
            view.stages=nullptr;
        }
        return view;
    }
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
struct ProjectionAttentionDiagnostics {
    static __device__ void start(const ProjectionAttentionGlobals& g,State& s){
        if(threadIdx.x==0){
            int node=s.instruction()[1];unsigned long long t;asm volatile("mov.u64 %0, %%globaltimer;":"=l"(t)::"memory");g.starts[node]=t;
            if(atomicExch(g.visits+node,static_cast<unsigned long long>(g.queue.epoch))==g.queue.epoch)atomicAdd(&g.queue.control->errors,1u);
        }
    }
    static __device__ void finish(const ProjectionAttentionGlobals& g,State& s){
        asm volatile("fence.proxy.async.global;":::"memory");native_attention::attention_sync();
        if(threadIdx.x==0){
            int node=s.instruction()[1];g.values[node]=g.queue.epoch;g.owners[node]=blockIdx.x;g.ordinals[node]=s.instruction_index;
            unsigned long long t;asm volatile("mov.u64 %0, %%globaltimer;":"=l"(t)::"memory");g.ends[node]=t;g.queue.complete(node);
        }
        native_attention::attention_sync();
    }
};
struct ScheduledProjectionOp {
    static constexpr int opcode=1;
    struct controller {
        static __device__ int release_lid(const ProjectionAttentionGlobals&,Cfg::instruction_t&,int& q){return q;}
        static __device__ int init_semaphores(const ProjectionAttentionGlobals&,State& s){for(int i=0;i<5;++i)kittens::init_semaphore(s.semaphores()[i],1);return 5;}
    };
    template<class Element> static __device__ const DenseGlobals<Element>& stage(const ProjectionAttentionGlobals& g){return *static_cast<const DenseGlobals<Element>*>(g.projection);}
    template<class Element> static __device__ void load(const ProjectionAttentionGlobals& g,State& s){
        const auto& source=stage<Element>(g);using Tile=typename Globals<Element>::Tile;
        if(kittens::laneid()==0){
            const auto *a=source.a.template get_tma<Tile,2>(),*b=source.b.template get_tma<Tile,2>();
            asm volatile("fence.proxy.tensormap::generic.acquire.gpu [%0], 128;"::"l"(a):"memory");
            asm volatile("fence.proxy.tensormap::generic.acquire.gpu [%0], 128;"::"l"(b):"memory");
        }
        __syncwarp();DenseView<Element> view(source,s.instruction()[2]);Matmul<Element>::loader::run(view,s);
    }
    template<class Element> static __device__ void launch(const ProjectionAttentionGlobals& g,State& s){DenseView<Element> view(stage<Element>(g),s.instruction()[2]);Matmul<Element>::launcher::run(view,s);}
    template<class Element> static __device__ void consume(const ProjectionAttentionGlobals& g,State& s){DenseView<Element> view(stage<Element>(g),s.instruction()[2]);DenseOp<Element>::consume(view,s);}
    struct loader {static __device__ void run(const ProjectionAttentionGlobals& g,State& s){auto view=g.select_stage(s.instruction()[1]);if(view.fp8)load<kittens::fp8e4m3>(view,s);else load<kittens::bf16>(view,s);}};
    struct launcher {static __device__ void run(const ProjectionAttentionGlobals& g,State& s){auto view=g.select_stage(s.instruction()[1]);if(view.fp8)launch<kittens::fp8e4m3>(view,s);else launch<kittens::bf16>(view,s);}};
    struct consumer {static __device__ void run(const ProjectionAttentionGlobals& g,State& s){
        ProjectionAttentionDiagnostics::start(g,s);
        auto view=g.select_stage(s.instruction()[1]);
        if(view.fp8)consume<kittens::fp8e4m3>(view,s);else consume<kittens::bf16>(view,s);
        ProjectionAttentionDiagnostics::finish(g,s);
    }};
    using storer=megakernel::NoOp<Cfg>::storer;
};
struct ScheduledAttentionOp {
    static constexpr int opcode=2;
    using controller=megakernel::NoOp<Cfg>::controller;
    using launcher=megakernel::NoOp<Cfg>::launcher;
    using storer=megakernel::NoOp<Cfg>::storer;
    struct loader {static __device__ void run(const ProjectionAttentionGlobals& g,State& s){
        auto view=g.select_stage(s.instruction()[1]);int used=AttentionOp::pages(*view.attention),lane=kittens::laneid();
        if(lane>=used && lane<Cfg::NUM_PAGES){int p=s.pid(lane);s.wait_page_ready(p);s.finish_page(p,4);}
    }};
    struct consumer {static __device__ void run(const ProjectionAttentionGlobals& g,State& s){
        ProjectionAttentionDiagnostics::start(g,s);
        auto stage=g.select_stage(s.instruction()[1]);AttentionGlobals view=*stage.attention;view.epoch=g.queue.epoch;
        AttentionOp::consume(view,s,s.instruction()[2]);
        ProjectionAttentionDiagnostics::finish(g,s);
    }};
};
struct ScheduledOutputProjectionOp {
    static constexpr int opcode=3;
    using controller=ScheduledProjectionOp::controller;
    using storer=megakernel::NoOp<Cfg>::storer;
    static __device__ ProjectionAttentionGlobals select(const ProjectionAttentionGlobals& g,State& s){
        auto view=g.select_stage(s.instruction()[1]);view.projection=view.output_projection;view.fp8=0;return view;
    }
    struct loader {static __device__ void run(const ProjectionAttentionGlobals& g,State& s){auto view=select(g,s);ScheduledProjectionOp::loader::run(view,s);}};
    struct launcher {static __device__ void run(const ProjectionAttentionGlobals& g,State& s){auto view=select(g,s);ScheduledProjectionOp::launcher::run(view,s);}};
    struct consumer {static __device__ void run(const ProjectionAttentionGlobals& g,State& s){auto view=select(g,s);ScheduledProjectionOp::consumer::run(view,s);}};
};
struct ProjectionAttentionContext {
    DagProbeContext* queue;DenseContextBase* projection;AttentionContext* attention;
    void* device_projection;AttentionGlobals* device_attention;int *kind,*task,device;
    DenseContextBase* output_projection;DenseGlobals<kittens::bf16>* device_output_projection;
};
template<class Element> static cudaError_t copy_projection(DenseContextBase* base,void** device,AttentionGlobals& attention,int* tasks){
    const auto& p=static_cast<DenseContext<Element>*>(base)->globals;
    if(!p.bf16_output||int64_t(attention.batches)*attention.nq>p.out.rows()||p.out.cols()<(attention.heads+2*attention.kv_heads)*attention.hd)return cudaErrorInvalidValue;
    attention.q=p.bf16_output;attention.k=p.bf16_output+attention.heads*attention.hd;
    attention.v=p.bf16_output+(attention.heads+attention.kv_heads)*attention.hd;
    attention.ldq=attention.ldkv=p.out.cols();*tasks=p.tasks;
    auto error=cudaMalloc(device,sizeof(p));if(error==cudaSuccess)error=cudaMemcpy(*device,&p,sizeof(p),cudaMemcpyHostToDevice);return error;
}
int create_projection_attention_impl(void** handle,void* queue_handle,void* projection_handle,void* attention_handle,const int* kind,const int* task,void* output_handle){
    if(!handle)return cudaErrorInvalidValue;*handle=nullptr;
    if(!queue_handle||!projection_handle||!attention_handle||!kind||!task)return cudaErrorInvalidValue;
    auto& q=*static_cast<DagProbeContext*>(queue_handle);auto* p=static_cast<DenseContextBase*>(projection_handle);auto* a=static_cast<AttentionContext*>(attention_handle);
    int device;auto error=cudaGetDevice(&device);if(error!=cudaSuccess)return error;
    if(device!=q.device||device!=p->device||device!=a->device)return cudaErrorInvalidDevice;
    if(a->globals.fp16||a->globals.nq!=a->globals.nk-a->globals.cached_prefix_rows)return cudaErrorInvalidValue;
    AttentionGlobals attention=a->globals;void* projection=nullptr;int projection_tasks=0;
    error=p->fp8?copy_projection<kittens::fp8e4m3>(p,&projection,attention,&projection_tasks):copy_projection<kittens::bf16>(p,&projection,attention,&projection_tasks);
    if(error!=cudaSuccess){cudaFree(projection);return error;}
    DenseContextBase* output=static_cast<DenseContextBase*>(output_handle);
    const DenseGlobals<kittens::bf16>* output_globals=nullptr;int output_tasks=0;
    if(output){
        if(output->device!=device||output->fp8){cudaFree(projection);return cudaErrorInvalidValue;}
        output_globals=&static_cast<DenseContext<kittens::bf16>*>(output)->globals;
        if(output_globals->a.rows()<int64_t(attention.batches)*attention.nq||output_globals->a.cols()<attention.heads*attention.hd){cudaFree(projection);return cudaErrorInvalidValue;}
        attention.output=output_globals->a.raw_ptr;attention.out_ld=output_globals->a.cols();output_tasks=output_globals->tasks;
    }
    std::vector<unsigned char> ps(projection_tasks,0),as(attention.tasks,0),os(output_tasks,0);
    for(int i=0;i<q.globals.queue.nodes;++i){
        int t=task[i];if(kind[i]==0){if(t<0||t>=projection_tasks||ps[t]){cudaFree(projection);return cudaErrorInvalidValue;}ps[t]=1;}
        else if(kind[i]==1){if(t<0||t>=attention.tasks||as[t]){cudaFree(projection);return cudaErrorInvalidValue;}as[t]=1;}
        else if(kind[i]==2){if(!output||t<0||t>=output_tasks||os[t]){cudaFree(projection);return cudaErrorInvalidValue;}os[t]=1;}
        else{cudaFree(projection);return cudaErrorInvalidValue;}
    }
    if(std::count(ps.begin(),ps.end(),1)!=projection_tasks||std::count(as.begin(),as.end(),1)!=attention.tasks||std::count(os.begin(),os.end(),1)!=output_tasks){cudaFree(projection);return cudaErrorInvalidValue;}
    AttentionGlobals* da=nullptr;DenseGlobals<kittens::bf16>* dop=nullptr;int *dk=nullptr,*dt=nullptr;size_t bytes=size_t(q.globals.queue.nodes)*4;
    error=cudaMalloc(&da,sizeof(attention));if(error==cudaSuccess)error=cudaMemcpy(da,&attention,sizeof(attention),cudaMemcpyHostToDevice);
    if(output&&error==cudaSuccess)error=cudaMalloc(&dop,sizeof(*output_globals));
    if(output&&error==cudaSuccess)error=cudaMemcpy(dop,output_globals,sizeof(*output_globals),cudaMemcpyHostToDevice);
    if(error==cudaSuccess)error=cudaMalloc(&dk,bytes);if(error==cudaSuccess)error=cudaMalloc(&dt,bytes);
    if(error==cudaSuccess)error=cudaMemcpy(dk,kind,bytes,cudaMemcpyHostToDevice);
    if(error==cudaSuccess)error=cudaMemcpy(dt,task,bytes,cudaMemcpyHostToDevice);
    auto kernel=megakernel::mk<Cfg,ProjectionAttentionGlobals,ScheduledProjectionOp,ScheduledAttentionOp,ScheduledOutputProjectionOp>;
    if(error==cudaSuccess)error=cudaFuncSetAttribute(kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,Cfg::DYNAMIC_SHARED_MEMORY);
    if(error==cudaSuccess){
        a->globals=attention;
        *handle=new ProjectionAttentionContext{&q,p,a,projection,da,dk,dt,device,output,dop};
    }else{cudaFree(projection);cudaFree(da);cudaFree(dk);cudaFree(dt);cudaFree(dop);}return error;
}
extern "C" int create_projection_attention(void** handle,void* queue,void* projection,void* attention,const int* kind,const int* task){
    return create_projection_attention_impl(handle,queue,projection,attention,kind,task,nullptr);
}
extern "C" int create_projection_attention_output(void** handle,void* queue,void* projection,void* attention,const int* kind,const int* task,void* output){
    if(!output)return cudaErrorInvalidValue;
    return create_projection_attention_impl(handle,queue,projection,attention,kind,task,output);
}
extern "C" int launch_projection_attention(void* handle,void* stream){
    if(!handle)return cudaErrorInvalidValue;auto& c=*static_cast<ProjectionAttentionContext*>(handle);int device;auto error=cudaGetDevice(&device);
    if(error!=cudaSuccess)return error;if(device!=c.device)return cudaErrorInvalidDevice;
    cudaStreamCaptureStatus capture;error=cudaStreamIsCapturing(static_cast<cudaStream_t>(stream),&capture);
    if(error!=cudaSuccess)return error;if(capture!=cudaStreamCaptureStatusNone)return cudaErrorStreamCaptureUnsupported;
    unsigned epoch=std::max(c.queue->epoch,c.attention->globals.epoch);if(epoch==UINT_MAX)return cudaErrorInvalidValue;
    c.queue->epoch=++epoch;c.queue->globals.queue.epoch=epoch;c.attention->globals.epoch=epoch;c.attention->started=true;
    const auto& q=c.queue->globals;
    ProjectionAttentionGlobals g{q.instructions,q.timings,q.queue,c.device_projection,c.device_attention,c.projection->fp8,c.device_output_projection,c.kind,c.task,
        q.values,q.owners,q.ordinals,q.exits,q.visits,q.starts,q.ends};
    megakernel::mk<Cfg,ProjectionAttentionGlobals,ScheduledProjectionOp,ScheduledAttentionOp,ScheduledOutputProjectionOp><<<c.queue->workers,Cfg::NUM_THREADS,Cfg::DYNAMIC_SHARED_MEMORY,static_cast<cudaStream_t>(stream)>>>(g);
    return cudaGetLastError();
}
extern "C" int destroy_projection_attention(void* handle){
    if(!handle)return cudaSuccess;auto* c=static_cast<ProjectionAttentionContext*>(handle);int device;auto error=cudaGetDevice(&device);
    if(error!=cudaSuccess)return error;if(device!=c->device)return cudaErrorInvalidDevice;
    for(void* p:{c->device_projection,static_cast<void*>(c->device_attention),static_cast<void*>(c->kind),static_cast<void*>(c->task),static_cast<void*>(c->device_output_projection)}){
        auto status=cudaFree(p);if(error==cudaSuccess)error=status;
    }
    delete c;return error;
}

struct CachedPrefixExpertContext {
    DagProbeContext* queue;ProjectionAttentionContext *prefix,*expert;int device;
    ProjectionAttentionStage* stages;int *kind,*task,*stage;
};
static int dense_tasks(DenseContextBase* base){
    if(!base)return 0;
    return base->fp8?static_cast<DenseContext<kittens::fp8e4m3>*>(base)->globals.tasks:static_cast<DenseContext<kittens::bf16>*>(base)->globals.tasks;
}
extern "C" int create_cached_prefix_expert(void** handle,void* queue_handle,void* prefix_handle,void* expert_handle,const int* kinds,const int* tasks,const int* stage_ids){
    if(!handle)return cudaErrorInvalidValue;*handle=nullptr;
    if(!queue_handle||!prefix_handle||!expert_handle||!kinds||!tasks||!stage_ids)return cudaErrorInvalidValue;
    auto* q=static_cast<DagProbeContext*>(queue_handle);auto* p=static_cast<ProjectionAttentionContext*>(prefix_handle);auto* e=static_cast<ProjectionAttentionContext*>(expert_handle);
    int device;auto error=cudaGetDevice(&device);if(error!=cudaSuccess)return error;
    if(q->device!=device||p->device!=device||e->device!=device||p==e)return cudaErrorInvalidDevice;
    const auto& pg=p->attention->globals;const auto& eg=e->attention->globals;
    if(!pg.cache_k||pg.cached_prefix_rows||pg.nk!=eg.cached_prefix_rows||pg.cache_k!=eg.cache_k||pg.cache_v!=eg.cache_v||pg.cache_rows!=eg.cache_rows
       ||pg.batches!=eg.batches||pg.kv_heads!=eg.kv_heads||pg.hd!=eg.hd||pg.fp16!=eg.fp16)return cudaErrorInvalidValue;
    int counts[2][3]={{dense_tasks(p->projection),pg.tasks,dense_tasks(p->output_projection)},{dense_tasks(e->projection),eg.tasks,dense_tasks(e->output_projection)}};
    std::vector<unsigned char> assigned[2][3];
    for(int stage=0;stage<2;++stage)for(int kind=0;kind<3;++kind)assigned[stage][kind].resize(counts[stage][kind],0);
    for(int i=0;i<q->globals.queue.nodes;++i){
        int stage=stage_ids[i],kind=kinds[i],task=tasks[i];
        if(stage<0||stage>1||kind<0||kind>2||task<0||task>=counts[stage][kind]||assigned[stage][kind][task])return cudaErrorInvalidValue;
        assigned[stage][kind][task]=1;
    }
    for(int stage=0;stage<2;++stage)for(int kind=0;kind<3;++kind)
        if(std::count(assigned[stage][kind].begin(),assigned[stage][kind].end(),1)!=counts[stage][kind])return cudaErrorInvalidValue;
    ProjectionAttentionStage host[]={{p->device_projection,p->device_attention,p->projection->fp8,p->device_output_projection},
        {e->device_projection,e->device_attention,e->projection->fp8,e->device_output_projection}};
    ProjectionAttentionStage* stages=nullptr;int *dk=nullptr,*dt=nullptr,*ds=nullptr;size_t bytes=size_t(q->globals.queue.nodes)*4;
    error=cudaMalloc(&stages,sizeof(host));if(error==cudaSuccess)error=cudaMemcpy(stages,host,sizeof(host),cudaMemcpyHostToDevice);
    if(error==cudaSuccess)error=cudaMalloc(&dk,bytes);if(error==cudaSuccess)error=cudaMalloc(&dt,bytes);if(error==cudaSuccess)error=cudaMalloc(&ds,bytes);
    if(error==cudaSuccess)error=cudaMemcpy(dk,kinds,bytes,cudaMemcpyHostToDevice);
    if(error==cudaSuccess)error=cudaMemcpy(dt,tasks,bytes,cudaMemcpyHostToDevice);
    if(error==cudaSuccess)error=cudaMemcpy(ds,stage_ids,bytes,cudaMemcpyHostToDevice);
    auto kernel=megakernel::mk<Cfg,ProjectionAttentionGlobals,ScheduledProjectionOp,ScheduledAttentionOp,ScheduledOutputProjectionOp>;
    if(error==cudaSuccess)error=cudaFuncSetAttribute(kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,Cfg::DYNAMIC_SHARED_MEMORY);
    if(error==cudaSuccess)*handle=new CachedPrefixExpertContext{q,p,e,device,stages,dk,dt,ds};
    else{cudaFree(stages);cudaFree(dk);cudaFree(dt);cudaFree(ds);}return error;
}
extern "C" int launch_cached_prefix_expert(void* handle,void* stream){
    if(!handle)return cudaErrorInvalidValue;auto& c=*static_cast<CachedPrefixExpertContext*>(handle);int device;auto error=cudaGetDevice(&device);
    if(error!=cudaSuccess)return error;if(device!=c.device)return cudaErrorInvalidDevice;
    cudaStreamCaptureStatus capture;error=cudaStreamIsCapturing(static_cast<cudaStream_t>(stream),&capture);
    if(error!=cudaSuccess)return error;if(capture!=cudaStreamCaptureStatusNone)return cudaErrorStreamCaptureUnsupported;
    unsigned epoch=std::max(c.queue->epoch,std::max(c.prefix->attention->globals.epoch,c.expert->attention->globals.epoch));
    if(epoch==UINT_MAX)return cudaErrorInvalidValue;
    c.queue->epoch=++epoch;c.queue->globals.queue.epoch=epoch;
    c.prefix->attention->globals.epoch=c.expert->attention->globals.epoch=epoch;
    c.prefix->attention->started=c.expert->attention->started=true;
    const auto& q=c.queue->globals;
    ProjectionAttentionGlobals g{q.instructions,q.timings,q.queue,nullptr,nullptr,0,nullptr,c.kind,c.task,
        q.values,q.owners,q.ordinals,q.exits,q.visits,q.starts,q.ends,c.stages,c.stage};
    megakernel::mk<Cfg,ProjectionAttentionGlobals,ScheduledProjectionOp,ScheduledAttentionOp,ScheduledOutputProjectionOp><<<c.queue->workers,Cfg::NUM_THREADS,Cfg::DYNAMIC_SHARED_MEMORY,static_cast<cudaStream_t>(stream)>>>(g);
    return cudaGetLastError();
}
extern "C" int destroy_cached_prefix_expert(void* handle){
    if(!handle)return cudaSuccess;auto* c=static_cast<CachedPrefixExpertContext*>(handle);int device;auto error=cudaGetDevice(&device);
    if(error!=cudaSuccess)return error;if(device!=c->device)return cudaErrorInvalidDevice;
    for(void* p:{static_cast<void*>(c->stages),static_cast<void*>(c->kind),static_cast<void*>(c->task),static_cast<void*>(c->stage)}){
        auto status=cudaFree(p);if(error==cudaSuccess)error=status;
    }
    delete c;return error;
}
