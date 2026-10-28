#pragma once
// Heterogeneous projection/preparation/attention instructions in the same VM.
static_assert(Cfg::NUM_PAGES==AC::NUM_PAGES && Cfg::PAGE_SIZE==AC::PAGE_SIZE);
struct GenerationAttentionGlobals:EdgeDenseGlobals {
    const EdgeRopeGlobals* preparation;
    const AttentionGlobals* attention;
};
struct GenerationAttentionDiagnostics {
    static __device__ void start(const GenerationAttentionGlobals& g,State& s){
        if(threadIdx.x==0){
            int node=s.instruction()[3];unsigned long long t;asm volatile("mov.u64 %0, %%globaltimer;":"=l"(t)::"memory");g.starts[node]=t;
            if(atomicExch(g.visits+node,static_cast<unsigned long long>(g.queue.epoch))==g.queue.epoch)atomicAdd(&g.queue.control->errors,1u);
        }
    }
    static __device__ void finish(const GenerationAttentionGlobals& g,State& s){
        asm volatile("fence.proxy.async.global;":::"memory");DagMatmulOp::sync();
        if(threadIdx.x==0){
            int node=s.instruction()[3];g.values[node]=g.queue.epoch;g.owners[node]=blockIdx.x;g.ordinals[node]=s.instruction_index;
            unsigned long long t;asm volatile("mov.u64 %0, %%globaltimer;":"=l"(t)::"memory");g.ends[node]=t;g.queue.complete(node);
        }
        DagMatmulOp::sync();
    }
};
struct ScheduledEdgePreparation {
    static constexpr int opcode=5;
    using controller=megakernel::NoOp<Cfg>::controller;
    using loader=megakernel::NoOp<Cfg>::loader;
    using launcher=megakernel::NoOp<Cfg>::launcher;
    using storer=megakernel::NoOp<Cfg>::storer;
    struct consumer {static __device__ void run(const GenerationAttentionGlobals& g,State& s){
        GenerationAttentionDiagnostics::start(g,s);
        const auto& p=*g.preparation;int first=s.instruction()[2]*EDGE_ROPE_VECTORS_PER_TASK;
        int end=min(first+EDGE_ROPE_VECTORS_PER_TASK,p.rows*(p.q_heads+p.k_heads));
        for(int vec=first+kittens::warpid();vec<end;vec+=Cfg::NUM_CONSUMER_WARPS)EdgeRopeOp::vector(p,vec);
        GenerationAttentionDiagnostics::finish(g,s);
    }};
};
struct ScheduledEdgeAttention {
    static constexpr int opcode=6;
    using controller=megakernel::NoOp<Cfg>::controller;
    using launcher=megakernel::NoOp<Cfg>::launcher;
    using storer=megakernel::NoOp<Cfg>::storer;
    struct loader {static __device__ void run(const GenerationAttentionGlobals& g,State& s){
        int used=AttentionOp::pages(*g.attention),lane=kittens::laneid();
        if(lane>=used && lane<Cfg::NUM_PAGES){int p=s.pid(lane);s.wait_page_ready(p);s.finish_page(p,4);}
    }};
    struct consumer {static __device__ void run(const GenerationAttentionGlobals& g,State& s){
        GenerationAttentionDiagnostics::start(g,s);
        auto view=*g.attention;view.epoch=g.queue.epoch;
        AttentionOp::consume(view,s,s.instruction()[2]);
        GenerationAttentionDiagnostics::finish(g,s);
    }};
};
struct ScheduledEdgeKVReady {
    static constexpr int opcode=7;
    using controller=megakernel::NoOp<Cfg>::controller;
    using loader=megakernel::NoOp<Cfg>::loader;
    using launcher=megakernel::NoOp<Cfg>::launcher;
    using storer=megakernel::NoOp<Cfg>::storer;
    struct consumer {static __device__ void run(const GenerationAttentionGlobals& g,State& s){
        GenerationAttentionDiagnostics::start(g,s);GenerationAttentionDiagnostics::finish(g,s);
    }};
};
struct GenerationAttentionContext {
    GenerationAttentionGlobals globals;DagProbeContext* queue;AttentionContext* attention;
    int device;std::vector<void*> allocations;
};
extern "C" int create_generation_attention(void** handle,void* queue,void* const* stages,void* preparation,
    void* attention,const int* node_stage,const int* node_task){
    if(!handle)return cudaErrorInvalidValue;*handle=nullptr;
    if(!queue||!stages||!preparation||!attention||!node_stage||!node_task)return cudaErrorInvalidValue;
    auto& q=*static_cast<DagProbeContext*>(queue);auto& p=*static_cast<EdgeRopeContext*>(preparation);
    auto& a=*static_cast<AttentionContext*>(attention);int device;auto error=cudaGetDevice(&device);
    if(error!=cudaSuccess)return error;
    if(device!=q.device||device!=p.device||device!=a.device)return cudaErrorInvalidDevice;
    const auto& r=p.globals;const auto& at=a.globals;
    if(r.q_heads!=16||r.k_heads!=8||!r.v||at.hd!=128||at.heads!=16||at.kv_heads!=8||at.fp16||at.batches!=1
       ||at.nk<r.rows||at.prefix!=-1||at.qb||at.kb||at.vb||at.qcos||at.kcos||at.mask)return cudaErrorInvalidValue;
    uintptr_t qb=reinterpret_cast<uintptr_t>(r.qo),aq=reinterpret_cast<uintptr_t>(at.q);
    if(aq<qb||(aq-qb)%(2048*2)||(aq-qb)/(2048*2)+at.nq>unsigned(r.rows)||at.ldq!=2048||r.qo_stride!=2048)return cudaErrorInvalidValue;
    int prefix=at.nk-r.rows;
    if(reinterpret_cast<uintptr_t>(at.k)+size_t(prefix)*at.ldkv*2!=reinterpret_cast<uintptr_t>(r.ko)
       ||reinterpret_cast<uintptr_t>(at.v)+size_t(prefix)*at.ldkv*2!=reinterpret_cast<uintptr_t>(r.vo)
       ||at.ldkv!=r.ko_stride||at.ldkv!=r.vo_stride)return cudaErrorInvalidValue;
    std::vector<void*> allocated,device_stages;std::vector<EdgeDenseInfo> info;
    auto cleanup=[&](){for(auto p:allocated)cudaFree(p);};
    for(int i=0;i<3;++i){
        if(!stages[i]){cleanup();return cudaErrorInvalidValue;}
        auto& h=*static_cast<EdgeProjectionHandle*>(stages[i]);
        auto f=h.precision==0?edge_dense_info<kittens::bf16>(h.context):edge_dense_info<kittens::fp8e4m3>(h.context);
        const void* expected=i==0?static_cast<const void*>(r.q):i==1?static_cast<const void*>(r.k):static_cast<const void*>(r.v);
        int stride=i==0?r.q_stride:i==1?r.k_stride:r.v_stride;
        if(f.device!=device||f.m<r.rows||f.output!=expected||f.n!=stride||f.n!=(i==0?2048:1024)){
            cleanup();return cudaErrorInvalidValue;
        }
        void* dest=nullptr;
        error=h.precision==0?upload_edge_dense_stage<kittens::bf16>(h.context,nullptr,&dest):upload_edge_dense_stage<kittens::fp8e4m3>(h.context,nullptr,&dest);
        if(dest)allocated.push_back(dest);if(error!=cudaSuccess){cleanup();return error;}
        info.push_back(f);device_stages.push_back(dest);
    }
    int prep_tasks=(r.rows*24+EDGE_ROPE_VECTORS_PER_TASK-1)/EDGE_ROPE_VECTORS_PER_TASK,nodes=q.globals.queue.nodes;
    std::vector<std::vector<bool>> assigned;
    for(auto& f:info)assigned.emplace_back(f.tasks,false);
    assigned.emplace_back(prep_tasks,false);assigned.emplace_back(at.tasks,false);assigned.emplace_back(1,false);
    std::vector<int> opcodes(nodes);
    for(int node=0;node<nodes;++node){
        int stage=node_stage[node],task=node_task[node],group=stage>=0?stage:stage==-1?3:stage==-2?4:stage==-3?5:-1;
        if(group<0||group>5||stage>2||task<0||task>=int(assigned[group].size())||assigned[group][task]){cleanup();return cudaErrorInvalidValue;}
        assigned[group][task]=true;
        opcodes[node]=stage>=0?(static_cast<EdgeProjectionHandle*>(stages[stage])->precision==0?3:4):stage==-1?5:stage==-2?6:7;
    }
    for(auto& group:assigned)for(bool seen:group)if(!seen){cleanup();return cudaErrorInvalidValue;}
    auto upload=[&](const void* src,size_t bytes)->void*{
        void* dest=nullptr;if(error==cudaSuccess)error=cudaMalloc(&dest,bytes);
        if(dest)allocated.push_back(dest);if(error==cudaSuccess)error=cudaMemcpy(dest,src,bytes,cudaMemcpyHostToDevice);return dest;
    };
    auto table=static_cast<const void**>(upload(device_stages.data(),3*sizeof(void*)));
    auto ns=static_cast<int*>(upload(node_stage,size_t(nodes)*4)),nt=static_cast<int*>(upload(node_task,size_t(nodes)*4));
    auto no=static_cast<int*>(upload(opcodes.data(),size_t(nodes)*4));
    auto dp=static_cast<EdgeRopeGlobals*>(upload(&r,sizeof(r)));auto da=static_cast<AttentionGlobals*>(upload(&at,sizeof(at)));
    auto kernel=megakernel::mk<Cfg,GenerationAttentionGlobals,EdgeBf16Op,EdgeFp8Op,ScheduledEdgePreparation,ScheduledEdgeAttention,ScheduledEdgeKVReady>;
    if(error==cudaSuccess)error=cudaFuncSetAttribute(kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,Cfg::DYNAMIC_SHARED_MEMORY);
    if(error==cudaSuccess){
        const auto& base=q.globals;
        EdgeDenseGlobals b{base.instructions,base.timings,base.queue,table,ns,nt,no,base.values,base.owners,base.ordinals,base.exits,base.visits,base.starts,base.ends};
        *handle=new GenerationAttentionContext{GenerationAttentionGlobals{b,dp,da},&q,&a,device,std::move(allocated)};
    }else cleanup();return error;
}
extern "C" int launch_generation_attention(void* handle,void* stream){
    if(!handle)return cudaErrorInvalidValue;auto& c=*static_cast<GenerationAttentionContext*>(handle);int device;
    auto error=cudaGetDevice(&device);if(error!=cudaSuccess)return error;if(device!=c.device)return cudaErrorInvalidDevice;
    cudaStreamCaptureStatus capture;error=cudaStreamIsCapturing(static_cast<cudaStream_t>(stream),&capture);
    if(error!=cudaSuccess)return error;if(capture!=cudaStreamCaptureStatusNone)return cudaErrorStreamCaptureUnsupported;
    unsigned epoch=std::max(c.queue->epoch,c.attention->globals.epoch);if(epoch==UINT_MAX)return cudaErrorInvalidValue;++epoch;
    c.queue->epoch=epoch;c.queue->globals.queue.epoch=epoch;c.globals.queue.epoch=epoch;
    c.attention->globals.epoch=epoch;c.attention->started=true;
    auto kernel=megakernel::mk<Cfg,GenerationAttentionGlobals,EdgeBf16Op,EdgeFp8Op,ScheduledEdgePreparation,ScheduledEdgeAttention,ScheduledEdgeKVReady>;
    kernel<<<c.queue->workers,Cfg::NUM_THREADS,Cfg::DYNAMIC_SHARED_MEMORY,static_cast<cudaStream_t>(stream)>>>(c.globals);return cudaGetLastError();
}
extern "C" unsigned generation_attention_epoch(void* handle){return handle?static_cast<GenerationAttentionContext*>(handle)->queue->epoch:0;}
extern "C" int destroy_generation_attention(void* handle){
    if(!handle)return cudaSuccess;auto* c=static_cast<GenerationAttentionContext*>(handle);int device;auto error=cudaGetDevice(&device);
    if(error!=cudaSuccess)return error;if(device!=c->device)return cudaErrorInvalidDevice;
    error=cudaDeviceSynchronize();if(error!=cudaSuccess)return error;
    for(auto p:c->allocations){auto e=cudaFree(p);if(error==cudaSuccess)error=e;}delete c;return error;
}
