#pragma once
// General model instruction schedule. Operators remain the existing VM roles.
#ifndef EDGE_ENGINE_NORM_ROWS
#define EDGE_ENGINE_NORM_ROWS 4
#endif
static_assert(EDGE_ENGINE_NORM_ROWS>0);
struct EdgeEngineGlobals:GenerationAttentionGlobals {};
struct EnginePreparation {
    static constexpr int opcode=5;
    using controller=megakernel::NoOp<Cfg>::controller;
    using loader=megakernel::NoOp<Cfg>::loader;
    using launcher=megakernel::NoOp<Cfg>::launcher;
    using storer=megakernel::NoOp<Cfg>::storer;
    struct consumer {static __device__ void run(const EdgeEngineGlobals& g,State& s){
        GenerationAttentionDiagnostics::start(g,s);
        const auto& p=*static_cast<const EdgeRopeGlobals*>(g.stages[s.instruction()[1]]);
        int first=s.instruction()[2]*EDGE_ROPE_VECTORS_PER_TASK,end=min(first+EDGE_ROPE_VECTORS_PER_TASK,p.rows*(p.q_heads+p.k_heads));
        for(int vec=first+kittens::warpid();vec<end;vec+=Cfg::NUM_CONSUMER_WARPS)EdgeRopeOp::vector(p,vec);
        GenerationAttentionDiagnostics::finish(g,s);
    }};
};
struct EngineAttention {
    static constexpr int opcode=6;
    using controller=megakernel::NoOp<Cfg>::controller;
    using launcher=megakernel::NoOp<Cfg>::launcher;
    using storer=megakernel::NoOp<Cfg>::storer;
    struct loader {static __device__ void run(const EdgeEngineGlobals& g,State& s){
        const auto& a=*static_cast<const AttentionGlobals*>(g.stages[s.instruction()[1]]);
        int used=AttentionOp::pages(a),lane=kittens::laneid();
        if(lane>=used && lane<Cfg::NUM_PAGES){int p=s.pid(lane);s.wait_page_ready(p);s.finish_page(p,4);}
    }};
    struct consumer {static __device__ void run(const EdgeEngineGlobals& g,State& s){
        GenerationAttentionDiagnostics::start(g,s);
        auto view=*static_cast<const AttentionGlobals*>(g.stages[s.instruction()[1]]);view.epoch=g.queue.epoch;
        AttentionOp::consume(view,s,s.instruction()[2]);GenerationAttentionDiagnostics::finish(g,s);
    }};
};
struct EngineNorm {
    static constexpr int opcode=8;
    using controller=megakernel::NoOp<Cfg>::controller;
    using loader=megakernel::NoOp<Cfg>::loader;
    using launcher=megakernel::NoOp<Cfg>::launcher;
    using storer=megakernel::NoOp<Cfg>::storer;
    struct consumer {static __device__ void run(const EdgeEngineGlobals& g,State& s){
        GenerationAttentionDiagnostics::start(g,s);
        const auto& norm=*static_cast<const NormGlobals*>(g.stages[s.instruction()[1]]);
        int first=s.instruction()[2]*EDGE_ENGINE_NORM_ROWS;
        for(int row=first;row<min(first+EDGE_ENGINE_NORM_ROWS,norm.rows);++row)NormOp::run_row(norm,s,row);
        GenerationAttentionDiagnostics::finish(g,s);
    }};
};
struct EdgeEngineContext {
    EdgeEngineGlobals globals;DagProbeContext* queue;std::vector<AttentionContext*> attention;
    int device;std::vector<void*> allocations;
};
extern "C" int edge_engine_norm_rows(){return EDGE_ENGINE_NORM_ROWS;}
extern "C" int create_edge_engine(void** handle,void* queue,int count,const int* kinds,void* const* handles,
    const int* node_stage,const int* node_task){
    if(!handle)return cudaErrorInvalidValue;*handle=nullptr;
    if(!queue||count<1||!kinds||!handles||!node_stage||!node_task)return cudaErrorInvalidValue;
    auto& q=*static_cast<DagProbeContext*>(queue);int device;auto error=cudaGetDevice(&device);
    if(error!=cudaSuccess)return error;if(device!=q.device)return cudaErrorInvalidDevice;
    std::vector<void*> objects,allocated;std::vector<int> tasks;std::vector<AttentionContext*> attention;
    auto cleanup=[&](){for(auto p:allocated)cudaFree(p);};
    auto upload=[&](const void* src,size_t bytes)->void*{
        void* dest=nullptr;if(error==cudaSuccess)error=cudaMalloc(&dest,bytes);
        if(dest)allocated.push_back(dest);if(error==cudaSuccess)error=cudaMemcpy(dest,src,bytes,cudaMemcpyHostToDevice);return dest;
    };
    for(int i=0;i<count;++i){
        int kind=kinds[i];void* ptr=handles[i];void* object=nullptr;int task_count=0;
        if(kind!=7&&!ptr){cleanup();return cudaErrorInvalidValue;}
        if(kind==3||kind==4){
            auto& h=*static_cast<EdgeProjectionHandle*>(ptr);
            if(h.precision!=kind-3){cleanup();return cudaErrorInvalidValue;}
            auto f=kind==3?edge_dense_info<kittens::bf16>(h.context):edge_dense_info<kittens::fp8e4m3>(h.context);
            if(f.device!=device){cleanup();return cudaErrorInvalidDevice;}
            error=kind==3?upload_edge_dense_stage<kittens::bf16>(h.context,nullptr,&object):upload_edge_dense_stage<kittens::fp8e4m3>(h.context,nullptr,&object);
            if(object)allocated.push_back(object);task_count=f.tasks;
        }else if(kind==5){
            const auto& c=*static_cast<EdgeRopeContext*>(ptr);if(c.device!=device){cleanup();return cudaErrorInvalidDevice;}
            object=upload(&c.globals,sizeof(c.globals));
            task_count=(c.globals.rows*(c.globals.q_heads+c.globals.k_heads)+EDGE_ROPE_VECTORS_PER_TASK-1)/EDGE_ROPE_VECTORS_PER_TASK;
        }else if(kind==6){
            auto& c=*static_cast<AttentionContext*>(ptr);if(c.device!=device){cleanup();return cudaErrorInvalidDevice;}
            object=upload(&c.globals,sizeof(c.globals));task_count=c.globals.tasks;attention.push_back(&c);
        }else if(kind==8){
            const auto& c=*static_cast<NormContext*>(ptr);if(c.device!=device){cleanup();return cudaErrorInvalidDevice;}
            object=upload(&c.globals,sizeof(c.globals));task_count=(c.globals.rows+EDGE_ENGINE_NORM_ROWS-1)/EDGE_ENGINE_NORM_ROWS;
        }else if(kind==7)task_count=1;
        else{cleanup();return cudaErrorInvalidValue;}
        if(error!=cudaSuccess){cleanup();return error;}objects.push_back(object);tasks.push_back(task_count);
    }
    int nodes=q.globals.queue.nodes;std::vector<int> opcodes(nodes);std::vector<std::vector<bool>> assigned;
    for(int task_count:tasks)assigned.emplace_back(task_count,false);
    for(int node=0;node<nodes;++node){
        int stage=node_stage[node],task=node_task[node];
        if(stage<0||stage>=count||task<0||task>=tasks[stage]||assigned[stage][task]){cleanup();return cudaErrorInvalidValue;}
        assigned[stage][task]=true;opcodes[node]=kinds[stage];
    }
    for(auto& group:assigned)for(bool seen:group)if(!seen){cleanup();return cudaErrorInvalidValue;}
    auto table=static_cast<const void**>(upload(objects.data(),size_t(count)*sizeof(void*)));
    auto ns=static_cast<int*>(upload(node_stage,size_t(nodes)*4)),nt=static_cast<int*>(upload(node_task,size_t(nodes)*4));
    auto no=static_cast<int*>(upload(opcodes.data(),size_t(nodes)*4));
    auto kernel=megakernel::mk<Cfg,EdgeEngineGlobals,EdgeBf16Op,EdgeFp8Op,EnginePreparation,EngineAttention,ScheduledEdgeKVReady,EngineNorm>;
    if(error==cudaSuccess)error=cudaFuncSetAttribute(kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,Cfg::DYNAMIC_SHARED_MEMORY);
    if(error==cudaSuccess){
        const auto& b=q.globals;
        EdgeDenseGlobals base{b.instructions,b.timings,b.queue,table,ns,nt,no,b.values,b.owners,b.ordinals,b.exits,b.visits,b.starts,b.ends};
        *handle=new EdgeEngineContext{EdgeEngineGlobals{GenerationAttentionGlobals{base,nullptr,nullptr}},&q,std::move(attention),device,std::move(allocated)};
    }else cleanup();return error;
}
extern "C" int launch_edge_engine(void* handle,void* stream){
    if(!handle)return cudaErrorInvalidValue;auto& c=*static_cast<EdgeEngineContext*>(handle);int device;
    auto error=cudaGetDevice(&device);if(error!=cudaSuccess)return error;if(device!=c.device)return cudaErrorInvalidDevice;
    cudaStreamCaptureStatus capture;error=cudaStreamIsCapturing(static_cast<cudaStream_t>(stream),&capture);
    if(error!=cudaSuccess)return error;if(capture!=cudaStreamCaptureStatusNone)return cudaErrorStreamCaptureUnsupported;
    unsigned epoch=c.queue->epoch;for(auto* a:c.attention)epoch=std::max(epoch,a->globals.epoch);
    if(epoch==UINT_MAX)return cudaErrorInvalidValue;++epoch;
    c.queue->epoch=epoch;c.queue->globals.queue.epoch=epoch;c.globals.queue.epoch=epoch;
    for(auto* a:c.attention){a->globals.epoch=epoch;a->started=true;}
    auto kernel=megakernel::mk<Cfg,EdgeEngineGlobals,EdgeBf16Op,EdgeFp8Op,EnginePreparation,EngineAttention,ScheduledEdgeKVReady,EngineNorm>;
    kernel<<<c.queue->workers,Cfg::NUM_THREADS,Cfg::DYNAMIC_SHARED_MEMORY,static_cast<cudaStream_t>(stream)>>>(c.globals);return cudaGetLastError();
}
extern "C" unsigned edge_engine_epoch(void* handle){return handle?static_cast<EdgeEngineContext*>(handle)->queue->epoch:0;}
extern "C" int destroy_edge_engine(void* handle){
    if(!handle)return cudaSuccess;auto* c=static_cast<EdgeEngineContext*>(handle);int device;auto error=cudaGetDevice(&device);
    if(error!=cudaSuccess)return error;if(device!=c->device)return cudaErrorInvalidDevice;
    error=cudaDeviceSynchronize();if(error!=cudaSuccess)return error;
    for(auto p:c->allocations){auto e=cudaFree(p);if(error==cudaSuccess)error=e;}delete c;return error;
}
