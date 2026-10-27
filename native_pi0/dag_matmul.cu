// Native matrix stages scheduled by the VM's dynamic ready queue.
#include "nvfp4_probe.cu"
#include "dag_probe.cu"
static_assert(Cfg::NUM_CONSUMER_WARPS==4);
struct DagMatmulGlobals {
    static constexpr bool DYNAMIC_INSTRUCTIONS=true;
    megakernel::instruction_layout<Cfg> instructions;
    RingTimings timings;
    native_pi0::DagScheduler queue;
    const Fp4Globals* stages;
    const int *node_stage,*node_task;
    unsigned *values,*owners,*ordinals,*exits;
    unsigned long long *visits,*starts,*ends;
    __device__ void fetch_instruction(int* instruction,int index)const{
        if(kittens::laneid()==0){
            int node=queue.claim();instruction[0]=node<0?-1:1;
            if(node<0)exits[blockIdx.x]=index;
            else {instruction[1]=node_stage[node];instruction[2]=node_task[node];instruction[3]=node;}
        }
        __syncwarp();
    }
};
struct DagMatmulOp {
    static constexpr int opcode=1;
    static __device__ const Fp4Globals& stage(const DagMatmulGlobals& g,State& s){return g.stages[s.instruction()[1]];}
    struct controller {
        static __device__ int release_lid(const DagMatmulGlobals&,Cfg::instruction_t&,int& q){return q;}
        static __device__ int init_semaphores(const DagMatmulGlobals& g,State& s){return Fp4Matmul::controller::init_semaphores(stage(g,s),s);}
    };
    struct loader {static __device__ void run(const DagMatmulGlobals& g,State& s){
        const auto& f=stage(g,s);
        if(kittens::laneid()==0){
            auto* a=f.a.template get_tma<PackedTile,2>();auto* b=f.b.template get_tma<PackedTile,2>();
            // Descriptors reside in global device stage storage, not a local
            // copied struct. Acquire their generic writes into the map proxy.
            asm volatile("fence.proxy.tensormap::generic.acquire.gpu [%0], 128;"::"l"(a):"memory");
            asm volatile("fence.proxy.tensormap::generic.acquire.gpu [%0], 128;"::"l"(b):"memory");
        }
        __syncwarp();Fp4Matmul::loader::run(f,s);
    }};
    struct launcher {static __device__ void run(const DagMatmulGlobals& g,State& s){Fp4Matmul::launcher::run(stage(g,s),s);}};
    using storer=megakernel::NoOp<Cfg>::storer;
    static __device__ void sync(){asm volatile("bar.sync 6, 128;":::"memory");}
    struct consumer {static __device__ void run(const DagMatmulGlobals& g,State& s){
        int node=s.instruction()[3];
        if(threadIdx.x==0){
            unsigned long long start;asm volatile("mov.u64 %0, %%globaltimer;":"=l"(start)::"memory");g.starts[node]=start;
            auto previous=atomicExch(g.visits+node,static_cast<unsigned long long>(g.queue.epoch));
            if(previous==g.queue.epoch)atomicAdd(&g.queue.control->errors,1u);
        }
        Fp4Matmul::consumer::run(stage(g,s),s);
        // All payload/scale writers participate before a successor is ready.
        asm volatile("fence.proxy.async.global;":::"memory");sync();
        if(threadIdx.x==0){
            g.values[node]=g.queue.epoch;g.owners[node]=blockIdx.x;g.ordinals[node]=s.instruction_index;
            unsigned long long end;asm volatile("mov.u64 %0, %%globaltimer;":"=l"(end)::"memory");g.ends[node]=end;
            g.queue.complete(node);
        }
        sync();
    }};
};
struct DagMatmulContext {
    DagProbeContext* queue;
    Fp4Globals* stages;
    int *node_stage,*node_task,device,workers;
    std::vector<Fp4Globals> host_stages;
};
extern "C" int create_dag_matmul(void** handle,void* queue_handle,int stage_count,void* const* stage_handles,
    const int* node_stage,const int* node_task,const int* source_stage){
    if(!handle)return cudaErrorInvalidValue;*handle=nullptr;
    if(!queue_handle||stage_count<1||!stage_handles||!node_stage||!node_task||!source_stage)return cudaErrorInvalidValue;
    auto& q=*static_cast<DagProbeContext*>(queue_handle);int device;auto error=cudaGetDevice(&device);
    if(error!=cudaSuccess)return error;
    if(device!=q.device)return cudaErrorInvalidDevice;
    std::vector<Fp4Globals> stages;stages.reserve(stage_count);
    for(int i=0;i<stage_count;++i){
        if(!stage_handles[i]||source_stage[i]<-1||source_stage[i]>=i)return cudaErrorInvalidValue;
        auto& c=*static_cast<ResidentFp4*>(stage_handles[i]);
        if(c.device!=device)return cudaErrorInvalidDevice;
        auto f=c.globals;
        if(source_stage[i]>=0){
            const auto& source=stages[source_stage[i]];
            int width=source.out.cols()/(source.output_gelu==2?2:1);
            if(!source.fp4_output||!source.output_scales||width!=f.a.cols()*2||source.out.rows()!=f.a.rows())return cudaErrorInvalidValue;
            f.a=PackedInput(reinterpret_cast<kittens::fp8e4m3*>(source.fp4_output),nullptr,nullptr,f.a.rows(),f.a.cols());
            f.sfa=source.output_scales;
        }
        f.task_from_instruction=true;stages.push_back(f);
    }
    std::vector<int> seen(size_t(stage_count),0);
    for(int node=0;node<q.globals.queue.nodes;++node){
        int i=node_stage[node],task=node_task[node];
        if(i<0||i>=stage_count||task<0||task>=stages[i].tasks)return cudaErrorInvalidValue;
        seen[i]++;
    }
    for(int i=0;i<stage_count;++i)if(seen[i]!=stages[i].tasks)return cudaErrorInvalidValue;
    Fp4Globals* device_stages=nullptr;int *ns=nullptr,*nt=nullptr;size_t bytes=size_t(q.globals.queue.nodes)*4;
    error=cudaMalloc(&device_stages,size_t(stage_count)*sizeof(Fp4Globals));
    if(error==cudaSuccess)error=cudaMemcpy(device_stages,stages.data(),size_t(stage_count)*sizeof(Fp4Globals),cudaMemcpyHostToDevice);
    if(error==cudaSuccess)error=cudaMalloc(&ns,bytes);
    if(error==cudaSuccess)error=cudaMalloc(&nt,bytes);
    if(error==cudaSuccess)error=cudaMemcpy(ns,node_stage,bytes,cudaMemcpyHostToDevice);
    if(error==cudaSuccess)error=cudaMemcpy(nt,node_task,bytes,cudaMemcpyHostToDevice);
    auto kernel=megakernel::mk<Cfg,DagMatmulGlobals,DagMatmulOp>;
    if(error==cudaSuccess)error=cudaFuncSetAttribute(kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,Cfg::DYNAMIC_SHARED_MEMORY);
    if(error==cudaSuccess)*handle=new DagMatmulContext{&q,device_stages,ns,nt,device,q.workers,std::move(stages)};
    else {cudaFree(device_stages);cudaFree(ns);cudaFree(nt);}
    return error;
}
extern "C" int launch_dag_matmul(void* handle,void* stream){
    if(!handle)return cudaErrorInvalidValue;auto& c=*static_cast<DagMatmulContext*>(handle);
    int device;auto error=cudaGetDevice(&device);
    if(error!=cudaSuccess)return error;
    if(device!=c.device)return cudaErrorInvalidDevice;
    cudaStreamCaptureStatus capture;error=cudaStreamIsCapturing(static_cast<cudaStream_t>(stream),&capture);
    if(error!=cudaSuccess)return error;
    if(capture!=cudaStreamCaptureStatusNone)return cudaErrorStreamCaptureUnsupported;
    if(c.queue->epoch==UINT_MAX)return cudaErrorInvalidValue;
    auto queue=c.queue->globals.queue;queue.epoch=++c.queue->epoch;c.queue->globals.queue.epoch=queue.epoch;
    auto& q=c.queue->globals;
    DagMatmulGlobals g{q.instructions,q.timings,queue,c.stages,c.node_stage,c.node_task,q.values,q.owners,q.ordinals,q.exits,q.visits,q.starts,q.ends};
    auto kernel=megakernel::mk<Cfg,DagMatmulGlobals,DagMatmulOp>;
    kernel<<<c.workers,Cfg::NUM_THREADS,Cfg::DYNAMIC_SHARED_MEMORY,static_cast<cudaStream_t>(stream)>>>(g);
    return cudaGetLastError();
}
extern "C" int destroy_dag_matmul(void* handle){
    if(!handle)return cudaSuccess;auto* c=static_cast<DagMatmulContext*>(handle);int device;auto error=cudaGetDevice(&device);
    if(error!=cudaSuccess)return error;
    if(device!=c->device)return cudaErrorInvalidDevice;
    for(void* p:{static_cast<void*>(c->stages),static_cast<void*>(c->node_stage),static_cast<void*>(c->node_task)}){
        auto e=cudaFree(p);if(error==cudaSuccess)error=e;
    }
    delete c;return error;
}
#ifdef NATIVE_DAG_MATMUL_VALIDATION
extern "C" int validation_split_dag_matmul(void* handle,void* stream){
    if(!handle)return cudaErrorInvalidValue;auto& c=*static_cast<DagMatmulContext*>(handle);
    for(auto g:c.host_stages){
        g.task_from_instruction=false;
        auto kernel=megakernel::mk<Cfg,Fp4Globals,Fp4Matmul>;
        kernel<<<g.workers,Cfg::NUM_THREADS,Cfg::DYNAMIC_SHARED_MEMORY,static_cast<cudaStream_t>(stream)>>>(g);
        auto error=cudaGetLastError();if(error!=cudaSuccess)return error;
    }
    return cudaSuccess;
}
#endif
