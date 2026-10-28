#pragma once
// BF16/FP8 projections use the same original VM and shared epoch-aware DAG.
struct EdgeDenseGlobals {
    static constexpr bool DYNAMIC_INSTRUCTIONS=true;
    megakernel::instruction_layout<Cfg> instructions;
    RingTimings timings;
    native_pi0::DagScheduler queue;
    const void* const* stages;
    const int *node_stage,*node_task,*node_opcode;
    unsigned *values,*owners,*ordinals,*exits;
    unsigned long long *visits,*starts,*ends;
    __device__ void fetch_instruction(int* instruction,int index)const{
        if(index==0)queue.initialize_warp();
        if(kittens::laneid()==0){
            int node=queue.claim(false);instruction[0]=node<0?-1:node_opcode[node];
            if(node<0)exits[blockIdx.x]=index;
            else{instruction[1]=node_stage[node];instruction[2]=node_task[node];instruction[3]=node;}
        }
        __syncwarp();
    }
};
template<class E> struct EdgeDenseView {
    using Tile=typename Globals<E>::Tile;
    struct InputMap {
        using identifier=kittens::ducks::gl::identifier;
        const CUtensorMap* map;int width;
        __device__ int cols()const{return width;}
        template<class T,int axis> __device__ const CUtensorMap* get_tma()const{
            static_assert(std::is_same_v<T,Tile> && axis==2);return map;
        }
    } a,b;
    Output out;float alpha;int workers,tasks,m_tiles,n_tiles,fixed_task;
    kittens::bf16* bf16_output;const kittens::bf16 *bias,*residual;int activation,round_before_residual;
    __device__ EdgeDenseView(const Globals<E>& f,int task):
        a{f.a.template get_tma<Tile,2>(),f.a.cols()},b{f.b.template get_tma<Tile,2>(),f.b.cols()},
        out(f.out),alpha(f.alpha),workers(f.workers),tasks(f.tasks),m_tiles(f.m_tiles),n_tiles(f.n_tiles),fixed_task(task),
        bf16_output(f.bf16_output),bias(f.bias),residual(f.residual),activation(f.activation),round_before_residual(f.round_before_residual){}
};
template<class E,int Opcode> struct EdgeDenseOp {
    static constexpr int opcode=Opcode;
    using G=Globals<E>;using Base=Matmul<E>;
    static __device__ const G& stage(const EdgeDenseGlobals& g,State& s){return *static_cast<const G*>(g.stages[s.instruction()[1]]);}
    struct controller {
        static __device__ int release_lid(const EdgeDenseGlobals&,Cfg::instruction_t&,int& q){return q;}
        static __device__ int init_semaphores(const EdgeDenseGlobals& g,State& s){return Base::controller::init_semaphores(stage(g,s),s);}
    };
    struct loader {static __device__ void run(const EdgeDenseGlobals& g,State& s){
        const auto& f=stage(g,s);
        if(kittens::laneid()==0){
            auto* a=f.a.template get_tma<typename G::Tile,2>();auto* b=f.b.template get_tma<typename G::Tile,2>();
            asm volatile("fence.proxy.tensormap::generic.acquire.gpu [%0], 128;"::"l"(a):"memory");
            asm volatile("fence.proxy.tensormap::generic.acquire.gpu [%0], 128;"::"l"(b):"memory");
        }
        __syncwarp();const EdgeDenseView<E> view(f,s.instruction()[2]);Base::loader::run(view,s);
    }};
    struct launcher {static __device__ void run(const EdgeDenseGlobals& g,State& s){const EdgeDenseView<E> view(stage(g,s),s.instruction()[2]);Base::launcher::run(view,s);}};
    using storer=megakernel::NoOp<Cfg>::storer;
    struct consumer {static __device__ void run(const EdgeDenseGlobals& g,State& s){
        int node=s.instruction()[3];
        if(threadIdx.x==0){
            unsigned long long start;asm volatile("mov.u64 %0, %%globaltimer;":"=l"(start)::"memory");g.starts[node]=start;
            auto old=atomicExch(g.visits+node,static_cast<unsigned long long>(g.queue.epoch));
            if(old==g.queue.epoch)atomicAdd(&g.queue.control->errors,1u);
        }
        const EdgeDenseView<E> view(stage(g,s),s.instruction()[2]);Base::consumer::run(view,s);
        asm volatile("fence.proxy.async.global;":::"memory");DagMatmulOp::sync();
        if(threadIdx.x==0){
            g.values[node]=g.queue.epoch;g.owners[node]=blockIdx.x;g.ordinals[node]=s.instruction_index;
            unsigned long long end;asm volatile("mov.u64 %0, %%globaltimer;":"=l"(end)::"memory");g.ends[node]=end;
            g.queue.complete(node);
        }
        DagMatmulOp::sync();
    }};
};
using EdgeBf16Op=EdgeDenseOp<kittens::bf16,3>;
using EdgeFp8Op=EdgeDenseOp<kittens::fp8e4m3,4>;
struct EdgeDenseContext {
    DagProbeContext* queue;const void** stages;int *node_stage,*node_task,*node_opcode;
    int device;std::vector<void*> allocations;
};
struct EdgeDenseInfo {int m,n,k,tasks,device;void* output;};
template<class E> EdgeDenseInfo edge_dense_info(void* context){
    const auto& c=*static_cast<EdgeProjection<E>*>(context);const auto& f=c.globals;
    return {f.a.rows(),f.out.cols(),f.a.cols(),f.tasks,c.device,f.bf16_output};
}
template<class E> cudaError_t upload_edge_dense_stage(void* context,void* source,void** destination){
    auto f=static_cast<EdgeProjection<E>*>(context)->globals;f.task_from_instruction=true;
    if(source){
        if constexpr(std::is_same_v<E,kittens::bf16>)f.a=typename Globals<E>::Input(static_cast<E*>(source),nullptr,nullptr,f.a.rows(),f.a.cols());
        else return cudaErrorInvalidValue;
    }
    auto error=cudaMalloc(destination,sizeof(f));
    if(error==cudaSuccess)error=cudaMemcpy(*destination,&f,sizeof(f),cudaMemcpyHostToDevice);
    return error;
}
extern "C" int create_edge_dense_dag(void** handle,void* queue,int stage_count,void* const* handles,
    const int* node_stage,const int* node_task,const int* sources){
    if(!handle)return cudaErrorInvalidValue;*handle=nullptr;
    if(!queue||stage_count<1||!handles||!node_stage||!node_task||!sources)return cudaErrorInvalidValue;
    auto& q=*static_cast<DagProbeContext*>(queue);int device;auto error=cudaGetDevice(&device);
    if(error!=cudaSuccess)return error;if(device!=q.device)return cudaErrorInvalidDevice;
    std::vector<EdgeDenseInfo> info;std::vector<void*> stages,allocations;
    auto cleanup=[&](){for(auto p:allocations)cudaFree(p);};
    for(int i=0;i<stage_count;++i){
        if(!handles[i]||sources[i]<-1||sources[i]>=i){cleanup();return cudaErrorInvalidValue;}
        auto& h=*static_cast<EdgeProjectionHandle*>(handles[i]);
        auto f=h.precision==0?edge_dense_info<kittens::bf16>(h.context):edge_dense_info<kittens::fp8e4m3>(h.context);
        if(f.device!=device){cleanup();return cudaErrorInvalidDevice;}
        void* source=nullptr;
        if(sources[i]>=0){
            const auto& parent=info[sources[i]];
            if(h.precision!=0||parent.m!=f.m||parent.n!=f.k){cleanup();return cudaErrorInvalidValue;}
            source=parent.output;
        }
        void* stage=nullptr;
        error=h.precision==0?upload_edge_dense_stage<kittens::bf16>(h.context,source,&stage):upload_edge_dense_stage<kittens::fp8e4m3>(h.context,source,&stage);
        if(stage)allocations.push_back(stage);
        if(error!=cudaSuccess){cleanup();return error;}
        stages.push_back(stage);info.push_back(f);
    }
    int nodes=q.globals.queue.nodes;std::vector<int> opcodes(nodes);std::vector<std::vector<bool>> assigned;
    for(auto& i:info)assigned.emplace_back(i.tasks,false);
    for(int node=0;node<nodes;++node){
        int stage=node_stage[node],task=node_task[node];
        if(stage<0||stage>=stage_count||task<0||task>=info[stage].tasks||assigned[stage][task]){cleanup();return cudaErrorInvalidValue;}
        assigned[stage][task]=true;opcodes[node]=static_cast<EdgeProjectionHandle*>(handles[stage])->precision==0?3:4;
    }
    for(auto& row:assigned)for(bool seen:row)if(!seen){cleanup();return cudaErrorInvalidValue;}
    auto upload=[&](const void* data,size_t bytes)->void*{
        void* p=nullptr;if(error==cudaSuccess)error=cudaMalloc(&p,bytes);
        if(p)allocations.push_back(p);if(error==cudaSuccess)error=cudaMemcpy(p,data,bytes,cudaMemcpyHostToDevice);return p;
    };
    auto table=static_cast<const void**>(upload(stages.data(),stages.size()*sizeof(void*)));
    auto ns=static_cast<int*>(upload(node_stage,size_t(nodes)*4));auto nt=static_cast<int*>(upload(node_task,size_t(nodes)*4));
    auto no=static_cast<int*>(upload(opcodes.data(),size_t(nodes)*4));
    auto kernel=megakernel::mk<Cfg,EdgeDenseGlobals,EdgeBf16Op,EdgeFp8Op>;
    if(error==cudaSuccess)error=cudaFuncSetAttribute(kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,Cfg::DYNAMIC_SHARED_MEMORY);
    if(error==cudaSuccess)*handle=new EdgeDenseContext{&q,table,ns,nt,no,device,std::move(allocations)};
    else cleanup();return error;
}
extern "C" int launch_edge_dense_dag(void* handle,void* stream){
    if(!handle)return cudaErrorInvalidValue;auto& c=*static_cast<EdgeDenseContext*>(handle);int device;
    auto error=cudaGetDevice(&device);if(error!=cudaSuccess)return error;if(device!=c.device)return cudaErrorInvalidDevice;
    cudaStreamCaptureStatus capture;error=cudaStreamIsCapturing(static_cast<cudaStream_t>(stream),&capture);
    if(error!=cudaSuccess)return error;if(capture!=cudaStreamCaptureStatusNone)return cudaErrorStreamCaptureUnsupported;
    if(c.queue->epoch==UINT_MAX)return cudaErrorInvalidValue;
    auto& q=c.queue->globals;auto queue=q.queue;queue.epoch=++c.queue->epoch;q.queue.epoch=queue.epoch;
    EdgeDenseGlobals g{q.instructions,q.timings,queue,c.stages,c.node_stage,c.node_task,c.node_opcode,
        q.values,q.owners,q.ordinals,q.exits,q.visits,q.starts,q.ends};
    auto kernel=megakernel::mk<Cfg,EdgeDenseGlobals,EdgeBf16Op,EdgeFp8Op>;
    kernel<<<c.queue->workers,Cfg::NUM_THREADS,Cfg::DYNAMIC_SHARED_MEMORY,static_cast<cudaStream_t>(stream)>>>(g);
    return cudaGetLastError();
}
extern "C" int destroy_edge_dense_dag(void* handle){
    if(!handle)return cudaSuccess;auto* c=static_cast<EdgeDenseContext*>(handle);int device;auto error=cudaGetDevice(&device);
    if(error!=cudaSuccess)return error;if(device!=c->device)return cudaErrorInvalidDevice;
    error=cudaDeviceSynchronize();if(error!=cudaSuccess)return error;
    for(auto p:c->allocations){auto e=cudaFree(p);if(error==cudaSuccess)error=e;}
    delete c;return error;
}
