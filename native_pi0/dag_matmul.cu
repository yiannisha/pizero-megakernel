// Native matrix stages scheduled by the VM's dynamic ready queue.
#include "nvfp4_probe.cu"
#include "dag_probe.cu"
#include "norm.cu"
#ifndef NATIVE_DAG_NORM_ROWS
#define NATIVE_DAG_NORM_ROWS 4
#endif
static_assert(NATIVE_DAG_NORM_ROWS==1||NATIVE_DAG_NORM_ROWS==4||NATIVE_DAG_NORM_ROWS==8);
static_assert(Cfg::NUM_CONSUMER_WARPS==4);
#ifndef NATIVE_DAG_PER_TASK_TENSORMAP_ACQUIRE
#define NATIVE_DAG_PER_TASK_TENSORMAP_ACQUIRE 0
#endif
#ifndef NATIVE_DAG_CACHE_COMPUTE_METADATA
#define NATIVE_DAG_CACHE_COMPUTE_METADATA 1
#endif
// Compute roles need scalar metadata, not copies of the embedded tensor maps.
// A local immutable view prevents output stores/asm memory clobbers from
// forcing repeated global descriptor and shared task-index loads.
struct Fp4ComputeView {
    struct InputMap {
        using identifier=kittens::ducks::gl::identifier;
        const CUtensorMap* map;
        int width;
        __device__ int cols()const{return width;}
        template<class Tile,int axis> __device__ const CUtensorMap* get_tma()const{
            static_assert(std::is_same_v<Tile,PackedTile> && axis==2);return map;
        }
    } a,b;
    const unsigned char *sfa,*sfb;
    Output out;
    float alpha;
    int workers,tasks,m_tiles,n_tiles,fixed_task;
    const kittens::bf16 *bias,*residual;
    kittens::bf16* bf16_output;
    unsigned char *fp4_output,*output_scales;
    float output_global_scale;
    int output_gelu;
    __device__ Fp4ComputeView(const Fp4Globals& g,int task):
        a{g.a.template get_tma<PackedTile,2>(),g.a.cols()},b{g.b.template get_tma<PackedTile,2>(),g.b.cols()},
        sfa(g.sfa),sfb(g.sfb),out(g.out),alpha(g.alpha),
        workers(g.workers),tasks(g.tasks),m_tiles(g.m_tiles),n_tiles(g.n_tiles),fixed_task(task),
        bias(g.bias),residual(g.residual),bf16_output(g.bf16_output),fp4_output(g.fp4_output),
        output_scales(g.output_scales),output_global_scale(g.output_global_scale),output_gelu(g.output_gelu){}
};
struct DagMatmulGlobals {
    static constexpr bool DYNAMIC_INSTRUCTIONS=true;
    megakernel::instruction_layout<Cfg> instructions;
    RingTimings timings;
    native_pi0::DagScheduler queue;
    const Fp4Globals* stages;
    const NormGlobals* normalization;
    unsigned* map_epochs;
    int stage_count;
    const int *node_stage,*node_task;
    unsigned *values,*owners,*ordinals,*exits;
    unsigned long long *visits,*starts,*ends;
    __device__ void fetch_instruction(int* instruction,int index)const{
        if(index==0)queue.initialize_warp();
        if(kittens::laneid()==0){
            int node=queue.claim(false);instruction[0]=node<0?-1:1;
            if(node<0)exits[blockIdx.x]=index;
            else {instruction[0]=node_stage[node]<0?2:1;instruction[1]=node_stage[node];instruction[2]=node_task[node];instruction[3]=node;}
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
            auto* seen=g.map_epochs+size_t(blockIdx.x)*g.stage_count+s.instruction()[1];
#if NATIVE_DAG_PER_TASK_TENSORMAP_ACQUIRE
            bool acquire=true;
#else
            // One loader lane owns this CTA/stage entry. Immutable maps need
            // acquisition once per invocation, even if the CTA revisits a stage.
            bool acquire=*seen!=g.queue.epoch;
#endif
            if(acquire){
            auto* a=f.a.template get_tma<PackedTile,2>();auto* b=f.b.template get_tma<PackedTile,2>();
            // Descriptors reside in global device stage storage, not a local
            // copied struct. Acquire their generic writes into the map proxy.
            asm volatile("fence.proxy.tensormap::generic.acquire.gpu [%0], 128;"::"l"(a):"memory");
            asm volatile("fence.proxy.tensormap::generic.acquire.gpu [%0], 128;"::"l"(b):"memory");
#if !NATIVE_DAG_PER_TASK_TENSORMAP_ACQUIRE
            *seen=g.queue.epoch;
#endif
            }
        }
        __syncwarp();
#if NATIVE_DAG_CACHE_COMPUTE_METADATA
        // Cache scalar fields and remote map addresses. Copying the map objects
        // into a local view would produce invalid TMA descriptor addresses.
        const Fp4ComputeView view(f,s.instruction()[2]);Fp4Matmul::loader::run(view,s);
#else
        Fp4Matmul::loader::run(f,s);
#endif
    }};
    struct launcher {static __device__ void run(const DagMatmulGlobals& g,State& s){
#if NATIVE_DAG_CACHE_COMPUTE_METADATA
        const Fp4ComputeView view(stage(g,s),s.instruction()[2]);Fp4Matmul::launcher::run(view,s);
#else
        Fp4Matmul::launcher::run(stage(g,s),s);
#endif
    }};
    using storer=megakernel::NoOp<Cfg>::storer;
    static __device__ void sync(){asm volatile("bar.sync 6, 128;":::"memory");}
    struct consumer {static __device__ void run(const DagMatmulGlobals& g,State& s){
        int node=s.instruction()[3];
        if(threadIdx.x==0){
            unsigned long long start;asm volatile("mov.u64 %0, %%globaltimer;":"=l"(start)::"memory");g.starts[node]=start;
            auto previous=atomicExch(g.visits+node,static_cast<unsigned long long>(g.queue.epoch));
            if(previous==g.queue.epoch)atomicAdd(&g.queue.control->errors,1u);
        }
#if NATIVE_DAG_CACHE_COMPUTE_METADATA
        const Fp4ComputeView view(stage(g,s),s.instruction()[2]);Fp4Matmul::consumer::run(view,s);
#else
        Fp4Matmul::consumer::run(stage(g,s),s);
#endif
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
struct DagNormOp {
    static constexpr int opcode=2;
    using controller=megakernel::NoOp<Cfg>::controller;
    using loader=megakernel::NoOp<Cfg>::loader;
    using launcher=megakernel::NoOp<Cfg>::launcher;
    using storer=megakernel::NoOp<Cfg>::storer;
    struct consumer {static __device__ void run(const DagMatmulGlobals& g,State& s){
        int node=s.instruction()[3],first=s.instruction()[2]*NATIVE_DAG_NORM_ROWS;
        if(threadIdx.x==0){
            unsigned long long start;asm volatile("mov.u64 %0, %%globaltimer;":"=l"(start)::"memory");g.starts[node]=start;
            auto previous=atomicExch(g.visits+node,static_cast<unsigned long long>(g.queue.epoch));
            if(previous==g.queue.epoch)atomicAdd(&g.queue.control->errors,1u);
        }
        const NormGlobals norm=*g.normalization;
        for(int row=first;row<min(first+NATIVE_DAG_NORM_ROWS,norm.rows);++row)NormOp::run_row(norm,s,row);
        asm volatile("fence.proxy.async.global;":::"memory");DagMatmulOp::sync();
        if(threadIdx.x==0){
            g.values[node]=g.queue.epoch;g.owners[node]=blockIdx.x;g.ordinals[node]=s.instruction_index;
            unsigned long long end;asm volatile("mov.u64 %0, %%globaltimer;":"=l"(end)::"memory");g.ends[node]=end;
            g.queue.complete(node);
        }
        DagMatmulOp::sync();
    }};
};
struct DagMatmulContext {
    DagProbeContext* queue;
    Fp4Globals* stages;
    int *node_stage,*node_task,device,workers;
    unsigned* map_epochs;
    std::vector<Fp4Globals> host_stages;
    NormContext* normalization;
    NormGlobals* norm_device;
};
int create_dag_matmul_impl(void** handle,void* queue_handle,int stage_count,void* const* stage_handles,
    const int* node_stage,const int* node_task,const int* source_stage,NormContext* normalization){
    if(!handle)return cudaErrorInvalidValue;*handle=nullptr;
    if(!queue_handle||stage_count<1||!stage_handles||!node_stage||!node_task||!source_stage)return cudaErrorInvalidValue;
    auto& q=*static_cast<DagProbeContext*>(queue_handle);int device;auto error=cudaGetDevice(&device);
    if(error!=cudaSuccess)return error;
    if(device!=q.device)return cudaErrorInvalidDevice;
    if(normalization && (normalization->device!=device||normalization->globals.format!=4))return cudaErrorInvalidValue;
    std::vector<Fp4Globals> stages;stages.reserve(stage_count);
    for(int i=0;i<stage_count;++i){
        if(!stage_handles[i]||source_stage[i]<-2||source_stage[i]>=i)return cudaErrorInvalidValue;
        auto& c=*static_cast<ResidentFp4*>(stage_handles[i]);
        if(c.device!=device)return cudaErrorInvalidDevice;
        auto f=c.globals;
        if(source_stage[i]==-2){
            if(!normalization)return cudaErrorInvalidValue;
            const auto& norm=normalization->globals;
            if(norm.width!=f.a.cols()*2||int64_t(f.a.rows())!=(int64_t(norm.rows)+127)/128*128)return cudaErrorInvalidValue;
            f.a=PackedInput(reinterpret_cast<kittens::fp8e4m3*>(norm.output),nullptr,nullptr,f.a.rows(),f.a.cols());f.sfa=norm.scales;
        }else if(source_stage[i]>=0){
            const auto& source=stages[source_stage[i]];
            int width=source.out.cols()/(source.output_gelu==2?2:1);
            if(!source.fp4_output||!source.output_scales||width!=f.a.cols()*2||source.out.rows()!=f.a.rows())return cudaErrorInvalidValue;
            f.a=PackedInput(reinterpret_cast<kittens::fp8e4m3*>(source.fp4_output),nullptr,nullptr,f.a.rows(),f.a.cols());
            f.sfa=source.output_scales;
        }
        f.task_from_instruction=true;stages.push_back(f);
    }
    std::vector<std::vector<unsigned char>> assigned;
    for(const auto& stage:stages)assigned.emplace_back(size_t(stage.tasks),0);
    std::vector<int> seen(size_t(stage_count),0);
    int norm_groups=normalization?(normalization->globals.rows+NATIVE_DAG_NORM_ROWS-1)/NATIVE_DAG_NORM_ROWS:0;
    std::vector<unsigned char> norm_assigned(norm_groups,0);
    for(int node=0;node<q.globals.queue.nodes;++node){
        int i=node_stage[node],task=node_task[node];
        if(i==-1){
            if(!normalization||task<0||task>=norm_groups||norm_assigned[task])return cudaErrorInvalidValue;
            norm_assigned[task]=1;continue;
        }
        if(i<0||i>=stage_count||task<0||task>=stages[i].tasks)return cudaErrorInvalidValue;
        if(assigned[i][task])return cudaErrorInvalidValue;
        assigned[i][task]=1;
        seen[i]++;
    }
    for(int i=0;i<stage_count;++i)if(seen[i]!=stages[i].tasks)return cudaErrorInvalidValue;
    if(normalization && std::count(norm_assigned.begin(),norm_assigned.end(),1)!=norm_groups)return cudaErrorInvalidValue;
    NormGlobals* norm_device=nullptr;
    Fp4Globals* device_stages=nullptr;int *ns=nullptr,*nt=nullptr;unsigned* map_epochs=nullptr;size_t bytes=size_t(q.globals.queue.nodes)*4;
    error=cudaMalloc(&device_stages,size_t(stage_count)*sizeof(Fp4Globals));
    if(error==cudaSuccess)error=cudaMemcpy(device_stages,stages.data(),size_t(stage_count)*sizeof(Fp4Globals),cudaMemcpyHostToDevice);
    if(error==cudaSuccess)error=cudaMalloc(&ns,bytes);
    if(error==cudaSuccess)error=cudaMalloc(&nt,bytes);
    if(error==cudaSuccess)error=cudaMemcpy(ns,node_stage,bytes,cudaMemcpyHostToDevice);
    if(error==cudaSuccess)error=cudaMemcpy(nt,node_task,bytes,cudaMemcpyHostToDevice);
    if(normalization && error==cudaSuccess)error=cudaMalloc(&norm_device,sizeof(NormGlobals));
    if(normalization && error==cudaSuccess)error=cudaMemcpy(norm_device,&normalization->globals,sizeof(NormGlobals),cudaMemcpyHostToDevice);
    size_t map_bytes=size_t(q.workers)*stage_count*sizeof(unsigned);
    if(error==cudaSuccess)error=cudaMalloc(&map_epochs,map_bytes);
    if(error==cudaSuccess)error=cudaMemset(map_epochs,0,map_bytes);
    if(error==cudaSuccess)error=cudaStreamSynchronize(nullptr);
    auto kernel=megakernel::mk<Cfg,DagMatmulGlobals,DagMatmulOp,DagNormOp>;
    if(error==cudaSuccess)error=cudaFuncSetAttribute(kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,Cfg::DYNAMIC_SHARED_MEMORY);
    if(error==cudaSuccess)*handle=new DagMatmulContext{&q,device_stages,ns,nt,device,q.workers,map_epochs,std::move(stages),normalization,norm_device};
    else {cudaFree(device_stages);cudaFree(ns);cudaFree(nt);cudaFree(map_epochs);cudaFree(norm_device);}
    return error;
}
extern "C" int create_dag_matmul(void** h,void* q,int count,void* const* stages,const int* ns,const int* nt,const int* sources){
    return create_dag_matmul_impl(h,q,count,stages,ns,nt,sources,nullptr);
}
extern "C" int create_dag_norm_matmul(void** h,void* q,int count,void* const* stages,const int* ns,const int* nt,const int* sources,void* norm){
    if(!norm)return cudaErrorInvalidValue;
    return create_dag_matmul_impl(h,q,count,stages,ns,nt,sources,static_cast<NormContext*>(norm));
}
extern "C" int dag_norm_group_rows(){return NATIVE_DAG_NORM_ROWS;}
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
    DagMatmulGlobals g{q.instructions,q.timings,queue,c.stages,c.norm_device,c.map_epochs,int(c.host_stages.size()),c.node_stage,c.node_task,q.values,q.owners,q.ordinals,q.exits,q.visits,q.starts,q.ends};
    auto kernel=megakernel::mk<Cfg,DagMatmulGlobals,DagMatmulOp,DagNormOp>;
    kernel<<<c.workers,Cfg::NUM_THREADS,Cfg::DYNAMIC_SHARED_MEMORY,static_cast<cudaStream_t>(stream)>>>(g);
    return cudaGetLastError();
}
extern "C" int destroy_dag_matmul(void* handle){
    if(!handle)return cudaSuccess;auto* c=static_cast<DagMatmulContext*>(handle);int device;auto error=cudaGetDevice(&device);
    if(error!=cudaSuccess)return error;
    if(device!=c->device)return cudaErrorInvalidDevice;
    for(void* p:{static_cast<void*>(c->stages),static_cast<void*>(c->node_stage),static_cast<void*>(c->node_task),static_cast<void*>(c->map_epochs),static_cast<void*>(c->norm_device)}){
        auto e=cudaFree(p);if(error==cudaSuccess)error=e;
    }
    delete c;return error;
}
extern "C" int dag_matmul_resources(int* values,int count){
    if(!values||count<6)return cudaErrorInvalidValue;
    int device,blocks=0;cudaFuncAttributes attributes{};cudaDeviceProp properties{};
    auto error=cudaGetDevice(&device);auto kernel=megakernel::mk<Cfg,DagMatmulGlobals,DagMatmulOp,DagNormOp>;
    if(error==cudaSuccess)error=cudaGetDeviceProperties(&properties,device);
    if(error==cudaSuccess)error=cudaFuncSetAttribute(kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,Cfg::DYNAMIC_SHARED_MEMORY);
    if(error==cudaSuccess)error=cudaFuncGetAttributes(&attributes,kernel);
    if(error==cudaSuccess)error=cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,kernel,Cfg::NUM_THREADS,Cfg::DYNAMIC_SHARED_MEMORY);
    if(error!=cudaSuccess)return error;
    values[0]=blocks;values[1]=properties.multiProcessorCount;values[2]=attributes.sharedSizeBytes;
    values[3]=Cfg::DYNAMIC_SHARED_MEMORY;values[4]=attributes.numRegs;values[5]=attributes.localSizeBytes;
    return cudaSuccess;
}
#ifdef NATIVE_DAG_MATMUL_VALIDATION
extern "C" int validation_split_dag_matmul(void* handle,void* stream){
    if(!handle)return cudaErrorInvalidValue;auto& c=*static_cast<DagMatmulContext*>(handle);
    if(c.normalization){int error=launch_norm(c.normalization,stream);if(error!=cudaSuccess)return error;}
    for(auto g:c.host_stages){
        g.task_from_instruction=false;
        auto kernel=megakernel::mk<Cfg,Fp4Globals,Fp4Matmul>;
        kernel<<<g.workers,Cfg::NUM_THREADS,Cfg::DYNAMIC_SHARED_MEMORY,static_cast<cudaStream_t>(stream)>>>(g);
        auto error=cudaGetLastError();if(error!=cudaSuccess)return error;
    }
    return cudaSuccess;
}
#endif
