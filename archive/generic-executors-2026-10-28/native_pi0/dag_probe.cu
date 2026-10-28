// Scalar dependency/VM retirement probe, not native model or GEMM validation.
#include "kittens.cuh"
#include "megakernel.cuh"
#include "dag_scheduler.cuh"
#include <vector>
#include <climits>
#include <map>
#include <algorithm>
struct DagConfig:megakernel::default_config {
    static constexpr int NUM_CONSUMER_WARPS=4,NUM_WARPS=8,NUM_THREADS=256,CONSUMER_REGISTERS=128;
};
using DS=megakernel::state<DagConfig>;
struct RingTimings {
    int* data;
    __device__ int& operator[](const kittens::coord<>& c)const{return data[c.d*256+(c.r%2)*128+c.c];}
};
struct DagGlobals {
    static constexpr bool DYNAMIC_INSTRUCTIONS=true;
    megakernel::instruction_layout<DagConfig> instructions;
    RingTimings timings;
    native_pi0::DagScheduler queue;
    const int *parent_offsets,*parents,*delays;
    unsigned *values,*owners,*ordinals,*exits;
    unsigned long long* visits;
    unsigned long long *starts,*ends;
    unsigned seed;
    __device__ void fetch_instruction(int* instruction,int index)const{
        if(index==0)queue.initialize_warp();
        if(kittens::laneid()==0){
            int node=queue.claim(false);
            instruction[0]=node<0?-1:1;instruction[1]=node;
            if(node<0)exits[blockIdx.x]=index;
        }
        __syncwarp();
    }
};
struct DagProbeOp {
    static constexpr int opcode=1;
    using controller=megakernel::NoOp<DagConfig>::controller;
    using loader=megakernel::NoOp<DagConfig>::loader;
    using launcher=megakernel::NoOp<DagConfig>::launcher;
    using storer=megakernel::NoOp<DagConfig>::storer;
    struct consumer {static __device__ void run(const DagGlobals& g,DS& s){
        if(threadIdx.x==0){
            int node=s.instruction()[1];
            unsigned long long start;asm volatile("mov.u64 %0, %%globaltimer;":"=l"(start)::"memory");g.starts[node]=start;
            auto previous=atomicExch(g.visits+node,static_cast<unsigned long long>(g.queue.epoch));
            if(previous==g.queue.epoch)atomicAdd(&g.queue.control->errors,1u);
            unsigned value=g.seed^(unsigned(node)*0x9e3779b9u);
            for(int i=g.parent_offsets[node];i<g.parent_offsets[node+1];++i)value=value*1664525u+g.values[g.parents[i]]+1013904223u;
            for(int i=0;i<g.delays[node];++i)value=value*1664525u+1013904223u;
            g.values[node]=value;g.owners[node]=blockIdx.x;g.ordinals[node]=s.instruction_index;
            unsigned long long end;asm volatile("mov.u64 %0, %%globaltimer;":"=l"(end)::"memory");g.ends[node]=end;
            g.queue.complete(node);
        }
    }};
};
__global__ void dag_slots(int* p){if(threadIdx.x==0)asm volatile("mov.u32 %0, %%nsmid;":"=r"(*p));}
struct DagProbeContext {
    DagGlobals globals;
    int workers,device;
    unsigned epoch=0;
    int dependency_links=0;
    std::vector<void*> allocations;
};
extern "C" int create_dag_probe(void** handle,int nodes,int edges,int root_count,int workers,
    const int* parent_offsets,const int* parents,const int* child_offsets,const int* children,const int* roots,const int* delays){
    if(!handle)return cudaErrorInvalidValue;*handle=nullptr;
    if(nodes<1||nodes==INT_MAX||edges<0||root_count<1||root_count>nodes||workers<1||!parent_offsets||!child_offsets||!roots||!delays
       ||(edges&&(!parents||!children)))return cudaErrorInvalidValue;
    int device;auto error=cudaGetDevice(&device);cudaDeviceProp properties{};
    if(error==cudaSuccess)error=cudaGetDeviceProperties(&properties,device);
    if(error!=cudaSuccess)return error;
    if(properties.major!=10||properties.minor!=0||workers>properties.multiProcessorCount)return cudaErrorInvalidConfiguration;
    // Host graph structure is validated by the Python wrapper before entering C.
    std::vector<void*> allocations;
    auto alloc=[&](size_t bytes)->void*{void* p=nullptr;if(error==cudaSuccess){error=cudaMalloc(&p,bytes);if(error==cudaSuccess)allocations.push_back(p);}return p;};
    auto upload=[&](const void* src,size_t bytes)->void*{void* p=alloc(bytes);if(error==cudaSuccess)error=cudaMemcpy(p,src,bytes,cudaMemcpyHostToDevice);return p;};
    int* slots_dev=static_cast<int*>(alloc(sizeof(int)));int slots=0;
    if(error==cudaSuccess){dag_slots<<<1,32>>>(slots_dev);error=cudaGetLastError();}
    if(error==cudaSuccess)error=cudaMemcpy(&slots,slots_dev,sizeof(int),cudaMemcpyDeviceToHost);
    int* po=static_cast<int*>(upload(parent_offsets,size_t(nodes+1)*4));
    int* pp=edges?static_cast<int*>(upload(parents,size_t(edges)*4)):nullptr;
    int* rr=static_cast<int*>(upload(roots,size_t(root_count)*4));
    int* dd=static_cast<int*>(upload(delays,size_t(nodes)*4));
    // Readiness depends on parent sets, even if a node's math uses parent order.
    // Consumers with identical prerequisites can share one completion counter.
    std::map<std::vector<int>,int> group_lookup;
    std::vector<int> indegree;
    std::vector<std::vector<int>> release_groups(nodes),ready_nodes;
    for(int node=0;node<nodes;++node){
        if(parent_offsets[node]==parent_offsets[node+1])continue;
        std::vector<int> key(parents+parent_offsets[node],parents+parent_offsets[node+1]);
        std::sort(key.begin(),key.end());int group=-1;
#if NATIVE_DAG_COALESCE_DEPENDENCIES
        auto found=group_lookup.find(key);if(found!=group_lookup.end())group=found->second;
#endif
        if(group<0){
            group=int(indegree.size());group_lookup[key]=group;indegree.push_back(int(key.size()));ready_nodes.emplace_back();
            for(int parent:key)release_groups[parent].push_back(group);
        }
        ready_nodes[group].push_back(node);
    }
    int group_count=int(indegree.size());if(indegree.empty())indegree.push_back(0);
    auto csr=[](const std::vector<std::vector<int>>& rows){
        std::pair<std::vector<int>,std::vector<int>> result;result.first.push_back(0);
        for(const auto& row:rows){result.second.insert(result.second.end(),row.begin(),row.end());result.first.push_back(int(result.second.size()));}
        return result;
    };
    auto releases=csr(release_groups),ready=csr(ready_nodes);
    int* co=static_cast<int*>(upload(releases.first.data(),releases.first.size()*4));
    int* cc=releases.second.empty()?nullptr:static_cast<int*>(upload(releases.second.data(),releases.second.size()*4));
    int* ro=static_cast<int*>(upload(ready.first.data(),ready.first.size()*4));
    int* rn=ready.second.empty()?nullptr:static_cast<int*>(upload(ready.second.data(),ready.second.size()*4));
    int* ii=static_cast<int*>(upload(indegree.data(),indegree.size()*4));
    auto* control=static_cast<native_pi0::DagControl*>(alloc(sizeof(native_pi0::DagControl)));
    auto* remaining=static_cast<unsigned long long*>(alloc(size_t(nodes)*8));
    auto* queue=static_cast<unsigned long long*>(alloc(size_t(nodes)*8));
    auto* visits=static_cast<unsigned long long*>(alloc(size_t(nodes)*8));
    auto* starts=static_cast<unsigned long long*>(alloc(size_t(nodes)*8));
    auto* ends=static_cast<unsigned long long*>(alloc(size_t(nodes)*8));
    auto* values=static_cast<unsigned*>(alloc(size_t(nodes)*4));
    auto* owners=static_cast<unsigned*>(alloc(size_t(nodes)*4));
    auto* ordinals=static_cast<unsigned*>(alloc(size_t(nodes)*4));
    auto* exits=static_cast<unsigned*>(alloc(size_t(workers)*4));
    auto* timing=static_cast<int*>(alloc(size_t(slots)*256*4));
    std::pair<void*,size_t> zeroes[]={{control,sizeof(*control)},{remaining,size_t(nodes)*8},{queue,size_t(nodes)*8},{visits,size_t(nodes)*8}};
    for(auto pair:zeroes)
        if(error==cudaSuccess)error=cudaMemset(pair.first,0,pair.second);
    if(error==cudaSuccess)error=cudaStreamSynchronize(nullptr);
    auto kernel=megakernel::mk<DagConfig,DagGlobals,DagProbeOp>;
    if(error==cudaSuccess)error=cudaFuncSetAttribute(kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,DagConfig::DYNAMIC_SHARED_MEMORY);
    if(error==cudaSuccess){
        DagGlobals g{megakernel::instruction_layout<DagConfig>(nullptr,nullptr,slots,nodes+1,nullptr),RingTimings{timing},
            native_pi0::DagScheduler{control,remaining,queue,ii,co,cc,rr,ro,rn,nodes,root_count,group_count,0},po,pp,dd,values,owners,ordinals,exits,visits,starts,ends,0};
        *handle=new DagProbeContext{g,workers,device,0,int(releases.second.size()),std::move(allocations)};
    }else for(void* p:allocations)cudaFree(p);
    return error;
}
extern "C" int launch_dag_probe(void* handle,void* stream,unsigned seed){
    if(!handle)return cudaErrorInvalidValue;
    auto& c=*static_cast<DagProbeContext*>(handle);
    int device;auto error=cudaGetDevice(&device);
    if(error!=cudaSuccess)return error;
    if(device!=c.device)return cudaErrorInvalidDevice;
    cudaStreamCaptureStatus capture;error=cudaStreamIsCapturing(static_cast<cudaStream_t>(stream),&capture);
    if(error!=cudaSuccess)return error;
    if(capture!=cudaStreamCaptureStatusNone)return cudaErrorStreamCaptureUnsupported;
    if(c.epoch==UINT_MAX)return cudaErrorInvalidValue;
    c.globals.queue.epoch=++c.epoch;c.globals.seed=seed;
    auto kernel=megakernel::mk<DagConfig,DagGlobals,DagProbeOp>;
    kernel<<<c.workers,DagConfig::NUM_THREADS,DagConfig::DYNAMIC_SHARED_MEMORY,static_cast<cudaStream_t>(stream)>>>(c.globals);
    return cudaGetLastError();
}
extern "C" int read_dag_probe(void* handle,void* values,void* owners,void* ordinals,void* visits,void* exits,void* starts,void* ends,unsigned* counts){
    if(!handle||!values||!owners||!ordinals||!visits||!exits||!starts||!ends||!counts)return cudaErrorInvalidValue;
    auto& c=*static_cast<DagProbeContext*>(handle);
    auto error=cudaDeviceSynchronize();int n=c.globals.queue.nodes;
    auto copy=[&](void* dst,const void* src,size_t bytes){if(error==cudaSuccess)error=cudaMemcpy(dst,src,bytes,cudaMemcpyDeviceToHost);};
    copy(values,c.globals.values,size_t(n)*4);copy(owners,c.globals.owners,size_t(n)*4);copy(ordinals,c.globals.ordinals,size_t(n)*4);
    copy(visits,c.globals.visits,size_t(n)*8);copy(exits,c.globals.exits,size_t(c.workers)*4);
    copy(starts,c.globals.starts,size_t(n)*8);copy(ends,c.globals.ends,size_t(n)*8);
    native_pi0::DagControl control{};copy(&control,c.globals.queue.control,sizeof(control));
    counts[0]=control.head;counts[1]=control.tail;counts[2]=control.completed;counts[3]=control.errors;counts[4]=c.epoch;
    return error;
}
extern "C" int destroy_dag_probe(void* handle){
    if(!handle)return cudaSuccess;
    auto* c=static_cast<DagProbeContext*>(handle);int device;auto error=cudaGetDevice(&device);
    if(error!=cudaSuccess)return error;
    if(device!=c->device)return cudaErrorInvalidDevice;
    for(void* p:c->allocations){auto e=cudaFree(p);if(error==cudaSuccess)error=e;}
    delete c;return error;
}
extern "C" int dag_probe_resources(int* values,int count){
    if(!values||count<6)return cudaErrorInvalidValue;
    int device,blocks=0;cudaFuncAttributes attributes{};cudaDeviceProp properties{};
    auto error=cudaGetDevice(&device);
    auto kernel=megakernel::mk<DagConfig,DagGlobals,DagProbeOp>;
    if(error==cudaSuccess)error=cudaGetDeviceProperties(&properties,device);
    if(error==cudaSuccess)error=cudaFuncSetAttribute(kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,DagConfig::DYNAMIC_SHARED_MEMORY);
    if(error==cudaSuccess)error=cudaFuncGetAttributes(&attributes,kernel);
    if(error==cudaSuccess)error=cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,kernel,DagConfig::NUM_THREADS,DagConfig::DYNAMIC_SHARED_MEMORY);
    if(error!=cudaSuccess)return error;
    values[0]=blocks;values[1]=properties.multiProcessorCount;
    values[2]=attributes.sharedSizeBytes;values[3]=DagConfig::DYNAMIC_SHARED_MEMORY;
    values[4]=attributes.numRegs;values[5]=attributes.localSizeBytes;
    return cudaSuccess;
}
extern "C" int dag_probe_plan(void* handle,int* values,int count){
    if(!handle||!values||count<3)return cudaErrorInvalidValue;
    const auto& c=*static_cast<DagProbeContext*>(handle);
    values[0]=c.globals.queue.groups;values[1]=c.dependency_links;values[2]=c.globals.queue.nodes;
    return cudaSuccess;
}
