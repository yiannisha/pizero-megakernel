#pragma once
#include <cuda_runtime.h>
#include <cstdint>

namespace native_pi0 {
__device__ inline unsigned long long acquire64(const unsigned long long* p){
    unsigned long long v;asm volatile("ld.acquire.gpu.global.u64 %0, [%1];":"=l"(v):"l"(p):"memory");return v;
}
__device__ inline void release64(unsigned long long* p,unsigned long long v){
    asm volatile("st.release.gpu.global.u64 [%0], %1;"::"l"(p),"l"(v):"memory");
}
__device__ inline unsigned long long cas_acq_rel(unsigned long long* p,unsigned long long old,unsigned long long value){
    unsigned long long result;
    asm volatile("atom.acq_rel.gpu.global.cas.b64 %0, [%1], %2, %3;":"=l"(result):"l"(p),"l"(old),"l"(value):"memory");
    return result;
}
__device__ inline unsigned acquire32(const unsigned* p){
    unsigned v;asm volatile("ld.acquire.gpu.global.u32 %0, [%1];":"=r"(v):"l"(p):"memory");return v;
}
__device__ inline unsigned increment_acq_rel(unsigned* p){
    unsigned v;asm volatile("atom.acq_rel.gpu.global.add.u32 %0, [%1], 1;":"=r"(v):"l"(p):"memory");return v;
}
struct alignas(64) DagControl {
    unsigned long long phase;
    unsigned head,tail,completed,errors;
};
// One enqueue per node, so queue tickets never wrap within an invocation.
// The host validates a DAG; the epoch advances as a scalar launch argument.
struct DagScheduler {
    DagControl* control;
    unsigned long long *remaining,*slots;
    const int *indegree,*child_offsets,*children,*roots;
    int nodes,root_count;
    unsigned epoch;
    __device__ unsigned long long tag(unsigned low)const{return (static_cast<unsigned long long>(epoch)<<32)|low;}
    __device__ void initialize()const{
        auto busy=2ull*epoch,ready=busy+1,old=acquire64(&control->phase);
        if(old<busy && cas_acq_rel(&control->phase,old,busy)==old){
            control->head=0;control->tail=root_count;control->completed=0;control->errors=0;
            for(int i=0;i<root_count;++i)release64(slots+i,tag(unsigned(roots[i])+1));
            release64(&control->phase,ready);
        }
        while(acquire64(&control->phase)!=ready)__nanosleep(32);
    }
    __device__ int claim()const{
        initialize();
        while(true){
            unsigned head=atomicAdd(&control->head,0u),tail=atomicAdd(&control->tail,0u);
            if(head<tail && atomicCAS(&control->head,head,head+1)==head){
                unsigned long long slot;
                do{slot=acquire64(slots+head);if(unsigned(slot>>32)!=epoch)__nanosleep(32);}while(unsigned(slot>>32)!=epoch);
                return int(unsigned(slot))-1;
            }
            if(acquire32(&control->completed)==unsigned(nodes))return -1;
            __nanosleep(32);
        }
    }
    __device__ void enqueue(int node)const{
        unsigned ticket=atomicAdd(&control->tail,1u);
        if(ticket>=unsigned(nodes))asm volatile("trap;");
        release64(slots+ticket,tag(unsigned(node)+1));
    }
    // Call only after all output writers have synchronized, including the
    // appropriate proxy fence if the successor will use TMA or tensor memory.
    __device__ void complete(int node)const{
        for(int edge=child_offsets[node];edge<child_offsets[node+1];++edge){
            int child=children[edge];auto old=acquire64(remaining+child);
            while(true){
                unsigned count=unsigned(old>>32)==epoch?unsigned(old):unsigned(indegree[child]);
                if(count==0)asm volatile("trap;");
                auto desired=tag(count-1),observed=cas_acq_rel(remaining+child,old,desired);
                if(observed==old){if(count==1)enqueue(child);break;}
                old=observed;
            }
        }
        increment_acq_rel(&control->completed);
    }
};
} // namespace native_pi0
