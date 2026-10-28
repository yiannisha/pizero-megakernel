#pragma once
#include <cuda_runtime.h>
#include <cstdint>
#ifndef NATIVE_DAG_BASELINE_QUEUE
#define NATIVE_DAG_BASELINE_QUEUE 0
#endif
#ifndef NATIVE_DAG_IDLE_SLEEP_NS
#define NATIVE_DAG_IDLE_SLEEP_NS 512
#endif
#ifndef NATIVE_DAG_CAS_DECREMENT
#define NATIVE_DAG_CAS_DECREMENT 0
#endif
#ifndef NATIVE_DAG_COALESCE_DEPENDENCIES
#define NATIVE_DAG_COALESCE_DEPENDENCIES 1
#endif
#ifndef NATIVE_DAG_WARP_BOOTSTRAP
#define NATIVE_DAG_WARP_BOOTSTRAP 1
#endif

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
__device__ inline unsigned long long decrement_acq_rel(unsigned long long* p){
    unsigned long long v;
    asm volatile("atom.acq_rel.gpu.global.add.u64 %0, [%1], %2;":"=l"(v):"l"(p),"l"(~0ull):"memory");return v;
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
    const int *ready_offsets,*ready_nodes;
    int nodes,root_count,groups;
    unsigned epoch;
    __device__ unsigned long long tag(unsigned low)const{return (static_cast<unsigned long long>(epoch)<<32)|low;}
    __device__ void initialize()const{
        auto busy=2ull*epoch,ready=busy+1,old=acquire64(&control->phase);
        if(old<busy && cas_acq_rel(&control->phase,old,busy)==old){
            control->head=0;control->tail=root_count;control->completed=0;control->errors=0;
            for(int i=0;i<root_count;++i){
#if NATIVE_DAG_BASELINE_QUEUE
                release64(slots+i,tag(unsigned(roots[i])+1));
#else
                // No dequeuer can pass the phase acquire until this initializer
                // publishes ready below, so one release covers all root slots.
                slots[i]=tag(unsigned(roots[i])+1);
#endif
            }
            release64(&control->phase,ready);
        }
        while(acquire64(&control->phase)!=ready)__nanosleep(32);
    }
    __device__ void initialize_warp()const{
#if NATIVE_DAG_WARP_BOOTSTRAP
        int lane=threadIdx.x%32,winner=0;auto busy=2ull*epoch,ready=busy+1;
        if(lane==0){
            auto old=acquire64(&control->phase);
            if(old<busy && cas_acq_rel(&control->phase,old,busy)==old){
                control->head=0;control->tail=root_count;control->completed=0;control->errors=0;winner=1;
            }
        }
        winner=__shfl_sync(0xffffffff,winner,0);
        if(winner){
            for(int i=lane;i<root_count;i+=32){
#if NATIVE_DAG_BASELINE_QUEUE
                release64(slots+i,tag(unsigned(roots[i])+1));
#else
                slots[i]=tag(unsigned(roots[i])+1);
#endif
            }
            // All root writers synchronize before the leader's release flag.
            __syncwarp();
            if(lane==0)release64(&control->phase,ready);
        }
        if(lane==0)while(acquire64(&control->phase)!=ready)__nanosleep(32);
        __syncwarp();
#else
        if(threadIdx.x%32==0)initialize();
        __syncwarp();
#endif
    }
    __device__ int claim(bool bootstrap=true)const{
        if(bootstrap)initialize();
        while(true){
#if NATIVE_DAG_BASELINE_QUEUE
            unsigned head=atomicAdd(&control->head,0u),tail=atomicAdd(&control->tail,0u);
#else
            // Read-only polls need no RMW. CAS below still owns each ticket;
            // slot acquire, rather than the tail read, observes publication.
            unsigned head=acquire32(&control->head),tail=acquire32(&control->tail);
#endif
            if(head<tail && atomicCAS(&control->head,head,head+1)==head){
                unsigned long long slot;
                do{slot=acquire64(slots+head);if(unsigned(slot>>32)!=epoch)__nanosleep(32);}while(unsigned(slot>>32)!=epoch);
                return int(unsigned(slot))-1;
            }
            if(head<tail){__nanosleep(32);continue;}
            if(acquire32(&control->completed)==unsigned(nodes))return -1;
            __nanosleep(NATIVE_DAG_IDLE_SLEEP_NS);
        }
    }
    __device__ void enqueue(int node)const{
        unsigned ticket=atomicAdd(&control->tail,1u);
        if(ticket>=unsigned(nodes))asm volatile("trap;");
        release64(slots+ticket,tag(unsigned(node)+1));
    }
    __device__ void publish_group(int group)const{
        for(int i=ready_offsets[group];i<ready_offsets[group+1];++i)enqueue(ready_nodes[i]);
    }
    // Call only after all output writers have synchronized, including the
    // appropriate proxy fence if the successor will use TMA or tensor memory.
    __device__ void complete(int node)const{
        for(int edge=child_offsets[node];edge<child_offsets[node+1];++edge){
            int group=children[edge];auto old=acquire64(remaining+group);
#if NATIVE_DAG_CAS_DECREMENT
            while(true){
                unsigned count=unsigned(old>>32)==epoch?unsigned(old):unsigned(indegree[group]);
                if(count==0)asm volatile("trap;");
                auto desired=tag(count-1),observed=cas_acq_rel(remaining+group,old,desired);
                if(observed==old){if(count==1)publish_group(group);break;}
                old=observed;
            }
#else
            // Only first use in an epoch needs CAS. Once initialized, fetch-add
            // decrements avoid competing CAS retry loops at wide fan-in nodes.
            bool accounted=false;
            while(unsigned(old>>32)!=epoch){
                unsigned count=unsigned(indegree[group]);
                if(count==0)asm volatile("trap;");
                auto desired=tag(count-1),observed=cas_acq_rel(remaining+group,old,desired);
                if(observed==old){if(count==1)publish_group(group);accounted=true;break;}
                old=observed;
            }
            if(!accounted){
                auto previous=decrement_acq_rel(remaining+group);
                if(unsigned(previous>>32)!=epoch||unsigned(previous)==0)asm volatile("trap;");
                if(unsigned(previous)==1)publish_group(group);
            }
#endif
        }
        increment_acq_rel(&control->completed);
    }
};
} // namespace native_pi0
