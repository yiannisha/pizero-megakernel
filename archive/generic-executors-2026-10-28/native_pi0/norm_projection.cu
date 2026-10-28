// On-demand normalization producers feeding native TMA/TCGen05 projection.
// Row ownership never waits: any resident consumer can produce an unclaimed row.
// No grid barrier and no requirement that every CTA be simultaneously resident.
#include "nvfp4_probe.cu"
#include "norm.cu"
#include <limits>
static_assert(Cfg::NUM_CONSUMER_WARPS==4);
#ifndef NATIVE_NORM_PRODUCER_ROWS
#define NATIVE_NORM_PRODUCER_ROWS 1
#endif
static_assert(NATIVE_NORM_PRODUCER_ROWS==1||NATIVE_NORM_PRODUCER_ROWS==4||NATIVE_NORM_PRODUCER_ROWS==8);
struct NormProjectionGlobals : Fp4Globals {
    NormGlobals norm;
    unsigned long long* row_state;
    unsigned long long epoch;
};
__device__ unsigned long long load_row_state(const unsigned long long* p){
    unsigned long long value;
    asm volatile("ld.acquire.gpu.global.u64 %0, [%1];":"=l"(value):"l"(p):"memory");
    return value;
}
struct NormProjection {
    static constexpr int opcode=1;
    using controller=Fp4Matmul::controller;
    using launcher=Fp4Matmul::launcher;
    using storer=Fp4Matmul::storer;
    struct loader {static __device__ void run(const NormProjectionGlobals& g,State& s){
        if(Fp4Matmul::task(g,s)<g.tasks && kittens::laneid()==0){
            int first=Fp4Matmul::row(g,s)*128/NATIVE_NORM_PRODUCER_ROWS;
            int end=(min(Fp4Matmul::row(g,s)*128+128,g.norm.rows)+NATIVE_NORM_PRODUCER_ROWS-1)/NATIVE_NORM_PRODUCER_ROWS;
            for(int group=first;group<end;++group)
                while(load_row_state(g.row_state+group)!=2*g.epoch+1)__nanosleep(32);
        }
        __syncwarp();
        Fp4Matmul::loader::run(g,s);
    }};
    static __device__ void produce_owned(const NormProjectionGlobals& g,State& s,int group){
        for(int row=group*NATIVE_NORM_PRODUCER_ROWS;row<min((group+1)*NATIVE_NORM_PRODUCER_ROWS,g.norm.rows);++row)
            NormOp::run_row(g.norm,s,row);
        // Every writing thread orders generic stores before the TMA async proxy.
        asm volatile("fence.proxy.async.global;":::"memory");
        // This barrier carries all writers' accesses into thread 0's release
        // store below. Loader acquire loads complete the cross-CTA handoff;
        // an additional per-thread GPU fence is redundant (PTX ISA 8.9.4–5).
        NormOp::sync();
        if(threadIdx.x==0){
            auto ready=2*g.epoch+1;
            asm volatile("st.release.gpu.global.u64 [%0], %1;"::"l"(g.row_state+group),"l"(ready):"memory");
        }
        NormOp::sync();
    }
    struct consumer {static __device__ void run(const NormProjectionGlobals& g,State& s){
        int* scratch=static_cast<int*>(s.scratch());
        if(s.instruction_index==0){
            // Even CTAs without a projection tile help with normalization.
            // Strided hints balance work; ownership is acquired only by resident
            // consumers. Missing hints can always be claimed below, so neither
            // stage depends on a not-yet-resident CTA making progress.
            int groups=(g.norm.rows+NATIVE_NORM_PRODUCER_ROWS-1)/NATIVE_NORM_PRODUCER_ROWS;
            for(int group=blockIdx.x;group<groups;group+=g.workers){
                if(threadIdx.x==0){
                    auto old=load_row_state(g.row_state+group);
                    scratch[4]=old<2*g.epoch && atomicCAS(g.row_state+group,old,2*g.epoch)==old;
                }
                NormOp::sync();
                bool owner=scratch[4];
                NormOp::sync();
                if(owner)produce_owned(g,s,group);
            }
        }
        if(Fp4Matmul::task(g,s)>=g.tasks)return;
        int cursor=Fp4Matmul::row(g,s)*128/NATIVE_NORM_PRODUCER_ROWS;
        int end=(min(Fp4Matmul::row(g,s)*128+128,g.norm.rows)+NATIVE_NORM_PRODUCER_ROWS-1)/NATIVE_NORM_PRODUCER_ROWS;
        while(true){
            if(threadIdx.x==0){
                int group=cursor;
                for(;group<end;++group){
                    auto old=load_row_state(g.row_state+group);
                    if(old<2*g.epoch && atomicCAS(g.row_state+group,old,2*g.epoch)==old)break;
                }
                scratch[4]=group;
            }
            NormOp::sync();
            int group=scratch[4];
            NormOp::sync();
            if(group>=end)break;
            produce_owned(g,s,group);
            cursor=group+1;
        }
        Fp4Matmul::consumer::run(g,s);
    }};
};
struct NormProjectionContext {
    ResidentFp4* projection;
    NormGlobals norm;
    unsigned long long* row_state;
    unsigned long long epoch=0;
    int device;
};
extern "C" int create_norm_projection(void** handle,void* norm_handle,void* weight,const void* weight_scales,
    void* output,int n,int workers,float weight_scale,int bf16_output){
    if(!handle)return cudaErrorInvalidValue;*handle=nullptr;
    if(!norm_handle||!isfinite(weight_scale)||weight_scale<=0||(bf16_output!=0&&bf16_output!=1))return cudaErrorInvalidValue;
    auto& norm=*static_cast<NormContext*>(norm_handle);
    int device;int error=cudaGetDevice(&device);
    if(error!=cudaSuccess)return error;
    if(device!=norm.device)return cudaErrorInvalidDevice;
    if(norm.globals.format!=4||norm.globals.width%64)return cudaErrorInvalidValue;
    int64_t padded=(int64_t(norm.globals.rows)+127)/128*128;
    if(padded>INT_MAX)return cudaErrorInvalidValue;
    int m=int(padded);
    void* projection=nullptr;
    error=create_resident_fp4(&projection,norm.globals.output,weight,norm.globals.scales,weight_scales,output,
        m,n,norm.globals.width,workers,norm.globals.global_scale*weight_scale);
    if(error!=cudaSuccess)return error;
    if(bf16_output)error=configure_resident_fp4_bf16(projection,nullptr,nullptr,output);
    unsigned long long* state=nullptr;
    size_t groups=(size_t(norm.globals.rows)+NATIVE_NORM_PRODUCER_ROWS-1)/NATIVE_NORM_PRODUCER_ROWS;
    if(error==cudaSuccess)error=cudaMalloc(&state,groups*sizeof(*state));
    if(error==cudaSuccess)error=cudaMemset(state,0,groups*sizeof(*state));
    if(error==cudaSuccess)error=cudaStreamSynchronize(nullptr);
    auto kernel=megakernel::mk<Cfg,NormProjectionGlobals,NormProjection>;
    if(error==cudaSuccess)error=cudaFuncSetAttribute(kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,Cfg::DYNAMIC_SHARED_MEMORY);
    if(error==cudaSuccess)*handle=new NormProjectionContext{static_cast<ResidentFp4*>(projection),norm.globals,state,0,device};
    else {cudaFree(state);destroy_resident_fp4(projection);}
    return error;
}
extern "C" int launch_norm_projection(void* handle,void* stream){
    if(!handle)return cudaErrorInvalidValue;
    auto& c=*static_cast<NormProjectionContext*>(handle);
    int device;auto error=cudaGetDevice(&device);
    if(error!=cudaSuccess)return error;
    if(device!=c.device)return cudaErrorInvalidDevice;
    cudaStreamCaptureStatus capture;
    error=cudaStreamIsCapturing(static_cast<cudaStream_t>(stream),&capture);
    if(error!=cudaSuccess)return error;
    if(capture!=cudaStreamCaptureStatusNone)return cudaErrorStreamCaptureUnsupported;
    if(c.epoch>=std::numeric_limits<unsigned long long>::max()/2)return cudaErrorInvalidValue;
    NormProjectionGlobals g{c.projection->globals,c.norm,c.row_state,++c.epoch};
    auto kernel=megakernel::mk<Cfg,NormProjectionGlobals,NormProjection>;
    kernel<<<g.workers,Cfg::NUM_THREADS,Cfg::DYNAMIC_SHARED_MEMORY,static_cast<cudaStream_t>(stream)>>>(g);
    return cudaGetLastError();
}
extern "C" int destroy_norm_projection(void* handle){
    if(!handle)return cudaSuccess;
    auto* c=static_cast<NormProjectionContext*>(handle);
    int device;auto error=cudaGetDevice(&device);
    if(error!=cudaSuccess)return error;
    if(device!=c->device)return cudaErrorInvalidDevice;
    error=cudaFree(c->row_state);
    auto other=destroy_resident_fp4(c->projection);delete c;
    return error==cudaSuccess?other:error;
}

#ifdef NATIVE_NORM_PROJECTION_VALIDATION
// Matched two-native-launch baseline; never used by the inference forward API.
extern "C" int validation_split_norm_projection(void* handle,void* stream){
    if(!handle)return cudaErrorInvalidValue;
    auto& c=*static_cast<NormProjectionContext*>(handle);
    int device;auto error=cudaGetDevice(&device);
    if(error!=cudaSuccess)return error;
    if(device!=c.device)return cudaErrorInvalidDevice;
    auto kernel=megakernel::mk<NC,NormGlobals,NormOp>;
    kernel<<<c.norm.workers,NC::NUM_THREADS,NC::DYNAMIC_SHARED_MEMORY,static_cast<cudaStream_t>(stream)>>>(c.norm);
    error=cudaGetLastError();
    return error==cudaSuccess?launch_resident_fp4(c.projection,stream):int(error);
}
#endif
