#pragma once
// FlashRT head-128 warp algorithm adapted as an original VM consumer.
// Each warp normalizes/rotates one head; K and optional V publish directly
// into the requested cache rows. Position tables are prepared at setup.
#ifndef EDGE_ROPE_VECTORS_PER_TASK
#define EDGE_ROPE_VECTORS_PER_TASK 192
#endif
extern "C" int edge_rope_vectors_per_task(){return EDGE_ROPE_VECTORS_PER_TASK;}
static_assert(EDGE_ROPE_VECTORS_PER_TASK>0 && EDGE_ROPE_VECTORS_PER_TASK%4==0);
struct EdgeRopeGlobals {
    megakernel::instruction_layout<Cfg> instructions;
    megakernel::timing_layout<Cfg> timings;
    const __nv_bfloat16 *q,*k,*v,*qw,*kw,*cos,*sin;
    __nv_bfloat16 *qo,*ko,*vo;
    int rows,q_heads,k_heads,q_stride,k_stride,v_stride,qo_stride,ko_stride,vo_stride,workers;
    float epsilon;int reference_rounding;
};
struct EdgeRopeOp {
    static constexpr int opcode=5;
    using controller=megakernel::NoOp<Cfg>::controller;
    using loader=megakernel::NoOp<Cfg>::loader;
    using launcher=megakernel::NoOp<Cfg>::launcher;
    using storer=megakernel::NoOp<Cfg>::storer;
    static __device__ float bf(float x){return __bfloat162float(__float2bfloat16_rn(x));}
    static __device__ void vector(const EdgeRopeGlobals& g,int vec){
        int lane=kittens::laneid(),q_vecs=g.rows*g.q_heads;
        bool query=vec<q_vecs;int heads=query?g.q_heads:g.k_heads;
        int local=query?vec:vec-q_vecs,row=local/heads,head=local%heads;
        const auto* x=(query?g.q:g.k)+size_t(row)*(query?g.q_stride:g.k_stride)+head*128;
        const auto* w=query?g.qw:g.kw;
        auto* out=(query?g.qo:g.ko)+size_t(row)*(query?g.qo_stride:g.ko_stride)+head*128;
        int e=lane*2;
        auto a=reinterpret_cast<const __nv_bfloat162*>(x)[e];
        auto b=reinterpret_cast<const __nv_bfloat162*>(x)[e+1];
        float val[4]={__bfloat162float(a.x),__bfloat162float(a.y),__bfloat162float(b.x),__bfloat162float(b.y)};
        float ss=val[0]*val[0]+val[1]*val[1]+val[2]*val[2]+val[3]*val[3];
        for(int off=16;off;off>>=1)ss+=__shfl_xor_sync(0xffffffff,ss,off);
        float rms=rsqrtf(ss*(1.f/128.f)+g.epsilon);
        auto wa=reinterpret_cast<const __nv_bfloat162*>(w)[e],wb=reinterpret_cast<const __nv_bfloat162*>(w)[e+1];
        float weight[4]={__bfloat162float(wa.x),__bfloat162float(wa.y),__bfloat162float(wb.x),__bfloat162float(wb.y)};
        float norm[4];
        #pragma unroll
        for(int i=0;i<4;i++){norm[i]=val[i]*rms*weight[i];if(g.reference_rounding)norm[i]=bf(norm[i]);}
        auto ca=reinterpret_cast<const __nv_bfloat162*>(g.cos+size_t(row)*128)[e];
        auto cb=reinterpret_cast<const __nv_bfloat162*>(g.cos+size_t(row)*128)[e+1];
        auto sa=reinterpret_cast<const __nv_bfloat162*>(g.sin+size_t(row)*128)[e];
        auto sb=reinterpret_cast<const __nv_bfloat162*>(g.sin+size_t(row)*128)[e+1];
        float co[4]={__bfloat162float(ca.x),__bfloat162float(ca.y),__bfloat162float(cb.x),__bfloat162float(cb.y)};
        float si[4]={__bfloat162float(sa.x),__bfloat162float(sa.y),__bfloat162float(sb.x),__bfloat162float(sb.y)};
        float result[4],sign=lane<16?-1.f:1.f;
        #pragma unroll
        for(int i=0;i<4;i++){
            float peer=__shfl_xor_sync(0xffffffff,norm[i],16);
            result[i]=g.reference_rounding?bf(norm[i]*co[i])+bf(sign*peer*si[i]):norm[i]*co[i]+sign*peer*si[i];
        }
        auto* out2=reinterpret_cast<__nv_bfloat162*>(out);
        out2[e]=__floats2bfloat162_rn(result[0],result[1]);out2[e+1]=__floats2bfloat162_rn(result[2],result[3]);
        if(!query && g.v){
            auto* src=reinterpret_cast<const uint2*>(g.v+size_t(row)*g.v_stride+head*128);
            auto* dst=reinterpret_cast<uint2*>(g.vo+size_t(row)*g.vo_stride+head*128);
            dst[lane]=src[lane];
        }
    }
    struct consumer {static __device__ void run(const EdgeRopeGlobals& g,State& s){
        int task=s.instruction_index*g.workers+blockIdx.x;
        int begin=task*EDGE_ROPE_VECTORS_PER_TASK,end=min(begin+EDGE_ROPE_VECTORS_PER_TASK,g.rows*(g.q_heads+g.k_heads));
        for(int vec=begin+kittens::warpid();vec<end;vec+=Cfg::NUM_CONSUMER_WARPS)vector(g,vec);
    }};
};
struct EdgeRopeContext {EdgeRopeGlobals globals;int *program,*timing;int device;};
extern "C" int create_edge_rope(void** handle,const void* q,const void* k,const void* v,const void* qw,const void* kw,
    const void* cos,const void* sin,void* qo,void* ko,void* vo,
    int rows,int q_heads,int k_heads,int q_stride,int k_stride,int v_stride,int qo_stride,int ko_stride,int vo_stride,
    int workers,float epsilon,int reference_rounding){
    if(!handle)return cudaErrorInvalidValue;*handle=nullptr;
    if(!k||!kw||!cos||!sin||!ko||rows<1||q_heads<0||k_heads<1||workers<1||!isfinite(epsilon)||epsilon<=0
       ||reference_rounding<0||reference_rounding>1||(q_heads&&(!q||!qw||!qo))||(v&&!vo)
       ||int64_t(rows)*(q_heads+k_heads)>INT_MAX||k_stride<k_heads*128||ko_stride<k_heads*128
       ||q_stride<q_heads*128||qo_stride<q_heads*128||(v&&(v_stride<k_heads*128||vo_stride<k_heads*128)))return cudaErrorInvalidValue;
    for(int stride:{q_stride,k_stride,v_stride,qo_stride,ko_stride,vo_stride})if(stride%4)return cudaErrorInvalidValue;
    int device;cudaDeviceProp prop{};auto error=cudaGetDevice(&device);
    if(error==cudaSuccess)error=cudaGetDeviceProperties(&prop,device);
    if(error!=cudaSuccess)return error;if(prop.major!=10||prop.minor!=0||workers>prop.multiProcessorCount)return cudaErrorInvalidConfiguration;
    int vectors=rows*(q_heads+k_heads),tasks=(vectors+EDGE_ROPE_VECTORS_PER_TASK-1)/EDGE_ROPE_VECTORS_PER_TASK;
    int instruction_rows=(tasks+workers-1)/workers,slots=0;int *slot_device=nullptr,*program=nullptr,*timing=nullptr;
    error=cudaMalloc(&slot_device,sizeof(int));
    if(error==cudaSuccess){slots_kernel<<<1,32>>>(slot_device);error=cudaGetLastError();}
    if(error==cudaSuccess)error=cudaMemcpy(&slots,slot_device,sizeof(int),cudaMemcpyDeviceToHost);cudaFree(slot_device);
    std::vector<int> code(size_t(slots)*instruction_rows*32,0);
    for(size_t i=0;i<code.size();i+=32)code[i]=EdgeRopeOp::opcode;
    if(error==cudaSuccess)error=cudaMalloc(&program,code.size()*4);
    if(error==cudaSuccess)error=cudaMalloc(&timing,size_t(slots)*instruction_rows*128*4);
    if(error==cudaSuccess)error=cudaMemcpy(program,code.data(),code.size()*4,cudaMemcpyHostToDevice);
    auto kernel=megakernel::mk<Cfg,EdgeRopeGlobals,EdgeRopeOp>;
    if(error==cudaSuccess)error=cudaFuncSetAttribute(kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,Cfg::DYNAMIC_SHARED_MEMORY);
    if(error==cudaSuccess){
        using B=__nv_bfloat16;
        EdgeRopeGlobals g{megakernel::instruction_layout<Cfg>(program,nullptr,slots,instruction_rows,nullptr),
            megakernel::timing_layout<Cfg>(timing,nullptr,slots,instruction_rows,nullptr),
            static_cast<const B*>(q),static_cast<const B*>(k),static_cast<const B*>(v),static_cast<const B*>(qw),static_cast<const B*>(kw),
            static_cast<const B*>(cos),static_cast<const B*>(sin),static_cast<B*>(qo),static_cast<B*>(ko),static_cast<B*>(vo),
            rows,q_heads,k_heads,q_stride,k_stride,v_stride,qo_stride,ko_stride,vo_stride,workers,epsilon,reference_rounding};
        *handle=new EdgeRopeContext{g,program,timing,device};
    }else{cudaFree(program);cudaFree(timing);}return error;
}
extern "C" int launch_edge_rope(void* handle,void* stream){
    if(!handle)return cudaErrorInvalidValue;auto& c=*static_cast<EdgeRopeContext*>(handle);int device;auto error=cudaGetDevice(&device);
    if(error!=cudaSuccess)return error;if(device!=c.device)return cudaErrorInvalidDevice;
    auto kernel=megakernel::mk<Cfg,EdgeRopeGlobals,EdgeRopeOp>;
    kernel<<<c.globals.workers,Cfg::NUM_THREADS,Cfg::DYNAMIC_SHARED_MEMORY,static_cast<cudaStream_t>(stream)>>>(c.globals);return cudaGetLastError();
}
extern "C" int destroy_edge_rope(void* handle){
    if(!handle)return cudaSuccess;auto* c=static_cast<EdgeRopeContext*>(handle);int device;auto error=cudaGetDevice(&device);
    if(error!=cudaSuccess)return error;if(device!=c->device)return cudaErrorInvalidDevice;
    error=cudaDeviceSynchronize();if(error!=cudaSuccess)return error;
    auto a=cudaFree(c->program),b=cudaFree(c->timing);delete c;return a!=cudaSuccess?a:b;
}
