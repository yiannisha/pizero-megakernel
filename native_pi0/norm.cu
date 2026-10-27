// FlashRT-derived register-cached vector normalization as an original VM op.
#include "kittens.cuh"
#include "megakernel.cuh"
#include <cuda_fp8.h>
#include <vector>

struct NormConfig:megakernel::default_config {
    static constexpr int NUM_CONSUMER_WARPS=4,NUM_WARPS=8,NUM_THREADS=256;
    static constexpr int CONSUMER_REGISTERS=128;
};
#ifndef NATIVE_NORM_ROW_TILE
#define NATIVE_NORM_ROW_TILE 1
#endif
static_assert(NATIVE_NORM_ROW_TILE==1||NATIVE_NORM_ROW_TILE==4||NATIVE_NORM_ROW_TILE==8);
using NC=NormConfig;
using NS=megakernel::state<NC>;
struct NormGlobals {
    megakernel::instruction_layout<NC> instructions;
    megakernel::timing_layout<NC> timings;
    const void *input,*residual;
    const float *gamma,*beta,*inverse_scale;
    void* residual_output;
    unsigned char *output,*scales;
    int rows,width,workers,fp16,format,layer,round_norm,round_residual,legacy_ties;
    float epsilon,global_scale;
};
__device__ float decode16(unsigned short bits,int fp16){return fp16?__half2float(__ushort_as_half(bits)):__bfloat162float(__ushort_as_bfloat16(bits));}
__device__ unsigned short encode16(float value,int fp16){return fp16?__half_as_ushort(__float2half_rn(value)):__bfloat16_as_ushort(__float2bfloat16_rn(value));}
struct NormOp {
    static constexpr int opcode=1;
    using controller=megakernel::NoOp<NC>::controller;
    using loader=megakernel::NoOp<NC>::loader;
    using launcher=megakernel::NoOp<NC>::launcher;
    using storer=megakernel::NoOp<NC>::storer;
    static __device__ void sync(){asm volatile("bar.sync 6, 128;":::"memory");}
    template<class StateT> static __device__ float reduce(float value,StateT& s){
        int lane=kittens::laneid(),warp=kittens::warpid();
        float* scratch=reinterpret_cast<float*>(s.scratch());
        #pragma unroll
        for(int offset=16;offset;offset>>=1)value+=__shfl_xor_sync(0xffffffff,value,offset);
        if(lane==0)scratch[warp]=value;
        sync();
        if(warp==0){
            value=lane<4?scratch[lane]:0.f;
            #pragma unroll
            for(int offset=16;offset;offset>>=1)value+=__shfl_xor_sync(0xffffffff,value,offset);
            if(lane==0)scratch[0]=value;
        }
        sync();
        value=scratch[0];sync();return value;
    }
    static __device__ unsigned char legacy(float x){
        unsigned char sign=x<0?8:0;float a=fabsf(x);
        return sign|(a<=.25f?0:a<=.75f?1:a<=1.25f?2:a<=1.75f?3:a<=2.5f?4:a<=3.5f?5:a<=5.f?6:7);
    }
    static __device__ unsigned char pair(float x,float y){
        unsigned short bits;
        asm("{ .reg .b8 b; cvt.rn.satfinite.e2m1x2.f32 b, %2, %1; cvt.u16.u8 %0, b; }":"=h"(bits):"f"(x),"f"(y));
        return static_cast<unsigned char>(bits);
    }
    template<class StateT> static __device__ void run_row(const NormGlobals& g,StateT& s,int row){
        if(row>=g.rows)return;
        int block=threadIdx.x,blocks=g.width/16;
        bool active=block<blocks;
        float values[16];uint4 raw[2]{},residual[2]{};
        size_t base=size_t(row)*g.width+block*16;
        if(active){
            auto* input=static_cast<const uint4*>(g.input)+base/8;
            raw[0]=input[0];raw[1]=input[1];
            if(g.residual){auto* r=static_cast<const uint4*>(g.residual)+base/8;residual[0]=r[0];residual[1]=r[1];}
        }
        auto* bits=reinterpret_cast<unsigned short*>(raw);
        auto* resbits=reinterpret_cast<unsigned short*>(residual);
        float sum=0;
        #pragma unroll
        for(int i=0;i<16;i++){
            float v=active?decode16(bits[i],g.fp16):0.f;
            if(active&&g.residual){
                v+=decode16(resbits[i],g.fp16);resbits[i]=encode16(v,g.fp16);
                if(g.round_residual)v=decode16(resbits[i],g.fp16);
            }
            values[i]=v;sum+=v;
        }
        if(active&&g.residual_output){auto* out=static_cast<uint4*>(g.residual_output)+base/8;out[0]=residual[0];out[1]=residual[1];}
        float mean=g.layer?reduce(sum,s)/g.width:0.f;
        float squares=0;
        if(active){
            #pragma unroll
            for(int i=0;i<16;i++){float v=values[i]-mean;squares+=v*v;}
        }
        float inverse=rsqrtf(reduce(squares,s)/g.width+g.epsilon);
        if(!active)return;
        float maximum=0,inv_global=1.f/g.global_scale;
        #pragma unroll
        for(int i=0;i<16;i++){
            float v=(values[i]-mean)*inverse;
            if(g.gamma&&g.beta)v=fmaf(v,g.gamma[block*16+i],g.beta[block*16+i]);
            else if(g.gamma)v*=g.gamma[block*16+i];
            else if(g.beta)v+=g.beta[block*16+i];
            if(g.inverse_scale)v*=g.inverse_scale[block*16+i];
            if(g.round_norm)v=decode16(encode16(v,g.fp16),g.fp16);
            values[i]=v*inv_global;maximum=fmaxf(maximum,fabsf(values[i]));
        }
        if(g.format==8){
            uint4 packed;auto* bytes=reinterpret_cast<unsigned char*>(&packed);
            #pragma unroll
            for(int i=0;i<16;i++){__nv_fp8_e4m3 value(values[i]);bytes[i]=value.__x;}
            *reinterpret_cast<uint4*>(g.output+base)=packed;
        }else{
            float desired=g.legacy_ties?fmaxf(maximum/6.f,1e-12f):maximum*(1.f/6.f);
            __nv_fp8_e4m3 scale(desired);float decoded=float(scale),reciprocal=decoded>0?1.f/decoded:0;
            uint2 packed;auto* bytes=reinterpret_cast<unsigned char*>(&packed);
            #pragma unroll
            for(int i=0;i<8;i++){
                float a=values[2*i]*reciprocal,b=values[2*i+1]*reciprocal;
                bytes[i]=decoded>0?(g.legacy_ties?(legacy(a)|(legacy(b)<<4)):pair(a,b)):0;
            }
            *reinterpret_cast<uint2*>(g.output+base/2)=packed;
            size_t offset=((size_t(row)/128)*((blocks+3)/4)+block/4)*512+(row%32)*16+((row%128)/32)*4+block%4;
            g.scales[offset]=scale.__x;
        }
    }
    struct consumer {static __device__ void run(const NormGlobals& g,NS& s){
        int64_t first=(int64_t(s.instruction_index)*g.workers+blockIdx.x)*NATIVE_NORM_ROW_TILE;
        for(int offset=0;offset<NATIVE_NORM_ROW_TILE && first+offset<g.rows;offset++)run_row(g,s,int(first+offset));
    }};
};
__global__ void norm_slots(int* slots){if(threadIdx.x==0)asm volatile("mov.u32 %0, %%nsmid;":"=r"(*slots));}
struct NormContext {NormGlobals globals;int *program,*timing;int device;};
extern "C" int create_norm(void** handle,const void* input,const void* residual,const float* gamma,const float* beta,const float* inverse_scale,
    void* residual_output,void* output,void* scales,int rows,int width,int workers,int fp16,int format,int layer,int round_norm,int round_residual,int legacy_ties,float epsilon,float global_scale){
    if(!handle)return cudaErrorInvalidValue;*handle=nullptr;
    if(!input||!output||rows<1||width<16||width>2048||width%16||workers<1||(format!=4&&format!=8)||(format==4&&!scales)
       ||!isfinite(epsilon)||epsilon<=0||!isfinite(global_scale)||global_scale<=0||(fp16!=0&&fp16!=1)||(!residual&&residual_output))return cudaErrorInvalidValue;
    int device;cudaDeviceProp properties{};cudaError_t error=cudaGetDevice(&device);
    if(error==cudaSuccess)error=cudaGetDeviceProperties(&properties,device);
    if(error!=cudaSuccess)return error;
    if(properties.major!=10||properties.minor!=0||workers>properties.multiProcessorCount)return cudaErrorInvalidConfiguration;
    int *slots_dev=nullptr,*program=nullptr,*timing=nullptr;int slots=0,count=int((int64_t(rows)+workers*NATIVE_NORM_ROW_TILE-1)/(workers*NATIVE_NORM_ROW_TILE));
    error=cudaMalloc(&slots_dev,sizeof(int));
    if(error==cudaSuccess){norm_slots<<<1,32>>>(slots_dev);error=cudaGetLastError();}
    if(error==cudaSuccess)error=cudaMemcpy(&slots,slots_dev,sizeof(int),cudaMemcpyDeviceToHost);
    cudaFree(slots_dev);std::vector<int> code(size_t(slots)*count*32,0);
    for(size_t i=0;i<code.size();i+=32)code[i]=1;
    if(error==cudaSuccess)error=cudaMalloc(&program,code.size()*sizeof(int));
    if(error==cudaSuccess)error=cudaMalloc(&timing,size_t(slots)*count*128*sizeof(int));
    if(error==cudaSuccess)error=cudaMemcpy(program,code.data(),code.size()*sizeof(int),cudaMemcpyHostToDevice);
    auto kernel=megakernel::mk<NC,NormGlobals,NormOp>;
    if(error==cudaSuccess)error=cudaFuncSetAttribute(kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,NC::DYNAMIC_SHARED_MEMORY);
    if(error==cudaSuccess){
        NormGlobals g{megakernel::instruction_layout<NC>(program,nullptr,slots,count,nullptr),megakernel::timing_layout<NC>(timing,nullptr,slots,count,nullptr),
            input,residual,gamma,beta,inverse_scale,residual_output,static_cast<unsigned char*>(output),static_cast<unsigned char*>(scales),
            rows,width,workers,fp16,format,layer,round_norm,round_residual,legacy_ties,epsilon,global_scale};
        *handle=new NormContext{g,program,timing,device};
    }else{cudaFree(program);cudaFree(timing);}
    return error;
}
extern "C" int launch_norm(void* handle,void* stream){
    if(!handle)return cudaErrorInvalidValue;auto& ctx=*static_cast<NormContext*>(handle);
    int device;cudaError_t error=cudaGetDevice(&device);if(error!=cudaSuccess)return error;if(device!=ctx.device)return cudaErrorInvalidDevice;
    megakernel::mk<NC,NormGlobals,NormOp><<<ctx.globals.workers,NC::NUM_THREADS,NC::DYNAMIC_SHARED_MEMORY,static_cast<cudaStream_t>(stream)>>>(ctx.globals);
    return cudaGetLastError();
}
extern "C" int destroy_norm(void* handle){
    if(!handle)return cudaSuccess;auto* ctx=static_cast<NormContext*>(handle);
    int device;cudaError_t error=cudaGetDevice(&device);if(error!=cudaSuccess)return error;if(device!=ctx->device)return cudaErrorInvalidDevice;
    error=cudaFree(ctx->program);auto second=cudaFree(ctx->timing);delete ctx;return error==cudaSuccess?second:error;
}
