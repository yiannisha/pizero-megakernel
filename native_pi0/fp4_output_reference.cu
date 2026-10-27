// Independent development oracle: pinned CUTLASS GELU + FlashRT quant_block16.
// Never called by the native inference/primitive forward path.
#include <cuda_runtime.h>
#include "cutlass/epilogue/thread/activation.h"
#include "cutlass/arch/barrier.h"
#include "gemm/fp4/nvfp4_skinny_common.cuh"
#include "gemm/fp4/sm100_gelu_mul_blockscale_visitor.hpp"

__global__ void reference_compact_quant(const float* input,unsigned char* packed,unsigned char* scales,int blocks,float global_scale){
    using Visitor=cutlass::epilogue::fusion::Sm100GeluMulCompactBlockScaleFactorRowStore<
        16,cute::Shape<cute::_128,cute::_128>,cutlass::float_e2m1_t,float,cutlass::float_ue4m3_t>;
    int block=blockIdx.x*blockDim.x+threadIdx.x;
    if(block>=blocks)return;
    float values[16],maximum=0,inverse_global=1.f/global_scale;
    #pragma unroll
    for(int i=0;i<16;i++){
        values[i]=Visitor::gelu_tanh_(input[block*32+i*2])*input[block*32+i*2+1]*inverse_global;
        maximum=fmaxf(maximum,fabsf(values[i]));
    }
    __nv_fp8_e4m3 scale(fmaxf(maximum/6.f,1e-12f));scales[block]=scale.__x;
    if(float(scale)==0){for(int i=0;i<8;i++)packed[block*8+i]=0;return;}
    float inverse=1.f/float(scale);cutlass::Array<float,16> normalized;
    #pragma unroll
    for(int i=0;i<16;i++)normalized[i]=values[i]*inverse;
    auto codes=cutlass::NumericArrayConverter<cutlass::float_e2m1_t,float,16,cutlass::FloatRoundStyle::round_to_nearest>{}(normalized);
    *reinterpret_cast<uint2*>(packed+block*8)=*reinterpret_cast<const uint2*>(codes.data());
}

__global__ void reference_quant(const float* input,unsigned char* packed,unsigned char* scales,int blocks,float global_scale,int gelu){
    int block=blockIdx.x*blockDim.x+threadIdx.x;
    if(block>=blocks)return;
    float values[16],inverse=1.f/global_scale;
    #pragma unroll
    for(int i=0;i<16;i++){
        float value=input[block*16+i];
        if(gelu)value=cutlass::epilogue::thread::GELU_taylor<float>{}(value);
        values[i]=value*inverse;
    }
    flash_rt::nvfp4_skinny::quant_block16(values,packed+block*8,scales+block);
    // Canonical zero bytes for scales rounded to zero (rather than -0 codes).
    if(scales[block]==0)for(int i=0;i<8;i++)packed[block*8+i]=0;
}
extern "C" int reference_fp4_output(const float* input,void* packed,void* scales,int m,int n,float global_scale,int gelu){
    if(!input||!packed||!scales||m<1||n<16||n%16||global_scale<=0||gelu<0||gelu>2||(gelu==2&&n%32))return cudaErrorInvalidValue;
    float* x=nullptr;unsigned char *q=nullptr,*sf=nullptr;size_t elements=size_t(m)*n;
    size_t output_elements=gelu==2?elements/2:elements;
    cudaError_t error=cudaMalloc(&x,elements*sizeof(float));
    if(error==cudaSuccess)error=cudaMalloc(&q,output_elements/2);
    if(error==cudaSuccess)error=cudaMalloc(&sf,output_elements/16);
    if(error==cudaSuccess)error=cudaMemcpy(x,input,elements*sizeof(float),cudaMemcpyHostToDevice);
    if(error==cudaSuccess){
        if(gelu==2)reference_compact_quant<<<(output_elements/16+127)/128,128>>>(x,q,sf,output_elements/16,global_scale);
        else reference_quant<<<(elements/16+127)/128,128>>>(x,q,sf,elements/16,global_scale,gelu);
        error=cudaGetLastError();
    }
    if(error==cudaSuccess)error=cudaMemcpy(packed,q,output_elements/2,cudaMemcpyDeviceToHost);
    if(error==cudaSuccess)error=cudaMemcpy(scales,sf,output_elements/16,cudaMemcpyDeviceToHost);
    cudaFree(x);cudaFree(q);cudaFree(sf);return error;
}
