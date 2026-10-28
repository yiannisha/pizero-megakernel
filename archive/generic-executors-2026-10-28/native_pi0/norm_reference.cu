// Standalone FlashRT references for development only, never native forward.
#include "fused_fp4/siglip_ln_vec.cuh"
#include "gemm/fp4/nvfp4_skinny_common.cuh"

__global__ void quant_norm_reference(const float* input,unsigned char* output,unsigned char* scales,int blocks,int format,float global_scale){
    int block=blockIdx.x*blockDim.x+threadIdx.x;if(block>=blocks)return;
    float values[16],inverse=1.f/global_scale;
    #pragma unroll
    for(int i=0;i<16;i++)values[i]=input[block*16+i]*inverse;
    if(format==4){
        flash_rt::nvfp4_skinny::quant_block16(values,output+block*8,scales+block);
        if(scales[block]==0)for(int i=0;i<8;i++)output[block*8+i]=0;
    }else{
        #pragma unroll
        for(int i=0;i<16;i++){__nv_fp8_e4m3 value(values[i]);output[block*16+i]=value.__x;}
    }
}
extern "C" int reference_norm_quant(const void* input,void* output,void* scales,int elements,int format,float global_scale){
    if(!input||!output||elements<16||elements%16||(format!=4&&format!=8)||(format==4&&!scales))return cudaErrorInvalidValue;
    quant_norm_reference<<<(elements/16+127)/128,128>>>(static_cast<const float*>(input),static_cast<unsigned char*>(output),
        static_cast<unsigned char*>(scales),elements/16,format,global_scale);return cudaGetLastError();
}
extern "C" int reference_layer_norm(const void* input,const void* gamma,const void* beta,const void* inverse_scale,
                                    void* output,void* scales,int rows,int width,int format){
    if(format==8)return flash_rt::fused_fp4::layer_norm_fp8_vec_fp16(static_cast<const __half*>(input),
        static_cast<const __half*>(gamma),static_cast<const __half*>(beta),output,rows,width,1e-6f,nullptr);
    return flash_rt::fused_fp4::layer_norm_mul_fp4_sfa_vec_fp16(static_cast<const __half*>(input),
        static_cast<const __half*>(gamma),static_cast<const __half*>(beta),static_cast<const __half*>(inverse_scale),output,scales,rows,width,1e-6f,nullptr);
}
