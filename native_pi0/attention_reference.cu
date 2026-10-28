// Validation-only bridge to the actual pinned standalone FlashRT kernel.
#include "kernels/fused_qkv_attention.cuh"
extern "C" int reference_attention(int dtype,const void* q,const void* k,const void* v,int ld,const void* qb,const void* kb,const void* vb,
 const void* cos,const void* sin,void* out,int out_ld,int batches,int rows,int heads,int hd,float scale,void* stream){
 return flash_rt::prefill_attention::fused_qkv_attention(dtype,q,k,v,ld,qb,kb,vb,cos,sin,out,out_ld,batches,rows,heads,hd,scale,static_cast<cudaStream_t>(stream));
}
