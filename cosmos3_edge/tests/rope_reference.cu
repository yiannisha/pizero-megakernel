// Validation-only binding to the pinned original FlashRT CUDA implementation.
#include "kernels/cosmos3_edge_misc.cuh"
extern "C" int reference_edge_rope(const void* q,const void* k,const void* qw,const void* kw,
    const void* cos,const void* sin,void* qo,void* ko,int rows,int qh,int kh,int qs,int ks,void* stream){
    using B=__nv_bfloat16;
    flash_rt::kernels::cosmos3_edge_qk_norm_rope_strided_bf16(
        static_cast<const B*>(q),static_cast<const B*>(k),static_cast<const B*>(qw),static_cast<const B*>(kw),
        static_cast<const B*>(cos),static_cast<const B*>(sin),static_cast<B*>(qo),static_cast<B*>(ko),
        rows,qh,kh,qs,ks,1e-5f,static_cast<cudaStream_t>(stream));
    return cudaGetLastError();
}
