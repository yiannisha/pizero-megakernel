#pragma once
// Resident BF16 / FP8 operator setup for the shared TMA/TCGen05 VM operator.
#include <climits>

template<class E> struct EdgeProjection {
    Globals<E> globals;
    int *program,*timing;
    int device;
};
struct EdgeProjectionHandle {int precision;void* context;};

template<class E> int create_edge_projection_impl(void** handle,void* a,void* b,void* output,
    const void* bias,const void* residual,int m,int n,int k,int workers,float alpha,int activation,int precision){
    using G=Globals<E>;using Input=typename G::Input;
    int device;cudaDeviceProp prop{};auto error=cudaGetDevice(&device);
    if(error==cudaSuccess)error=cudaGetDeviceProperties(&prop,device);
    if(error!=cudaSuccess)return error;
    if(prop.major!=10||prop.minor!=0||workers>prop.multiProcessorCount)return cudaErrorInvalidConfiguration;
    int64_t count=int64_t(m/128)*(n/128);if(count>INT_MAX)return cudaErrorInvalidValue;
    int tasks=int(count),rows=(tasks+workers-1)/workers,slots=0;
    int *slot_device=nullptr,*program=nullptr,*timing=nullptr;
    error=cudaMalloc(&slot_device,sizeof(int));
    if(error==cudaSuccess){slots_kernel<<<1,32>>>(slot_device);error=cudaGetLastError();}
    if(error==cudaSuccess)error=cudaMemcpy(&slots,slot_device,sizeof(int),cudaMemcpyDeviceToHost);
    cudaFree(slot_device);
    std::vector<int> code(size_t(slots)*rows*32,0);
    for(size_t i=0;i<code.size();i+=32)code[i]=1;
    if(error==cudaSuccess)error=cudaMalloc(&program,code.size()*sizeof(int));
    if(error==cudaSuccess)error=cudaMalloc(&timing,size_t(slots)*rows*128*sizeof(int));
    if(error==cudaSuccess)error=cudaMemcpy(program,code.data(),code.size()*sizeof(int),cudaMemcpyHostToDevice);
    auto kernel=megakernel::mk<Cfg,G,Matmul<E>>;
    if(error==cudaSuccess)error=cudaFuncSetAttribute(kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,Cfg::DYNAMIC_SHARED_MEMORY);
    if(error==cudaSuccess){
        G g{megakernel::instruction_layout<Cfg>(program,nullptr,slots,rows,nullptr),
            megakernel::timing_layout<Cfg>(timing,nullptr,slots,rows,nullptr),
            Input(static_cast<E*>(a),nullptr,nullptr,m,k),Input(static_cast<E*>(b),nullptr,nullptr,n,k),
            Output(reinterpret_cast<float*>(output),nullptr,nullptr,m,n),alpha,workers,tasks,m/128,n/128,
            static_cast<kittens::bf16*>(output),static_cast<const kittens::bf16*>(bias),
            static_cast<const kittens::bf16*>(residual),activation};
        *handle=new EdgeProjectionHandle{precision,new EdgeProjection<E>{g,program,timing,device}};
    }else{cudaFree(program);cudaFree(timing);}
    return error;
}
extern "C" int create_edge_projection(void** handle,void* a,void* b,void* output,const void* bias,const void* residual,
    int m,int n,int k,int workers,float alpha,int activation,int precision){
    if(!handle)return cudaErrorInvalidValue;*handle=nullptr;
    if(!a||!b||!output||m<128||m%128||n<128||n%128||k<64||k%64||workers<1||!isfinite(alpha)||alpha<=0||activation<0||activation>2)
        return cudaErrorInvalidValue;
    if(precision==0)return create_edge_projection_impl<kittens::bf16>(handle,a,b,output,bias,residual,m,n,k,workers,alpha,activation,precision);
    if(precision==1)return create_edge_projection_impl<kittens::fp8e4m3>(handle,a,b,output,bias,residual,m,n,k,workers,alpha,activation,precision);
    return cudaErrorInvalidValue;
}
template<class E> int launch_edge_projection_impl(void* context,void* stream){
    auto& c=*static_cast<EdgeProjection<E>*>(context);int device;auto error=cudaGetDevice(&device);
    if(error!=cudaSuccess)return error;if(device!=c.device)return cudaErrorInvalidDevice;
    auto kernel=megakernel::mk<Cfg,Globals<E>,Matmul<E>>;
    kernel<<<c.globals.workers,Cfg::NUM_THREADS,Cfg::DYNAMIC_SHARED_MEMORY,static_cast<cudaStream_t>(stream)>>>(c.globals);
    return cudaGetLastError();
}
extern "C" int launch_edge_projection(void* handle,void* stream){
    if(!handle)return cudaErrorInvalidValue;auto& h=*static_cast<EdgeProjectionHandle*>(handle);
    return h.precision==0?launch_edge_projection_impl<kittens::bf16>(h.context,stream):launch_edge_projection_impl<kittens::fp8e4m3>(h.context,stream);
}
template<class E> int destroy_edge_projection_impl(void* context){
    auto* c=static_cast<EdgeProjection<E>*>(context);int device;auto error=cudaGetDevice(&device);
    if(error!=cudaSuccess)return error;if(device!=c->device)return cudaErrorInvalidDevice;
    error=cudaDeviceSynchronize();if(error!=cudaSuccess)return error;
    auto first=cudaFree(c->program),second=cudaFree(c->timing);delete c;
    return first!=cudaSuccess?first:second;
}
extern "C" int destroy_edge_projection(void* handle){
    if(!handle)return cudaSuccess;auto* h=static_cast<EdgeProjectionHandle*>(handle);
    int error=h->precision==0?destroy_edge_projection_impl<kittens::bf16>(h->context):destroy_edge_projection_impl<kittens::fp8e4m3>(h->context);
    if(error==cudaSuccess)delete h;return error;
}

extern "C" int configure_edge_projection_reference(void* handle){
    if(!handle)return cudaErrorInvalidValue;auto& h=*static_cast<EdgeProjectionHandle*>(handle);
    if(h.precision==0){auto& g=static_cast<EdgeProjection<kittens::bf16>*>(h.context)->globals;if(g.bias||!g.residual)return cudaErrorInvalidValue;g.round_before_residual=1;}
    else{auto& g=static_cast<EdgeProjection<kittens::fp8e4m3>*>(h.context)->globals;if(g.bias||!g.residual)return cudaErrorInvalidValue;g.round_before_residual=1;}
    return cudaSuccess;
}
