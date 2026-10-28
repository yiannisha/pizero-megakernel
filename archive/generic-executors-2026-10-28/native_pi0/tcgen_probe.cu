#pragma once
// Development BF16/FP8 TMA/TCGen05 data-path test in the original VM.
#include "kittens.cuh"
#include "megakernel.cuh"
#include <vector>

#ifndef NATIVE_COMPUTE_CONSUMER_WARPS
#define NATIVE_COMPUTE_CONSUMER_WARPS 16
#endif
struct ComputeConfig:megakernel::default_config {
    static constexpr int NUM_CONSUMER_WARPS=NATIVE_COMPUTE_CONSUMER_WARPS;
    static_assert(NUM_CONSUMER_WARPS==4||NUM_CONSUMER_WARPS==16);
    static constexpr int NUM_WARPS=NUM_CONSUMER_WARPS+4;
    static constexpr int NUM_THREADS=NUM_WARPS*32;
    static constexpr int CONSUMER_REGISTERS=NUM_CONSUMER_WARPS==4?200:104;
};
using Cfg = ComputeConfig;
using Output = kittens::gl<float,1,1,-1,-1>;
template<class Element> struct Globals {
    using Tile = kittens::st<Element,128,64>;
    using Input = kittens::gl<Element,1,1,-1,-1,Tile>;
    megakernel::instruction_layout<Cfg> instructions;
    megakernel::timing_layout<Cfg> timings;
    Input a,b;
    Output out;
    float alpha;
    int workers,tasks,m_tiles,n_tiles;
    kittens::bf16* bf16_output=nullptr;
    const kittens::bf16 *bias=nullptr,*residual=nullptr;
    int activation=0; // 0: identity; 1: ReLU²
    bool task_from_instruction=false;
    int round_before_residual=0,round_before_bias=0,zero_tail_from=-1;

};
using State = megakernel::state<Cfg>;
template<class Element> struct Matmul {
    using G = Globals<Element>;
    using Tile = typename G::Tile;
    static constexpr int opcode=1;
    template<class View> static __device__ int task(const View& g,State& s){
        if constexpr(requires{g.fixed_task;})return g.fixed_task;
        else return g.task_from_instruction?s.instruction()[2]:s.instruction_index*g.workers+blockIdx.x;
    }
    template<class View> static __device__ int row(const View& g,State& s){return (task(g,s)/g.n_tiles)%g.m_tiles;}
    template<class View> static __device__ int col(const View& g,State& s){return task(g,s)%g.n_tiles;}
    static __device__ Tile& tile(State& s,int i) { return *reinterpret_cast<Tile*>(s.pages[s.pid(i)].ptr()); }
    struct controller {
        static __device__ int release_lid(const G&,Cfg::instruction_t&,int& q){return q;}
        static __device__ int init_semaphores(const G&,State& s){
            for(int i=0;i<5;i++)kittens::init_semaphore(s.semaphores()[i],1);
            return 5;
        }
    };
    struct loader { template<class View> static __device__ void run(const View& g,State& s){
        if(task(g,s)>=g.tasks){megakernel::NoOp<Cfg>::loader::run(g,s);return;}
        int lane=kittens::laneid();
        int iterations=g.a.cols()/64,used=2*min(iterations,2);
        if(lane>=used && lane<Cfg::NUM_PAGES){int p=s.pid(lane);s.wait_page_ready(p);s.finish_page(p,Cfg::NUM_CONSUMER_WARPS);}
        if(lane==0){
            for(int i=0;i<iterations;i++){
                int ring=i%2;
                kittens::wait(s.semaphores()[2+ring],1-((i/2)&1));
                if(i<2){s.wait_page_ready(s.pid(2*ring));s.wait_page_ready(s.pid(2*ring+1));}
                kittens::tma::expect_bytes(s.semaphores()[ring],2*sizeof(Tile));
                kittens::tma::load_async(tile(s,2*ring),g.a,{row(g,s),i},s.semaphores()[ring]);
                kittens::tma::load_async(tile(s,2*ring+1),g.b,{col(g,s),i},s.semaphores()[ring]);
            }
        }
    }};
    struct launcher { template<class View> static __device__ void run(const View& g,State& s){
        if(task(g,s)>=g.tasks){megakernel::NoOp<Cfg>::launcher::run(g,s);return;}
        if(kittens::laneid()==0){
            s.wait_tensor_ready();
            auto acc=s.tensor_alloc.template allocate<kittens::tt<float,128,128>>(0);
            int iterations=g.a.cols()/64;
            for(int i=0;i<iterations;i++){
                int ring=i%2,phase=(i/2)&1;
                kittens::wait(s.semaphores()[ring],phase);
                if(i==0)kittens::mm<kittens::transpose::N,kittens::transpose::T>(acc,tile(s,2*ring),tile(s,2*ring+1),s.semaphores()[2+ring]);
                else kittens::mma<kittens::transpose::N,kittens::transpose::T>(acc,tile(s,2*ring),tile(s,2*ring+1),s.semaphores()[2+ring]);
                kittens::wait(s.semaphores()[2+ring],phase);
            }
            kittens::arrive(s.semaphores()[4]);
            for(int i=0;i<2*min(iterations,2);i++)s.finish_page(s.pid(i),Cfg::NUM_CONSUMER_WARPS);
        }
    }};
    struct consumer { template<class View> static __device__ void run(const View& g,State& s){
        if(task(g,s)>=g.tasks)return;
        kittens::wait(s.semaphores()[4],0);
        if(kittens::warpid()<4){
            auto acc=s.tensor_alloc.template allocate<kittens::tt<float,128,128>>(0);
            kittens::rt_fl<32,128> values;
            kittens::warpgroup::load_async(values,acc);
            kittens::tensor_load_wait();
            kittens::warp::mul(values,values,g.alpha);
            if(g.bf16_output){
                int lane=kittens::laneid(),warp=kittens::warpid();
                int row_base=(task(g,s)/g.n_tiles)*128,col_base=col(g,s)*128,n=g.out.cols();
                #pragma unroll
                for(int i=0;i<values.height;i++){
                    #pragma unroll
                    for(int j=0;j<values.width;j++){
                        #pragma unroll
                        for(int v=0;v<4;v++){
                            int r=row_base+warp*32+i*16+lane/4+((v&1)?8:0);
                            int c=col_base+j*16+2*(lane%4)+((v&2)?8:0);
                            auto pair=values.tiles[i][j].data[v];size_t index=size_t(r)*n+c;
                            if(g.residual){
                                if(g.round_before_residual){pair.x=__bfloat162float(__float2bfloat16_rn(pair.x));pair.y=__bfloat162float(__float2bfloat16_rn(pair.y));}
                                pair.x+=float(g.residual[index]);pair.y+=float(g.residual[index+1]);
                            }
                            if(g.bias){
                                if(g.round_before_bias){pair.x=__bfloat162float(__float2bfloat16_rn(pair.x));pair.y=__bfloat162float(__float2bfloat16_rn(pair.y));}
                                pair.x+=float(g.bias[c]);pair.y+=float(g.bias[c+1]);
                            }
                            if(g.activation){
                                if(g.activation==2){pair.x=__bfloat162float(__float2bfloat16_rn(pair.x));pair.y=__bfloat162float(__float2bfloat16_rn(pair.y));}
                                pair.x=fmaxf(pair.x,0.f);pair.y=fmaxf(pair.y,0.f);pair.x*=pair.x;pair.y*=pair.y;
                            }
                            if(g.zero_tail_from>=0){if(c>=g.zero_tail_from)pair.x=0.f;if(c+1>=g.zero_tail_from)pair.y=0.f;}
                            *reinterpret_cast<__nv_bfloat162*>(g.bf16_output+index)=__floats2bfloat162_rn(pair.x,pair.y);
                        }
                    }
                }
            }else kittens::warpgroup::store(g.out,values,{task(g,s)/g.n_tiles,col(g,s)});
        }
        kittens::warp::arrive(s.tensor_finished);
    }};
    using storer=megakernel::NoOp<Cfg>::storer;
};
__global__ void slots_kernel(int* slots){if(threadIdx.x==0)asm volatile("mov.u32 %0, %%nsmid;":"=r"(*slots));}

template<class Element> int run_probe(const void* a,const void* b,float* out,int repetitions,float alpha,int k=64,int m=128,int n=128,int workers=1){
    using G = Globals<Element>;
    using Input = typename G::Input;
    if(!a || !b || !out || repetitions<1 || k<64 || k%64 || m<128 || m%128 || n<128 || n%128 || workers<1)return cudaErrorInvalidValue;
    size_t a_bytes=size_t(m)*k*sizeof(Element),b_bytes=size_t(n)*k*sizeof(Element);
    int tasks=repetitions*(m/128)*(n/128),rows=(tasks+workers-1)/workers;
    int *slots_dev=nullptr,*code_dev=nullptr,*timing=nullptr;
    Element *a_dev=nullptr,*b_dev=nullptr;float* out_dev=nullptr;
    cudaError_t error=cudaMalloc(&slots_dev,sizeof(int));int slots=0;
    if(error==cudaSuccess){slots_kernel<<<1,32>>>(slots_dev);error=cudaGetLastError();}
    if(error==cudaSuccess)error=cudaMemcpy(&slots,slots_dev,sizeof(int),cudaMemcpyDeviceToHost);
    std::vector<int> code(size_t(slots)*rows*32,0);
    for(size_t i=0;i<code.size();i+=32)code[i]=1;
    if(error==cudaSuccess)error=cudaMalloc(&code_dev,code.size()*4);
    if(error==cudaSuccess)error=cudaMalloc(&timing,size_t(slots)*rows*128*4);
    if(error==cudaSuccess)error=cudaMalloc(&a_dev,a_bytes);
    if(error==cudaSuccess)error=cudaMalloc(&b_dev,b_bytes);
    if(error==cudaSuccess)error=cudaMalloc(&out_dev,size_t(repetitions)*m*n*4);
    if(error==cudaSuccess)error=cudaMemcpy(code_dev,code.data(),code.size()*4,cudaMemcpyHostToDevice);
    if(error==cudaSuccess)error=cudaMemcpy(a_dev,a,a_bytes,cudaMemcpyHostToDevice);
    if(error==cudaSuccess)error=cudaMemcpy(b_dev,b,b_bytes,cudaMemcpyHostToDevice);
    if(error==cudaSuccess){
        G g{megakernel::instruction_layout<Cfg>(code_dev,nullptr,slots,rows,nullptr),
                  megakernel::timing_layout<Cfg>(timing,nullptr,slots,rows,nullptr),
                  Input(a_dev,nullptr,nullptr,m,k),Input(b_dev,nullptr,nullptr,n,k),
                  Output(out_dev,nullptr,nullptr,repetitions*m,n),alpha,workers,tasks,m/128,n/128};
        auto kernel=megakernel::mk<Cfg,G,Matmul<Element>>;
        error=cudaFuncSetAttribute(kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,Cfg::DYNAMIC_SHARED_MEMORY);
        if(error==cudaSuccess){kernel<<<workers,Cfg::NUM_THREADS,Cfg::DYNAMIC_SHARED_MEMORY>>>(g);error=cudaGetLastError();}
        if(error==cudaSuccess)error=cudaMemcpy(out,out_dev,size_t(repetitions)*m*n*4,cudaMemcpyDeviceToHost);
    }
    cudaFree(slots_dev);cudaFree(code_dev);cudaFree(timing);cudaFree(a_dev);cudaFree(b_dev);cudaFree(out_dev);
    return error;
}
extern "C" int run_k_probe(const void* a,const void* b,float* out,int repetitions,int k,int precision,float activation_scale,float weight_scale){
    float alpha=activation_scale*weight_scale;
    if(!isfinite(alpha) || activation_scale<=0 || weight_scale<=0 || alpha<=0)return cudaErrorInvalidValue;
    if(precision==0)return run_probe<kittens::bf16>(a,b,out,repetitions,alpha,k);
    if(precision==1)return run_probe<kittens::fp8e4m3>(a,b,out,repetitions,alpha,k);
    return cudaErrorInvalidValue;
}

extern "C" int run_tcgen_probe(const void* a,const void* b,float* out,int repetitions){
    return run_probe<kittens::bf16>(a,b,out,repetitions,1.f);
}
extern "C" int run_fp8_probe(const void* a,const void* b,float* out,int repetitions,float activation_scale,float weight_scale){
    float alpha=activation_scale*weight_scale;
    if(!isfinite(alpha) || activation_scale<=0 || weight_scale<=0 || alpha<=0)return cudaErrorInvalidValue;
    return run_probe<kittens::fp8e4m3>(a,b,out,repetitions,alpha);
}

extern "C" int run_grid_probe(const void* a,const void* b,float* out,int repetitions,int m,int n,int k,int workers,int precision,float activation_scale,float weight_scale){
    float alpha=activation_scale*weight_scale;
    if(!isfinite(alpha) || activation_scale<=0 || weight_scale<=0 || alpha<=0)return cudaErrorInvalidValue;
    if(precision==0)return run_probe<kittens::bf16>(a,b,out,repetitions,alpha,k,m,n,workers);
    if(precision==1)return run_probe<kittens::fp8e4m3>(a,b,out,repetitions,alpha,k,m,n,workers);
    return cudaErrorInvalidValue;
}
