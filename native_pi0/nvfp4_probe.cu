// Development block-scaled FP4 data path in the original native VM.
// Instruction fields/descriptors follow pinned CUTLASS SM100 device primitives;
// no CUTLASS host GEMM adapter or external model kernel is used.
#include "tcgen_probe.cu"
#include <climits>
#ifndef NVFP4_OVERLAP_MMA
#define NVFP4_OVERLAP_MMA 0
#endif

#ifndef NATIVE_FP4_K_TILE
#define NATIVE_FP4_K_TILE 64
#endif
static_assert(NATIVE_FP4_K_TILE==64||NATIVE_FP4_K_TILE==128||NATIVE_FP4_K_TILE==256);
constexpr int FP4_K_GROUPS=NATIVE_FP4_K_TILE/64;
using PackedTile = kittens::st_fp8e4m3<128,NATIVE_FP4_K_TILE/2>; // opaque packed E2M1 bytes
using PackedInput = kittens::gl<kittens::fp8e4m3,1,1,-1,-1,PackedTile>;
struct Fp4Globals {
    megakernel::instruction_layout<Cfg> instructions;
    megakernel::timing_layout<Cfg> timings;
    PackedInput a,b;
    const unsigned char *sfa,*sfb;
    Output out;
    float alpha;
    int workers,tasks,m_tiles,n_tiles;
    const kittens::bf16 *bias=nullptr,*residual=nullptr;
    kittens::bf16* bf16_output=nullptr;
    unsigned char *fp4_output=nullptr,*output_scales=nullptr;
    float output_global_scale=1.f;
    int output_gelu=0;
    bool task_from_instruction=false;
};
struct Fp4Matmul {
    static constexpr int opcode=1;
    static_assert(sizeof(PackedTile)<=Cfg::PAGE_SIZE);
    template<class G> static __device__ int iterations(const G& g){return (g.a.cols()/32+FP4_K_GROUPS-1)/FP4_K_GROUPS;}
    template<class G> static __device__ int task(const G& g,State& s){
        if constexpr(requires { g.fixed_task; })return g.fixed_task;
        else return g.task_from_instruction?s.instruction()[2]:s.instruction_index*g.workers+blockIdx.x;
    }
    template<class G> static __device__ int row(const G& g,State& s){return (task(g,s)/g.n_tiles)%g.m_tiles;}
    template<class G> static __device__ int col(const G& g,State& s){return task(g,s)%g.n_tiles;}
    static __device__ PackedTile& tile(State& s,int i){return *reinterpret_cast<PackedTile*>(s.pages[s.pid(i)].ptr());}
    static __device__ unsigned char* sf(State& s,int ring){return reinterpret_cast<unsigned char*>(s.pages[s.pid(3*ring+2)].ptr());}
    struct controller {
        static __device__ int release_lid(const Fp4Globals&,Cfg::instruction_t&,int& q){return q;}
        static __device__ int init_semaphores(const Fp4Globals&,State& s){
            for(int i=0;i<7;i++)kittens::init_semaphore(s.semaphores()[i],1);
            return 7;
        }
    };
    struct loader {template<class G> static __device__ void run(const G& g,State& s){
        if(task(g,s)>=g.tasks){megakernel::NoOp<Cfg>::loader::run(g,s);return;}
        int lane=kittens::laneid();
        int groups=g.a.cols()/32,count=iterations(g),used=3*min(count,2);
        if(lane>=used && lane<Cfg::NUM_PAGES){int p=s.pid(lane);s.wait_page_ready(p);s.finish_page(p,Cfg::NUM_CONSUMER_WARPS);}
        for(int iteration=0;iteration<count;iteration++){
            int ring=iteration%2,phase=(iteration/2)&1;
            if(lane==0){
                kittens::wait(s.semaphores()[4+ring],1-phase);
                if(iteration<2)for(int i=0;i<3;i++)s.wait_page_ready(s.pid(3*ring+i));
                kittens::tma::expect_bytes(s.semaphores()[ring],2*sizeof(PackedTile));
                kittens::tma::load_async(tile(s,3*ring),g.a,{row(g,s),iteration},s.semaphores()[ring]);
                kittens::tma::load_async(tile(s,3*ring+1),g.b,{col(g,s),iteration},s.semaphores()[ring]);
            }
            __syncwarp();
            auto* destination=sf(s,ring);
            int a_row=row(g,s),b_row=col(g,s);
            #pragma unroll
            for(int inner=0;inner<FP4_K_GROUPS;inner++){
                int group=iteration*FP4_K_GROUPS+inner;
                uint4 av=make_uint4(0,0,0,0),bv=av;
                if(group<groups){
                    av=*reinterpret_cast<const uint4*>(g.sfa+(size_t(a_row)*groups+group)*512+lane*16);
                    bv=*reinterpret_cast<const uint4*>(g.sfb+(size_t(b_row)*groups+group)*512+lane*16);
                }
                *reinterpret_cast<uint4*>(destination+inner*512+lane*16)=av;
                *reinterpret_cast<uint4*>(destination+(FP4_K_GROUPS+inner)*512+lane*16)=bv;
            }
            asm volatile("fence.proxy.async.shared::cta;":::"memory");
            __syncwarp();
            if(lane==0)kittens::arrive(s.semaphores()[2+ring]);
        }
    }};
    static __device__ uint64_t scale_desc(const void* ptr){
        uint64_t address=__cvta_generic_to_shared(ptr);
        return (address>>4) | (1ull<<16) | (8ull<<32) | (1ull<<46);
    }
    static __device__ float gelu(float x){
        // CUTLASS GELU_taylor<float> operation order and fast_tanh primitive.
        constexpr float k0=.7978845608028654f,k1=k0*.044715f;
        float argument=x*fmaf(k1*x,x,k0),t;
        asm("tanh.approx.f32 %0, %1;":"=f"(t):"f"(argument));
        return .5f*fmaf(x,t,x);
    }
    static __device__ unsigned char packed_pair(float x,float y){
        unsigned short bits;
        asm("{ .reg .b8 b; cvt.rn.satfinite.e2m1x2.f32 b, %2, %1; cvt.u16.u8 %0, b; }"
            :"=h"(bits):"f"(x),"f"(y));
        return static_cast<unsigned char>(bits);
    }
    static __device__ float compact_gelu(float x){
        // FlashRT Sm100GeluMulCompactBlockScaleFactorRowStore::gelu_tanh_.
        return x/(1.f+expf(-1.5957691216057308f*x*(1.f+.044715f*x*x)));
    }
    struct launcher {template<class G> static __device__ void run(const G& g,State& s){
        if(task(g,s)>=g.tasks){megakernel::NoOp<Cfg>::launcher::run(g,s);return;}
        if(kittens::laneid()==0){
            s.wait_tensor_ready();
            int count=iterations(g);
            for(int iteration=0;iteration<count;iteration++){
                int ring=iteration%2,phase=(iteration/2)&1;
                kittens::wait(s.semaphores()[ring],phase);kittens::wait(s.semaphores()[2+ring],phase);
                auto* scales=sf(s,ring);
                kittens::st_descriptor<PackedTile,0> ad(tile(s,3*ring)),bd(tile(s,3*ring+1));
                #pragma unroll
                for(int inner=0;inner<FP4_K_GROUPS;inner++){
                    int bank=inner+(NVFP4_OVERLAP_MMA?ring*FP4_K_GROUPS:0);
                    uint32_t acc=s.tensor_alloc.addr,sa=acc+128+bank*8,sb=sa+4;
                    uint64_t as=scale_desc(scales+inner*512),bs=scale_desc(scales+(FP4_K_GROUPS+inner)*512);
                    asm volatile("tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;"::"r"(sa),"l"(as));
                    asm volatile("tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;"::"r"(sb),"l"(bs));
                    uint64_t a=ad.chunk_descriptor(inner),b=bd.chunk_descriptor(inner);
                    constexpr uint32_t desc=(5u<<7)|(5u<<10)|(16u<<17)|(8u<<24);
                    int accumulate=iteration*FP4_K_GROUPS+inner;
                    asm volatile("{ .reg .pred p; setp.ne.b32 p, %6, 0; tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::4X [%0], %1, %2, %3, [%4], [%5], p; }"
                                 ::"r"(acc),"l"(a),"l"(b),"r"(desc),"r"(sa),"r"(sb),"r"(accumulate));
                }
                kittens::detail::tcgen05::commit<1>(s.semaphores()[4+ring]);
                if constexpr (!NVFP4_OVERLAP_MMA)kittens::wait(s.semaphores()[4+ring],phase);
            }
            // Drain both slots before releasing shared pages or publishing TMEM.
            // The loader's finished waits protect slot reuse during the loop.
            if constexpr (NVFP4_OVERLAP_MMA){
                for(int ring=0;ring<min(count,2);ring++){
                    int last=count-1-((count-1-ring)&1);
                    kittens::wait(s.semaphores()[4+ring],(last/2)&1);
                }
            }
            kittens::arrive(s.semaphores()[6]);
            for(int i=0;i<3*min(count,2);i++)s.finish_page(s.pid(i),Cfg::NUM_CONSUMER_WARPS);
        }
    }};
    struct consumer {template<class G> static __device__ void run(const G& g,State& s){
        if(task(g,s)>=g.tasks)return;
        kittens::wait(s.semaphores()[6],0);
        if(kittens::warpid()<4){
            auto acc=s.tensor_alloc.template allocate<kittens::tt<float,128,128>>(0);
            kittens::rt_fl<32,128> values;kittens::warpgroup::load_async(values,acc);kittens::tensor_load_wait();
            kittens::warp::mul(values,values,g.alpha);
            if(g.fp4_output && g.output_gelu==2){
                int lane=kittens::laneid(),warp=kittens::warpid(),n=g.out.cols()/2;
                int row_base=(task(g,s)/g.n_tiles)*128,col_base=col(g,s)*64;
                float inv_global=1.f/g.output_global_scale;
                #pragma unroll
                for(int i=0;i<values.height;i++){
                    #pragma unroll
                    for(int j=0;j<values.width;j+=2){
                        #pragma unroll
                        for(int half=0;half<2;half++){
                            int r=row_base+warp*32+i*16+lane/4+half*8;
                            int c=col_base+j*8;
                            auto a=values.tiles[i][j].data[half],b=values.tiles[i][j].data[half+2];
                            auto d=values.tiles[i][j+1].data[half],e=values.tiles[i][j+1].data[half+2];
                            float v[4]={compact_gelu(a.x)*a.y*inv_global,compact_gelu(b.x)*b.y*inv_global,
                                        compact_gelu(d.x)*d.y*inv_global,compact_gelu(e.x)*e.y*inv_global};
                            float maximum=fmaxf(fmaxf(fabsf(v[0]),fabsf(v[1])),fmaxf(fabsf(v[2]),fabsf(v[3])));
                            maximum=fmaxf(maximum,__shfl_xor_sync(0xffffffff,maximum,1));
                            maximum=fmaxf(maximum,__shfl_xor_sync(0xffffffff,maximum,2));
                            __nv_fp8_e4m3 scale(fmaxf(maximum/6.f,1e-12f));
                            float decoded=float(scale),inverse=decoded>0?1.f/decoded:0.f;
                            #pragma unroll
                            for(int k=0;k<4;k++){
                                unsigned int code=decoded>0?(packed_pair(v[k]*inverse,0.f)&15):0;
                                unsigned int next=__shfl_down_sync(0xffffffff,code,1);
                                if((lane&1)==0)g.fp4_output[(size_t(r)*n+c+k*4+(lane&3))/2]=static_cast<unsigned char>(code|(next<<4));
                            }
                            if((lane&3)==0){
                                int block=c/16;
                                size_t offset=((size_t(r)/128)*((n/16+3)/4)+block/4)*512+(r%32)*16+((r%128)/32)*4+block%4;
                                g.output_scales[offset]=scale.__x;
                            }
                        }
                    }
                }
            }else if(g.fp4_output){
                int lane=kittens::laneid(),warp=kittens::warpid(),n=g.out.cols();
                int row_base=(task(g,s)/g.n_tiles)*128,col_base=col(g,s)*128;
                float inv_global=1.f/g.output_global_scale;
                #pragma unroll
                for(int i=0;i<values.height;i++){
                    #pragma unroll
                    for(int j=0;j<values.width;j++){
                        #pragma unroll
                        for(int half=0;half<2;half++){
                            int r=row_base+warp*32+i*16+lane/4+half*8;
                            int c=col_base+j*16+2*(lane%4);
                            auto lo=values.tiles[i][j].data[half];
                            auto hi=values.tiles[i][j].data[half+2];
                            size_t index=size_t(r)*n+c;
                            if(g.residual){lo.x+=float(g.residual[index]);lo.y+=float(g.residual[index+1]);hi.x+=float(g.residual[index+8]);hi.y+=float(g.residual[index+9]);}
                            if(g.bias){lo.x+=float(g.bias[c]);lo.y+=float(g.bias[c+1]);hi.x+=float(g.bias[c+8]);hi.y+=float(g.bias[c+9]);}
                            if(g.output_gelu){lo.x=gelu(lo.x);lo.y=gelu(lo.y);hi.x=gelu(hi.x);hi.y=gelu(hi.y);}
                            lo.x*=inv_global;lo.y*=inv_global;hi.x*=inv_global;hi.y*=inv_global;
                            float maximum=fmaxf(fmaxf(fabsf(lo.x),fabsf(lo.y)),fmaxf(fabsf(hi.x),fabsf(hi.y)));
                            maximum=fmaxf(maximum,__shfl_xor_sync(0xffffffff,maximum,1));
                            maximum=fmaxf(maximum,__shfl_xor_sync(0xffffffff,maximum,2));
                            __nv_fp8_e4m3 scale(maximum*(1.f/6.f));
                            float decoded=float(scale),inverse=decoded>0?1.f/decoded:0.f;
                            // Zero scales emit deterministic positive zero codes.
                            g.fp4_output[index/2]=decoded>0?packed_pair(lo.x*inverse,lo.y*inverse):0;
                            g.fp4_output[index/2+4]=decoded>0?packed_pair(hi.x*inverse,hi.y*inverse):0;
                            if((lane&3)==0){
                                int block=c/16;
                                size_t offset=((size_t(r)/128)*((n/16+3)/4)+block/4)*512+(r%32)*16+((r%128)/32)*4+block%4;
                                g.output_scales[offset]=scale.__x;
                            }
                        }
                    }
                }
            }else if(g.bf16_output){
                // Same register mapping as TK's row-major warpgroup store.
                // FlashRT LinCombPerColBias: alpha*acc + residual + bias,
                // one BF16 rounding at the final store, no global FP32 result.
                int lane=kittens::laneid(),warp=kittens::warpid();
                int row_base=(task(g,s)/g.n_tiles)*128,col_base=col(g,s)*128;
                #pragma unroll
                for(int i=0;i<values.height;i++){
                    #pragma unroll
                    for(int j=0;j<values.width;j++){
                        #pragma unroll
                        for(int v=0;v<4;v++){
                            int r=row_base+warp*32+i*16+lane/4+((v&1)?8:0);
                            int c=col_base+j*16+2*(lane%4)+((v&2)?8:0);
                            auto pair=values.tiles[i][j].data[v];
                            size_t index=size_t(r)*g.out.cols()+c;
                            if(g.residual){pair.x+=float(g.residual[index]);pair.y+=float(g.residual[index+1]);}
                            if(g.bias){pair.x+=float(g.bias[c]);pair.y+=float(g.bias[c+1]);}
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

int nvfp4_probe(const void* a,const void* b,const void* sa,const void* sb,float* out,int repetitions,float alpha,int k,int m=128,int n=128,int workers=1){
    if(!a||!b||!sa||!sb||!out||repetitions<1||k<64||k%64||m<128||m%128||n<128||n%128||workers<1||!isfinite(alpha)||alpha<=0)return cudaErrorInvalidValue;
    size_t a_bytes=size_t(m)*k/2,b_bytes=size_t(n)*k/2;
    size_t sa_bytes=size_t(m)*k/16,sb_bytes=size_t(n)*k/16;
    int tasks=repetitions*(m/128)*(n/128),rows=(tasks+workers-1)/workers;
    int *slots_dev=nullptr,*code_dev=nullptr,*timing=nullptr;int slots=0;
    kittens::fp8e4m3 *a_dev=nullptr,*b_dev=nullptr;unsigned char* scales=nullptr;float* output=nullptr;
    cudaError_t error=cudaMalloc(&slots_dev,4);
    if(error==cudaSuccess){slots_kernel<<<1,32>>>(slots_dev);error=cudaGetLastError();}
    if(error==cudaSuccess)error=cudaMemcpy(&slots,slots_dev,4,cudaMemcpyDeviceToHost);
    std::vector<int> code(size_t(slots)*rows*32,0);
    for(size_t i=0;i<code.size();i+=32)code[i]=1;
    if(error==cudaSuccess)error=cudaMalloc(&code_dev,code.size()*4);
    if(error==cudaSuccess)error=cudaMalloc(&timing,size_t(slots)*rows*128*4);
    if(error==cudaSuccess)error=cudaMalloc(&a_dev,a_bytes);
    if(error==cudaSuccess)error=cudaMalloc(&b_dev,b_bytes);
    if(error==cudaSuccess)error=cudaMalloc(&scales,sa_bytes+sb_bytes);
    if(error==cudaSuccess)error=cudaMalloc(&output,size_t(repetitions)*m*n*4);
    if(error==cudaSuccess)error=cudaMemcpy(code_dev,code.data(),code.size()*4,cudaMemcpyHostToDevice);
    if(error==cudaSuccess)error=cudaMemcpy(a_dev,a,a_bytes,cudaMemcpyHostToDevice);
    if(error==cudaSuccess)error=cudaMemcpy(b_dev,b,b_bytes,cudaMemcpyHostToDevice);
    if(error==cudaSuccess)error=cudaMemcpy(scales,sa,sa_bytes,cudaMemcpyHostToDevice);
    if(error==cudaSuccess)error=cudaMemcpy(scales+sa_bytes,sb,sb_bytes,cudaMemcpyHostToDevice);
    if(error==cudaSuccess){
        Fp4Globals g{megakernel::instruction_layout<Cfg>(code_dev,nullptr,slots,rows,nullptr),
            megakernel::timing_layout<Cfg>(timing,nullptr,slots,rows,nullptr),
            PackedInput(a_dev,nullptr,nullptr,m,k/2),PackedInput(b_dev,nullptr,nullptr,n,k/2),
            scales,scales+sa_bytes,Output(output,nullptr,nullptr,repetitions*m,n),alpha,workers,tasks,m/128,n/128};
        auto kernel=megakernel::mk<Cfg,Fp4Globals,Fp4Matmul>;
        error=cudaFuncSetAttribute(kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,Cfg::DYNAMIC_SHARED_MEMORY);
        if(error==cudaSuccess){kernel<<<workers,Cfg::NUM_THREADS,Cfg::DYNAMIC_SHARED_MEMORY>>>(g);error=cudaGetLastError();}
        if(error==cudaSuccess)error=cudaMemcpy(out,output,size_t(repetitions)*m*n*4,cudaMemcpyDeviceToHost);
    }
    cudaFree(slots_dev);cudaFree(code_dev);cudaFree(timing);cudaFree(a_dev);cudaFree(b_dev);cudaFree(scales);cudaFree(output);
    return error;
}

extern "C" int run_nvfp4_probe(const void* a,const void* b,const void* sa,const void* sb,float* out,int repetitions,float alpha,int k){
    return nvfp4_probe(a,b,sa,sb,out,repetitions,alpha,k);
}
extern "C" int run_nvfp4_grid_probe(const void* a,const void* b,const void* sa,const void* sb,float* out,int repetitions,int m,int n,int k,int workers,float alpha){
    return nvfp4_probe(a,b,sa,sb,out,repetitions,alpha,k,m,n,workers);
}

// Resident development primitive: inputs/output are supplied device pointers.
// Only program/timing storage is owned here. Tensor owners must outlive handle.
struct ResidentFp4 {
    Fp4Globals globals;
    int *program,*timing;
    int device;
};
extern "C" int create_resident_fp4(void** handle,void* a,void* b,const void* sa,const void* sb,void* out,
                                   int m,int n,int k,int workers,float alpha){
    if(!handle)return cudaErrorInvalidValue;
    *handle=nullptr;
    if(!a||!b||!sa||!sb||!out||m<128||m%128||n<128||n%128||k<64||k%64||workers<1||!isfinite(alpha)||alpha<=0)
        return cudaErrorInvalidValue;
    int device=0;cudaDeviceProp properties{};
    cudaError_t error=cudaGetDevice(&device);
    if(error==cudaSuccess)error=cudaGetDeviceProperties(&properties,device);
    if(error!=cudaSuccess)return error;
    if(properties.major!=10||properties.minor!=0||workers>properties.multiProcessorCount)return cudaErrorInvalidConfiguration;
    int64_t task_count=int64_t(m/128)*(n/128);
    if(task_count>INT_MAX)return cudaErrorInvalidValue;
    int tasks=int(task_count),rows=(tasks+workers-1)/workers;
    int *slots_dev=nullptr,*program=nullptr,*timing=nullptr;int slots=0;
    error=cudaMalloc(&slots_dev,sizeof(int));
    if(error==cudaSuccess){slots_kernel<<<1,32>>>(slots_dev);error=cudaGetLastError();}
    if(error==cudaSuccess)error=cudaMemcpy(&slots,slots_dev,sizeof(int),cudaMemcpyDeviceToHost);
    cudaFree(slots_dev);
    std::vector<int> code(size_t(slots)*rows*32,0);
    for(size_t i=0;i<code.size();i+=32)code[i]=1;
    if(error==cudaSuccess)error=cudaMalloc(&program,code.size()*sizeof(int));
    if(error==cudaSuccess)error=cudaMalloc(&timing,size_t(slots)*rows*128*sizeof(int));
    if(error==cudaSuccess)error=cudaMemcpy(program,code.data(),code.size()*sizeof(int),cudaMemcpyHostToDevice);
    auto kernel=megakernel::mk<Cfg,Fp4Globals,Fp4Matmul>;
    if(error==cudaSuccess)error=cudaFuncSetAttribute(kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,Cfg::DYNAMIC_SHARED_MEMORY);
    if(error==cudaSuccess){
        Fp4Globals g{megakernel::instruction_layout<Cfg>(program,nullptr,slots,rows,nullptr),
            megakernel::timing_layout<Cfg>(timing,nullptr,slots,rows,nullptr),
            PackedInput(static_cast<kittens::fp8e4m3*>(a),nullptr,nullptr,m,k/2),
            PackedInput(static_cast<kittens::fp8e4m3*>(b),nullptr,nullptr,n,k/2),
            static_cast<const unsigned char*>(sa),static_cast<const unsigned char*>(sb),
            Output(static_cast<float*>(out),nullptr,nullptr,m,n),alpha,workers,tasks,m/128,n/128};
        *handle=new ResidentFp4{g,program,timing,device};
    }else{cudaFree(program);cudaFree(timing);}
    return error;
}
extern "C" int launch_resident_fp4(void* handle,void* stream){
    if(!handle)return cudaErrorInvalidValue;
    auto& context=*static_cast<ResidentFp4*>(handle);
    int device;cudaError_t error=cudaGetDevice(&device);
    if(error!=cudaSuccess)return error;
    if(device!=context.device)return cudaErrorInvalidDevice;
    auto kernel=megakernel::mk<Cfg,Fp4Globals,Fp4Matmul>;
    kernel<<<context.globals.workers,Cfg::NUM_THREADS,Cfg::DYNAMIC_SHARED_MEMORY,static_cast<cudaStream_t>(stream)>>>(context.globals);
    return cudaGetLastError();
}
extern "C" int configure_resident_fp4_bf16(void* handle,const void* bias,const void* residual,void* output){
    if(!handle||!output)return cudaErrorInvalidValue;
    auto& context=*static_cast<ResidentFp4*>(handle);
    context.globals.bias=static_cast<const kittens::bf16*>(bias);
    context.globals.residual=static_cast<const kittens::bf16*>(residual);
    context.globals.bf16_output=static_cast<kittens::bf16*>(output);
    context.globals.fp4_output=nullptr;context.globals.output_scales=nullptr;
    return cudaSuccess;
}
extern "C" int configure_resident_fp4_quantized(void* handle,const void* bias,const void* residual,void* output,void* scales,float global_scale,int gelu){
    if(!handle||!output||!scales||!isfinite(global_scale)||global_scale<=0||gelu<0||gelu>2)return cudaErrorInvalidValue;
    if(gelu==2&&(bias||residual))return cudaErrorInvalidValue;
    auto& g=static_cast<ResidentFp4*>(handle)->globals;
    g.bias=static_cast<const kittens::bf16*>(bias);g.residual=static_cast<const kittens::bf16*>(residual);
    g.fp4_output=static_cast<unsigned char*>(output);g.output_scales=static_cast<unsigned char*>(scales);
    g.output_global_scale=global_scale;g.output_gelu=gelu;
    g.bf16_output=nullptr;
    return cudaSuccess;
}
extern "C" int resident_fp4_resources(void* handle,int* values,int count){
    if(!handle||!values||count<8)return cudaErrorInvalidValue;
    auto& context=*static_cast<ResidentFp4*>(handle);
    int device;cudaError_t error=cudaGetDevice(&device);
    if(error!=cudaSuccess)return error;
    if(device!=context.device)return cudaErrorInvalidDevice;
    cudaFuncAttributes attributes{};cudaDeviceProp properties{};int blocks=0;
    auto kernel=megakernel::mk<Cfg,Fp4Globals,Fp4Matmul>;
    error=cudaFuncGetAttributes(&attributes,kernel);
    if(error==cudaSuccess)error=cudaGetDeviceProperties(&properties,device);
    if(error==cudaSuccess)error=cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,kernel,Cfg::NUM_THREADS,Cfg::DYNAMIC_SHARED_MEMORY);
    if(error!=cudaSuccess)return error;
    values[0]=attributes.numRegs;values[1]=attributes.sharedSizeBytes;
    values[2]=Cfg::DYNAMIC_SHARED_MEMORY;values[3]=attributes.localSizeBytes;
    values[4]=blocks;values[5]=Cfg::NUM_THREADS;
    values[6]=properties.maxThreadsPerMultiProcessor;values[7]=properties.multiProcessorCount;
    return cudaSuccess;
}
extern "C" int destroy_resident_fp4(void* handle){
    if(!handle)return cudaSuccess;
    auto* context=static_cast<ResidentFp4*>(handle);
    int device;cudaError_t error=cudaGetDevice(&device);
    if(error!=cudaSuccess)return error;
    if(device!=context->device)return cudaErrorInvalidDevice;
    // Freeing program storage waits for pending users before releasing tensors.
    error=cudaFree(context->program);cudaError_t timing_error=cudaFree(context->timing);
    delete context;
    return error==cudaSuccess?timing_error:error;
}
