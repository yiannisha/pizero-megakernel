// Native BF16/FP8 TCGen05 projections with direct BF16 bias/residual epilogues.
#include "tcgen_probe.cu"
#include <climits>
static_assert(Cfg::NUM_CONSUMER_WARPS==4);
template<class Element> struct DenseGlobals {
    megakernel::instruction_layout<Cfg> instructions;
    megakernel::timing_layout<Cfg> timings;
    typename Globals<Element>::Input a,b;
    Output out;
    float alpha;
    int workers,tasks,m_tiles,n_tiles;
    const kittens::bf16 *bias,*residual;
    kittens::bf16* bf16_output;
    int round_before_residual=0;
};
template<class Element> struct DenseView {
    using Tile=typename Globals<Element>::Tile;
    struct InputMap {
        using identifier=kittens::ducks::gl::identifier;
        const CUtensorMap* map;int width;
        __device__ int cols()const{return width;}
        template<class T,int axis> __device__ const CUtensorMap* get_tma()const{
            static_assert(std::is_same_v<T,Tile> && axis==2);return map;
        }
    } a,b;
    Output out;float alpha;int workers,tasks,m_tiles,n_tiles,fixed_task;
    const kittens::bf16 *bias,*residual; kittens::bf16* bf16_output;
    int round_before_residual;
    __device__ DenseView(const DenseGlobals<Element>& g,int task):
        a{g.a.template get_tma<Tile,2>(),g.a.cols()},b{g.b.template get_tma<Tile,2>(),g.b.cols()},out(g.out),alpha(g.alpha),
        workers(g.workers),tasks(g.tasks),m_tiles(g.m_tiles),n_tiles(g.n_tiles),fixed_task(task),bias(g.bias),residual(g.residual),bf16_output(g.bf16_output),round_before_residual(g.round_before_residual){}
};
template<class Element> struct DenseOp {
    static constexpr int opcode=1;
    using G=DenseGlobals<Element>;
    static __device__ int task(const G& g,State& s){return s.instruction_index*g.workers+blockIdx.x;}
    struct controller {
        static __device__ int release_lid(const G&,Cfg::instruction_t&,int& q){return q;}
        static __device__ int init_semaphores(const G&,State& s){
            for(int i=0;i<5;++i)kittens::init_semaphore(s.semaphores()[i],1);return 5;
        }
    };
    struct loader {static __device__ void run(const G& g,State& s){
        DenseView<Element> view(g,task(g,s));Matmul<Element>::loader::run(view,s);
    }};
    struct launcher {static __device__ void run(const G& g,State& s){
        DenseView<Element> view(g,task(g,s));Matmul<Element>::launcher::run(view,s);
    }};
    template<class View> static __device__ void consume(const View& g,State& s,int phase=0,bool retire_tensor=true){
        int task=Matmul<Element>::task(g,s);if(task>=g.tasks)return;
        kittens::wait(s.semaphores()[4],phase);
        auto acc=s.tensor_alloc.template allocate<kittens::tt<float,128,128>>(0);
        kittens::rt_fl<32,128> values;kittens::warpgroup::load_async(values,acc);kittens::tensor_load_wait();
        kittens::warp::mul(values,values,g.alpha);
        if(!g.bf16_output&&!g.bias&&!g.residual)kittens::warpgroup::store(g.out,values,{task/g.n_tiles,task%g.n_tiles});
        else{
            int lane=kittens::laneid(),warp=kittens::warpid(),n=g.out.cols();
            int row_base=(task/g.n_tiles)*128,col_base=(task%g.n_tiles)*128;
#pragma unroll
            for(int i=0;i<values.height;++i){
#pragma unroll
                for(int j=0;j<values.width;++j){
#pragma unroll
                    for(int half=0;half<2;++half){
                        int r=row_base+warp*32+i*16+lane/4+half*8,c=col_base+j*16+2*(lane%4);
                        size_t index=size_t(r)*n+c;
                        auto lo=values.tiles[i][j].data[half],hi=values.tiles[i][j].data[half+2];
                        if(g.residual&&g.round_before_residual&&g.bias){
                            lo.x+=float(g.bias[c]);lo.y+=float(g.bias[c+1]);
                            hi.x+=float(g.bias[c+8]);hi.y+=float(g.bias[c+9]);
                        }
                        if(g.residual&&g.round_before_residual){
                            lo.x=float(kittens::bf16(lo.x));lo.y=float(kittens::bf16(lo.y));
                            hi.x=float(kittens::bf16(hi.x));hi.y=float(kittens::bf16(hi.y));
                        }
                        if(g.residual){lo.x+=float(g.residual[index]);lo.y+=float(g.residual[index+1]);hi.x+=float(g.residual[index+8]);hi.y+=float(g.residual[index+9]);}
                        if(g.bias&&!(g.residual&&g.round_before_residual)){lo.x+=float(g.bias[c]);lo.y+=float(g.bias[c+1]);hi.x+=float(g.bias[c+8]);hi.y+=float(g.bias[c+9]);}
                        if(g.bf16_output){
                            *reinterpret_cast<__nv_bfloat162*>(g.bf16_output+index)=__floats2bfloat162_rn(lo.x,lo.y);
                            *reinterpret_cast<__nv_bfloat162*>(g.bf16_output+index+8)=__floats2bfloat162_rn(hi.x,hi.y);
                        }else{
                            *reinterpret_cast<float2*>(g.out.raw_ptr+index)=make_float2(lo.x,lo.y);
                            *reinterpret_cast<float2*>(g.out.raw_ptr+index+8)=make_float2(hi.x,hi.y);
                        }
                    }
                }
            }
        }
        if(retire_tensor)kittens::warp::arrive(s.tensor_finished);
    }
    struct consumer {static __device__ void run(const G& g,State& s){DenseView<Element> view(g,task(g,s));consume(view,s);}};
    using storer=megakernel::NoOp<Cfg>::storer;
};
struct DenseContextBase {int device,fp8;};
template<class Element> struct DenseContext:DenseContextBase {
    DenseGlobals<Element> globals;int *program,*timing;
    DenseContext(int device,int fp8,DenseGlobals<Element> g,int* p,int* t):DenseContextBase{device,fp8},globals(g),program(p),timing(t){}
};
template<class Element> int create_dense_impl(void** handle,const void* a,const void* b,void* output,const void* bias,const void* residual,
    int m,int n,int k,int workers,int bf16,float alpha,int device){
    using G=DenseGlobals<Element>;using Input=typename Globals<Element>::Input;
    int *slots_dev=nullptr,*program=nullptr,*timing=nullptr,slots=0;
    int64_t tasks64=int64_t(m/128)*(n/128);if(tasks64>INT_MAX)return cudaErrorInvalidValue;
    int tasks=int(tasks64),iterations=int((tasks64+workers-1)/workers);
    auto error=cudaMalloc(&slots_dev,4);
    if(error==cudaSuccess){slots_kernel<<<1,32>>>(slots_dev);error=cudaGetLastError();}
    if(error==cudaSuccess)error=cudaMemcpy(&slots,slots_dev,4,cudaMemcpyDeviceToHost);
    cudaFree(slots_dev);if(error!=cudaSuccess)return error;
    std::vector<int> code(size_t(slots)*iterations*Cfg::INSTRUCTION_WIDTH,0);
    for(size_t i=0;i<code.size();i+=Cfg::INSTRUCTION_WIDTH)code[i]=1;
    error=cudaMalloc(&program,code.size()*4);
    if(error==cudaSuccess)error=cudaMalloc(&timing,size_t(slots)*iterations*Cfg::TIMING_WIDTH*4);
    if(error==cudaSuccess)error=cudaMemcpy(program,code.data(),code.size()*4,cudaMemcpyHostToDevice);
    auto kernel=megakernel::mk<Cfg,G,DenseOp<Element>>;
    if(error==cudaSuccess)error=cudaFuncSetAttribute(kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,Cfg::DYNAMIC_SHARED_MEMORY);
    if(error==cudaSuccess){
        G g{megakernel::instruction_layout<Cfg>(program,nullptr,slots,iterations,nullptr),megakernel::timing_layout<Cfg>(timing,nullptr,slots,iterations,nullptr),
            Input((Element*)a,nullptr,nullptr,m,k),Input((Element*)b,nullptr,nullptr,n,k),Output(bf16?nullptr:static_cast<float*>(output),nullptr,nullptr,m,n),
            alpha,workers,tasks,m/128,n/128,static_cast<const kittens::bf16*>(bias),static_cast<const kittens::bf16*>(residual),bf16?static_cast<kittens::bf16*>(output):nullptr};
        *handle=new DenseContext<Element>(device,std::is_same_v<Element,kittens::fp8e4m3>,g,program,timing);
    }else{cudaFree(program);cudaFree(timing);}
    return error;
}
extern "C" int create_dense(void** handle,const void* a,const void* b,void* output,const void* bias,const void* residual,
    int m,int n,int k,int workers,int fp8,int bf16,float alpha){
    if(!handle)return cudaErrorInvalidValue;*handle=nullptr;
    if(!a||!b||!output||m<128||m%128||n<128||n%128||k<64||k%64||workers<1||(fp8!=0&&fp8!=1)||(bf16!=0&&bf16!=1)||!isfinite(alpha)||alpha<=0)return cudaErrorInvalidValue;
    int device;cudaDeviceProp prop{};auto error=cudaGetDevice(&device);
    if(error==cudaSuccess)error=cudaGetDeviceProperties(&prop,device);if(error!=cudaSuccess)return error;
    if(prop.major!=10||prop.minor!=0||workers>prop.multiProcessorCount)return cudaErrorInvalidConfiguration;
    if(fp8)return create_dense_impl<kittens::fp8e4m3>(handle,a,b,output,bias,residual,m,n,k,workers,bf16,alpha,device);
    return create_dense_impl<kittens::bf16>(handle,a,b,output,bias,residual,m,n,k,workers,bf16,alpha,device);
}
template<class Element> int launch_dense_impl(DenseContextBase* base,void* stream){
    auto& c=*static_cast<DenseContext<Element>*>(base);
    megakernel::mk<Cfg,DenseGlobals<Element>,DenseOp<Element>><<<c.globals.workers,Cfg::NUM_THREADS,Cfg::DYNAMIC_SHARED_MEMORY,static_cast<cudaStream_t>(stream)>>>(c.globals);
    return cudaGetLastError();
}
extern "C" int launch_dense(void* handle,void* stream){
    if(!handle)return cudaErrorInvalidValue;auto* base=static_cast<DenseContextBase*>(handle);int device;auto error=cudaGetDevice(&device);
    if(error!=cudaSuccess)return error;if(device!=base->device)return cudaErrorInvalidDevice;
    return base->fp8?launch_dense_impl<kittens::fp8e4m3>(base,stream):launch_dense_impl<kittens::bf16>(base,stream);
}
template<class Element> int destroy_dense_impl(DenseContextBase* base){
    auto* c=static_cast<DenseContext<Element>*>(base);auto a=cudaFree(c->program),b=cudaFree(c->timing);delete c;return a==cudaSuccess?b:a;
}
extern "C" int destroy_dense(void* handle){
    if(!handle)return cudaSuccess;auto* base=static_cast<DenseContextBase*>(handle);int device;auto error=cudaGetDevice(&device);
    if(error!=cudaSuccess)return error;if(device!=base->device)return cudaErrorInvalidDevice;
    return base->fp8?destroy_dense_impl<kittens::fp8e4m3>(base):destroy_dense_impl<kittens::bf16>(base);
}

extern "C" int configure_dense_round_before_residual(void* handle){
    if(!handle)return cudaErrorInvalidValue;
    auto* base=static_cast<DenseContextBase*>(handle);
    if(base->fp8)return cudaErrorInvalidValue;
    auto* c=static_cast<DenseContext<kittens::bf16>*>(base);
    if(!c->globals.residual)return cudaErrorInvalidValue;
    c->globals.round_before_residual=1;return cudaSuccess;
}
