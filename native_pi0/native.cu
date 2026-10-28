// Static native instruction streams: no dynamic fetch, task queue or DAG VM.
#include "dense.cu"
#include <limits>

// Original static VM fetch is unchanged. Logical CTA IDs identify streams;
// physical SM IDs can be sparse and must not identify unique work ownership.
struct NativeInstructions {
    int* data;int length;
    __device__ int rows()const{return length;}
    __device__ int& operator[](const kittens::coord<>& c)const{return data[(size_t(blockIdx.x)*length+c.r)*32+c.c];}
};
struct NativeTimings {
    int* data;int length;
    __device__ int& operator[](const kittens::coord<>& c)const{return data[(size_t(blockIdx.x)*length+c.r)*128+c.c];}
};
struct NativeMatrix {
    DenseGlobals<kittens::bf16> matrix;
    int input_counter,output_counter,input_tiles;
    const int* input_expected=nullptr;
};
struct NativeGlobals {
    static constexpr bool DYNAMIC_INSTRUCTIONS=false;
    NativeInstructions instructions;NativeTimings timings;
    const NativeMatrix* matrices;unsigned long long* ready;unsigned long long epoch;
};
static_assert(!megakernel::dynamic_instructions_v<NativeGlobals>);
__device__ inline unsigned long long native_acquire(const unsigned long long* p){
    unsigned long long value;asm volatile("ld.acquire.gpu.global.u64 %0, [%1];":"=l"(value):"l"(p):"memory");return value;
}
__device__ inline void native_publish(unsigned long long* p,unsigned count){
    unsigned long long prior;asm volatile("atom.acq_rel.gpu.global.add.u64 %0, [%1], %2;":"=l"(prior):"l"(p),"l"(static_cast<unsigned long long>(count)):"memory");
}
struct NativeLinearOp {
    static constexpr int opcode=1;
    using Tile=typename Globals<kittens::bf16>::Tile;
    static __device__ const NativeMatrix& stage(const NativeGlobals& g,State& s){return g.matrices[s.instruction()[1]];}
    static __device__ Tile& tile(State& s,int i){return *reinterpret_cast<Tile*>(s.pages[s.pid(i)].ptr());}
    struct controller {
        static __device__ int release_lid(const NativeGlobals& g,Cfg::instruction_t& code,int& q){
            int used=2*min(code[3]*(g.matrices[code[1]].matrix.a.cols()/64),2);
            return q<Cfg::NUM_PAGES-used?used+q:q-(Cfg::NUM_PAGES-used);
        }
        static __device__ int init_semaphores(const NativeGlobals&,State& s){
            for(int i=0;i<5;++i)kittens::init_semaphore(s.semaphores()[i],1);
            kittens::init_semaphore(s.semaphores()[5],4);return 6;
        }
    };
    struct loader {static __device__ void run(const NativeGlobals& g,State& s){
        const auto& st=stage(g,s);const auto& f=st.matrix;
        int count=s.instruction()[3],first=s.instruction()[2],iterations=f.a.cols()/64;
        int used=2*min(count*iterations,2),lane=kittens::laneid();
        if(lane>=used&&lane<Cfg::NUM_PAGES){int p=s.pid(lane);s.wait_page_ready(p);s.finish_page(p,4);}
        if(lane==0){
            const auto *a=f.a.template get_tma<Tile,2>(),*b=f.b.template get_tma<Tile,2>();
            asm volatile("fence.proxy.tensormap::generic.acquire.gpu [%0], 128;"::"l"(a):"memory");
            asm volatile("fence.proxy.tensormap::generic.acquire.gpu [%0], 128;"::"l"(b):"memory");
            int last_band=-1;
            for(int t=0;t<count;++t){
                int task=first+t,band=task/f.n_tiles;
                if(band!=last_band&&st.input_counter>=0){
                    auto expected=g.epoch*(st.input_expected?st.input_expected[band]:st.input_tiles);
                    while(native_acquire(g.ready+st.input_counter+band)<expected)__nanosleep(20);
                }
                last_band=band;
                for(int k=0;k<iterations;++k){
                    int index=t*iterations+k,ring=index&1,phase=(index/2)&1;
                    kittens::wait(s.semaphores()[2+ring],1-phase);
                    if(index<2){s.wait_page_ready(s.pid(2*ring));s.wait_page_ready(s.pid(2*ring+1));}
                    kittens::tma::expect_bytes(s.semaphores()[ring],2*sizeof(Tile));
                    kittens::tma::load_async(tile(s,2*ring),f.a,{band,k},s.semaphores()[ring]);
                    kittens::tma::load_async(tile(s,2*ring+1),f.b,{task%f.n_tiles,k},s.semaphores()[ring]);
                }
            }
        }
    }};
    struct launcher {static __device__ void run(const NativeGlobals& g,State& s){
        if(kittens::laneid()==0){
            const auto& f=stage(g,s).matrix;int count=s.instruction()[3],iterations=f.a.cols()/64;
            s.wait_tensor_ready();auto acc=s.tensor_alloc.template allocate<kittens::tt<float,128,128>>(0);
            for(int t=0;t<count;++t){
                // The consumer acknowledges its TMEM load/store before the
                // next tile resets the same accumulator. Pages remain owned.
                kittens::wait(s.semaphores()[5],1-(t&1));
                for(int k=0;k<iterations;++k){
                    int index=t*iterations+k,ring=index&1,phase=(index/2)&1;
                    kittens::wait(s.semaphores()[ring],phase);
                    if(k==0)kittens::mm<kittens::transpose::N,kittens::transpose::T>(acc,tile(s,2*ring),tile(s,2*ring+1),s.semaphores()[2+ring]);
                    else kittens::mma<kittens::transpose::N,kittens::transpose::T>(acc,tile(s,2*ring),tile(s,2*ring+1),s.semaphores()[2+ring]);
                    kittens::wait(s.semaphores()[2+ring],phase);
                }
                kittens::arrive(s.semaphores()[4]);
            }
            for(int i=0;i<2*min(count*iterations,2);++i)s.finish_page(s.pid(i),4);
        }
    }};
    struct consumer {static __device__ void run(const NativeGlobals& g,State& s){
        const auto& st=stage(g,s);int first=s.instruction()[2],count=s.instruction()[3],contribution=0;
        for(int t=0;t<count;++t){
            int task=first+t,band=task/st.matrix.n_tiles;
            const DenseView<kittens::bf16> view(st.matrix,task);
            DenseOp<kittens::bf16>::consume(view,s,t&1,false);
            ++contribution;
            if(t+1==count||(task+1)/st.matrix.n_tiles!=band){
                asm volatile("fence.proxy.async.global;":::"memory");
                asm volatile("bar.sync 6, 128;":::"memory");
                if(threadIdx.x==0)native_publish(g.ready+st.output_counter+band,contribution);
                contribution=0;
            }
            asm volatile("bar.sync 6, 128;":::"memory");
            kittens::warp::arrive(s.semaphores()[5]);
        }
        kittens::warp::arrive(s.tensor_finished);
    }};
    using storer=megakernel::NoOp<Cfg>::storer;
};
struct NativeLinearContext {
    NativeMatrix* matrices;unsigned long long* ready;int *program,*timings;
    int workers,length,counters,device,max_tiles;unsigned long long epoch=0;
};
extern "C" int create_native_linear(void** handle,int count,void* const* handles,const int* sources,const int* code,int workers,int length){
    if(!handle)return cudaErrorInvalidValue;*handle=nullptr;
    if(count<1||!handles||!sources||!code||workers<1||length<1)return cudaErrorInvalidValue;
    int device;auto error=cudaGetDevice(&device);cudaDeviceProp props{};
    if(error==cudaSuccess)error=cudaGetDeviceProperties(&props,device);if(error!=cudaSuccess)return error;
    auto kernel=megakernel::mk<Cfg,NativeGlobals,NativeLinearOp>;
    error=cudaFuncSetAttribute(kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,Cfg::DYNAMIC_SHARED_MEMORY);
    int occupancy=0;if(error==cudaSuccess)error=cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occupancy,kernel,Cfg::NUM_THREADS,Cfg::DYNAMIC_SHARED_MEMORY);
    if(error!=cudaSuccess)return error;if(occupancy<1||workers>props.multiProcessorCount)return cudaErrorInvalidConfiguration;
    std::vector<NativeMatrix> matrices;std::vector<std::vector<unsigned char>> seen;int counters=0,max_tiles=0;
    for(int i=0;i<count;++i){
        auto* base=static_cast<DenseContextBase*>(handles[i]);if(!base||base->device!=device||base->fp8||sources[i]<-1||sources[i]>=i)return cudaErrorInvalidValue;
        auto& f=static_cast<DenseContext<kittens::bf16>*>(base)->globals;if(!f.bf16_output)return cudaErrorInvalidValue;
        int input=-1,input_tiles=0;
        if(sources[i]>=0){
            const auto& parent=matrices[sources[i]];
            if(parent.matrix.out.rows()!=f.a.rows()||parent.matrix.out.cols()!=f.a.cols())return cudaErrorInvalidValue;
            f.a=decltype(f.a)(parent.matrix.bf16_output,nullptr,nullptr,f.a.rows(),f.a.cols());
            input=parent.output_counter;input_tiles=parent.matrix.n_tiles;
        }
        matrices.push_back({f,input,counters,input_tiles});counters+=f.m_tiles;max_tiles=std::max(max_tiles,f.n_tiles);
        seen.emplace_back(f.tasks,0);
    }
    for(int worker=0;worker<workers;++worker){
        int prior=-1;
        for(int row=0;row<length;++row){
            const int* in=code+(size_t(worker)*length+row)*32;if(in[0]==0)continue;
            int stage=in[1],first=in[2],tiles=in[3];
            if(in[0]!=1||stage<prior||stage<0||stage>=count||first<0||tiles<1||int64_t(first)+tiles>matrices[stage].matrix.tasks)return cudaErrorInvalidValue;
            prior=stage;
            for(int t=first;t<first+tiles;++t){if(seen[stage][t])return cudaErrorInvalidValue;seen[stage][t]=1;}
        }
    }
    for(const auto& row:seen)if(std::count(row.begin(),row.end(),1)!=row.size())return cudaErrorInvalidValue;
    NativeMatrix* dm=nullptr;unsigned long long* ready=nullptr;int *dp=nullptr,*dt=nullptr;
    error=cudaMalloc(&dm,matrices.size()*sizeof(matrices[0]));if(error==cudaSuccess)error=cudaMemcpy(dm,matrices.data(),matrices.size()*sizeof(matrices[0]),cudaMemcpyHostToDevice);
    if(error==cudaSuccess)error=cudaMalloc(&ready,size_t(counters)*8);if(error==cudaSuccess)error=cudaMemset(ready,0,size_t(counters)*8);
    size_t bytes=size_t(workers)*length*32*4;
    if(error==cudaSuccess)error=cudaMalloc(&dp,bytes);if(error==cudaSuccess)error=cudaMemcpy(dp,code,bytes,cudaMemcpyHostToDevice);
    if(error==cudaSuccess)error=cudaMalloc(&dt,size_t(workers)*length*128*4);
    if(error==cudaSuccess)*handle=new NativeLinearContext{dm,ready,dp,dt,workers,length,counters,device,max_tiles};
    else{cudaFree(dm);cudaFree(ready);cudaFree(dp);cudaFree(dt);}return error;
}
extern "C" int launch_native_linear(void* handle,void* stream){
    if(!handle)return cudaErrorInvalidValue;auto& c=*static_cast<NativeLinearContext*>(handle);int device;auto error=cudaGetDevice(&device);
    if(error!=cudaSuccess)return error;if(device!=c.device)return cudaErrorInvalidDevice;
    cudaStreamCaptureStatus capture;error=cudaStreamIsCapturing(static_cast<cudaStream_t>(stream),&capture);
    if(error!=cudaSuccess)return error;if(capture!=cudaStreamCaptureStatusNone)return cudaErrorStreamCaptureUnsupported;
    if(c.epoch>=std::numeric_limits<unsigned long long>::max()/c.max_tiles)return cudaErrorInvalidValue;
    NativeGlobals g{{c.program,c.length},{c.timings,c.length},c.matrices,c.ready,++c.epoch};
    megakernel::mk<Cfg,NativeGlobals,NativeLinearOp><<<c.workers,Cfg::NUM_THREADS,Cfg::DYNAMIC_SHARED_MEMORY,static_cast<cudaStream_t>(stream)>>>(g);
    return cudaGetLastError();
}
extern "C" int read_native_linear(void* handle,unsigned long long* output,int count,unsigned long long* epoch){
    if(!handle||!output||!epoch)return cudaErrorInvalidValue;auto& c=*static_cast<NativeLinearContext*>(handle);
    if(count!=c.counters)return cudaErrorInvalidValue;auto error=cudaMemcpy(output,c.ready,size_t(count)*8,cudaMemcpyDeviceToHost);*epoch=c.epoch;return error;
}
extern "C" int destroy_native_linear(void* handle){
    if(!handle)return cudaSuccess;auto* c=static_cast<NativeLinearContext*>(handle);int device;auto error=cudaGetDevice(&device);
    if(error!=cudaSuccess)return error;if(device!=c->device)return cudaErrorInvalidDevice;
    for(void* p:{static_cast<void*>(c->matrices),static_cast<void*>(c->ready),static_cast<void*>(c->program),static_cast<void*>(c->timings)}){auto e=cudaFree(p);if(error==cudaSuccess)error=e;}
    delete c;return error;
}
