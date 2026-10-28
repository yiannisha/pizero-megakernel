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
    int residual_counter=-1;const int* residual_expected=nullptr;
    int factor_counter=-1;const int* factor_expected=nullptr;
    int raw_qkv_counter=-1;
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
template<int TileRows> struct NativeLinearOpT {
    static_assert(TileRows==64||TileRows==128);
    static constexpr int opcode=TileRows==64?16:1;
    using TileA=kittens::st<kittens::bf16,TileRows,64>;
    using TileB=typename Globals<kittens::bf16>::Tile;
    static __device__ const NativeMatrix& stage(const NativeGlobals& g,State& s){return g.matrices[s.instruction()[1]];}
    static __device__ TileA& tile_a(State& s,int i){return *reinterpret_cast<TileA*>(s.pages[s.pid(i)].ptr());}
    static __device__ TileB& tile_b(State& s,int i){return *reinterpret_cast<TileB*>(s.pages[s.pid(i)].ptr());}
    static __device__ void finish_qkv(const NativeMatrix& st,int task){
        const auto& f=st.matrix;
        int band=task/f.n_tiles,col_base=(task%f.n_tiles)*128,row_base=band*128;
        auto* out=f.bf16_output;int n=f.out.cols(),limit=min(row_base+TileRows,f.rope_rows);
        if(col_base<f.rope_key_offset+256 && col_base%256==0){
            for(int i=threadIdx.x;i<(limit-row_base)*128;i+=128){
                int row=row_base+i/128,d=i%128;
                size_t lo=size_t(row)*n+col_base+d,hi=lo+128;
                float x=float(out[lo]),y=float(out[hi]);
                float c0=float(f.rope_cos[size_t(row)*256+d]),s0=float(f.rope_sin[size_t(row)*256+d]);
                float c1=float(f.rope_cos[size_t(row)*256+d+128]),s1=float(f.rope_sin[size_t(row)*256+d+128]);
                kittens::bf16 q0(float(kittens::bf16(x*c0))+float(kittens::bf16(-y*s0)));
                kittens::bf16 q1(float(kittens::bf16(y*c1))+float(kittens::bf16(x*s1)));
                out[lo]=q0;out[hi]=q1;
                if(col_base==f.rope_key_offset){
                    size_t cached=size_t(f.rope_cache_base+row)*256+d;
                    f.rope_cache_k[cached]=q0;f.rope_cache_k[cached+128]=q1;
                }
            }
        }else if(col_base>=f.rope_key_offset+256&&col_base<f.rope_key_offset+512){
            for(int i=threadIdx.x;i<(limit-row_base)*128;i+=128){
                int row=row_base+i/128,d=i%128;
                size_t source=size_t(row)*n+col_base+d;
                size_t cached=size_t(f.rope_cache_base+row)*256+col_base-(f.rope_key_offset+256)+d;
                f.rope_cache_v[cached]=out[source];
            }
        }
    }
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
            const auto *a=f.a.template get_tma<TileA,2>(),*b=f.b.template get_tma<TileB,2>();
            asm volatile("fence.proxy.tensormap::generic.acquire.gpu [%0], 128;"::"l"(a):"memory");
            asm volatile("fence.proxy.tensormap::generic.acquire.gpu [%0], 128;"::"l"(b):"memory");
            int last_band=-1;
            for(int t=0;t<count;++t){
                int task=first+t,band=task/f.n_tiles;
                if(band!=last_band&&st.input_counter>=0){
                    auto expected=g.epoch*(st.input_expected?st.input_expected[band]:st.input_tiles);
                    while(native_acquire(g.ready+st.input_counter+band)<expected)__nanosleep(20);
                }
                if(band!=last_band&&st.residual_counter>=0){
                    auto expected=g.epoch*st.residual_expected[band];
                    while(native_acquire(g.ready+st.residual_counter+band)<expected)__nanosleep(20);
                }
                if(band!=last_band&&st.factor_counter>=0){
                    auto expected=g.epoch*st.factor_expected[band];
                    while(native_acquire(g.ready+st.factor_counter+band)<expected)__nanosleep(20);
                }
                last_band=band;
                for(int k=0;k<iterations;++k){
                    int index=t*iterations+k,ring=index&1,phase=(index/2)&1;
                    kittens::wait(s.semaphores()[2+ring],1-phase);
                    if(index<2){s.wait_page_ready(s.pid(2*ring));s.wait_page_ready(s.pid(2*ring+1));}
                    kittens::tma::expect_bytes(s.semaphores()[ring],sizeof(TileA)+sizeof(TileB));
                    kittens::tma::load_async(tile_a(s,2*ring),f.a,{band,k},s.semaphores()[ring]);
                    kittens::tma::load_async(tile_b(s,2*ring+1),f.b,{task%f.n_tiles,k},s.semaphores()[ring]);
                }
            }
        }
    }};
    struct launcher {static __device__ void run(const NativeGlobals& g,State& s){
        if(kittens::laneid()==0){
            const auto& f=stage(g,s).matrix;int count=s.instruction()[3],iterations=f.a.cols()/64;
            s.wait_tensor_ready();auto acc=dense_accumulator<TileRows>(s);
            for(int t=0;t<count;++t){
                // The consumer acknowledges its TMEM load/store before the
                // next tile resets the same accumulator. Pages remain owned.
                kittens::wait(s.semaphores()[5],1-(t&1));
                for(int k=0;k<iterations;++k){
                    int index=t*iterations+k,ring=index&1,phase=(index/2)&1;
                    kittens::wait(s.semaphores()[ring],phase);
                    if(k==0)kittens::mm<kittens::transpose::N,kittens::transpose::T>(acc,tile_a(s,2*ring),tile_b(s,2*ring+1),s.semaphores()[2+ring]);
                    else kittens::mma<kittens::transpose::N,kittens::transpose::T>(acc,tile_a(s,2*ring),tile_b(s,2*ring+1),s.semaphores()[2+ring]);
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
            DenseOp<kittens::bf16>::template consume<TileRows>(view,s,t&1,false);
            if(st.raw_qkv_counter>=0){
                asm volatile("fence.proxy.async.global;":::"memory");
                asm volatile("bar.sync 6, 128;":::"memory");
                if(threadIdx.x==0){
                    native_publish(g.ready+st.raw_qkv_counter+band,1);
                    while(native_acquire(g.ready+st.raw_qkv_counter+band)<g.epoch*st.matrix.n_tiles)__nanosleep(20);
                }
                asm volatile("bar.sync 6, 128;":::"memory");
                finish_qkv(st,task);
            }
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
using NativeLinearOp=NativeLinearOpT<128>;
using NativeSkinnyLinearOp=NativeLinearOpT<64>;
struct NativeLinearContext {
    NativeMatrix* matrices;unsigned long long* ready;int *program,*timings;
    int workers,length,counters,device,max_tiles;unsigned long long epoch=0;
};
extern "C" int create_native_linear(void** handle,int count,void* const* handles,const int* sources,const int* code,int workers,int length){
    if(!handle)return cudaErrorInvalidValue;*handle=nullptr;
    if(count<1||!handles||!sources||!code||workers<1||length<1)return cudaErrorInvalidValue;
    int device;auto error=cudaGetDevice(&device);cudaDeviceProp props{};
    if(error==cudaSuccess)error=cudaGetDeviceProperties(&props,device);if(error!=cudaSuccess)return error;
    auto kernel=megakernel::mk<Cfg,NativeGlobals,NativeLinearOp,NativeSkinnyLinearOp>;
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
            if(stage<0||stage>=count||in[0]!=(matrices[stage].matrix.skinny?16:1)||stage<prior||first<0||tiles<1||int64_t(first)+tiles>matrices[stage].matrix.tasks)return cudaErrorInvalidValue;
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
    megakernel::mk<Cfg,NativeGlobals,NativeLinearOp,NativeSkinnyLinearOp><<<c.workers,Cfg::NUM_THREADS,Cfg::DYNAMIC_SHARED_MEMORY,static_cast<cudaStream_t>(stream)>>>(g);
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
