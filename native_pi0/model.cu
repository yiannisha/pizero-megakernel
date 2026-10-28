// Complete BF16 pi0 on original Hazy static worker streams.
#define NATIVE_ATTENTION_SCORE_SCALING 1
#include "native.cu"
#include "attention.cu"
#include "model_attention.cuh"
#include "norm.cu"

struct NativeScalar {
    const void *a,*b,*c,*d;void* output,*cache_v;
    int mode,m,n,k,lda,ldb,ldo,p0,p1,input_slot,cache_base;float alpha;
};
struct NativeSpec {int kind;void* handle;NativeScalar scalar;int sources[3];};
struct NativeData {int base,bands;};
struct NativeContribution {int counter,count;};
struct NativeModelGlobals:NativeGlobals {
    const AttentionGlobals* attention;const NormGlobals* norms;const NativeScalar* scalar;
    const NativeData* data;const int* sources;const int* expected;
    const int* task_offsets;const int* contribution_offsets;const NativeContribution* contributions;
    const float *images,*state,*noise;const long long* tokens;
};
static_assert(!megakernel::dynamic_instructions_v<NativeModelGlobals>);
__device__ inline void native_sync(){asm volatile("bar.sync 6, 128;":::"memory");}
__device__ inline void native_wait_inputs(const NativeModelGlobals& g,State& s){
    int stage=s.instruction()[1];
    if(threadIdx.x==0)for(int j=0;j<3;++j){
        int source=g.sources[stage*3+j];if(source<0)continue;auto d=g.data[source];
        for(int b=0;b<d.bands;++b)while(native_acquire(g.ready+d.base+b)<g.epoch*g.expected[d.base+b])__nanosleep(20);
    }
    native_sync();
}
__device__ inline void native_complete(const NativeModelGlobals& g,State& s){
    asm volatile("fence.proxy.async.global;":::"memory");native_sync();
    if(threadIdx.x==0){
        int first=g.task_offsets[s.instruction()[1]]+s.instruction()[2],last=first+s.instruction()[3];
        for(int t=first;t<last;++t)for(int i=g.contribution_offsets[t];i<g.contribution_offsets[t+1];++i){
            auto p=g.contributions[i];native_publish(g.ready+p.counter,p.count);
        }
    }
    native_sync();
}
struct NativeVisionAttentionOp {
    static constexpr int opcode=2;
    using controller=megakernel::NoOp<Cfg>::controller;
    using launcher=megakernel::NoOp<Cfg>::launcher;
    using storer=megakernel::NoOp<Cfg>::storer;
    struct loader {static __device__ void run(const NativeModelGlobals& g,State& s){
        int used=AttentionOp::pages(g.attention[s.instruction()[1]]),lane=kittens::laneid();
        if(lane>=used&&lane<Cfg::NUM_PAGES){int p=s.pid(lane);s.wait_page_ready(p);s.finish_page(p,4);}
    }};
    struct consumer {static __device__ void run(const NativeModelGlobals& g,State& s){
        native_wait_inputs(g,s);auto a=g.attention[s.instruction()[1]];int used=AttentionOp::pages(a);
        if(threadIdx.x==0)for(int p=0;p<used;++p)s.wait_page_ready(s.pid(p));native_sync();
        for(int task=s.instruction()[2];task<s.instruction()[2]+s.instruction()[3];++task)
            native_model_attention::flash_attend<kittens::bf16,72>(s,task,static_cast<const kittens::bf16*>(a.q),static_cast<const kittens::bf16*>(a.k),static_cast<const kittens::bf16*>(a.v),a.ldq,a.ldkv,
                nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,static_cast<kittens::bf16*>(a.output),a.out_ld,a.nq,a.nk,a.heads,a.kv_heads,a.scale_log2,a.prefix,a.mask,
                0,a.nk,nullptr,nullptr,nullptr,nullptr,0,0,false);
        native_sync();if(threadIdx.x==0)for(int p=0;p<used;++p)s.finish_page(s.pid(p),4);
        native_complete(g,s);
    }};
};
struct NativeGemmaAttentionOp {
    static constexpr int opcode=15;
    using controller=NativeVisionAttentionOp::controller;
    using loader=NativeVisionAttentionOp::loader;
    using launcher=NativeVisionAttentionOp::launcher;
    using storer=NativeVisionAttentionOp::storer;
    struct consumer {static __device__ void run(const NativeModelGlobals& g,State& s){
        native_wait_inputs(g,s);const auto a=g.attention[s.instruction()[1]];int used=AttentionOp::pages(a);
        if(threadIdx.x==0)for(int p=0;p<used;++p)s.wait_page_ready(s.pid(p));native_sync();
        for(int task=s.instruction()[2];task<s.instruction()[2]+s.instruction()[3];++task){
            if(g.scalar[s.instruction()[1]].p1)
                native_model_attention::flash_attend<kittens::bf16,256>(s,task,static_cast<const kittens::bf16*>(a.q),static_cast<const kittens::bf16*>(a.k),static_cast<const kittens::bf16*>(a.v),a.ldq,a.ldkv,
                    nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,static_cast<kittens::bf16*>(a.output),a.out_ld,a.nq,a.nk,a.heads,a.kv_heads,a.scale_log2,a.prefix,a.mask,
                    0,a.nk,nullptr,nullptr,nullptr,nullptr,0,0,false);
            else
                native_model_attention::eager_attend<kittens::bf16,256>(s,task,static_cast<const kittens::bf16*>(a.q),static_cast<const kittens::bf16*>(a.k),static_cast<const kittens::bf16*>(a.v),a.ldq,a.ldkv,
                    nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,static_cast<kittens::bf16*>(a.output),a.out_ld,a.nq,a.nk,a.heads,a.kv_heads,a.scale_log2,a.prefix,a.mask,
                    0,a.nk,nullptr,nullptr,static_cast<kittens::bf16*>(a.cache_k),static_cast<kittens::bf16*>(a.cache_v),a.cache_rows,a.cached_prefix_rows,false);
        }
        native_sync();if(threadIdx.x==0)for(int p=0;p<used;++p)s.finish_page(s.pid(p),4);native_complete(g,s);
    }};
};
struct NativeNormOp {
    static constexpr int opcode=3;
    using controller=megakernel::NoOp<Cfg>::controller;
    using loader=megakernel::NoOp<Cfg>::loader;
    using launcher=megakernel::NoOp<Cfg>::launcher;
    using storer=megakernel::NoOp<Cfg>::storer;
    struct consumer {static __device__ void run(const NativeModelGlobals& g,State& s){
        native_wait_inputs(g,s);const auto norm=g.norms[s.instruction()[1]];
        int first=s.instruction()[2]*8,last=min(first+s.instruction()[3]*8,norm.rows);
        for(int row=first;row<last;++row)NormOp::run_row(norm,s,row);native_complete(g,s);
    }};
};
template<int Mode> struct NativeTensorOp {
    static constexpr int opcode=Mode+(Mode>=12?5:3);
    using controller=megakernel::NoOp<Cfg>::controller;
    using loader=megakernel::NoOp<Cfg>::loader;
    using launcher=megakernel::NoOp<Cfg>::launcher;
    using storer=megakernel::NoOp<Cfg>::storer;
    static __device__ float read16(const void* p,size_t i){return float(static_cast<const kittens::bf16*>(p)[i]);}
    static __device__ float gelu(float x){return .5f*x*(1.f+tanhf(.7978845608028654f*(x+.044715f*x*x*x)));}
    struct consumer {static __device__ void run(const NativeModelGlobals& g,State& s){
        native_wait_inputs(g,s);
        const NativeScalar p=g.scalar[s.instruction()[1]];
        const void* input=p.input_slot==1?g.images:p.input_slot==2?g.state:p.input_slot==3?g.noise:p.a;
        constexpr int tile=(Mode==5||Mode==6)?128:(Mode==12||Mode==13)?4096:16384; int begin=s.instruction()[2]*tile,end=min(begin+s.instruction()[3]*tile,p.m*p.n);
        for(int i=begin+threadIdx.x;i<end;i+=128){
            int row=i/p.n,col=i%p.n;float value=0;
            if constexpr(Mode==1)value=read16(input,size_t(row)*p.lda+col)+read16(p.b,size_t(row)*p.ldb+col);
            else if constexpr(Mode==2)value=col<p.p0?gelu(read16(input,size_t(row)*p.lda+col)):0.f;
            else if constexpr(Mode==3){
                float gate=read16(input,size_t(row)*p.lda+col),up=read16(input,size_t(row)*p.lda+p.n+col);
                value=float(kittens::bf16(gelu(gate)))*up;
            }else if constexpr(Mode==4){float x=static_cast<const float*>(input)[i];value=x/(1.f+expf(-x));}
            else if constexpr(Mode==5){
                float acc=0;
                for(int k=0;k<p.k;++k){
                    float x=p.p0?read16(input,size_t(row)*p.lda+k):static_cast<const float*>(input)[size_t(row)*p.lda+k];
                    acc=fmaf(x,static_cast<const float*>(p.b)[size_t(k)*p.ldb+col],acc);
                }
                if(p.cache_v)acc*=static_cast<const float*>(p.cache_v)[row];
                value=acc+(p.c?static_cast<const float*>(p.c)[col]:0.f);
                if(p.p1&2)value=value/(1.f+expf(-value));
                if(p.p1&1){
                    const float* previous=p.d?static_cast<const float*>(p.d):g.noise;
                    value=__fadd_rn(previous[i],__fmul_rn(p.alpha,value));
                }
            }else if constexpr(Mode==6){
                int image=row/256,patch=row%256,pr=patch/16,pc=patch%16;float acc=0;
                for(int k=0;k<588;++k){
                    int channel=k/196,rr=(k%196)/14,cc=k%14;
                    size_t at=((size_t(image)*224+pr*14+rr)*224+pc*14+cc)*3+channel;
                    acc=fmaf(g.images[at],static_cast<const float*>(p.b)[size_t(k)*p.n+col],acc);
                }
                value=(acc+static_cast<const float*>(p.c)[col])+static_cast<const float*>(p.d)[size_t(patch)*p.n+col];
            }else if constexpr(Mode==7){
                value=col<p.p0?static_cast<const float*>(input)[size_t(row)*p.p0+col]:static_cast<const float*>(p.b)[col-p.p0];
            }else if constexpr(Mode==8){
                value=row==0?static_cast<const float*>(input)[col]:row<=p.p0?static_cast<const float*>(p.b)[size_t(row-1)*p.n+col]:0.f;
            }else if constexpr(Mode==9){
                value=__fadd_rn(static_cast<const float*>(input)[i],__fmul_rn(p.alpha,static_cast<const float*>(p.b)[i]));
            }else if constexpr(Mode==10){
                if(row<p.p0)value=read16(input,size_t(row)*p.lda+col);
                else if(row<p.p0+p.p1){long long token=g.tokens[row-p.p0];value=read16(p.b,size_t(token)*p.n+col)*p.alpha;}
            }else if constexpr(Mode==11){
                if(row<p.k){
                    value=read16(input,size_t(row)*p.lda+col);
                    if(col<p.p0){
                        int d=col%p.p1,partner=(col/p.p1)*p.p1+(d+p.p1/2)%p.p1;
                        float other=read16(input,size_t(row)*p.lda+partner)*(d<p.p1/2?-1.f:1.f);
                        value=float(kittens::bf16(value*read16(p.b,size_t(row)*p.p1+d)))+float(kittens::bf16(other*read16(p.c,size_t(row)*p.p1+d)));
                    }
                }
            }else if constexpr(Mode==12){
                if(col<588){
                    int image=row/256,patch=row%256,pr=patch/16,pc=patch%16;
                    int channel=col/196,rr=(col%196)/14,cc=col%14;
                    size_t at=((size_t(image)*224+pr*14+rr)*224+pc*14+cc)*3+channel;
                    value=g.images[at];
                }
            }else if constexpr(Mode==13){
                value=static_cast<const float*>(input)[size_t(row)*p.lda+col]
                      +static_cast<const float*>(p.b)[col]
                      +static_cast<const float*>(p.c)[size_t(row%256)*p.n+col];
            }
            size_t out=size_t(row)*p.ldo+col;
            if constexpr(Mode==4||Mode==5||Mode==7||Mode==9)static_cast<float*>(p.output)[out]=value;
            else static_cast<kittens::bf16*>(p.output)[out]=kittens::bf16(value);
            if constexpr(Mode==11){
                if(row<p.k&&p.d&&p.cache_v){
                    int key_start=p.p0-p.p1;
                    size_t cached=size_t(p.cache_base+row)*p.p1;
                    if(col>=key_start&&col<p.p0)
                        static_cast<kittens::bf16*>(const_cast<void*>(p.d))[cached+col-key_start]=kittens::bf16(value);
                    else if(col>=p.p0&&col<p.p0+p.p1)
                        static_cast<kittens::bf16*>(p.cache_v)[cached+col-p.p0]=kittens::bf16(value);
                }
            }
        }
        native_complete(g,s);
    }};
};
#define NATIVE_MODEL_KERNEL megakernel::mk<Cfg,NativeModelGlobals,NativeLinearOp,NativeVisionAttentionOp,NativeNormOp,NativeTensorOp<1>,NativeTensorOp<2>,NativeTensorOp<3>,NativeTensorOp<4>,NativeTensorOp<5>,NativeTensorOp<6>,NativeTensorOp<7>,NativeTensorOp<8>,NativeTensorOp<9>,NativeTensorOp<10>,NativeTensorOp<11>,NativeGemmaAttentionOp,NativeSkinnyLinearOp,NativeTensorOp<12>,NativeTensorOp<13>>
struct NativeModelContext {
    NativeModelGlobals globals;int workers,device,count;std::vector<void*> owned;unsigned long long epoch=0;
};
extern "C" int create_native_model(void** handle,int count,const NativeSpec* specs,const int* code,int workers,int length){
    if(!handle)return cudaErrorInvalidValue;*handle=nullptr;
    if(count<1||!specs||!code||workers<1||length<1)return cudaErrorInvalidValue;
    int device;auto error=cudaGetDevice(&device);cudaDeviceProp props{};
    if(error==cudaSuccess)error=cudaGetDeviceProperties(&props,device);if(error!=cudaSuccess)return error;
    if(workers>props.multiProcessorCount)return cudaErrorInvalidConfiguration;
    error=cudaFuncSetAttribute(NATIVE_MODEL_KERNEL,cudaFuncAttributeMaxDynamicSharedMemorySize,Cfg::DYNAMIC_SHARED_MEMORY);
    int occupancy=0;if(error==cudaSuccess)error=cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occupancy,NATIVE_MODEL_KERNEL,Cfg::NUM_THREADS,Cfg::DYNAMIC_SHARED_MEMORY);
    if(error!=cudaSuccess)return error;if(occupancy<1)return cudaErrorInvalidConfiguration;
    const DenseGlobals<kittens::bf16>* first_dense=nullptr;const AttentionGlobals* first_attention=nullptr;const NormGlobals* first_norm=nullptr;
    for(int i=0;i<count;++i){
        if(!specs[i].handle&&specs[i].kind!=3)return cudaErrorInvalidValue;
        if(specs[i].kind==0)first_dense=&static_cast<DenseContext<kittens::bf16>*>(specs[i].handle)->globals;
        if(specs[i].kind==1)first_attention=&static_cast<AttentionContext*>(specs[i].handle)->globals;
        if(specs[i].kind==2)first_norm=&static_cast<NormContext*>(specs[i].handle)->globals;
    }
    if(!first_dense||!first_attention||!first_norm)return cudaErrorInvalidValue;
    std::vector<NativeMatrix> matrices(count,NativeMatrix{*first_dense,-1,0,0});std::vector<AttentionGlobals> attention(count,*first_attention);std::vector<NormGlobals> norms(count,*first_norm);
    std::vector<NativeScalar> scalar(count);std::vector<NativeData> data(count);std::vector<int> sources(count*3),expected,tasks(count),opcodes(count),offsets(count+1),contribution_offsets(1,0);
    std::vector<NativeContribution> contributions;std::vector<std::vector<unsigned char>> seen(count);
    for(int stage=0;stage<count;++stage){
        const auto& spec=specs[stage];int rows=0,width=0;
        for(int j=0;j<3;++j){int p=spec.sources[j];if(p<-1||p>=stage)return cudaErrorInvalidValue;sources[stage*3+j]=p;}
        scalar[stage]=spec.scalar;
        if(spec.kind==0){
            auto* base=static_cast<DenseContextBase*>(spec.handle);if(!base||base->device!=device||base->fp8)return cudaErrorInvalidValue;
            const auto& f=static_cast<DenseContext<kittens::bf16>*>(base)->globals;
            if(!f.bf16_output&&(f.bias||f.residual||f.geglu_output||f.gelu_columns||f.rms_factors||f.rope_cos))return cudaErrorInvalidValue;
            rows=f.out.rows();width=f.out.cols();tasks[stage]=f.tasks;opcodes[stage]=f.skinny?16:1;matrices[stage]={f,-1,0,0};
        }else if(spec.kind==1){
            auto* a=static_cast<AttentionContext*>(spec.handle);if(!a||a->device!=device||a->globals.fp16||a->globals.splits!=1)return cudaErrorInvalidValue;
            attention[stage]=a->globals;const auto& f=a->globals;rows=f.batches*f.nq;width=f.heads*f.hd;
            if(f.nk>4096||(!spec.scalar.p0&&f.hd!=72))return cudaErrorInvalidValue;
            tasks[stage]=f.tasks;opcodes[stage]=spec.scalar.p0?15:2;
        }else if(spec.kind==2){
            auto* n=static_cast<NormContext*>(spec.handle);if(!n||n->device!=device||(n->globals.format!=16&&n->globals.format!=17))return cudaErrorInvalidValue;
            norms[stage]=n->globals;rows=n->globals.rows;width=n->globals.width;tasks[stage]=(rows+7)/8;opcodes[stage]=3;
        }else if(spec.kind==3){
            const auto& p=spec.scalar;if(p.mode<1||p.mode>13||p.m<1||p.n<1||!p.output)return cudaErrorInvalidValue;
            if(p.mode==11&&(bool(p.d)!=bool(p.cache_v)||p.cache_base<0))return cudaErrorInvalidValue;
            rows=p.m;width=p.n;int tile=p.mode==5||p.mode==6?128:p.mode>=12?4096:16384;tasks[stage]=(int64_t(rows)*width+tile-1)/tile;opcodes[stage]=p.mode+(p.mode>=12?5:3);
        }else return cudaErrorInvalidValue;
        if(tasks[stage]<1)return cudaErrorInvalidValue;
        data[stage]={int(expected.size()),(rows+127)/128};expected.resize(expected.size()+data[stage].bands,0);
        if(spec.kind==0){
            auto& m=matrices[stage];m.output_counter=data[stage].base;int parent=spec.sources[0];
            if(m.matrix.rope_cos){
                if(tasks[stage]>workers)return cudaErrorInvalidConfiguration;
                m.raw_qkv_counter=int(expected.size());expected.resize(expected.size()+data[stage].bands,0);
            }
            if(parent>=0){if(data[parent].bands!=data[stage].bands)return cudaErrorInvalidValue;m.input_counter=data[parent].base;}
            int residual=spec.sources[1];if(residual>=0&&m.matrix.residual){
                if(data[residual].bands!=data[stage].bands)return cudaErrorInvalidValue;
                m.residual_counter=data[residual].base;
            }
            int factor=spec.sources[m.matrix.residual?2:1];if(factor>=0&&m.matrix.rms_factors){
                if(data[factor].bands!=data[stage].bands)return cudaErrorInvalidValue;
                m.factor_counter=data[factor].base;
            }
        }
        offsets[stage]=int(contribution_offsets.size()-1);seen[stage].resize(tasks[stage],0);
        for(int t=0;t<tasks[stage];++t){
            std::vector<int> bands(data[stage].bands,0);
            auto add=[&](int first,int last){for(int b=first/128;b<=last/128;++b)++bands[b];};
            if(spec.kind==0)add((t/matrices[stage].matrix.n_tiles)*128,std::min(rows-1,(t/matrices[stage].matrix.n_tiles)*128+127));
            else if(spec.kind==2)add(t*8,std::min(rows-1,t*8+7));
            else if(spec.kind==3){int tile=spec.scalar.mode==5||spec.scalar.mode==6?128:spec.scalar.mode>=12?4096:16384;add(int(int64_t(t)*tile/width),int(std::min(int64_t(rows)*width-1,int64_t(t+1)*tile-1)/width));}
            else{
                const auto& a=attention[stage];int per=(a.nq+63)/64,row=(t%per)*64,batch=t/(per*a.heads);add(batch*a.nq+row,batch*a.nq+std::min(row+63,a.nq-1));
            }
            for(int b=0;b<data[stage].bands;++b)if(bands[b]){int counter=data[stage].base+b;expected[counter]+=bands[b];contributions.push_back({counter,bands[b]});}
            contribution_offsets.push_back(int(contributions.size()));
        }
    }
    offsets[count]=int(contribution_offsets.size()-1);
    for(int w=0;w<workers;++w){int prior=-1;for(int r=0;r<length;++r){
        auto* in=code+(size_t(w)*length+r)*32;if(in[0]==0)continue;int stage=in[1],first=in[2],n=in[3];
        if(stage<0||stage>=count||stage<prior||in[0]!=opcodes[stage]||first<0||n<1||int64_t(first)+n>tasks[stage])return cudaErrorInvalidValue;
        prior=stage;for(int t=first;t<first+n;++t){if(seen[stage][t])return cudaErrorInvalidValue;seen[stage][t]=1;}
    }}
    for(const auto& row:seen)if(std::count(row.begin(),row.end(),1)!=row.size())return cudaErrorInvalidValue;
    std::vector<void*> owned;
    auto alloc=[&](size_t bytes)->void*{void* p=nullptr;if(error==cudaSuccess)error=cudaMalloc(&p,bytes);if(p)owned.push_back(p);return p;};
    auto upload=[&](const auto& values){using T=typename std::decay_t<decltype(values)>::value_type;auto* p=static_cast<T*>(alloc(values.size()*sizeof(T)));if(error==cudaSuccess)error=cudaMemcpy(p,values.data(),values.size()*sizeof(T),cudaMemcpyHostToDevice);return p;};
    auto* de=upload(expected);
    for(int i=0;i<count;++i)if(specs[i].kind==0&&matrices[i].input_counter>=0)matrices[i].input_expected=de+matrices[i].input_counter;
    for(int i=0;i<count;++i)if(specs[i].kind==0&&matrices[i].residual_counter>=0)matrices[i].residual_expected=de+matrices[i].residual_counter;
    for(int i=0;i<count;++i)if(specs[i].kind==0&&matrices[i].factor_counter>=0)matrices[i].factor_expected=de+matrices[i].factor_counter;
    NativeModelGlobals g{};g.matrices=upload(matrices);g.attention=upload(attention);g.norms=upload(norms);g.scalar=upload(scalar);g.data=upload(data);g.sources=upload(sources);g.expected=de;
    g.task_offsets=upload(offsets);g.contribution_offsets=upload(contribution_offsets);g.contributions=upload(contributions);
    g.ready=static_cast<unsigned long long*>(alloc(expected.size()*8));if(error==cudaSuccess)error=cudaMemset(g.ready,0,expected.size()*8);
    size_t bytes=size_t(workers)*length*32*4;auto* dp=static_cast<int*>(alloc(bytes));if(error==cudaSuccess)error=cudaMemcpy(dp,code,bytes,cudaMemcpyHostToDevice);
    auto* dt=static_cast<int*>(alloc(size_t(workers)*length*128*4));g.instructions={dp,length};g.timings={dt,length};
    if(error==cudaSuccess)*handle=new NativeModelContext{g,workers,device,count,owned};else for(void* p:owned)cudaFree(p);return error;
}
extern "C" int launch_native_model(void* handle,void* stream,const float* images,const float* state,const float* noise,const long long* tokens){
    if(!handle||!images||!state||!noise)return cudaErrorInvalidValue;auto& c=*static_cast<NativeModelContext*>(handle);int device;auto error=cudaGetDevice(&device);
    if(error!=cudaSuccess)return error;if(device!=c.device)return cudaErrorInvalidDevice;
    cudaStreamCaptureStatus capture;error=cudaStreamIsCapturing(static_cast<cudaStream_t>(stream),&capture);
    if(error!=cudaSuccess)return error;if(capture!=cudaStreamCaptureStatusNone)return cudaErrorStreamCaptureUnsupported;
    if(c.epoch==UINT_MAX)return cudaErrorInvalidValue;
    auto g=c.globals;g.epoch=++c.epoch;g.images=images;g.state=state;g.noise=noise;g.tokens=tokens;
    NATIVE_MODEL_KERNEL<<<c.workers,Cfg::NUM_THREADS,Cfg::DYNAMIC_SHARED_MEMORY,static_cast<cudaStream_t>(stream)>>>(g);return cudaGetLastError();
}
extern "C" int read_native_model_timings(void* handle,int* output,size_t count){
    if(!handle||!output)return cudaErrorInvalidValue;
    auto& c=*static_cast<NativeModelContext*>(handle);
    size_t expected=size_t(c.workers)*c.globals.instructions.length*Cfg::TIMING_WIDTH;
    if(count!=expected)return cudaErrorInvalidValue;
    return cudaMemcpy(output,c.globals.timings.data,count*sizeof(int),cudaMemcpyDeviceToHost);
}
extern "C" int destroy_native_model(void* handle){
    if(!handle)return cudaSuccess;auto* c=static_cast<NativeModelContext*>(handle);int device;auto error=cudaGetDevice(&device);
    if(error!=cudaSuccess)return error;if(device!=c->device)return cudaErrorInvalidDevice;
    for(void* p:c->owned){auto e=cudaFree(p);if(error==cudaSuccess)error=e;}delete c;return error;
}
