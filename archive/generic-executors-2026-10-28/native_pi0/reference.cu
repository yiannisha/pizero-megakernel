// Complete-model reference execution program in the original Megakernels VM.
#define NATIVE_ATTENTION_SCORE_SCALING 1
#include "projection_attention.cu"
#include "norm.cu"

struct RefScalar {
    const void *a,*b,*c,*d;void* output;
    int mode,m,n,k,lda,ldb,ldo,p0,p1,input_slot;
    float alpha;
};
struct RefStage {int kind;void* handle;RefScalar scalar;};
struct RefWork {
    unsigned long long *phase,*claimed,*done;
    const int *counts,*bases,*kinds;int stages;unsigned epoch;
    __device__ bool claim(int& stage,int& task)const{
        unsigned long long tag=static_cast<unsigned long long>(epoch)<<32;
        for(;;){
            auto current=native_pi0::acquire64(phase);
            if(unsigned(current>>32)!=epoch){native_pi0::cas_acq_rel(phase,current,tag);continue;}
            stage=unsigned(current);if(stage>=stages)return false;
            auto next=native_pi0::acquire64(claimed+stage);
            if(unsigned(next>>32)!=epoch){native_pi0::cas_acq_rel(claimed+stage,next,tag);continue;}
            if(unsigned(next)<unsigned(counts[stage])){
                auto ticket=atomicAdd(claimed+stage,1ull);task=unsigned(ticket);
                if(task<counts[stage])return true;
            }else{
                auto completed=native_pi0::acquire64(done+stage);
                if(unsigned(completed>>32)==epoch&&unsigned(completed)==unsigned(counts[stage]))
                    native_pi0::cas_acq_rel(phase,current,tag|unsigned(stage+1));
                else __nanosleep(512);
            }
        }
    }
    __device__ void complete(int stage)const{
        unsigned long long tag=static_cast<unsigned long long>(epoch)<<32;
        auto old=native_pi0::acquire64(done+stage);
        while(unsigned(old>>32)!=epoch){auto actual=native_pi0::cas_acq_rel(done+stage,old,tag);if(actual==old)break;old=actual;}
        unsigned long long previous;
        asm volatile("atom.acq_rel.gpu.global.add.u64 %0, [%1], 1;":"=l"(previous):"l"(done+stage):"memory");
    }
};
struct ReferenceGlobals:ProjectionAttentionGlobals {
    RefWork work;
    const void* const* metadata;const RefScalar* scalar;
    const float *images,*state,*noise;const long long* tokens;
    __device__ void fetch_instruction(int* instruction,int)const{
        if(kittens::laneid()==0){
            int stage,task;if(!work.claim(stage,task))instruction[0]=-1;
            else{instruction[0]=work.kinds[stage]+1;instruction[1]=work.bases[stage]+task;instruction[2]=task;instruction[3]=stage;}
        }
        __syncwarp();
    }
};
struct ReferenceDiagnostics {
    static __device__ void start(const ReferenceGlobals&,State&){}
    static __device__ void finish(const ReferenceGlobals& g,State& s){
        asm volatile("fence.proxy.async.global;":::"memory");native_attention::attention_sync();
        if(threadIdx.x==0)g.work.complete(s.instruction()[3]);native_attention::attention_sync();
    }
};
struct RefDenseOp {
    static constexpr int opcode=1;
    using controller=ScheduledProjectionOp::controller;
    using storer=megakernel::NoOp<Cfg>::storer;
    static __device__ const DenseGlobals<kittens::bf16>& stage(const ReferenceGlobals& g,State& s){return *static_cast<const DenseGlobals<kittens::bf16>*>(g.metadata[s.instruction()[3]]);}
    struct loader {static __device__ void run(const ReferenceGlobals& g,State& s){
        DenseView<kittens::bf16> view(stage(g,s),s.instruction()[2]);
        if(kittens::laneid()==0){
            const auto *a=view.a.template get_tma<typename Globals<kittens::bf16>::Tile,2>(),*b=view.b.template get_tma<typename Globals<kittens::bf16>::Tile,2>();
            asm volatile("fence.proxy.tensormap::generic.acquire.gpu [%0], 128;"::"l"(a):"memory");
            asm volatile("fence.proxy.tensormap::generic.acquire.gpu [%0], 128;"::"l"(b):"memory");
        }
        __syncwarp();Matmul<kittens::bf16>::loader::run(view,s);
    }};
    struct launcher {static __device__ void run(const ReferenceGlobals& g,State& s){DenseView<kittens::bf16> view(stage(g,s),s.instruction()[2]);Matmul<kittens::bf16>::launcher::run(view,s);}};
    struct consumer {static __device__ void run(const ReferenceGlobals& g,State& s){DenseView<kittens::bf16> view(stage(g,s),s.instruction()[2]);DenseOp<kittens::bf16>::consume(view,s);ReferenceDiagnostics::finish(g,s);}};
};
// Reference attention preserves the eager BF16 score/probability rounding
// boundaries. Four query rows share a VM task; no inter-CTA barrier is used.
struct RefAttentionOp {
    static constexpr int opcode=2;
    using controller=megakernel::NoOp<Cfg>::controller;
    using launcher=megakernel::NoOp<Cfg>::launcher;
    using storer=megakernel::NoOp<Cfg>::storer;
    struct loader {static __device__ void run(const ReferenceGlobals& g,State& s){
        int stage=s.instruction()[3];int used=g.scalar[stage].p0?2:AttentionOp::pages(*static_cast<const AttentionGlobals*>(g.metadata[stage]));
        int lane=kittens::laneid();if(lane>=used&&lane<Cfg::NUM_PAGES){int p=s.pid(lane);s.wait_page_ready(p);s.finish_page(p,4);}
    }};
    static __device__ float reduce(float value,float* scratch,int hd,bool maximum){
        int lane=kittens::laneid(),warp=kittens::warpid();
        for(int d=16;d;d>>=1){float x=__shfl_xor_sync(0xffffffff,value,d);value=maximum?fmaxf(value,x):value+x;}
        if(lane==0)scratch[hd+warp]=value;native_attention::attention_sync();
        if(warp==0){
            value=lane<4?scratch[hd+lane]:(maximum?-INFINITY:0.f);
            for(int d=16;d;d>>=1){float x=__shfl_xor_sync(0xffffffff,value,d);value=maximum?fmaxf(value,x):value+x;}
            if(lane==0)scratch[hd+4]=value;
        }
        native_attention::attention_sync();value=scratch[hd+4];native_attention::attention_sync();return value;
    }
    struct consumer {static __device__ void run(const ReferenceGlobals& g,State& s){
        ReferenceDiagnostics::start(g,s);
        int stage=s.instruction()[3];const auto a=*static_cast<const AttentionGlobals*>(g.metadata[stage]);
        const auto policy=g.scalar[stage];int first=s.instruction()[2]*4;
        if(!policy.p0){auto view=a;view.epoch=g.work.epoch;AttentionOp::consume(view,s,s.instruction()[2]);ReferenceDiagnostics::finish(g,s);return;}
        int p0=s.pid(0),p1=s.pid(1);s.wait_page_ready(p0);s.wait_page_ready(p1);
        float* probability=reinterpret_cast<float*>(s.pages[p0].ptr());float* q=reinterpret_cast<float*>(s.pages[p1].ptr());
        const auto *query=static_cast<const kittens::bf16*>(a.q),*key=static_cast<const kittens::bf16*>(a.k),*value=static_cast<const kittens::bf16*>(a.v);
        auto *ck=static_cast<kittens::bf16*>(a.cache_k),*cv=static_cast<kittens::bf16*>(a.cache_v);
        int raw=a.nk-a.cached_prefix_rows;
        for(int index=first;index<min(first+4,a.batches*a.heads*a.nq);++index){
            int row=index%a.nq,head=(index/a.nq)%a.heads,batch=index/(a.nq*a.heads),kh=head/(a.heads/a.kv_heads);
            for(int d=threadIdx.x;d<a.hd;d+=128)q[d]=float(query[size_t(batch*a.nq+row)*a.ldq+head*a.hd+d]);
            native_attention::attention_sync();float maximum=-INFINITY;
            for(int j=threadIdx.x;j<a.nk;j+=128){
                bool cached=j<a.cached_prefix_rows;
                size_t ci=(size_t(batch)*a.cache_rows+j)*a.kv_heads*a.hd+kh*a.hd;
                size_t ri=size_t(batch*raw+j-a.cached_prefix_rows)*a.ldkv+kh*a.hd;float dot=0;
                bool writer=ck&&row==0&&head%(a.heads/a.kv_heads)==0&&!cached;
                for(int d=0;d<a.hd;++d){
                    float k=cached?float(ck[ci+d]):float(key[ri+d]);dot=fmaf(q[d],k,dot);
                    if(writer){ck[ci+d]=kittens::bf16(k);cv[ci+d]=value[ri+d];}
                }
                float score=policy.p0?float(kittens::bf16(float(kittens::bf16(dot))*policy.alpha)):dot*policy.alpha;
                if((a.prefix>=0&&row==0&&j>a.prefix)||(a.mask&&!a.mask[(size_t(batch)*a.nq+row)*a.nk+j]))score=-INFINITY;
                probability[j]=score;maximum=fmaxf(maximum,score);
            }
            maximum=reduce(maximum,q,a.hd,true);float total=0;
            for(int j=threadIdx.x;j<a.nk;j+=128){float p=expf(probability[j]-maximum);probability[j]=p;total+=p;}
            total=reduce(total,q,a.hd,false);
            for(int j=threadIdx.x;j<a.nk;j+=128){float p=probability[j]/total;probability[j]=policy.p0?float(kittens::bf16(p)):p;}
            native_attention::attention_sync();
            for(int d=threadIdx.x;d<a.hd;d+=128){
                float out=0;
                for(int j=0;j<a.nk;++j){
                    float v=j<a.cached_prefix_rows?float(cv[(size_t(batch)*a.cache_rows+j)*a.kv_heads*a.hd+kh*a.hd+d]):float(value[size_t(batch*raw+j-a.cached_prefix_rows)*a.ldkv+kh*a.hd+d]);
                    out=fmaf(probability[j],v,out);
                }
                static_cast<kittens::bf16*>(a.output)[size_t(batch*a.nq+row)*a.out_ld+head*a.hd+d]=kittens::bf16(out);
            }
            native_attention::attention_sync();
        }
        if(threadIdx.x==0){s.finish_page(p0,4);s.finish_page(p1,4);}
        ReferenceDiagnostics::finish(g,s);
    }};
};
struct RefNormOp {
    static constexpr int opcode=3;
    using controller=megakernel::NoOp<Cfg>::controller;
    using loader=megakernel::NoOp<Cfg>::loader;
    using launcher=megakernel::NoOp<Cfg>::launcher;
    using storer=megakernel::NoOp<Cfg>::storer;
    struct consumer {static __device__ void run(const ReferenceGlobals& g,State& s){
        ReferenceDiagnostics::start(g,s);
        const auto norm=*static_cast<const NormGlobals*>(g.metadata[s.instruction()[3]]);
        int first=s.instruction()[2]*8;
        for(int row=first;row<min(first+8,norm.rows);++row)NormOp::run_row(norm,s,row);
        ReferenceDiagnostics::finish(g,s);
    }};
};
struct RefScalarOp {
    static constexpr int opcode=4;
    using controller=megakernel::NoOp<Cfg>::controller;
    using loader=megakernel::NoOp<Cfg>::loader;
    using launcher=megakernel::NoOp<Cfg>::launcher;
    using storer=megakernel::NoOp<Cfg>::storer;
    static __device__ float read16(const void* p,size_t i){return float(static_cast<const kittens::bf16*>(p)[i]);}
    static __device__ float gelu(float x){return .5f*x*(1.f+tanhf(.7978845608028654f*(x+.044715f*x*x*x)));}
    struct consumer {static __device__ void run(const ReferenceGlobals& g,State& s){
        ReferenceDiagnostics::start(g,s);
        const RefScalar p=g.scalar[s.instruction()[3]];
        const void* input=p.input_slot==1?g.images:p.input_slot==2?g.state:p.input_slot==3?g.noise:p.a;
        int tile=(p.mode==5||p.mode==6)?128:16384,begin=s.instruction()[2]*tile,end=min(begin+tile,p.m*p.n);
        for(int i=begin+threadIdx.x;i<end;i+=128){
            int row=i/p.n,col=i%p.n;float value=0;
            if(p.mode==1)value=read16(input,size_t(row)*p.lda+col)+read16(p.b,size_t(row)*p.ldb+col);
            else if(p.mode==2)value=col<p.p0?gelu(read16(input,size_t(row)*p.lda+col)):0.f;
            else if(p.mode==3){
                float gate=read16(input,size_t(row)*p.lda+col),up=read16(input,size_t(row)*p.lda+p.n+col);
                value=float(kittens::bf16(gelu(gate)))*up;
            }else if(p.mode==4){float x=static_cast<const float*>(input)[i];value=x/(1.f+expf(-x));}
            else if(p.mode==5){
                float acc=0;
                for(int k=0;k<p.k;++k){
                    float x=p.p0?read16(input,size_t(row)*p.lda+k):static_cast<const float*>(input)[size_t(row)*p.lda+k];
                    acc=fmaf(x,static_cast<const float*>(p.b)[size_t(k)*p.ldb+col],acc);
                }
                value=acc+(p.c?static_cast<const float*>(p.c)[col]:0.f);
            }else if(p.mode==6){
                int image=row/256,patch=row%256,pr=patch/16,pc=patch%16;float acc=0;
                for(int k=0;k<588;++k){
                    int channel=k/196,rr=(k%196)/14,cc=k%14;
                    size_t at=((size_t(image)*224+pr*14+rr)*224+pc*14+cc)*3+channel;
                    acc=fmaf(g.images[at],static_cast<const float*>(p.b)[size_t(k)*p.n+col],acc);
                }
                value=(acc+static_cast<const float*>(p.c)[col])+static_cast<const float*>(p.d)[size_t(patch)*p.n+col];
            }else if(p.mode==7){
                value=col<p.p0?static_cast<const float*>(input)[size_t(row)*p.p0+col]:static_cast<const float*>(p.b)[col-p.p0];
            }else if(p.mode==8){
                value=row==0?static_cast<const float*>(input)[col]:row<=p.p0?static_cast<const float*>(p.b)[size_t(row-1)*p.n+col]:0.f;
            }else if(p.mode==9){
                value=__fadd_rn(static_cast<const float*>(input)[i],__fmul_rn(p.alpha,static_cast<const float*>(p.b)[i]));
            }else if(p.mode==10){
                if(row<p.p0)value=read16(input,size_t(row)*p.lda+col);
                else if(row<p.p0+p.p1){long long token=g.tokens[row-p.p0];value=read16(p.b,size_t(token)*p.n+col)*p.alpha;}
            }else if(p.mode==11){
                if(row<p.k){
                    value=read16(input,size_t(row)*p.lda+col);
                    if(col<p.p0){
                        int d=col%p.p1,partner=(col/p.p1)*p.p1+(d+p.p1/2)%p.p1;
                        float other=read16(input,size_t(row)*p.lda+partner)*(d<p.p1/2?-1.f:1.f);
                        value=float(kittens::bf16(value*read16(p.b,size_t(row)*p.p1+d)))+float(kittens::bf16(other*read16(p.c,size_t(row)*p.p1+d)));
                    }
                }
            }
            size_t out=size_t(row)*p.ldo+col;
            if(p.mode==4||p.mode==5||p.mode==7||p.mode==9)static_cast<float*>(p.output)[out]=value;
            else static_cast<kittens::bf16*>(p.output)[out]=kittens::bf16(value);
        }
        ReferenceDiagnostics::finish(g,s);
    }};
};
#define REFERENCE_KERNEL megakernel::mk<Cfg,ReferenceGlobals,RefDenseOp,RefAttentionOp,RefNormOp,RefScalarOp>
struct ReferenceContext {
    std::vector<AttentionContext*> attention;ProjectionAttentionStage* stages;
    const void** metadata;RefScalar* scalar;RefWork work;
    int *program,*timings,slots,workers,device,nodes;std::vector<void*> owned;
};
extern "C" int create_reference_program(void** handle,int count,const RefStage* specs,int workers){
    if(!handle)return cudaErrorInvalidValue;*handle=nullptr;
    if(count<1||!specs||workers<1)return cudaErrorInvalidValue;
    int device;auto error=cudaGetDevice(&device);cudaDeviceProp properties{};
    if(error==cudaSuccess)error=cudaGetDeviceProperties(&properties,device);
    if(error!=cudaSuccess)return error;if(workers>properties.multiProcessorCount)return cudaErrorInvalidValue;
    std::vector<void*> owned;std::vector<ProjectionAttentionStage> stages(count);
    std::vector<const void*> metadata(count,nullptr);std::vector<RefScalar> scalar(count);
    std::vector<AttentionContext*> attention;std::vector<int> expected(count),bases(count),kinds(count);
    auto alloc=[&](size_t bytes)->void*{void* p=nullptr;if(error==cudaSuccess)error=cudaMalloc(&p,bytes);if(p)owned.push_back(p);return p;};
    auto upload=[&](const void* source,size_t bytes)->void*{void* p=alloc(bytes);if(error==cudaSuccess)error=cudaMemcpy(p,source,bytes,cudaMemcpyHostToDevice);return p;};
    int64_t nodes=0;
    for(int i=0;i<count&&error==cudaSuccess;++i){
        const auto& spec=specs[i];kinds[i]=spec.kind;bases[i]=int(nodes);
        if(spec.kind==0){
            auto* c=static_cast<DenseContextBase*>(spec.handle);if(!c||c->device!=device||c->fp8){error=cudaErrorInvalidValue;break;}
            const auto& f=static_cast<DenseContext<kittens::bf16>*>(c)->globals;
            metadata[i]=upload(&f,sizeof(f));stages[i]={metadata[i],nullptr,0,nullptr};expected[i]=f.tasks;
        }else if(spec.kind==1){
            auto* c=static_cast<AttentionContext*>(spec.handle);if(!c||c->device!=device){error=cudaErrorInvalidValue;break;}
            metadata[i]=upload(&c->globals,sizeof(c->globals));stages[i]={nullptr,static_cast<const AttentionGlobals*>(metadata[i]),0,nullptr};
            if(c->globals.nk>4096||c->globals.fp16){error=cudaErrorInvalidValue;break;}
            scalar[i]=spec.scalar;expected[i]=spec.scalar.p0?(c->globals.batches*c->globals.heads*c->globals.nq+3)/4:c->globals.tasks;attention.push_back(c);
        }else if(spec.kind==2){
            auto* c=static_cast<NormContext*>(spec.handle);if(!c||c->device!=device){error=cudaErrorInvalidValue;break;}
            metadata[i]=upload(&c->globals,sizeof(c->globals));expected[i]=(c->globals.rows+7)/8;
        }else if(spec.kind==3){
            scalar[i]=spec.scalar;int tile=(scalar[i].mode==5||scalar[i].mode==6)?128:16384;
            if(scalar[i].mode<1||scalar[i].mode>11||scalar[i].m<1||scalar[i].n<1||!scalar[i].output){error=cudaErrorInvalidValue;break;}
            expected[i]=(int64_t(scalar[i].m)*scalar[i].n+tile-1)/tile;
        }else error=cudaErrorInvalidValue;
        nodes+=expected[i];if(nodes>=INT_MAX)error=cudaErrorInvalidValue;
    }
    auto* ds=static_cast<ProjectionAttentionStage*>(upload(stages.data(),stages.size()*sizeof(stages[0])));
    auto* dm=static_cast<const void**>(upload(metadata.data(),metadata.size()*sizeof(metadata[0])));
    auto* dp=static_cast<RefScalar*>(upload(scalar.data(),scalar.size()*sizeof(scalar[0])));
    auto* counts=static_cast<int*>(upload(expected.data(),expected.size()*4));auto* offsets=static_cast<int*>(upload(bases.data(),bases.size()*4));
    auto* operations=static_cast<int*>(upload(kinds.data(),kinds.size()*4));
    auto* phase=static_cast<unsigned long long*>(alloc(8));auto* claimed=static_cast<unsigned long long*>(alloc(size_t(count)*8));
    auto* done=static_cast<unsigned long long*>(alloc(size_t(count)*8));
    if(error==cudaSuccess)error=cudaMemset(phase,0,8);if(error==cudaSuccess)error=cudaMemset(claimed,0,size_t(count)*8);
    if(error==cudaSuccess)error=cudaMemset(done,0,size_t(count)*8);
    int* slots_device=static_cast<int*>(alloc(4));int slots=0;
    if(error==cudaSuccess){dag_slots<<<1,32>>>(slots_device);error=cudaGetLastError();}
    if(error==cudaSuccess)error=cudaMemcpy(&slots,slots_device,4,cudaMemcpyDeviceToHost);
    std::vector<int> code(size_t(slots)*32,0);for(size_t i=0;i<code.size();i+=32)code[i]=1;
    auto* program=static_cast<int*>(upload(code.data(),code.size()*4));auto* timings=static_cast<int*>(alloc(size_t(slots)*256*4));
    if(error==cudaSuccess)error=cudaFuncSetAttribute(REFERENCE_KERNEL,cudaFuncAttributeMaxDynamicSharedMemorySize,Cfg::DYNAMIC_SHARED_MEMORY);
    RefWork work{phase,claimed,done,counts,offsets,operations,count,0};
    if(error==cudaSuccess)*handle=new ReferenceContext{attention,ds,dm,dp,work,program,timings,slots,workers,device,int(nodes),owned};
    else for(void* p:owned)cudaFree(p);return error;
}
extern "C" int launch_reference_program(void* handle,void* stream,const float* images,const float* state,const float* noise,const long long* tokens){
    if(!handle||!images||!state||!noise)return cudaErrorInvalidValue;auto& c=*static_cast<ReferenceContext*>(handle);int device;auto error=cudaGetDevice(&device);
    if(error!=cudaSuccess)return error;if(device!=c.device)return cudaErrorInvalidDevice;
    cudaStreamCaptureStatus capture;error=cudaStreamIsCapturing(static_cast<cudaStream_t>(stream),&capture);
    if(error!=cudaSuccess)return error;if(capture!=cudaStreamCaptureStatusNone)return cudaErrorStreamCaptureUnsupported;
    unsigned epoch=c.work.epoch;for(auto* a:c.attention)epoch=std::max(epoch,a->globals.epoch);
    if(epoch==UINT_MAX)return cudaErrorInvalidValue;c.work.epoch=++epoch;
    for(auto* a:c.attention){a->globals.epoch=epoch;a->started=true;}
    ProjectionAttentionGlobals base{megakernel::instruction_layout<Cfg>(c.program,nullptr,c.slots,1,nullptr),RingTimings{c.timings},{},
        nullptr,nullptr,0,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,c.stages,nullptr};
    ReferenceGlobals g{base,c.work,c.metadata,c.scalar,images,state,noise,tokens};
    REFERENCE_KERNEL<<<c.workers,Cfg::NUM_THREADS,Cfg::DYNAMIC_SHARED_MEMORY,static_cast<cudaStream_t>(stream)>>>(g);return cudaGetLastError();
}
extern "C" int destroy_reference_program(void* handle){
    if(!handle)return cudaSuccess;auto* c=static_cast<ReferenceContext*>(handle);int device;auto error=cudaGetDevice(&device);
    if(error!=cudaSuccess)return error;if(device!=c->device)return cudaErrorInvalidDevice;
    for(void* p:c->owned){auto e=cudaFree(p);if(error==cudaSuccess)error=e;}delete c;return error;
}
