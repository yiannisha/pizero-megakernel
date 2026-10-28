// SPDX-License-Identifier: Apache-2.0
// Device math ported from pinned FlashRT csrc/kernels/fused_qkv_attention.cu.
// Native VM page ownership/barriers, GQA, rectangular queries and suffix masks
// replace the standalone launch, CTA shared arrays and non-causal-only indexing.
#include "kittens.cuh"
#include "megakernel.cuh"
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <type_traits>
#include <vector>
#include <climits>
#include <cuda/atomic>
#ifndef NATIVE_ATTENTION_CACHE_MERGE
#define NATIVE_ATTENTION_CACHE_MERGE 0
#endif
struct AttentionConfig:megakernel::default_config {
    static constexpr int NUM_CONSUMER_WARPS=4,NUM_WARPS=8,NUM_THREADS=256,CONSUMER_REGISTERS=240;
};
using AC=AttentionConfig;
using AS=megakernel::state<AC>;
namespace native_attention {
__device__ inline void attention_sync(){asm volatile("bar.sync 6, 128;":::"memory");}
#ifndef NATIVE_ATTENTION_CACHE_PAGES
#define NATIVE_ATTENTION_CACHE_PAGES 1
#endif
#ifndef NATIVE_ATTENTION_SCORE_SCALING
#define NATIVE_ATTENTION_SCORE_SCALING 0
#endif
// Cache physical page addresses once per tile. Small heads fit each matrix
// into one page; larger heads select among three independently mapped pages.
// No contiguous physical-page assumption is made.
template<class T,int Stride,int Base,int Rows,class StateT> struct PagedRows {
    static constexpr int PageElements=AC::PAGE_SIZE/sizeof(T);
    static constexpr int Pages=(Stride*Rows+PageElements-1)/PageElements;
    static_assert(Pages>=1 && Pages<=3);
#if NATIVE_ATTENTION_CACHE_PAGES
    T *p0,*p1,*p2;
    __device__ PagedRows(StateT& s){
        p0=reinterpret_cast<T*>(s.pages[s.pid(Base)].ptr());
        if constexpr(Pages>1)p1=reinterpret_cast<T*>(s.pages[s.pid(Base+1)].ptr());
        if constexpr(Pages>2)p2=reinterpret_cast<T*>(s.pages[s.pid(Base+2)].ptr());
    }
#else
    StateT& state;
    __device__ PagedRows(StateT& s):state(s){}
#endif
    struct Row {
        const PagedRows& owner;int row;
        __device__ T& operator[](int col)const{
            int offset=row*Stride+col;
#if NATIVE_ATTENTION_CACHE_PAGES
            if constexpr(Pages==1)return owner.p0[offset];
            else {
                int page=offset/PageElements;
                T* pointer=page==0?owner.p0:page==1?owner.p1:owner.p2;
                return pointer[offset%PageElements];
            }
#else
            return reinterpret_cast<T*>(owner.state.pages[owner.state.pid(Base+offset/PageElements)].ptr())[offset%PageElements];
#endif
        }
    };
    __device__ Row operator[](int row)const{return {*this,row};}
};
constexpr int kChunk = 64;   // keys per shared-memory chunk

template <typename T>
struct Op;

template <>
struct Op<__half> {
    static __device__ __forceinline__ float2 f2(uint32_t u) {
        return __half22float2(*reinterpret_cast<const __half2*>(&u));
    }
    static __device__ __forceinline__ uint32_t pack(float a, float b) {
        const __half2 v = __floats2half2_rn(a, b);
        return *reinterpret_cast<const uint32_t*>(&v);
    }
    static __device__ __forceinline__ void mma(float* c, const uint32_t* a, uint32_t b0, uint32_t b1) {
        asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
            "{%0,%1,%2,%3};\n"
            : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
            : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
    }
};

template <>
struct Op<__nv_bfloat16> {
    static __device__ __forceinline__ float2 f2(uint32_t u) {
        return __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&u));
    }
    static __device__ __forceinline__ uint32_t pack(float a, float b) {
        const __nv_bfloat162 v = __floats2bfloat162_rn(a, b);
        return *reinterpret_cast<const uint32_t*>(&v);
    }
    static __device__ __forceinline__ void mma(float* c, const uint32_t* a, uint32_t b0, uint32_t b1) {
        asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
            "{%0,%1,%2,%3};\n"
            : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
            : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
    }
};

template <typename T>
__device__ __forceinline__ uint32_t ld2(const T* p) {
    return *reinterpret_cast<const uint32_t*>(p);
}

// VW consecutive elements starting at c of the first half of a head row and
// their rotate-half partners (c + hd/2): raw loads, issued before use.
template <typename T, int VW>
struct Raw {
    using V = typename std::conditional<VW == 8, uint4, uint2>::type;
    V lo, hi, c, s;
};

template <typename T, int VW>
__device__ __forceinline__ void unpack(const typename Raw<T, VW>::V& r, float* f) {
    const uint32_t* u = reinterpret_cast<const uint32_t*>(&r);
#pragma unroll
    for (int j = 0; j < VW / 2; ++j) {
        const float2 x = Op<T>::f2(u[j]);
        f[2 * j] = x.x;
        f[2 * j + 1] = x.y;
    }
}

template <typename T, int VW>
__device__ __forceinline__ typename Raw<T, VW>::V pack_vec(const float* f) {
    typename Raw<T, VW>::V r;
    uint32_t* u = reinterpret_cast<uint32_t*>(&r);
#pragma unroll
    for (int j = 0; j < VW / 2; ++j) u[j] = Op<T>::pack(f[2 * j], f[2 * j + 1]);
    return r;
}

// Load the raw unit (row, c): x[c..c+VW), x[c+H2..), cos / sin[c..) (rope).
template <typename T, int HD, int VW>
__device__ __forceinline__ void load_unit(Raw<T, VW>& r, const T* x, const T* cos_row, const T* sin_row, int c,
                                          bool valid) {
    using V = typename Raw<T, VW>::V;
    constexpr int H2 = HD / 2;
    if (!valid) {
        r.lo = r.hi = r.c = r.s = V{};
        return;
    }
    r.lo = *reinterpret_cast<const V*>(x + c);
    r.hi = *reinterpret_cast<const V*>(x + c + H2);
    if (cos_row != nullptr) {
        r.c = *reinterpret_cast<const V*>(cos_row + c);
        r.s = *reinterpret_cast<const V*>(sin_row + c);
    }
}

// Finish a unit: + bias, RoPE (when rope), into lo[VW] / hi[VW] floats.
template <typename T, int HD, int VW>
__device__ __forceinline__ void finish_unit(const Raw<T, VW>& r, const T* bias, int c, bool rope, bool valid,
                                            float* lo, float* hi) {
    using V = typename Raw<T, VW>::V;
    constexpr int H2 = HD / 2;
    unpack<T, VW>(r.lo, lo);
    unpack<T, VW>(r.hi, hi);
    if (!valid) return;
    if (bias != nullptr) {
        float bl[VW], bh[VW];
        unpack<T, VW>(*reinterpret_cast<const V*>(bias + c), bl);
        unpack<T, VW>(*reinterpret_cast<const V*>(bias + c + H2), bh);
#pragma unroll
        for (int j = 0; j < VW; ++j) {
            lo[j] += bl[j];
            hi[j] += bh[j];
        }
    }
    if (rope) {
        float cs[VW], sn[VW];
        unpack<T, VW>(r.c, cs);
        unpack<T, VW>(r.s, sn);
#pragma unroll
        for (int j = 0; j < VW; ++j) {
            const float a = lo[j], b = hi[j];
            lo[j] = a * cs[j] - b * sn[j];
            hi[j] = b * cs[j] + a * sn[j];
        }
    }
}

template<class T,int HD,int NW=4,class StateT>
__device__ void attend(StateT& state,int task,const T* q,const T* k,const T* v,int ldq,int ldkv,
 const T* q_bias,const T* k_bias,const T* v_bias,const T* qcos,const T* qsin,const T* kcos,const T* ksin,
 T* out,int out_ld,int NQ,int NK,int heads,int kv_heads,float scale_log2,int prefix,const unsigned char* mask,
 int key_begin,int key_end,float* partial,float* statistics,
 T* cache_k,T* cache_v,int cache_rows,int cached_prefix_rows,bool causal){
    using O = Op<T>;
    constexpr int HDP = (HD + 15) / 16 * 16;
    constexpr int KS = HDP / 16;   // k-steps of Q K^T
    constexpr int NT = HD / 8;     // n-tiles of P V
    constexpr int H2 = HD / 2;
    constexpr int VW = H2 % 8 == 0 ? 8 : 4;   // elements per vector load
    constexpr int UK = H2 / VW;               // units per row
    constexpr int UNITS = kChunk * UK;        // units per 64-row chunk
    constexpr int kTileQ = NW * 16;           // query rows per CTA
    constexpr int UQ = kTileQ * UK;           // units of the Q tile
    constexpr int UPT = (UNITS + NW * 32 - 1) / (NW * 32);
    constexpr int UPTQ = (UQ + NW * 32 - 1) / (NW * 32);
    constexpr int KST = HDP + 8;
    using V = typename Raw<T, VW>::V;
    constexpr int KROWS = kChunk > kTileQ ? kChunk : kTileQ;
    PagedRows<T,KST,0,KROWS,StateT> ks{state};
    constexpr int kPages=(KROWS*KST*sizeof(T)+AC::PAGE_SIZE-1)/AC::PAGE_SIZE;
    PagedRows<T,kChunk+8,kPages,HD,StateT> vt{state};

    const int tiles=(NQ+kTileQ-1)/kTileQ;
    const int tile=task%tiles,h=(task/tiles)%heads,b=task/(tiles*heads);
    const int kh=h/(heads/kv_heads);
    const int w = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    const size_t base = static_cast<size_t>(b) * NQ;
    const int col0 = h * HD;
    const bool rope = qcos != nullptr;
    const bool key_rope=kcos!=nullptr;
    const bool cache_writer=cache_k && tile==0 && h%(heads/kv_heads)==0;
    const int raw_key_rows=NK-cached_prefix_rows;
    const T* qbh = q_bias ? q_bias + col0 : nullptr;
    const T* kbh = k_bias ? k_bias + kh*HD : nullptr;
    const T* vbh = v_bias ? v_bias + kh*HD : nullptr;

    if constexpr (HDP > HD) {
        for (int i = threadIdx.x; i < KROWS * (HDP - HD); i += (NW*32))
            ks[i / (HDP - HD)][HD + i % (HDP - HD)] = T(0.f);
    }

    // ── Q tile through shared memory: bias, RoPE, softmax scale * log2 e ──
    const int q0 = tile * kTileQ;
    {
        Raw<T, VW> rq[UPTQ];
#pragma unroll
        for (int u = 0; u < UPTQ; ++u) {
            const int i = u * NW * 32 + threadIdx.x;
            const int r = i / UK, c = (i % UK) * VW;
            const bool valid = i < UQ && q0 + r < NQ;
            const size_t row = base + q0 + r;
            load_unit<T, HD, VW>(rq[u], q + row * ldq + col0, rope ? qcos + row * HD : nullptr,
                                 rope ? qsin + row * HD : nullptr, c, valid);
        }
        attention_sync();
#pragma unroll
        for (int u = 0; u < UPTQ; ++u) {
            const int i = u * NW * 32 + threadIdx.x;
            if (i >= UQ) continue;
            const int r = i / UK, c = (i % UK) * VW;
            float lo[VW], hi[VW];
            finish_unit<T, HD, VW>(rq[u], qbh, c, rope, q0 + r < NQ, lo, hi);
#pragma unroll
            for (int j = 0; j < VW; ++j) {
                // Cosmos keeps Q in BF16 without pre-scaling. Scaling the
                // FP32 QK scores avoids another lossy BF16 Q rounding.
                if constexpr(HD!=128&&!NATIVE_ATTENTION_SCORE_SCALING){lo[j] *= scale_log2;hi[j] *= scale_log2;}
            }
            *reinterpret_cast<V*>(&ks[r][c]) = pack_vec<T, VW>(lo);
            *reinterpret_cast<V*>(&ks[r][c + H2]) = pack_vec<T, VW>(hi);
        }
        attention_sync();
    }
    uint32_t qa[KS][4];
#pragma unroll
    for (int kk = 0; kk < KS; ++kk) {
        qa[kk][0] = ld2(&ks[w * 16 + g][kk * 16 + t * 2]);
        qa[kk][1] = ld2(&ks[w * 16 + g + 8][kk * 16 + t * 2]);
        qa[kk][2] = ld2(&ks[w * 16 + g][kk * 16 + 8 + t * 2]);
        qa[kk][3] = ld2(&ks[w * 16 + g + 8][kk * 16 + 8 + t * 2]);
    }

    float m[2] = {-INFINITY, -INFINITY}, l[2] = {0.f, 0.f};
    float o[NT][4];
#pragma unroll
    for (int n = 0; n < NT; ++n) o[n][0] = o[n][1] = o[n][2] = o[n][3] = 0.f;

    // ── K / V chunks, the next chunk's loads in flight during the current one ──
    Raw<T, VW> rk[UPT], rv[UPT];
    auto load_chunk = [&](int c0) {
#pragma unroll
        for (int u = 0; u < UPT; ++u) {
            const int i = u * NW * 32 + threadIdx.x;
            const int r = i / UK, c = (i % UK) * VW;
            const bool valid = i < UNITS && c0 + r < key_end;
            const int key=c0+r;const bool cached=key<cached_prefix_rows;
            const size_t row=size_t(b)*raw_key_rows+key-cached_prefix_rows;
            const size_t cached_row=size_t(b)*cache_rows+key;
            const T* kp=cached?cache_k+cached_row*(kv_heads*HD)+kh*HD:k+row*ldkv+kh*HD;
            const T* vp=cached?cache_v+cached_row*(kv_heads*HD)+kh*HD:v+row*ldkv+kh*HD;
            load_unit<T, HD, VW>(rk[u],kp,!cached&&key_rope?kcos+row*HD:nullptr,
                                 !cached&&key_rope?ksin+row*HD:nullptr,c,valid);
            load_unit<T, HD, VW>(rv[u],vp,nullptr,nullptr,c,valid);
        }
    };
    load_chunk(key_begin);
    for (int c0 = key_begin; c0 < key_end; c0 += kChunk) {
        attention_sync();   // previous chunk (or the Q tile) fully consumed
#pragma unroll
        for (int u = 0; u < UPT; ++u) {
            const int i = u * NW * 32 + threadIdx.x;
            if (i >= UNITS) continue;
            const int r = i / UK, c = (i % UK) * VW;
            const bool valid = c0 + r < key_end;
            float lo[VW], hi[VW];
            const int key=c0+r;const bool cached=key<cached_prefix_rows;
            finish_unit<T, HD, VW>(rk[u],cached?nullptr:kbh,c,!cached&&key_rope,valid,lo,hi);
            V packed_lo=pack_vec<T,VW>(lo),packed_hi=pack_vec<T,VW>(hi);
            *reinterpret_cast<V*>(&ks[r][c])=packed_lo;
            *reinterpret_cast<V*>(&ks[r][c+H2])=packed_hi;
            size_t cache_offset=(size_t(b)*cache_rows+key)*(kv_heads*HD)+kh*HD+c;
            if(cache_writer&&valid&&!cached){
                *reinterpret_cast<V*>(cache_k+cache_offset)=packed_lo;
                *reinterpret_cast<V*>(cache_k+cache_offset+H2)=packed_hi;
            }
            finish_unit<T, HD, VW>(rv[u],cached?nullptr:vbh,c,false,valid,lo,hi);
            if(cache_writer&&valid&&!cached){
                *reinterpret_cast<V*>(cache_v+cache_offset)=pack_vec<T,VW>(lo);
                *reinterpret_cast<V*>(cache_v+cache_offset+H2)=pack_vec<T,VW>(hi);
            }
#pragma unroll
            for (int j = 0; j < VW; ++j) {
                vt[c + j][r] = T(lo[j]);
                vt[c + H2 + j][r] = T(hi[j]);
            }
        }
        attention_sync();
        if (c0 + kChunk < key_end) load_chunk(c0 + kChunk);

        float s[kChunk / 8][4];
#pragma unroll
        for (int n = 0; n < kChunk / 8; ++n) {
            s[n][0] = s[n][1] = s[n][2] = s[n][3] = 0.f;
#pragma unroll
            for (int kk = 0; kk < KS; ++kk)
                O::mma(s[n], qa[kk], ld2(&ks[n * 8 + g][kk * 16 + t * 2]), ld2(&ks[n * 8 + g][kk * 16 + 8 + t * 2]));
        }
        if constexpr(HD==128||NATIVE_ATTENTION_SCORE_SCALING){
            #pragma unroll
            for(int n=0;n<kChunk/8;++n){
                #pragma unroll
                for(int j=0;j<4;++j)s[n][j]*=scale_log2;
            }
        }
        float mx[2] = {-INFINITY, -INFINITY};
#pragma unroll
        for (int n = 0; n < kChunk / 8; ++n) {
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int key=c0+n*8+t*2+(j&1),query=q0+w*16+g+(j>>1)*8;
                const bool valid=key<key_end && query<NQ && (prefix<0 || query!=0 || key<=prefix)
                    && (!causal || key<=query)
                    && (!mask || mask[(size_t(b)*NQ+query)*NK+key]);
                if(!valid)s[n][j]=-INFINITY;
                mx[j >> 1] = fmaxf(mx[j >> 1], s[n][j]);
            }
        }
        float corr[2];
#pragma unroll
        for (int r = 0; r < 2; ++r) {
            mx[r] = fmaxf(mx[r], __shfl_xor_sync(0xffffffffu, mx[r], 1));
            mx[r] = fmaxf(mx[r], __shfl_xor_sync(0xffffffffu, mx[r], 2));
            const float mn = fmaxf(m[r], mx[r]);
            corr[r] = mn == -INFINITY ? 0.f : exp2f(m[r] - mn);
            m[r] = mn;
            l[r] *= corr[r];
        }
#pragma unroll
        for (int n = 0; n < NT; ++n) {
            o[n][0] *= corr[0]; o[n][1] *= corr[0];
            o[n][2] *= corr[1]; o[n][3] *= corr[1];
        }
        uint32_t pa[kChunk / 16][4];
        // Head-128 correctness path: preserve the residual of the BF16
        // probability conversion and accumulate it with a second tensor MMA.
        uint32_t pa_low[kChunk / 16][4];
#pragma unroll
        for (int n = 0; n < kChunk / 8; ++n) {
            const float p0 = (s[n][0]==-INFINITY?0.f:exp2f(s[n][0] - m[0])), p1 = (s[n][1]==-INFINITY?0.f:exp2f(s[n][1] - m[0]));
            const float p2 = (s[n][2]==-INFINITY?0.f:exp2f(s[n][2] - m[1])), p3 = (s[n][3]==-INFINITY?0.f:exp2f(s[n][3] - m[1]));
            l[0] += p0 + p1;
            l[1] += p2 + p3;
            pa[n >> 1][(n & 1) * 2 + 0] = O::pack(p0, p1);
            pa[n >> 1][(n & 1) * 2 + 1] = O::pack(p2, p3);
            if constexpr(HD==128 && std::is_same_v<T,__nv_bfloat16>){
                pa_low[n >> 1][(n & 1)*2]=O::pack(p0-float(T(p0)),p1-float(T(p1)));
                pa_low[n >> 1][(n & 1)*2+1]=O::pack(p2-float(T(p2)),p3-float(T(p3)));
            }
        }
#pragma unroll
        for (int kk = 0; kk < kChunk / 16; ++kk) {
#pragma unroll
            for (int n = 0; n < NT; ++n){
                auto lo=ld2(&vt[n * 8 + g][kk * 16 + t * 2]),hi=ld2(&vt[n * 8 + g][kk * 16 + 8 + t * 2]);
                O::mma(o[n], pa[kk], lo, hi);
                if constexpr(HD==128 && std::is_same_v<T,__nv_bfloat16>){
                    // Causal understanding follows the BF16 probability/value
                    // product of the pinned BF16 SDPA reference. Generation
                    // retains the existing higher-precision residual product.
                    if(!causal)O::mma(o[n],pa_low[kk],lo,hi);
                }
            }
        }
    }
#pragma unroll
    for (int r = 0; r < 2; ++r) {
        l[r] += __shfl_xor_sync(0xffffffffu, l[r], 1);
        l[r] += __shfl_xor_sync(0xffffffffu, l[r], 2);
    }
#pragma unroll
    for (int r = 0; r < 2; ++r) {
        const int row = q0 + w * 16 + g + r * 8;
        if (row >= NQ) continue;
        const float inv = l[r]>0.f ? 1.f / l[r] : 0.f;
        if(partial){
            const int local_row=row-q0;
            if(t==0){statistics[local_row*2]=m[r];statistics[local_row*2+1]=l[r];}
#pragma unroll
            for(int n=0;n<NT;++n)
                *reinterpret_cast<float2*>(partial+local_row*HD+n*8+t*2)=make_float2(o[n][2*r],o[n][2*r+1]);
        }else{
            T* dst = out + (base + row) * out_ld + col0;
#pragma unroll
            for (int n = 0; n < NT; ++n)
                *reinterpret_cast<uint32_t*>(dst + n * 8 + t * 2) = O::pack(o[n][2 * r] * inv, o[n][2 * r + 1] * inv);
        }
    }
}
} // namespace native_attention
struct AttentionGlobals {
    megakernel::instruction_layout<AC> instructions;
    megakernel::timing_layout<AC> timings;
    const void *q,*k,*v,*qb,*kb,*vb,*qcos,*qsin,*kcos,*ksin;
    void* output;const unsigned char* mask;
    int batches,nq,nk,heads,kv_heads,hd,ldq,ldkv,out_ld,fp16,workers,prefix,tasks;
    float scale_log2;
    int splits=1;
    unsigned epoch=0;
    float *partial=nullptr,*statistics=nullptr;
    unsigned long long* ready=nullptr;
    void *cache_k=nullptr,*cache_v=nullptr;
    int cache_rows=0,cached_prefix_rows=0;
    bool causal=false;
};
struct AttentionOp {
    static constexpr int opcode=1;
    using controller=megakernel::NoOp<AC>::controller;
    using launcher=megakernel::NoOp<AC>::launcher;
    using storer=megakernel::NoOp<AC>::storer;
    static __device__ int task(const AttentionGlobals& g,AS& s){return s.instruction_index*g.workers+blockIdx.x;}
    static __device__ int pages(const AttentionGlobals& g){
        int kp=(64*((g.hd+15)/16*16+8)*2+AC::PAGE_SIZE-1)/AC::PAGE_SIZE;
        int vp=(g.hd*72*2+AC::PAGE_SIZE-1)/AC::PAGE_SIZE;return kp+vp;
    }
    struct loader {static __device__ void run(const AttentionGlobals& g,AS& s){
        int used=task(g,s)<g.tasks?pages(g):0,lane=kittens::laneid();
        if(lane>=used && lane<AC::NUM_PAGES){int p=s.pid(lane);s.wait_page_ready(p);s.finish_page(p,4);}
    }};
    template<class T,int HD,class StateT> static __device__ void compute(const AttentionGlobals& g,StateT& s,int index){
        int logical=index/g.splits,part=index%g.splits;
        int chunks=(g.nk+63)/64,per_split=(chunks+g.splits-1)/g.splits;
        int begin=part*per_split*64,end=min(g.nk,begin+per_split*64);
        native_attention::attend<T,HD>(s,logical,static_cast<const T*>(g.q),static_cast<const T*>(g.k),static_cast<const T*>(g.v),g.ldq,g.ldkv,
            static_cast<const T*>(g.qb),static_cast<const T*>(g.kb),static_cast<const T*>(g.vb),
            static_cast<const T*>(g.qcos),static_cast<const T*>(g.qsin),static_cast<const T*>(g.kcos),static_cast<const T*>(g.ksin),
            static_cast<T*>(g.output),g.out_ld,g.nq,g.nk,g.heads,g.kv_heads,g.scale_log2,g.prefix,g.mask,
            begin,end,g.splits>1?g.partial+size_t(index)*64*HD:nullptr,g.splits>1?g.statistics+size_t(index)*128:nullptr,
            static_cast<T*>(g.cache_k),static_cast<T*>(g.cache_v),g.cache_rows,g.cached_prefix_rows,g.causal);
    }
    // FlashRT skinny-attention max/denominator/partial-output merge, moved
    // from a warp group to independently scheduled native VM tile producers.
    template<class T,class StateT> static __device__ void merge(const AttentionGlobals& g,StateT& state,int logical){
        int tiles=(g.nq+63)/64,tile=logical%tiles,head=(logical/tiles)%g.heads,batch=logical/(tiles*g.heads);
        // Up to 15 splits plus one reciprocal per row fit the instruction's
        // 4 KiB scratch. Larger experiments retain the direct merge fallback.
        bool cache=NATIVE_ATTENTION_CACHE_MERGE && g.splits<=15;
        float* factors=static_cast<float*>(state.scratch());
        float* inverse_denominator=factors+64*g.splits;
        if(cache){
            int row=threadIdx.x;
            if(row<64 && tile*64+row<g.nq){
                float maximum=-INFINITY;
                for(int part=0;part<g.splits;++part)
                    maximum=fmaxf(maximum,g.statistics[(size_t(logical)*g.splits+part)*128+row*2]);
                float denominator=0.f;
                for(int part=0;part<g.splits;++part){
                    size_t offset=(size_t(logical)*g.splits+part)*128+row*2;
                    float m=g.statistics[offset],l=g.statistics[offset+1];
                    float factor=l>0.f?exp2f(m-maximum):0.f;
                    factors[row*g.splits+part]=factor;denominator+=factor*l;
                }
                inverse_denominator[row]=denominator>0.f?1.f/denominator:0.f;
            }
            native_attention::attention_sync();
        }
        for(int i=threadIdx.x;i<64*(g.hd/2);i+=128){
            int row=i/(g.hd/2),col=(i%(g.hd/2))*2,query=tile*64+row;
            if(query>=g.nq)continue;
            float maximum=-INFINITY;
            if(!cache)for(int part=0;part<g.splits;++part)
                maximum=fmaxf(maximum,g.statistics[(size_t(logical)*g.splits+part)*128+row*2]);
            float denominator=0.f,a=0.f,b=0.f;
            for(int part=0;part<g.splits;++part){
                size_t index=size_t(logical)*g.splits+part,offset=index*128+row*2;
                float factor;
                if(cache)factor=factors[row*g.splits+part];
                else{
                    float m=g.statistics[offset],l=g.statistics[offset+1];
                    factor=l>0.f?exp2f(m-maximum):0.f;denominator+=factor*l;
                }
                float2 value=*reinterpret_cast<const float2*>(g.partial+(index*64+row)*g.hd+col);
                a+=factor*value.x;b+=factor*value.y;
            }
            float inverse=cache?inverse_denominator[row]:(denominator>0.f?1.f/denominator:0.f);
            T* destination=static_cast<T*>(g.output)+(size_t(batch)*g.nq+query)*g.out_ld+head*g.hd+col;
            *reinterpret_cast<uint32_t*>(destination)=native_attention::Op<T>::pack(a*inverse,b*inverse);
        }
    }
    static __device__ bool publish(const AttentionGlobals& g,int logical){
        cuda::atomic_ref<unsigned long long,cuda::thread_scope_device> counter(g.ready[logical]);
        unsigned long long old=counter.load(cuda::memory_order_acquire);
        unsigned long long initial=(static_cast<unsigned long long>(g.epoch)<<32)|1ull;
        for(;;){
            if(unsigned(old>>32)!=g.epoch){
                if(counter.compare_exchange_weak(old,initial,cuda::memory_order_acq_rel,cuda::memory_order_acquire))return g.splits==1;
            }else{
                auto previous=counter.fetch_add(1ull,cuda::memory_order_acq_rel);
                return unsigned(previous)+1u==unsigned(g.splits);
            }
        }
    }
    template<class StateT> static __device__ void consume(const AttentionGlobals& g,StateT& s,int index){
        static_assert(sizeof(s.pages[0])==AC::PAGE_SIZE);
        if(index>=g.tasks)return;
        if(threadIdx.x==0)for(int i=0;i<pages(g);++i)s.wait_page_ready(s.pid(i));
        native_attention::attention_sync();
        if(g.fp16){
            if(g.hd==64)compute<__half,64>(g,s,index);
            else if(g.hd==72)compute<__half,72>(g,s,index);
            else if(g.hd==128)compute<__half,128>(g,s,index);
            else compute<__half,256>(g,s,index);
        }else{
            if(g.hd==64)compute<__nv_bfloat16,64>(g,s,index);
            else if(g.hd==72)compute<__nv_bfloat16,72>(g,s,index);
            else if(g.hd==128)compute<__nv_bfloat16,128>(g,s,index);
            else compute<__nv_bfloat16,256>(g,s,index);
        }
        native_attention::attention_sync();
        if(threadIdx.x==0)for(int i=0;i<pages(g);++i)s.finish_page(s.pid(i),4);
        if(g.splits>1){
            // Every partial writer participates before the leader publishes.
            __threadfence();native_attention::attention_sync();
            int* last=static_cast<int*>(s.scratch());
            if(threadIdx.x==0)*last=publish(g,index/g.splits);
            native_attention::attention_sync();
            int merge_last=*last;
            // All warps snapshot the flag before merge coefficients reuse it.
            native_attention::attention_sync();
            if(merge_last){
                if(g.fp16)merge<__half>(g,s,index/g.splits);
                else merge<__nv_bfloat16>(g,s,index/g.splits);
            }
            native_attention::attention_sync();
        }
    }
    struct consumer {static __device__ void run(const AttentionGlobals& g,AS& s){consume(g,s,task(g,s));}};
};
struct AttentionContext {AttentionGlobals globals;int *program,*timing,device,slots;bool started=false;};
__global__ void attention_slots(int* p){if(threadIdx.x==0)asm volatile("mov.u32 %0, %%nsmid;":"=r"(*p));}
extern "C" int create_attention(void** handle,const void* q,const void* k,const void* v,const void* qb,const void* kb,const void* vb,
    const void* qcos,const void* qsin,const void* kcos,const void* ksin,void* output,const unsigned char* mask,
    int batches,int nq,int nk,int heads,int kv_heads,int hd,int ldq,int ldkv,int out_ld,int fp16,int workers,int prefix,float scale){
    if(!handle)return cudaErrorInvalidValue;*handle=nullptr;
    if(!q||!k||!v||!output||batches<1||nq<1||nk<1||heads<1||kv_heads<1||heads%kv_heads||(hd!=64&&hd!=72&&hd!=128&&hd!=256)
       ||ldq<int64_t(heads)*hd||ldkv<int64_t(kv_heads)*hd||out_ld<int64_t(heads)*hd||ldq%8||ldkv%8||out_ld%8||workers<1
       ||(fp16!=0&&fp16!=1)||!isfinite(scale)||scale<=0||prefix<-1||(prefix>=0&&int64_t(prefix)+nq!=nk)
       ||bool(qcos)!=bool(qsin)||bool(kcos)!=bool(ksin))return cudaErrorInvalidValue;
    int64_t tasks64=int64_t(batches)*heads*((int64_t(nq)+63)/64);if(tasks64>INT_MAX)return cudaErrorInvalidValue;
    int device;cudaDeviceProp prop{};auto error=cudaGetDevice(&device);
    if(error==cudaSuccess)error=cudaGetDeviceProperties(&prop,device);
    if(error!=cudaSuccess)return error;
    if(prop.major!=10||prop.minor!=0||workers>prop.multiProcessorCount)return cudaErrorInvalidConfiguration;
    int *slot_device=nullptr,*program=nullptr,*timing=nullptr,slots=0,tasks=int(tasks64),iterations=int((tasks64+workers-1)/workers);
    error=cudaMalloc(&slot_device,4);
    if(error==cudaSuccess){attention_slots<<<1,32>>>(slot_device);error=cudaGetLastError();}
    if(error==cudaSuccess)error=cudaMemcpy(&slots,slot_device,4,cudaMemcpyDeviceToHost);
    cudaFree(slot_device);if(error!=cudaSuccess)return error;
    std::vector<int> code(size_t(slots)*iterations*AC::INSTRUCTION_WIDTH,0);
    for(size_t i=0;i<code.size();i+=AC::INSTRUCTION_WIDTH)code[i]=1;
    error=cudaMalloc(&program,code.size()*4);
    if(error==cudaSuccess)error=cudaMalloc(&timing,size_t(slots)*iterations*AC::TIMING_WIDTH*4);
    if(error==cudaSuccess)error=cudaMemcpy(program,code.data(),code.size()*4,cudaMemcpyHostToDevice);
    auto kernel=megakernel::mk<AC,AttentionGlobals,AttentionOp>;
    if(error==cudaSuccess)error=cudaFuncSetAttribute(kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,AC::DYNAMIC_SHARED_MEMORY);
    if(error==cudaSuccess){
        AttentionGlobals g{megakernel::instruction_layout<AC>(program,nullptr,slots,iterations,nullptr),
            megakernel::timing_layout<AC>(timing,nullptr,slots,iterations,nullptr),q,k,v,qb,kb,vb,qcos,qsin,kcos,ksin,output,mask,
            batches,nq,nk,heads,kv_heads,hd,ldq,ldkv,out_ld,fp16,workers,prefix,tasks,scale*1.4426950408889634f};
        *handle=new AttentionContext{g,program,timing,device,slots};
    }else{cudaFree(program);cudaFree(timing);}
    return error;
}
extern "C" int launch_attention(void* handle,void* stream){
    if(!handle)return cudaErrorInvalidValue;auto& c=*static_cast<AttentionContext*>(handle);int device;auto error=cudaGetDevice(&device);
    if(error!=cudaSuccess)return error;if(device!=c.device)return cudaErrorInvalidDevice;
    if(c.globals.splits>1){
        cudaStreamCaptureStatus capture;error=cudaStreamIsCapturing(static_cast<cudaStream_t>(stream),&capture);
        if(error!=cudaSuccess)return error;if(capture!=cudaStreamCaptureStatusNone)return cudaErrorStreamCaptureUnsupported;
        if(c.globals.epoch==UINT_MAX)return cudaErrorInvalidValue;
        ++c.globals.epoch;
    }
    c.started=true;
    megakernel::mk<AC,AttentionGlobals,AttentionOp><<<c.globals.workers,AC::NUM_THREADS,AC::DYNAMIC_SHARED_MEMORY,static_cast<cudaStream_t>(stream)>>>(c.globals);
    return cudaGetLastError();
}
extern "C" int destroy_attention(void* handle){
    if(!handle)return cudaSuccess;auto* c=static_cast<AttentionContext*>(handle);int device;auto error=cudaGetDevice(&device);
    if(error!=cudaSuccess)return error;if(device!=c->device)return cudaErrorInvalidDevice;
    auto a=cudaFree(c->program),b=cudaFree(c->timing);
    for(void* p:{static_cast<void*>(c->globals.partial),static_cast<void*>(c->globals.statistics),static_cast<void*>(c->globals.ready)}){
        auto status=cudaFree(p);if(a==cudaSuccess)a=status;
    }
    delete c;return a==cudaSuccess?b:a;
}
extern "C" int attention_resources(int* values,int count){
    if(!values||count<6)return cudaErrorInvalidValue;
    int device,blocks=0;cudaFuncAttributes attributes{};cudaDeviceProp properties{};
    auto kernel=megakernel::mk<AC,AttentionGlobals,AttentionOp>;
    auto error=cudaGetDevice(&device);
    if(error==cudaSuccess)error=cudaGetDeviceProperties(&properties,device);
    if(error==cudaSuccess)error=cudaFuncSetAttribute(kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,AC::DYNAMIC_SHARED_MEMORY);
    if(error==cudaSuccess)error=cudaFuncGetAttributes(&attributes,kernel);
    if(error==cudaSuccess)error=cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,kernel,AC::NUM_THREADS,AC::DYNAMIC_SHARED_MEMORY);
    if(error==cudaSuccess){values[0]=blocks;values[1]=properties.multiProcessorCount;values[2]=attributes.numRegs;
        values[3]=attributes.localSizeBytes;values[4]=attributes.sharedSizeBytes;values[5]=AC::DYNAMIC_SHARED_MEMORY;}
    return error;
}
// Setup-only: scratch and instructions are allocated before any forward.
extern "C" int configure_attention_splits(void* handle,int splits){
    if(!handle||splits<1||splits>64)return cudaErrorInvalidValue;
    auto& c=*static_cast<AttentionContext*>(handle);int device;auto error=cudaGetDevice(&device);
    if(error!=cudaSuccess)return error;if(device!=c.device)return cudaErrorInvalidDevice;
    if(c.started||c.globals.splits!=1)return cudaErrorInvalidValue;
    if(splits==1)return cudaSuccess;
    int logical=c.globals.tasks;
    int64_t total64=int64_t(logical)*splits;if(total64>INT_MAX)return cudaErrorInvalidValue;
    int total=int(total64),iterations=int((total64+c.globals.workers-1)/c.globals.workers);
    std::vector<int> code(size_t(c.slots)*iterations*AC::INSTRUCTION_WIDTH,0);
    for(size_t i=0;i<code.size();i+=AC::INSTRUCTION_WIDTH)code[i]=1;
    int *program=nullptr,*timing=nullptr;float *partial=nullptr,*statistics=nullptr;unsigned long long* ready=nullptr;
    error=cudaMalloc(&program,code.size()*4);
    if(error==cudaSuccess)error=cudaMalloc(&timing,size_t(c.slots)*iterations*AC::TIMING_WIDTH*4);
    if(error==cudaSuccess)error=cudaMalloc(&partial,size_t(total)*64*c.globals.hd*4);
    if(error==cudaSuccess)error=cudaMalloc(&statistics,size_t(total)*128*4);
    if(error==cudaSuccess)error=cudaMalloc(&ready,size_t(logical)*8);
    if(error==cudaSuccess)error=cudaMemset(partial,0,size_t(total)*64*c.globals.hd*4);
    if(error==cudaSuccess)error=cudaMemset(statistics,0,size_t(total)*128*4);
    if(error==cudaSuccess)error=cudaMemset(ready,0,size_t(logical)*8);
    if(error==cudaSuccess)error=cudaMemcpy(program,code.data(),code.size()*4,cudaMemcpyHostToDevice);
    if(error!=cudaSuccess){cudaFree(program);cudaFree(timing);cudaFree(partial);cudaFree(statistics);cudaFree(ready);return error;}
    cudaFree(c.program);cudaFree(c.timing);c.program=program;c.timing=timing;
    c.globals.instructions=megakernel::instruction_layout<AC>(program,nullptr,c.slots,iterations,nullptr);
    c.globals.timings=megakernel::timing_layout<AC>(timing,nullptr,c.slots,iterations,nullptr);
    c.globals.splits=splits;c.globals.tasks=total;c.globals.partial=partial;c.globals.statistics=statistics;c.globals.ready=ready;
    return cudaSuccess;
}
// Development diagnostics, never called by forward.
extern "C" int attention_split_diagnostics(void* handle,unsigned long long* counts,int capacity,unsigned* epoch){
    if(!handle||!counts||!epoch)return cudaErrorInvalidValue;
    auto& c=*static_cast<AttentionContext*>(handle);int logical=c.globals.tasks/c.globals.splits,device;
    if(c.globals.splits<2||capacity<logical)return cudaErrorInvalidValue;
    auto error=cudaGetDevice(&device);if(error!=cudaSuccess)return error;if(device!=c.device)return cudaErrorInvalidDevice;
    *epoch=c.globals.epoch;return cudaMemcpy(counts,c.globals.ready,size_t(logical)*8,cudaMemcpyDeviceToHost);
}
extern "C" int attention_poison_partials(void* handle,void* stream){
    if(!handle)return cudaErrorInvalidValue;auto& c=*static_cast<AttentionContext*>(handle);
    if(c.globals.splits<2)return cudaErrorInvalidValue;
    int device;auto error=cudaGetDevice(&device);if(error!=cudaSuccess)return error;if(device!=c.device)return cudaErrorInvalidDevice;
    error=cudaMemsetAsync(c.globals.partial,255,size_t(c.globals.tasks)*64*c.globals.hd*4,static_cast<cudaStream_t>(stream));
    if(error==cudaSuccess)error=cudaMemsetAsync(c.globals.statistics,255,size_t(c.globals.tasks)*128*4,static_cast<cudaStream_t>(stream));
    return error;
}

// Cache storage is caller-owned and remains resident across prefix/expert calls.
extern "C" int configure_attention_cache(void* handle,void* key,void* value,int rows,int prefix_rows){
    if(!handle||!key||!value||rows<1||prefix_rows<0)return cudaErrorInvalidValue;
    auto& c=*static_cast<AttentionContext*>(handle);int device;auto error=cudaGetDevice(&device);
    if(error!=cudaSuccess)return error;if(device!=c.device)return cudaErrorInvalidDevice;
    if(c.started||c.globals.cache_k||rows<c.globals.nk||prefix_rows>=c.globals.nk)return cudaErrorInvalidValue;
    c.globals.cache_k=key;c.globals.cache_v=value;c.globals.cache_rows=rows;c.globals.cached_prefix_rows=prefix_rows;
    return cudaSuccess;
}

extern "C" int configure_attention_causal(void* handle){
    if(!handle)return cudaErrorInvalidValue;auto& c=*static_cast<AttentionContext*>(handle);
    if(c.globals.nq!=c.globals.nk||c.globals.cached_prefix_rows)return cudaErrorInvalidValue;
    c.globals.causal=true;return cudaSuccess;
}
