// Eager Gemma BF16 rounding with tensor MMA, within native range instructions.
#pragma once
namespace native_model_attention {
using namespace native_attention;
template<class T,int HD,int NW=4,class StateT>
__device__ void eager_attend(StateT& state,int task,const T* q,const T* k,const T* v,int ldq,int ldkv,
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
    for(int pass=0;pass<2;++pass){
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
        // Match eager Gemma's BF16 QK output and separate scale rounding.
        #pragma unroll
        for(int n=0;n<kChunk/8;++n){
            #pragma unroll
            for(int j=0;j<4;++j)s[n][j]=float(T(float(T(s[n][j]))*.0625f))*1.4426950408889634f;
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
        if(pass==0){
            #pragma unroll
            for(int r=0;r<2;++r){
                mx[r]=fmaxf(mx[r],__shfl_xor_sync(0xffffffffu,mx[r],1));
                mx[r]=fmaxf(mx[r],__shfl_xor_sync(0xffffffffu,mx[r],2));
                float next=fmaxf(m[r],mx[r]);l[r]*=next==-INFINITY?0.f:exp2f(m[r]-next);m[r]=next;
            }
            #pragma unroll
            for(int n=0;n<kChunk/8;++n){
                l[0]+=(s[n][0]==-INFINITY?0.f:exp2f(s[n][0]-m[0]))+(s[n][1]==-INFINITY?0.f:exp2f(s[n][1]-m[0]));
                l[1]+=(s[n][2]==-INFINITY?0.f:exp2f(s[n][2]-m[1]))+(s[n][3]==-INFINITY?0.f:exp2f(s[n][3]-m[1]));
            }
            continue;
        }
        uint32_t pa[kChunk / 16][4];
        // Head-128 correctness path: preserve the residual of the BF16
        // probability conversion and accumulate it with a second tensor MMA.

#pragma unroll
        for (int n = 0; n < kChunk / 8; ++n) {
            const float p0 = (s[n][0]==-INFINITY?0.f:exp2f(s[n][0] - m[0])), p1 = (s[n][1]==-INFINITY?0.f:exp2f(s[n][1] - m[0]));
            const float p2 = (s[n][2]==-INFINITY?0.f:exp2f(s[n][2] - m[1])), p3 = (s[n][3]==-INFINITY?0.f:exp2f(s[n][3] - m[1]));

            pa[n >> 1][(n & 1) * 2 + 0] = O::pack(p0/l[0], p1/l[0]);
            pa[n >> 1][(n & 1) * 2 + 1] = O::pack(p2/l[1], p3/l[1]);

        }
#pragma unroll
        for (int kk = 0; kk < kChunk / 16; ++kk) {
#pragma unroll
            for (int n = 0; n < NT; ++n){
                auto lo=ld2(&vt[n * 8 + g][kk * 16 + t * 2]),hi=ld2(&vt[n * 8 + g][kk * 16 + 8 + t * 2]);
                O::mma(o[n], pa[kk], lo, hi);

            }
        }
    }
    if(pass==0){
        #pragma unroll
        for(int r=0;r<2;++r){l[r]+=__shfl_xor_sync(0xffffffffu,l[r],1);l[r]+=__shfl_xor_sync(0xffffffffu,l[r],2);}
    }
    } // two passes: statistics, then normalized BF16 P/V product
#pragma unroll
    for (int r = 0; r < 2; ++r) {
        const int row = q0 + w * 16 + g + r * 8;
        if (row >= NQ) continue;
        const float inv = 1.f;
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
// FlashAttention traverses key blocks from last to first.
template<class T,int HD,int NW=4,class StateT>
__device__ void flash_attend(StateT& state,int task,const T* q,const T* k,const T* v,int ldq,int ldkv,
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
    load_chunk(key_end-kChunk);
    for (int c0 = key_end-kChunk; c0 >= key_begin; c0 -= kChunk) {
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
        if (c0-kChunk >= key_begin) load_chunk(c0-kChunk);

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
            if constexpr((HD==128) && std::is_same_v<T,__nv_bfloat16>){
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
                if constexpr((HD==128) && std::is_same_v<T,__nv_bfloat16>){
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
}
