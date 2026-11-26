#include "waltz_hip.h"
#include "utils/Blob.hpp"
#include <hip/hip_runtime.h>
#include <rocprim/rocprim.hpp>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include "wzp_hip.cuh"
#include "fused_hip.cuh"
#include "reorder_hip.cuh"
#include "../../src/batch_decode.cuh"
#include "xy_wave64.cuh"

namespace native {
namespace tiles = WALTZ::allhalo;
namespace wzp = WALTZ::lossless;
constexpr unsigned Threads=256, Brick=2048;
using u64 = unsigned long long;
void require(bool ok,const char* why){if(!ok)throw std::runtime_error(why);}
void ck(hipError_t code){if(code!=hipSuccess)throw std::runtime_error(hipGetErrorString(code));}
void launch_check(){ck(hipGetLastError());}
// Diagnostic build only. Release benchmarks compile these markers out.
struct Trace {
#ifdef WALTZ_HIP_TRACE
    const char* operation;
    const char* names[16]{};
    hipEvent_t events[16]{};
    unsigned used=0;
    Trace(const char* op):operation(op){mark("begin");}
    void mark(const char* name){
        if(used==16)return;
        ck(hipEventCreate(events+used));names[used]=name;
        ck(hipEventRecord(events[used++],nullptr));
    }
    void finish(){
        ck(hipEventSynchronize(events[used-1]));
        std::fprintf(stderr,"HIPTRACE,%s",operation);
        for(unsigned j=1;j<used;++j){
            float ms=0;ck(hipEventElapsedTime(&ms,events[j-1],events[j]));
            std::fprintf(stderr,",%s=%.4f",names[j],ms);
        }
        std::fputc('\n',stderr);
    }
    ~Trace(){for(auto e:events)if(e)(void)hipEventDestroy(e);}
#else
    Trace(const char*){}
    void mark(const char*){}
    void finish(){}
#endif
};
unsigned blocks(size_t n,unsigned b=Threads){return unsigned((n+b-1)/b);}
unsigned half(unsigned n){return (n+1)/2;}
struct Profile {dim3 dims;uint32_t n,bz,lx,lz;bool dyadic;};
bool regular_morton(Profile p){
    return p.dims.x%16==0&&p.dims.y%16==0&&p.dims.z%p.bz==0&&!(p.bz&(p.bz-1));
}
Profile profile(const waltz_hip_config& c){
    require(c.width==4||c.width==8,"unsupported dtype");
    require(c.x&&c.y&&c.z&&uint64_t(c.x)*c.y<=uint64_t(INT32_MAX)/unsigned(c.width)/c.z,"invalid or oversized shape");
    const unsigned lx=WALTZ::dwt_levels(std::min(c.x,c.y)),lz=WALTZ::dwt_levels(c.z);
    const dim3 dims(c.x,c.y,c.z);
    return {dims,c.x*c.y*c.z,wzp::select_block_zband(dims,lz,lx==lz),lx,lz,lx==lz};
}
struct Device {
    void* ptr=nullptr;size_t capacity=0;
    ~Device(){if(ptr)(void)hipFree(ptr);}
    template<class T> T* get(size_t count){
        size_t bytes=count*sizeof(T);
        if(bytes>capacity){
            if(ptr){ck(hipFree(ptr));ptr=nullptr;capacity=0;}
            ck(hipMalloc(&ptr,bytes));capacity=bytes;
        }
        return static_cast<T*>(ptr);
    }
};
using Header=WALTZ::WaltzBlobHeader;
using WALTZ::BlobSize;
struct Stats {u64 side;unsigned payload,status;};
struct HostReceipt {Header header;Stats stats;unsigned status;};
}

struct waltz_hip_session {
    int device,sms;
    native::Device masks,first,second,coeff,zbuf,correction,qout,mag,sign,table,rank_table,stats,status,scan,
        inverse_stage,
        side_count,side_offset,chunk_count,wzp_count,wzp_offset,slots,sign_slots,meta,wzp_scan;
    hipStream_t auxiliary=nullptr;
    hipEvent_t quant_done=nullptr,wzp_done=nullptr;
    native::Profile table_key{};
    char error[240]{};
    native::HostReceipt* host=nullptr;
    waltz_hip_session(int d,int sm):device(d),sms(sm){
        native::ck(hipHostMalloc(reinterpret_cast<void**>(&host),sizeof(*host)));
    }
    void enable_overlap(){
        if(!auxiliary)native::ck(hipStreamCreateWithFlags(&auxiliary,hipStreamNonBlocking));
        if(!quant_done)native::ck(hipEventCreateWithFlags(&quant_done,hipEventDisableTiming));
        if(!wzp_done)native::ck(hipEventCreateWithFlags(&wzp_done,hipEventDisableTiming));
    }
    ~waltz_hip_session(){
        if(auxiliary)(void)hipStreamSynchronize(auxiliary);
        if(quant_done)(void)hipEventDestroy(quant_done);
        if(wzp_done)(void)hipEventDestroy(wzp_done);
        if(auxiliary)(void)hipStreamDestroy(auxiliary);
        if(host)(void)hipHostFree(host);
    }
};

namespace native {
using tiles::Read;
using tiles::Store;
using tiles::xy;
using tiles::zline;

dim3 shrink(dim3 n,int axes){if(axes&2){n.x=half(n.x);n.y=half(n.y);}if(axes&1)n.z=half(n.z);return n;}

template<class T> struct BandStore {
    T *bands,*low;
    dim3 inner;
    bool last;
    __device__ void operator()(T v,size_t at,int x,int y,int z)const{
        if(last||x>=int(inner.x)||y>=int(inner.y)||z>=int(inner.z))bands[at]=v;
        else low[at]=v;
    }
};
template<bool Inv,int Width,int Rows,class T,class Out>
void launch_strip(Read<T> src,dim3 n,Out out) {
    // Prefer adjacent planes for locality. Pack Y strips when more than half
    // the subgroups would otherwise be idle and Y has enough independent work.
    constexpr unsigned groups=256/Width;
    const unsigned gx=blocks(n.x,2*Width-8),strips=blocks(n.y,Rows);
    if(n.z<groups/2 && strips>=groups)
        xy_strip<Inv,Width,Rows,true><<<dim3(gx,blocks(strips,groups),n.z),256>>>(src,n,out);
    else
        xy_strip<Inv,Width,Rows><<<dim3(gx,strips,blocks(n.z,groups)),256>>>(src,n,out);
}
template<bool Inv,class T,class Out> void run_xy(Read<T> src, dim3 n, Out out) {
    const dim3 grid(blocks(n.x,56),blocks(n.y,8),blocks(n.z,8));
    if ((!src.low || (Inv && src.low_dims.x==half(n.x) && src.low_dims.y==half(n.y) && src.low_dims.z==n.z)) && n.y>=64 && n.x>=64) {
        if constexpr(!Inv || sizeof(T)==4) {
            if(n.x>=128) {
                constexpr int rows=Inv?8:16;
                launch_strip<Inv,64,rows>(src,n,out);
                return;
            }
        }
        launch_strip<Inv,32,8>(src,n,out);
    } else xy<T,Inv><<<grid,256>>>(src,n,out);
}
template<class T> void forward_level(Read<T> src,dim3 n,BandStore<T> emit,int axes,T* stage){
    if(axes==3){
        constexpr unsigned Z=sizeof(T)==4?8:4;
        run_xy<false>(src,n,Store<T>{stage});
        launch_check();
        zline<Z,T,false><<<dim3(blocks(size_t(n.x)*n.y,128),blocks(n.z,Z)),128>>>
            (Read<T>{stage,nullptr,src.full,{}},n,emit);
    }else if(axes==2)
        run_xy<false>(src,n,emit);
    else{
        constexpr unsigned Z=sizeof(T)==4?16:8;
        zline<Z,T,false><<<dim3(blocks(size_t(n.x)*n.y,128),blocks(n.z,Z)),128>>>(src,n,emit);
    }
    launch_check();
}
template<class T> const T* forward_bands(waltz_hip_session& w,const T* src,Profile p,
                                         T* bands,unsigned count,int axes,T* stage=nullptr){
    T* a=w.first.get<T>(p.n);T* b=w.second.get<T>(p.n);
    dim3 n=p.dims;
    for(unsigned j=0;j<count;++j){
        const dim3 inner=shrink(n,axes);
        T* dst=(j&1)?b:a;
        forward_level(Read<T>{src,nullptr,p.dims,{}},n,BandStore<T>{bands,dst,inner,j+1==count},axes,stage);
        n=inner;src=dst;
    }
    return bands;
}
template<class T> const T* forward(waltz_hip_session& w,const T* input,Profile p){
    if(!p.lx&&!p.lz)return input;
    if(p.dyadic){
        T* stage=w.inverse_stage.get<T>(p.n);
        return forward_bands(w,input,p,w.coeff.get<T>(p.n),p.lx,3,stage);
    }
    const T* src=p.lz?forward_bands(w,input,p,w.zbuf.get<T>(p.n),p.lz,1):input;
    return p.lx?forward_bands(w,src,p,w.coeff.get<T>(p.n),p.lx,2):src;
}
template<class T> __device__ bool strict(double a,double b,double eb){
    const double high=a-b,m=fabs(high);
    if(!isfinite(high)||m>eb)return false;
    if(m<eb)return true;
    const double v=a-high,low=(a-(high+v))+(v-b);
    return !((high>0&&low>0)||(high<0&&low<0));
}
struct QuantExceptions {
    uint32_t* value;
    uint32_t* present;
    __device__ void put(size_t i,unsigned mag) const {
        if(mag>65535u){value[i]=mag;atomicOr(present+(i>>5),1u<<(i&31));}
    }
};
template<class T,bool Fast=false> struct Verify {
    const T* input;T* correction;double bound;unsigned* status;unsigned* present;
    __device__ T error_value(T sample,T src) const {
        const T err=sample-src;
        if(fabs(err)<T(bound))return T(0);
        const bool patch=fabs(err)>T(bound);
        const T v=patch?sample-err:sample;
        bool ok=false;
        if constexpr(Fast&&sizeof(T)==4) {
            // If the RN FP32 difference is strictly below RN(bound), the
            // exact difference is <= their midpoint, which is <= bound.
            // Equality, nonfinite and subnormal edge cases use TwoDiff.
            const T hi=patch?v-src:err;
            ok=fabs(hi)<T(bound);
        }
        if(!isfinite(err)||(!ok&&!strict<T>(double(v),double(src),bound)))atomicOr(status,2u);
        return patch?err:T(0);
    }
    __device__ void operator()(T sample,size_t at) const {
        const T e=error_value(sample,input[at]);
        if(e!=T(0)){correction[at]=e;atomicOr(present+(at>>5),1u<<(at&31));}
    }
    __device__ void pair(float a,float b,unsigned at) const{
        const float2 v=reinterpret_cast<const float2*>(input)[at/2];
        const float da=error_value(a,v.x),db=error_value(b,v.y);
        if(da!=0)correction[at]=da;
        if(db!=0)correction[at+1]=db;
        const unsigned m=unsigned(da!=0)|(unsigned(db!=0)<<1);
        if(m)atomicOr(present+(at>>5),m<<(at&31));
    }
};
struct PairStore {
    float* output;
    __device__ void pair(float a,float b,unsigned at)const {
        reinterpret_cast<float2*>(output)[at/2]=make_float2(a,b);
    }
};
template<class T,bool F> auto pair_out(Verify<T,F> v){return v;}
inline auto pair_out(Store<float> v){return PairStore{v.output};}
template<class T,bool F> bool pair_aligned(Verify<T,F> v){return !(reinterpret_cast<uintptr_t>(v.input)&7);}
inline bool pair_aligned(Store<float> v){return !(reinterpret_cast<uintptr_t>(v.output)&7);}
template<class T,class Out> __global__ void identity(const T* src,unsigned n,Out emit){
    size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(i<n)emit(src[i],i);
}
template<class T,class Out> void inverse_level(Read<T> src,dim3 n,Out emit,int axes,T* stage,bool finest=false){
    if(axes==3){
        constexpr unsigned Z=sizeof(T)==4?16:8;
        if constexpr(sizeof(T)==4){
            // Both X subbands must start on pair boundaries. Only the
            // finest level has tightly packed rows in src and stage.
            if(finest && n.x>=128 && n.y>=128 && n.z>=64 && !(n.x&3u)){
                inverse_zpairs<<<dim3(blocks(n.x,256),n.y,blocks(n.z,16)),128>>>(src,n,stage);
                launch_check();
                const Read<T> from_z{stage,nullptr,src.full,{}};
                if(pair_aligned(emit))
                    xy_strip<true,64,8,false,true><<<dim3(blocks(n.x,120),blocks(n.y,8),blocks(n.z,4)),256>>>(
                        from_z,n,pair_out(emit));
                else run_xy<true>(from_z,n,emit);
                launch_check();return;
            }
        }
        inverse_zstrip<Z><<<dim3(blocks(size_t(n.x)*n.y,128),blocks(n.z,Z)),128>>>(src,n,Store<T>{stage});
        launch_check();
        run_xy<true>(Read<T>{stage,nullptr,src.full,{}},n,emit);
    }else if(axes==2)
        run_xy<true>(src,n,emit);
    else{
        constexpr unsigned Z=sizeof(T)==4?16:8;
        inverse_zstrip<Z><<<dim3(blocks(size_t(n.x)*n.y,128),blocks(n.z,Z)),128>>>(src,n,emit);
    }
    launch_check();
}
template<class T,class Out> void inverse(waltz_hip_session& w,const T* coeff,Profile p,Out out){
    if(!p.lx&&!p.lz){identity<<<blocks(p.n),Threads>>>(coeff,p.n,out);launch_check();return;}
    T* a=w.first.get<T>(p.n);T* b=w.second.get<T>(p.n);
    T* stage=p.dyadic?w.inverse_stage.get<T>(p.n):nullptr;
    auto phase=[&](const T* src,unsigned count,int axes,auto emit){
        dim3 dims[5],n=p.dims,low;
        for(unsigned j=0;j<count;++j){dims[j]=n;n=shrink(n,axes);}
        const T* prior=nullptr;
        for(unsigned step=0;step<count;++step){
            n=dims[count-1-step];
            Read<T> read{src,prior,p.dims,low};
            if(step+1==count)inverse_level(read,n,emit,axes,stage,true);
            else{
                T* temp=(step&1)?b:a;
                inverse_level(read,n,Store<T>{temp},axes,stage);prior=temp;low=n;
            }
        }
    };
    if(p.dyadic)phase(coeff,p.lx,3,out);
    else if(p.lx&&p.lz){
        T* stage=w.zbuf.get<T>(p.n);
        phase(coeff,p.lx,2,Store<T>{stage});phase(stage,p.lz,1,out);
    }else phase(coeff,p.lx?p.lx:p.lz,p.lx?2:1,out);
}

template<class T>
__device__ int32_t quantize_value(T v,unsigned* status){
    if(fabs(v)<T(2147483647.0)){
        if constexpr(sizeof(T)==4)return int32_t(nearbyintf(v));
        else return int32_t(nearbyint(v));
    }
    if(!isfinite(v)||v<=T(-0x1p63)||v>=T(0x1p63)){atomicOr(status,1u);return 0;}
    return v<0?-INT32_MAX:INT32_MAX;
}

// A workgroup reads and writes contiguous row-major coefficients, then emits
// contiguous packed magnitudes/signs after an LDS Morton permutation.
template<class T>
__global__ void quant_natural(const T* in,T* coeff,uint16_t* mag,uint32_t* signs,QuantExceptions qout,
                              const uint16_t* rank,dim3 n,unsigned bz,T q,T inv,unsigned* status){
    extern __shared__ uint32_t perm[];
    const unsigned bx=blockIdx.x*16,by=blockIdx.y*16,bz0=blockIdx.z*bz;
    const unsigned nx=min(16u,n.x-bx),ny=min(16u,n.y-by),nz=min(bz,n.z-bz0);
    const unsigned count=nx*ny*nz;
    const unsigned shape=(nx!=16)|((ny!=16)<<1)|((nz!=bz)<<2);
    const size_t plane=size_t(n.x)*n.y;
    const size_t base=plane*bz0+size_t(by)*n.x+bx;
    const size_t packed=plane*bz0+size_t(n.x)*nz*by+size_t(ny)*nz*bx;
    const bool aligned=(packed%32==0&&count%32==0);
    rank+=shape*256*bz;
    for(unsigned i=threadIdx.x;i<count;i+=Threads){
        const size_t at=base+(nx==16&&ny==16?
            (i&15)+size_t((i>>4)&15)*n.x+size_t(i>>8)*plane:
            i%nx+size_t(i/nx%ny)*n.x+size_t(i/(nx*ny))*plane);
        const int32_t a=quantize_value(in[at]*inv,status);
        const unsigned m=a<0?unsigned(-a):unsigned(a);
        coeff[at]=a?T(a)*q:T(0);qout.put(at,m);
        perm[rank[i]]=min(65535u,m)|(a<0?1u<<31:0u);
    }
    __syncthreads();
    for(unsigned i=threadIdx.x;i<(count+63u)/64u*64u;i+=Threads){
        bool neg=false;
        if(i<count){
            const unsigned v=perm[i];neg=bool(v>>31);mag[packed+i]=uint16_t(v);
            if(!aligned&&neg)atomicOr(signs+(packed+i)/32,1u<<((packed+i)&31));
        }
        if(aligned){
            const unsigned mask=wzp::ballot32(neg);
            if(!(threadIdx.x&31)&&i<count)signs[(packed+i)/32]=mask;
        }
    }
}

// Dequantized coefficients feed local synthesis. For WZP, a zeroed Morton
// buffer holds a 15-bit magnitude plus sign; 0x7fff escapes to the full
// transport magnitude. This staging convention never reaches the archive.
template<class T> struct QuantBand {
    T *coeff,*low;
    uint16_t *codes,*escapes;
    QuantExceptions qout;uint32_t* status;
    const uint16_t* rank;
    dim3 full,inner;
    unsigned bz,zshift;
    bool last;
    T q,inv;
    __device__ void operator()(T v,size_t at,int x,int y,int z)const{
        if(!last&&x<int(inner.x)&&y<int(inner.y)&&z<int(inner.z)){low[at]=v;return;}
        const int32_t a=quantize_value(v*inv,status);
        coeff[at]=a?T(a)*q:T(0);
        if(!a)return; // packed magnitudes are cleared before the transform
        const unsigned bx=unsigned(x)&~15u,by=unsigned(y)&~15u;
        const unsigned bz0=(unsigned(z)>>zshift)<<zshift;
        const unsigned coords=(unsigned(z)-bz0)*256+(unsigned(y)&15u)*16+(unsigned(x)&15u);
        const unsigned packed=full.x*full.y*bz0+full.x*bz*by+16u*bz*bx+rank[coords];
        const unsigned m=a<0?unsigned(-a):unsigned(a);
        qout.put(at,m);
        const unsigned kept=min(32767u,m);
        codes[packed]=uint16_t(kept|(a<0?32768u:0u));
        if(m>=32767u)escapes[packed]=uint16_t(min(65535u,m));
    }
};
template<class T>
void forward_quantized(waltz_hip_session& w,const T* src,T* coeff,uint16_t* codes,uint16_t* escapes,
                       QuantExceptions qout,unsigned* status,const uint16_t* rank,Profile p,T q,T inv){
    T* a=w.first.get<T>(p.n);T* b=w.second.get<T>(p.n);
    dim3 n=p.dims;
    unsigned zshift=0;while((1u<<zshift)<p.bz)++zshift;
    if(!p.dyadic){
        // Complete the plane Z transform before any XY analysis. Only final
        // XY subbands are quantized; the recursively transformed LL stays
        // at its original precision until the next level consumes it.
        if(p.lz)src=forward_bands(w,src,p,w.zbuf.get<T>(p.n),p.lz,1);
        for(unsigned l=0;l<p.lx;++l){
            const dim3 inner=shrink(n,2);
            T* low=(l&1)?b:a;
            run_xy<false>(Read<T>{src,nullptr,p.dims,{}},n,
                QuantBand<T>{coeff,low,codes,escapes,qout,status,rank,p.dims,inner,p.bz,zshift,l+1==p.lx,q,inv});
            launch_check();n=inner;src=low;
        }
        return;
    }
    T* stage=w.inverse_stage.get<T>(p.n);
    constexpr unsigned Z=sizeof(T)==4?16:8;
    for(unsigned lev=0;lev<p.lx;++lev){
        T* low=(lev&1)?b:a;
        const dim3 inner=shrink(n,3);
        run_xy<false>(Read<T>{src,nullptr,p.dims,{}},n,Store<T>{stage});
        launch_check();
        zline<Z,T,false><<<dim3(blocks(size_t(n.x)*n.y,128),blocks(n.z,Z)),128>>>(
            Read<T>{stage,nullptr,p.dims,{}},n,
            QuantBand<T>{coeff,low,codes,escapes,qout,status,rank,p.dims,inner,p.bz,zshift,lev+1==p.lx,q,inv});
        launch_check();n=inner;src=low;
    }
}

const uint32_t* table(waltz_hip_session& w,Profile p){
    auto* d=w.table.get<uint32_t>(8*256*p.bz);
    const auto k=w.table_key;
    if(k.bz!=p.bz||k.dims.x!=p.dims.x||k.dims.y!=p.dims.y||k.dims.z!=p.dims.z){
        wzp::d_build_block_ioffset<<<dim3(p.bz,8),256>>>(d,16,16,p.bz,p.dims.x,size_t(p.dims.x)*p.dims.y,p.dims);
        launch_check();
        wzp::d_build_block_rank<<<dim3(p.bz,8),256>>>(w.rank_table.get<uint16_t>(8*256*p.bz),16,16,p.bz,p.dims);
        launch_check();w.table_key=p;
    }
    return d;
}

template<class V> void scan(Device& scratch,const V* in,V* out,unsigned n,hipStream_t stream=nullptr){
    size_t sz=0;
    ck(rocprim::exclusive_scan(nullptr,sz,in,out,V(0),n,rocprim::plus<V>(),stream));
    ck(rocprim::exclusive_scan(scratch.get<uint8_t>(sz),sz,in,out,V(0),n,rocprim::plus<V>(),stream));
}

// Two naturally ordered indicator bitmaps avoid scanning dense value arrays.
// One wave counts or emits a 2048-value tile (64 words); padding bits are zero.
static_assert(Brick == 64*32 && Threads%64 == 0);
template<bool Write,class T>
__global__ void side_masks(const uint32_t* pm,const uint32_t* qm,unsigned words,unsigned tiles,
    u64* count,const u64* offset=nullptr,const T* errors=nullptr,const uint32_t* qo=nullptr,
    uint32_t* pi=nullptr,T* pv=nullptr,uint32_t* qi=nullptr,uint32_t* qv=nullptr){
    const unsigned word=blockIdx.x*blockDim.x+threadIdx.x, tile=word>>6,lane=word&63;
    if(tile>=tiles)return; // wave-uniform
    if constexpr(Write){if(!count[tile])return;}
    unsigned p=word<words?pm[word]:0,q=word<words?qm[word]:0;
    // Each total is <= 2048, so adding packed 16-bit counts cannot carry.
    const unsigned self=__popc(p)+(unsigned(__popc(q))<<16);
    unsigned sum=self;
    #pragma unroll
    for(unsigned d=1;d<64;d*=2){
        const unsigned before=__shfl_up(sum,d,64);
        if(lane>=d)sum+=before;
    }
    if constexpr(Write){
        const u64 base=offset[tile];
        unsigned pos=unsigned(base)+((sum-self)&65535u),qpos=unsigned(base>>32)+((sum-self)>>16);
        while(p){const unsigned i=word*32+__ffs(p)-1;pi[pos]=i;pv[pos++]=errors[i];p&=p-1;}
        while(q){const unsigned i=word*32+__ffs(q)-1;qi[qpos]=i;qv[qpos++]=qo[i];q&=q-1;}
    } else if(lane==63) count[tile]=(sum&65535u)+(u64(sum>>16)<<32);
}
__device__ unsigned chunk_len(const uint8_t* meta,unsigned chunk,unsigned n){
    const unsigned code=wzp::load_meta24(meta,chunk);
    return wzp::decode_mag_bytes(code,min(16384u,2*n-chunk*16384u))+wzp::decode_sign_bytes(code);
}
__global__ void wzp_groups(const uint8_t* meta,unsigned n,unsigned chunks,unsigned groups,unsigned* sizes){
    const unsigned g=blockIdx.x*blockDim.x+threadIdx.x;
    if(g>=groups)return;
    unsigned sum=0;
    for(unsigned c=g*16;c<min(g*16+16,chunks);++c)sum+=chunk_len(meta,c,n);
    sizes[g]=sum;
}
__global__ void collect(Stats* s,const u64* ns,const u64* os,unsigned sn,
                        const unsigned* nw,const unsigned* ow,unsigned wg,const unsigned* status){
    s->side=ns[sn-1]+os[sn-1];
    s->payload=nw[wg-1]+ow[wg-1];s->status=*status;
}
__device__ __forceinline__ void copy_slot(uint8_t* dst,const uint8_t* src,unsigned bytes,unsigned lane){
    // Align the destination without writing into the neighboring chunk.
    const unsigned head=min(bytes,unsigned(-reinterpret_cast<uintptr_t>(dst))&3u);
    if(lane<head)dst[lane]=src[lane];
    const unsigned words=(bytes-head)/4u;
    for(unsigned k=lane;k<words;k+=64){
        uint32_t v;
        __builtin_memcpy(&v,src+head+4*k,4);
        reinterpret_cast<uint32_t*>(dst+head)[k]=v;
    }
    const unsigned tail=head+4*words+lane;
    if(tail<bytes)dst[tail]=src[tail];
}
__global__ void write_wzp(uint8_t* wire,const uint8_t* meta,const uint8_t* mag,const uint8_t* signs,
                          const unsigned* sizes,const unsigned* offsets,unsigned n,unsigned chunks,unsigned groups){
    const unsigned c=blockIdx.x*4+threadIdx.x/64,lane=threadIdx.x&63;
    if(c>=chunks)return;
    const unsigned g=c/16,first=g*16,local=c-first;
    if(!local && !lane)reinterpret_cast<uint32_t*>(wire)[g]=sizes[g]?offsets[g]:0;
    auto* ometa=wire+size_t(groups)*4;
    if(lane<3)ometa[c*3+lane]=meta[c*3+lane];
    const unsigned code=wzp::load_meta24(meta,c);
    if(!code)return;
    unsigned v=lane<local?chunk_len(meta,first+lane,n):0;
    #pragma unroll
    for(unsigned d=8;d;d>>=1)v+=__shfl_xor(v,d,16);
    const unsigned off=offsets[g]+__shfl(v,0,64);
    const unsigned m=wzp::decode_mag_bytes(code,min(16384u,2*n-c*16384u));
    const unsigned s=wzp::decode_sign_bytes(code);
    uint8_t* dest=ometa+size_t(chunks)*3+off;
    copy_slot(dest,mag+size_t(c)*16384,m,lane);
    copy_slot(dest+m,signs+size_t(c)*1024,s,lane);
}
__global__ void prefix_header(uint8_t* dest,Header h,size_t pi,size_t pv,size_t qi,size_t qv){
    unsigned t=threadIdx.x;
    if(t<BlobSize)dest[t]=reinterpret_cast<const uint8_t*>(&h)[t];
    if(t<8){
        const size_t ends[5]={BlobSize+h.mag_comp_bytes,pi+4ull*h.n_pwe,pv+h.pwe_bytes,qi+4ull*h.n_qo,qv+4ull*h.n_qo};
        for(unsigned j=0;j<5;++j)if(t<((8-(ends[j]&7))&7))dest[ends[j]+t]=0;
    }
}
template<class T,bool Quant>
__global__ void patch(T* data,const uint32_t* index,const void* values,unsigned count,
                      unsigned n,T q,unsigned* status){
    const unsigned j=blockIdx.x*blockDim.x+threadIdx.x;
    if(j>=count)return;
    const unsigned at=index[j];
    if(at>=n||(j&&index[j-1]>=at)){atomicOr(status,4u);return;}
    if constexpr(Quant){
        unsigned v=static_cast<const uint32_t*>(values)[j];
        if(v<=65535||v>INT32_MAX){atomicOr(status,4u);return;}
        const T x=T(v)*q;data[at]=data[at]<T(0)?-x:x;
    }else data[at]-=static_cast<const T*>(values)[j];
}

template<class T>
size_t encode(waltz_hip_session& w,const waltz_hip_config& c,const T* src,uint8_t* dst,size_t cap){
    const Profile p=profile(c);
    const unsigned n=p.n,chunks=blocks(n,8192),groups=blocks(chunks,16),nb=blocks(n,Brick);
    const double quantum=1.5*c.bound;const T q=T(quantum),inv=T(1.0/quantum);
    require(c.bound>0&&std::isfinite(c.bound)&&q>0&&std::isfinite(q)&&std::isfinite(inv),"invalid quantizer");
    unsigned* st=w.status.get<unsigned>(1);ck(hipMemsetAsync(st,0,4));
    Trace trace("encode");
    const bool fuse_quant=p.lx&&regular_morton(p);
    const T* transformed=fuse_quant?nullptr:forward(w,src,p);
    trace.mark("forward");
    // Keep the dequantized values separate from the source so that a block's
    // Morton gather can never race another block's coefficient write.
    T* coeff=w.correction.get<T>(n);
    // After quantization the forward coefficient buffer is free for native repairs.
    T* errors=w.coeff.get<T>(n);
    table(w,p);
    auto* mag=w.mag.get<uint16_t>(n);
    auto* sign=w.sign.get<uint32_t>(fuse_quant?blocks(n,2):blocks(n,32));
    const unsigned words=blocks(n,32);
    auto* pm=w.masks.get<uint32_t>(2ull*words);auto* qm=pm+words;
    ck(hipMemsetAsync(pm,0,2ull*words*sizeof(uint32_t)));
    auto* qo=w.qout.get<uint32_t>(n);const QuantExceptions exceptions{qo,qm};
    if(!fuse_quant)ck(hipMemsetAsync(sign,0,4ull*blocks(n,32)));
    const dim3 qgrid(blocks(p.dims.x,16),blocks(p.dims.y,16),blocks(p.dims.z,p.bz));
    if(fuse_quant){
        // QuantBand only scatters nonzero transport codes.
        ck(hipMemsetAsync(mag,0,size_t(n)*sizeof(uint16_t)));
        const auto* rank=w.rank_table.get<uint16_t>(8*256*p.bz);
        auto* escapes=reinterpret_cast<uint16_t*>(sign);
        forward_quantized(w,src,coeff,mag,escapes,exceptions,st,rank,p,q,inv);
    }else
        quant_natural<<<qgrid,Threads,256*p.bz*sizeof(uint32_t)>>>
            (transformed,coeff,mag,sign,exceptions,w.rank_table.get<uint16_t>(8*256*p.bz),p.dims,p.bz,q,inv,st);
    launch_check();
    trace.mark("quant_morton");
    auto* slots=w.slots.get<uint8_t>(size_t(chunks)*16384);
    auto* ss=w.sign_slots.get<uint8_t>(size_t(chunks)*1024);
    auto* meta=w.meta.get<uint8_t>(size_t(chunks)*3);
    auto* next=w.chunk_count.get<unsigned>(1);
    auto* nw=w.wzp_count.get<unsigned>(groups);auto* ow=w.wzp_offset.get<unsigned>(groups);
    w.enable_overlap();
    const hipStream_t stream=w.auxiliary;
    ck(hipEventRecord(w.quant_done,nullptr));
    ck(hipStreamWaitEvent(stream,w.quant_done,0));
    ck(hipMemsetAsync(next,0,4,stream));
    const auto grid=min(chunks,unsigned(w.sms*4));
    const auto* magnitudes=reinterpret_cast<const byte*>(mag);
    if(fuse_quant)wzp::encode_kernel<true><<<grid,TPB,0,stream>>>(magnitudes,n*2,sign,slots,ss,meta,next);
    else wzp::encode_kernel<false><<<grid,TPB,0,stream>>>(magnitudes,n*2,sign,slots,ss,meta,next);
    launch_check();
    wzp_groups<<<blocks(groups),Threads,0,stream>>>(meta,n,chunks,groups,nw);
    launch_check();scan(w.wzp_scan,nw,ow,groups,stream);
    ck(hipEventRecord(w.wzp_done,stream));
    if(!p.dyadic||!p.lx)inverse(w,coeff,p,Verify<T,true>{src,errors,c.bound,st,pm});
    else inverse(w,coeff,p,Verify<T>{src,errors,c.bound,st,pm});
    trace.mark("inverse_verify");
    auto* ns=w.side_count.get<u64>(nb);auto* os=w.side_offset.get<u64>(nb);
    side_masks<false,T><<<blocks(nb,Threads/64),Threads>>>(pm,qm,words,nb,ns);
    launch_check();scan(w.scan,ns,os,nb);
    trace.mark("side_tally_scan");
    ck(hipStreamWaitEvent(nullptr,w.wzp_done,0));
    trace.mark("wzp_with_prefix");
    auto* ds=w.stats.get<Stats>(1);Stats& host=w.host->stats;
    collect<<<1,1>>>(ds,ns,os,nb,nw,ow,groups,st);launch_check();
    ck(hipMemcpy(&host,ds,sizeof(host),hipMemcpyDeviceToHost));
    trace.mark("wzp_prefix_and_status");
    require(!host.status,"quantization or strict-bound check failed");
    Header h{};
    h.flags=Header::pack_flags(p.dyadic,sizeof(T),WALTZ::PweEncoding::Native,WALTZ::Synthesis::HaloAllLevels);
    h.levels_xy=p.lx;h.levels_z=p.lz;h.reorder_bz=p.bz;h.q=quantum;
    h.mag_comp_bytes=size_t(groups)*4+size_t(chunks)*3+host.payload;
    h.n_pwe=uint32_t(host.side);h.n_qo=uint32_t(host.side>>32);h.pwe_bytes=h.n_pwe*sizeof(T);
    require(!(n>(1u<<20)&&h.n_qo>=(1u<<20))&&!(n>(1u<<22)&&h.n_pwe>=(1u<<22)),"portable side-channel capacity reached");
    const auto l=WALTZ::BlobLayout::checked(h);require(l.end<=cap,"archive capacity too small");
    prefix_header<<<1,64>>>(dst,h,l.pi,l.pv,l.qi,l.qv);launch_check();
    write_wzp<<<blocks(chunks,4),Threads>>>(dst+BlobSize,meta,slots,ss,nw,ow,n,chunks,groups);launch_check();
    trace.mark("write_records");
    if(h.n_pwe||h.n_qo){
        side_masks<true,T><<<blocks(nb,Threads/64),Threads>>>(pm,qm,words,nb,ns,os,errors,qo,reinterpret_cast<uint32_t*>(dst+l.pi),
            reinterpret_cast<T*>(dst+l.pv),reinterpret_cast<uint32_t*>(dst+l.qi),reinterpret_cast<uint32_t*>(dst+l.qv));
        launch_check();
    }
    trace.mark("side_compact");trace.finish();
    ck(hipStreamSynchronize(nullptr));return l.end;
}

template<class T> void decode(waltz_hip_session& w,const waltz_hip_config& c,const uint8_t* src,size_t size,T* dst,size_t dst_size){
    const Profile p=profile(c);
    const unsigned chunks=blocks(p.n,8192),groups=blocks(chunks,16);
    require(size>=BlobSize&&dst_size>=size_t(p.n)*sizeof(T),"short archive or output");
    Trace trace("decode");
    Header& h=w.host->header;ck(hipMemcpy(&h,src,sizeof(h),hipMemcpyDeviceToHost));
    trace.mark("header");
    require(h.flags==Header::pack_flags(p.dyadic,sizeof(T),WALTZ::PweEncoding::Native,WALTZ::Synthesis::HaloAllLevels)
        &&h.levels_xy==p.lx&&h.levels_z==p.lz&&h.reorder_bz==p.bz,"canonical Waltz header required");
    require(h.mag_comp_bytes>=size_t(groups)*4+size_t(chunks)*3&&h.mag_comp_bytes<=size-BlobSize&&h.n_pwe<=p.n&&h.n_qo<=p.n
        &&h.pwe_bytes==size_t(h.n_pwe)*sizeof(T),"invalid section lengths");
    const T q=T(h.q);require(q>0&&std::isfinite(q),"invalid quantum");
    const auto l=WALTZ::BlobLayout::checked(h);require(l.end==size,"inconsistent archive size");
    auto* coeff=w.correction.get<T>(p.n);const auto* ranks=table(w,p);const auto d=p.dims;
    const unsigned bytes=p.n*2;
    const bool full=d.x%16==0&&d.y%16==0&&d.z%p.bz==0;
    if(full&&p.bz==2)
        wzp::decode_kernel<2,T><<<chunks,TPB>>>(src+BlobSize,chunks,groups,bytes,nullptr,nullptr,coeff,ranks,q,d.x,d.y,d.z,p.bz);
    else if(sizeof(T)==4&&full&&p.bz==16)
        wzp::decode_kernel<3,T><<<chunks,TPB>>>(src+BlobSize,chunks,groups,bytes,nullptr,nullptr,coeff,ranks,q,d.x,d.y,d.z,p.bz);
    else wzp::decode_kernel<1,T><<<chunks,TPB>>>(src+BlobSize,chunks,groups,bytes,nullptr,nullptr,coeff,ranks,q,d.x,d.y,d.z,p.bz);
    launch_check();
    unsigned* st=nullptr;
    if(h.n_qo||h.n_pwe){st=w.status.get<unsigned>(1);ck(hipMemsetAsync(st,0,sizeof(*st)));}
    if(h.n_qo){
        patch<T,true><<<blocks(h.n_qo),Threads>>>(coeff,reinterpret_cast<const uint32_t*>(src+l.qi),src+l.qv,h.n_qo,p.n,q,st);
        launch_check();
    }
    trace.mark("wzp_decode_and_qout");
    inverse(w,coeff,p,Store<T>{dst});
    trace.mark("inverse");
    if(h.n_pwe){
        patch<T,false><<<blocks(h.n_pwe),Threads>>>(dst,reinterpret_cast<const uint32_t*>(src+l.pi),src+l.pv,h.n_pwe,p.n,q,st);
        launch_check();
    }
    if(st){
        auto& status=w.host->status;
        ck(hipMemcpy(&status,st,sizeof(status),hipMemcpyDeviceToHost));
        require(!status,"invalid repair indices or value");
    }else ck(hipStreamSynchronize(nullptr));
    trace.mark("patch_and_status");trace.finish();
}
}

extern "C" waltz_hip_session* waltz_hip_create(int dev){
    try{
        native::ck(hipSetDevice(dev));hipDeviceProp_t p{};native::ck(hipGetDeviceProperties(&p,dev));
        native::require(p.warpSize==64,"this build requires wave64");
        return new waltz_hip_session(dev,p.multiProcessorCount);
    }catch(...){return nullptr;}
}
extern "C" void waltz_hip_destroy(waltz_hip_session* w){if(w){(void)hipSetDevice(w->device);delete w;}}
extern "C" const char* waltz_hip_error(const waltz_hip_session* w){return w?w->error:"null session";}
extern "C" size_t waltz_hip_max_bytes(const waltz_hip_config* c){
    if(!c)return 0;
    try{
        const auto p=native::profile(*c);
        const auto chunks=native::blocks(p.n,8192),groups=native::blocks(chunks,16);
        const size_t wzp=size_t(chunks)*3+size_t(groups)*4+size_t(p.n)*2+(p.n+7)/8;
        return WALTZ::BlobLayout::checked(wzp,p.n,p.n*c->width,p.n).end;
    }catch(...){return 0;}
}
extern "C" int waltz_hip_encode(waltz_hip_session* w,const waltz_hip_config* c,const void* in,void* out,size_t cap,size_t* n){
    if(n)*n=0;if(!w)return -1;
    try{
        native::require(c&&in&&out&&n,"null encode argument");native::ck(hipSetDevice(w->device));
        auto* dst=static_cast<uint8_t*>(out);
        if(c->width==4)*n=native::encode(*w,*c,static_cast<const float*>(in),dst,cap);
        else *n=native::encode(*w,*c,static_cast<const double*>(in),dst,cap);
        w->error[0]=0;return 0;
    }catch(const std::exception& e){
        if(w->auxiliary)(void)hipStreamSynchronize(w->auxiliary);
        (void)hipStreamSynchronize(nullptr);std::snprintf(w->error,sizeof(w->error),"%s",e.what());return -1;
    }
}
extern "C" int waltz_hip_decode(waltz_hip_session* w,const waltz_hip_config* c,const void* in,size_t n,void* out,size_t cap){
    if(!w)return -1;
    try{
        native::require(c&&in&&out,"null decode argument");native::ck(hipSetDevice(w->device));
        auto* src=static_cast<const uint8_t*>(in);
        if(c->width==4)native::decode(*w,*c,src,n,static_cast<float*>(out),cap);
        else native::decode(*w,*c,src,n,static_cast<double*>(out),cap);
        w->error[0]=0;return 0;
    }catch(const std::exception& e){(void)hipStreamSynchronize(nullptr);std::snprintf(w->error,sizeof(w->error),"%s",e.what());return -1;}
}
