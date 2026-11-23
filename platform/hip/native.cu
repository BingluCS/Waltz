#include "waltz_hip.h"
#include "utils/Blob.hpp"
#include <hip/hip_runtime.h>
#include <rocprim/rocprim.hpp>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <stdexcept>
#include "wzp_hip.cuh"
#include "fused_hip.cuh"
#include "reorder_hip.cuh"

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
size_t align8(size_t bytes){return (bytes+7)&~size_t(7);}
unsigned blocks(size_t n,unsigned b=Threads){return unsigned((n+b-1)/b);}
unsigned half(unsigned n){return (n+1)/2;}
unsigned levels(unsigned n){unsigned l=0;while(n>8&&l<5){n=half(n);++l;}return l;}
struct Profile {dim3 dims;uint32_t n,bz,lx,lz;bool dyadic;};
enum Tune {
    FastVerify=1, SeparateVerify=2, LdsMorton=4, SplitInverse=8,
    SplitForward=16, OverlapWzp=32, FuseQuantization=128,
};
bool regular_morton(Profile p){
    return p.dims.x%16==0&&p.dims.y%16==0&&p.dims.z%p.bz==0&&!(p.bz&(p.bz-1));
}
int default_tune(Profile p){
    if(p.dyadic&&p.lx)
        return SplitInverse|OverlapWzp|(regular_morton(p)?FuseQuantization:LdsMorton|SplitForward);
    return FastVerify|LdsMorton|OverlapWzp;
}
Profile profile(const waltz_hip_config& c){
    require(c.width==4||c.width==8,"unsupported dtype");
    require(c.x&&c.y&&c.z&&uint64_t(c.x)*c.y<=uint64_t(INT32_MAX)/unsigned(c.width)/c.z,"invalid or oversized shape");
    const unsigned lx=levels(std::min(c.x,c.y)),lz=levels(c.z);
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
struct Layout{
    size_t pi,pv,qi,qv,end;
    Layout(const Header& h):pi(align8(BlobSize+h.mag_comp_bytes)),pv(align8(pi+size_t(h.n_pwe)*4)),
        qi(align8(pv+h.pwe_bytes)),qv(align8(qi+size_t(h.n_qo)*4)),end(align8(qv+size_t(h.n_qo)*4)){}
};
struct Stats {u64 side;unsigned payload,status;};
}

struct waltz_hip_session {
    int device,sms,tune=0,tune_override=-1;
    native::Device first,second,coeff,zbuf,correction,qout,mag,sign,table,rank_table,stats,status,scan,
        inverse_output,inverse_stage,
        side_count,side_offset,chunk_count,wzp_count,wzp_offset,slots,sign_slots,meta,wzp_scan;
    hipStream_t auxiliary=nullptr;
    hipEvent_t quant_done=nullptr,wzp_done=nullptr;
    native::Profile table_key{};
    bool has_rank=false;
    char error[240]{};
    waltz_hip_session(int d,int sm):device(d),sms(sm){
        if(const auto* val=std::getenv("WALTZ_HIP_TUNE"))tune_override=std::atoi(val);
    }
    void select_tune(native::Profile p){
        tune=tune_override<0?native::default_tune(p):tune_override;
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
    }
};

namespace native {
using tiles::Read;
using tiles::Store;
using tiles::forward_xyz;
using tiles::inverse_xyz;
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
template<class T> void forward_level(Read<T> src,dim3 n,BandStore<T> emit,int axes,T* stage){
    if(axes==3){
        constexpr unsigned Z=sizeof(T)==4?8:4;
        if(stage){
            xy<T,false><<<dim3(blocks(n.x,56),blocks(n.y,8),blocks(n.z,8)),256>>>(src,n,Store<T>{stage});
            launch_check();
            zline<Z,T,false><<<dim3(blocks(size_t(n.x)*n.y,128),blocks(n.z,Z)),128>>>
                (Read<T>{stage,nullptr,src.full,{}},n,emit);
        }else forward_xyz<Z><<<dim3(blocks(n.x,56),blocks(n.y,8),blocks(n.z,Z)),256,56*8*(Z+8)*sizeof(T)>>>(src,n,emit);
    }else if(axes==2)
        xy<T,false><<<dim3(blocks(n.x,56),blocks(n.y,8),blocks(n.z,8)),256>>>(src,n,emit);
    else{
        constexpr unsigned Z=sizeof(T)==4?16:8;
        zline<Z,T,false><<<dim3(blocks(size_t(n.x)*n.y,128),blocks(n.z,Z)),128>>>(src,n,emit);
    }
    launch_check();
}
template<class T> const T* forward(waltz_hip_session& w,const T* input,Profile p){
    if(!p.lx&&!p.lz)return input;
    T* a=w.first.get<T>(p.n);T* b=w.second.get<T>(p.n);
    T* stage=(w.tune&SplitForward)&&p.dyadic?w.inverse_stage.get<T>(p.n):nullptr;
    auto phase=[&](const T* src,T* bands,unsigned count,int axes){
        dim3 n=p.dims;
        for(unsigned j=0;j<count;++j){
            dim3 inner=shrink(n,axes);
            T* dst=(j&1)?b:a;
            forward_level(Read<T>{src,nullptr,p.dims,{}},n,BandStore<T>{bands,dst,inner,j+1==count},axes,stage);
            n=inner;src=dst;
        }
        return static_cast<const T*>(bands);
    };
    if(p.dyadic)return phase(input,w.coeff.get<T>(p.n),p.lx,3);
    const T* src=p.lz?phase(input,w.zbuf.get<T>(p.n),p.lz,1):input;
    return p.lx?phase(src,w.coeff.get<T>(p.n),p.lx,2):src;
}
template<class T> __device__ bool strict(double a,double b,double eb){
    const double high=a-b,m=fabs(high);
    if(!isfinite(high)||m>eb)return false;
    if(m<eb)return true;
    const double v=a-high,low=(a-(high+v))+(v-b);
    return !((high>0&&low>0)||(high<0&&low<0));
}
template<class T,bool Fast=false> struct Verify {
    const T* input;T* correction;double bound;unsigned* status;
    __device__ T error(T sample,size_t at) const {
        const T src=input[at],err=sample-src;
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
    __device__ void operator()(T sample,size_t at) const {correction[at]=error(sample,at);}
};
template<class T,class Out> __global__ void identity(const T* src,unsigned n,Out emit){
    size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(i<n)emit(src[i],i);
}
template<class T,class Out> void inverse_level(Read<T> src,dim3 n,Out emit,int axes,T* stage=nullptr){
    if(axes==3){
        constexpr unsigned Z=sizeof(T)==4?16:8;
        const dim3 grid(blocks(n.x,56),blocks(n.y,8),blocks(n.z,Z));
        if(stage){
            zline<Z,T,true><<<dim3(blocks(size_t(n.x)*n.y,128),blocks(n.z,Z)),128>>>(src,n,Store<T>{stage});
            launch_check();
            xy<T,true><<<dim3(blocks(n.x,56),blocks(n.y,8),blocks(n.z,8)),256>>>
                (Read<T>{stage,nullptr,src.full,{}},n,emit);
        }else if(n.x>=64&&n.y>=64&&n.z>=64)
            inverse_xyz<Z,T,Out,true><<<grid,256,65536>>>(src,n,emit);
        else inverse_xyz<Z,T,Out,false><<<grid,256,65536>>>(src,n,emit);
    }else if(axes==2)
        xy<T,true><<<dim3(blocks(n.x,56),blocks(n.y,8),blocks(n.z,8)),256>>>(src,n,emit);
    else{
        constexpr unsigned Z=sizeof(T)==4?16:8;
        zline<Z,T,true><<<dim3(blocks(size_t(n.x)*n.y,128),blocks(n.z,Z)),128>>>(src,n,emit);
    }
    launch_check();
}
template<class T,class Out> void inverse(waltz_hip_session& w,const T* coeff,Profile p,Out out){
    if(!p.lx&&!p.lz){identity<<<blocks(p.n),Threads>>>(coeff,p.n,out);launch_check();return;}
    T* a=w.first.get<T>(p.n);T* b=w.second.get<T>(p.n);
    T* stage=(w.tune&SplitInverse)&&p.dyadic?w.inverse_stage.get<T>(p.n):nullptr;
    auto phase=[&](const T* src,unsigned count,int axes,auto emit){
        dim3 dims[5],n=p.dims,low;
        for(unsigned j=0;j<count;++j){dims[j]=n;n=shrink(n,axes);}
        const T* prior=nullptr;
        for(unsigned step=0;step<count;++step){
            n=dims[count-1-step];
            Read<T> read{src,prior,p.dims,low};
            if(step+1==count)inverse_level(read,n,emit,axes,stage);
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

template<class T>
__global__ void quant(const T* in,T* coeff,uint16_t* mag,uint32_t* signs,uint32_t* qout,
                      const uint32_t* table,dim3 n,unsigned bz,T q,T inv,unsigned* status){
    const unsigned bx=blockIdx.x*16,by=blockIdx.y*16,bz0=blockIdx.z*bz;
    const unsigned nx=min(16u,n.x-bx),ny=min(16u,n.y-by),nz=min(bz,n.z-bz0);
    const unsigned count=nx*ny*nz;
    const unsigned shape=(nx!=16)|((ny!=16)<<1)|((nz!=bz)<<2);
    const size_t plane=size_t(n.x)*n.y;
    const size_t natural=plane*bz0+size_t(by)*n.x+bx;
    const size_t packed=plane*bz0+size_t(n.x)*nz*by+size_t(ny)*nz*bx;
    const bool aligned=(packed%32==0&&count%32==0);
    table+=shape*256*bz;
    for(unsigned i=threadIdx.x;i<(count+63u)/64u*64u;i+=Threads){
        bool neg=false;
        if(i<count){
            size_t at=natural+table[i];
            const T v=in[at]*inv;
            int32_t a=0;
            if(!isfinite(v)||v<=T(-0x1p63)||v>=T(0x1p63))atomicOr(status,1u);
            else {
                const long long x=sizeof(T)==4?static_cast<long long>(nearbyintf(float(v))):static_cast<long long>(nearbyint(double(v)));
                a=x>INT32_MAX?INT32_MAX:x< -INT32_MAX?-INT32_MAX:int32_t(x);
            }
            unsigned m=a<0?unsigned(-a):unsigned(a);
            neg=a<0;
            coeff[at]=a?T(a)*q:T(0);
            mag[packed+i]=uint16_t(min(65535u,m));qout[at]=m>65535?m:0;
            if(!aligned&&neg)atomicOr(signs+(packed+i)/32,1u<<((packed+i)&31));
        }
        if(aligned){
            const unsigned mask=wzp::ballot32(neg);
            if(!(threadIdx.x&31)&&i<count)signs[(packed+i)/32]=mask;
        }
    }
}

// A workgroup reads and writes contiguous row-major coefficients, then emits
// contiguous packed magnitudes/signs after an LDS Morton permutation.
template<class T>
__global__ void quant_natural(const T* in,T* coeff,uint16_t* mag,uint32_t* signs,uint32_t* qout,
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
        coeff[at]=a?T(a)*q:T(0);qout[at]=m>65535?m:0;
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

// Write a completed subband directly to both consumers. The Z-line threads
// naturally own consecutive x/y columns. Byte signs avoid cross-tile atomics;
// WZP compacts them to the same format as its ordinary uint32 bitmap input.
template<class T,bool Regular> struct QuantBand {
    T *coeff,*low;
    uint16_t *mag;
    uint8_t* sign;
    uint32_t *qout,*status;
    const uint16_t* rank;
    dim3 full,inner;
    unsigned bz,zshift;
    bool last;
    T q,inv;
    __device__ void operator()(T v,size_t at,int x,int y,int z)const{
        if(!last&&x<int(inner.x)&&y<int(inner.y)&&z<int(inner.z)){low[at]=v;return;}
        const unsigned bx=unsigned(x)&~15u,by=unsigned(y)&~15u;
        unsigned nx,ny,nz,bz0;
        if constexpr(Regular){
            nx=16;ny=16;nz=bz;bz0=(unsigned(z)>>zshift)<<zshift;
        }else{
            bz0=(unsigned(z)/bz)*bz;
            nx=min(16u,full.x-bx);ny=min(16u,full.y-by);nz=min(bz,full.z-bz0);
        }
        const unsigned coords=(unsigned(z)-bz0)*ny*nx+(unsigned(y)&15u)*nx+(unsigned(x)&15u);
        const unsigned shape=(nx!=16)|((ny!=16)<<1)|((nz!=bz)<<2);
        const unsigned packed=full.x*full.y*bz0+full.x*nz*by+ny*nz*bx+rank[shape*256*bz+coords];
        const int32_t a=quantize_value(v*inv,status);
        const unsigned m=a<0?unsigned(-a):unsigned(a);
        coeff[at]=a?T(a)*q:T(0);qout[at]=m>65535?m:0;
        mag[packed]=uint16_t(min(65535u,m));sign[packed]=a<0;
    }
};
template<class T,bool Regular>
void forward_quantized(waltz_hip_session& w,const T* src,T* coeff,uint16_t* mag,uint8_t* sign,
                       uint32_t* qout,unsigned* status,const uint16_t* rank,Profile p,T q,T inv){
    T* a=w.first.get<T>(p.n);T* b=w.second.get<T>(p.n);T* stage=w.inverse_stage.get<T>(p.n);
    dim3 n=p.dims;
    unsigned zshift=0;while((1u<<zshift)<p.bz)++zshift;
    constexpr unsigned Z=sizeof(T)==4?8:4;
    for(unsigned lev=0;lev<p.lx;++lev){
        T* low=(lev&1)?b:a;
        const dim3 inner=shrink(n,3);
        xy<T,false><<<dim3(blocks(n.x,56),blocks(n.y,8),blocks(n.z,8)),256>>>(
            Read<T>{src,nullptr,p.dims,{}},n,Store<T>{stage});
        launch_check();
        zline<Z,T,false><<<dim3(blocks(size_t(n.x)*n.y,128),blocks(n.z,Z)),128>>>(
            Read<T>{stage,nullptr,p.dims,{}},n,
            QuantBand<T,Regular>{coeff,low,mag,sign,qout,status,rank,p.dims,inner,p.bz,zshift,lev+1==p.lx,q,inv});
        launch_check();n=inner;src=low;
    }
}

const uint32_t* table(waltz_hip_session& w,Profile p){
    auto* d=w.table.get<uint32_t>(8*256*p.bz);
    const auto k=w.table_key;
    if(k.bz!=p.bz||k.dims.x!=p.dims.x||k.dims.y!=p.dims.y||k.dims.z!=p.dims.z){
        wzp::d_build_block_ioffset<<<dim3(p.bz,8),256>>>(d,16,16,p.bz,p.dims.x,size_t(p.dims.x)*p.dims.y,p.dims);
        launch_check();w.table_key=p;w.has_rank=false;
    }
    if(!w.has_rank&&(w.tune&(LdsMorton|FuseQuantization))){
        wzp::d_build_block_rank<<<dim3(p.bz,8),256>>>(w.rank_table.get<uint16_t>(8*256*p.bz),16,16,p.bz,p.dims);
        launch_check();w.has_rank=true;
    }
    return d;
}

template<class V> void scan(Device& scratch,const V* in,V* out,unsigned n,hipStream_t stream=nullptr){
    size_t sz=0;
    ck(rocprim::exclusive_scan(nullptr,sz,in,out,V(0),n,rocprim::plus<V>(),stream));
    ck(rocprim::exclusive_scan(scratch.get<uint8_t>(sz),sz,in,out,V(0),n,rocprim::plus<V>(),stream));
}

// Side records remain sorted in native index order: one block tally per dense
// tile, a small device prefix, and stable compaction only in nonempty tiles.
template<class T,bool Fused=false>
__global__ void tally(const T* e,const unsigned* q,unsigned n,u64* count,
                      Verify<T,true> verify={},const T* recon=nullptr){
    using Reduce=rocprim::block_reduce<u64,Threads>;
    __shared__ typename Reduce::storage_type scratch;
    size_t at=size_t(blockIdx.x)*Brick+threadIdx.x;
    u64 v=0;
    #pragma unroll
    for(unsigned i=0;i<Brick/Threads;++i){
        const size_t j=at+i*Threads;
        if(j<n){
            T err;
            if constexpr(Fused){err=verify.error(recon[j],j);verify.correction[j]=err;}
            else err=e[j];
            v+=u64(err!=T(0))+(u64(q[j]!=0)<<32);
        }
    }
    u64 total;Reduce().reduce(v,total,scratch,rocprim::plus<u64>());
    if(!threadIdx.x)count[blockIdx.x]=total;
}
template<class T>
__global__ void compact(const T* error,const unsigned* q,unsigned n,const u64* count,
                        const u64* offset,uint32_t* pi,T* pv,uint32_t* qi,uint32_t* qv){
    if(count[blockIdx.x]==0)return;
    using Scan=rocprim::block_scan<u64,Threads>;
    __shared__ typename Scan::storage_type scratch;
    const size_t base=size_t(blockIdx.x)*Brick+threadIdx.x;
    u64 prefix=offset[blockIdx.x];
    for(unsigned i=0;i<Brick/Threads;++i){
        const size_t j=base+i*Threads;
        const T e=j<n?error[j]:T(0);const unsigned v=j<n?q[j]:0;
        u64 before,total;
        Scan().exclusive_scan(u64(e!=T(0))+(u64(v!=0)<<32),before,u64(0),total,scratch,rocprim::plus<u64>());
        before+=prefix;
        if(e!=T(0)){pi[uint32_t(before)]=uint32_t(j);pv[uint32_t(before)]=e;}
        if(v){qi[before>>32]=uint32_t(j);qv[before>>32]=v;}
        prefix+=total;
        __syncthreads();
    }
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
__global__ void write_wzp(uint8_t* wire,const uint8_t* meta,const uint8_t* mag,const uint8_t* signs,
                          const unsigned* sizes,const unsigned* offsets,unsigned n,unsigned chunks,unsigned groups){
    const unsigned g=blockIdx.x,t=threadIdx.x,first=g*16;
    __shared__ unsigned group_starts[16];
    if(!t){
        reinterpret_cast<uint32_t*>(wire)[g]=sizes[g]?offsets[g]:0;
        unsigned off=offsets[g];
        for(unsigned j=0;j<min(16u,chunks-first);++j){group_starts[j]=off;off+=chunk_len(meta,first+j,n);}
    }
    auto* ometa=wire+size_t(groups)*4;
    const unsigned span=min(16u,chunks-first);
    if(t<span*3)ometa[first*3+t]=meta[first*3+t];
    if(!sizes[g])return;
    __syncthreads();
    unsigned part=t>>4, lane=t&15;
    if(part>=span)return;
    const unsigned off=group_starts[part];
    const unsigned c=first+part,code=wzp::load_meta24(meta,c);
    const unsigned m=wzp::decode_mag_bytes(code,min(16384u,2*n-c*16384u));
    const unsigned s=wzp::decode_sign_bytes(code);
    uint8_t* dest=ometa+size_t(chunks)*3+off;
    for(unsigned i=lane;i<m;i+=16)dest[i]=mag[size_t(c)*16384+i];
    for(unsigned i=lane;i<s;i+=16)dest[m+i]=signs[size_t(c)*1024+i];
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
    w.select_tune(p);
    const unsigned n=p.n,chunks=blocks(n,8192),groups=blocks(chunks,16),nb=blocks(n,Brick);
    const double quantum=1.5*c.bound;const T q=T(quantum),inv=T(1.0/quantum);
    require(c.bound>0&&std::isfinite(c.bound)&&q>0&&std::isfinite(q)&&std::isfinite(inv),"invalid quantizer");
    unsigned* st=w.status.get<unsigned>(1);ck(hipMemsetAsync(st,0,4));
    Trace trace("encode");
    const bool fuse_quant=(w.tune&FuseQuantization)&&p.dyadic&&p.lx;
    const T* transformed=fuse_quant?nullptr:forward(w,src,p);
    trace.mark("forward");
    // Keep the dequantized values separate from the source so that a block's
    // Morton gather can never race another block's coefficient write.
    T* coeff=w.correction.get<T>(n);
    T* errors=w.coeff.get<T>(n);
    if(transformed==errors)errors=w.zbuf.get<T>(n);
    // zbuf may be needed by inverse plane synthesis, so route errors to a
    // separate buffer in that case.
    if(p.lx&&p.lz&&!p.dyadic)errors=w.coeff.get<T>(n);
    const uint32_t* ranks=table(w,p);
    auto* mag=w.mag.get<uint16_t>(n);
    auto* sign=w.sign.get<uint32_t>(fuse_quant?blocks(n,4):blocks(n,32));
    auto* qo=w.qout.get<uint32_t>(n);
    if(!fuse_quant)ck(hipMemsetAsync(sign,0,4ull*blocks(n,32)));
    const dim3 qgrid(blocks(p.dims.x,16),blocks(p.dims.y,16),blocks(p.dims.z,p.bz));
    if(fuse_quant){
        const auto* rank=w.rank_table.get<uint16_t>(8*256*p.bz);
        if(regular_morton(p))forward_quantized<T,true>(w,src,coeff,mag,reinterpret_cast<uint8_t*>(sign),qo,st,rank,p,q,inv);
        else forward_quantized<T,false>(w,src,coeff,mag,reinterpret_cast<uint8_t*>(sign),qo,st,rank,p,q,inv);
    }else if(w.tune&LdsMorton)
        quant_natural<<<qgrid,Threads,256*p.bz*sizeof(uint32_t)>>>
            (transformed,coeff,mag,sign,qo,w.rank_table.get<uint16_t>(8*256*p.bz),p.dims,p.bz,q,inv,st);
    else quant<<<qgrid,Threads>>>(transformed,coeff,mag,sign,qo,ranks,p.dims,p.bz,q,inv,st);
    launch_check();
    trace.mark("quant_morton");
    auto* slots=w.slots.get<uint8_t>(size_t(chunks)*16384);
    auto* ss=w.sign_slots.get<uint8_t>(size_t(chunks)*1024);
    auto* meta=w.meta.get<uint8_t>(size_t(chunks)*3);
    auto* next=w.chunk_count.get<unsigned>(1);
    auto* nw=w.wzp_count.get<unsigned>(groups);auto* ow=w.wzp_offset.get<unsigned>(groups);
    const bool overlap=(w.tune&OverlapWzp);
    auto run_wzp=[&](hipStream_t stream){
        ck(hipMemsetAsync(next,0,4,stream));
        const auto grid=min(chunks,unsigned(w.sms*4));
        const auto* magnitudes=reinterpret_cast<const byte*>(mag);
        if(fuse_quant)wzp::encode_kernel<true><<<grid,TPB,0,stream>>>(magnitudes,n*2,sign,slots,ss,meta,next);
        else wzp::encode_kernel<false><<<grid,TPB,0,stream>>>(magnitudes,n*2,sign,slots,ss,meta,next);
        launch_check();
        wzp_groups<<<blocks(groups),Threads,0,stream>>>(meta,n,chunks,groups,nw);
        launch_check();scan(w.wzp_scan,nw,ow,groups,stream);
    };
    if(overlap){
        w.enable_overlap();
        ck(hipEventRecord(w.quant_done,nullptr));
        ck(hipStreamWaitEvent(w.auxiliary,w.quant_done,0));
        run_wzp(w.auxiliary);
        ck(hipEventRecord(w.wzp_done,w.auxiliary));
    }
    T* reconstructed=nullptr;
    if(w.tune&SeparateVerify){
        reconstructed=w.inverse_output.get<T>(n);
        inverse(w,coeff,p,Store<T>{reconstructed});
    }else if(w.tune&FastVerify)inverse(w,coeff,p,Verify<T,true>{src,errors,c.bound,st});
    else inverse(w,coeff,p,Verify<T>{src,errors,c.bound,st});
    trace.mark("inverse_verify");
    auto* ns=w.side_count.get<u64>(nb);auto* os=w.side_offset.get<u64>(nb);
    if(reconstructed)tally<T,true><<<nb,Threads>>>(errors,qo,n,ns,Verify<T,true>{src,errors,c.bound,st},reconstructed);
    else tally<<<nb,Threads>>>(errors,qo,n,ns);
    launch_check();scan(w.scan,ns,os,nb);
    trace.mark("side_tally_scan");
    if(overlap)ck(hipStreamWaitEvent(nullptr,w.wzp_done,0));
    else run_wzp(nullptr);
    trace.mark("wzp_with_prefix");
    auto* ds=w.stats.get<Stats>(1);Stats host{};
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
    Layout l(h);require(l.end<=cap,"archive capacity too small");
    prefix_header<<<1,64>>>(dst,h,l.pi,l.pv,l.qi,l.qv);launch_check();
    write_wzp<<<groups,Threads>>>(dst+BlobSize,meta,slots,ss,nw,ow,n,chunks,groups);launch_check();
    trace.mark("write_records");
    if(h.n_pwe||h.n_qo){
        compact<<<nb,Threads>>>(errors,qo,n,ns,os,reinterpret_cast<uint32_t*>(dst+l.pi),
            reinterpret_cast<T*>(dst+l.pv),reinterpret_cast<uint32_t*>(dst+l.qi),reinterpret_cast<uint32_t*>(dst+l.qv));
        launch_check();
    }
    trace.mark("side_compact");trace.finish();
    ck(hipStreamSynchronize(nullptr));return l.end;
}

template<class T> void decode(waltz_hip_session& w,const waltz_hip_config& c,const uint8_t* src,size_t size,T* dst,size_t dst_size){
    const Profile p=profile(c);
    w.select_tune(p);
    const unsigned chunks=blocks(p.n,8192),groups=blocks(chunks,16);
    require(size>=BlobSize&&dst_size>=size_t(p.n)*sizeof(T),"short archive or output");
    Trace trace("decode");
    Header h;ck(hipMemcpy(&h,src,sizeof(h),hipMemcpyDeviceToHost));
    trace.mark("header");
    require(h.flags==Header::pack_flags(p.dyadic,sizeof(T),WALTZ::PweEncoding::Native,WALTZ::Synthesis::HaloAllLevels)
        &&h.levels_xy==p.lx&&h.levels_z==p.lz&&h.reorder_bz==p.bz,"canonical Waltz header required");
    require(h.mag_comp_bytes>=size_t(groups)*4+size_t(chunks)*3&&h.mag_comp_bytes<=size-BlobSize&&h.n_pwe<=p.n&&h.n_qo<=p.n
        &&h.pwe_bytes==size_t(h.n_pwe)*sizeof(T),"invalid section lengths");
    const T q=T(h.q);require(q>0&&std::isfinite(q),"invalid quantum");
    const Layout l(h);require(l.end==size,"inconsistent archive size");
    auto* coeff=w.correction.get<T>(p.n);const auto* ranks=table(w,p);const auto d=p.dims;
    const unsigned bytes=p.n*2;
    const bool full=d.x%16==0&&d.y%16==0&&d.z%p.bz==0;
    if(full&&p.bz==2)
        wzp::decode_kernel<2,T><<<chunks,TPB>>>(src+BlobSize,chunks,groups,bytes,nullptr,nullptr,coeff,ranks,q,d.x,d.y,d.z,p.bz);
    else if(sizeof(T)==4&&full&&p.bz==16)
        wzp::decode_kernel<3,T><<<chunks,TPB>>>(src+BlobSize,chunks,groups,bytes,nullptr,nullptr,coeff,ranks,q,d.x,d.y,d.z,p.bz);
    else wzp::decode_kernel<1,T><<<chunks,TPB>>>(src+BlobSize,chunks,groups,bytes,nullptr,nullptr,coeff,ranks,q,d.x,d.y,d.z,p.bz);
    launch_check();
    unsigned* st=w.status.get<unsigned>(1);ck(hipMemsetAsync(st,0,4));
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
    unsigned status=0;ck(hipMemcpy(&status,st,4,hipMemcpyDeviceToHost));
    trace.mark("patch_and_status");trace.finish();
    require(!status,"invalid repair indices or value");
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
        return native::align8(native::BlobSize+size_t(chunks)*3+size_t(groups)*4+size_t(p.n)*2+(p.n+7)/8)
            +native::align8(size_t(p.n)*4)*3+native::align8(size_t(p.n)*c->width);
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
