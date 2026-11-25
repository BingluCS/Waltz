#include "waltz_cpu.h"
#include "transform.hpp"
#include "wzp_cpu.hpp"
#include "quantize_avx2.hpp"
#include "morton_decode_avx2.hpp"
#include "repair_avx2.hpp"
#include "../../include/utils/Blob.hpp"
#include <algorithm>
#include <cfenv>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <new>
#include <stdexcept>
#include <type_traits>
#if defined(__x86_64__) || defined(__i386__)
#include <xmmintrin.h>
#endif

// No containers or allocation inside line/chunk/sample loops. A caller-owned
// workspace caches one aligned allocation and the eight bounded Morton tables.
struct waltz_cpu_workspace {
    void* memory;
    size_t capacity;
    uint16_t rank[8 * 8192];
    uint32_t nx, ny, nz, bz;
    char error[256];
};
namespace {
using waltz_cpu::Shape;
using namespace waltz_cpu_detail;
using namespace WALTZ;
[[noreturn]] void fail(const char* message) { throw std::runtime_error(message); }
size_t add(size_t a,size_t b) { if (b>SIZE_MAX-a) fail("size overflow"); return a+b; }
size_t mul(size_t a,size_t b) { if (a && b>SIZE_MAX/a) fail("size overflow"); return a*b; }
size_t align(size_t a,size_t by=8) { return add(a,by-1)&~(by-1); }
uint32_t rd32(const uint8_t* p) { return uint32_t(p[0])|(uint32_t(p[1])<<8)|(uint32_t(p[2])<<16)|(uint32_t(p[3])<<24); }
uint64_t rd64(const uint8_t* p) { return rd32(p)|(uint64_t(rd32(p+4))<<32); }
void wr32(uint8_t* p,uint32_t v) { for(unsigned i=0;i<4;++i)p[i]=uint8_t(v>>(8*i)); }
void wr64(uint8_t* p,uint64_t v) { wr32(p,uint32_t(v));wr32(p+4,uint32_t(v>>32)); }
template<class T> T read_real(const uint8_t* p) {
    T v; if constexpr(sizeof(T)==4) { uint32_t u=rd32(p);std::memcpy(&v,&u,4); }
    else { uint64_t u=rd64(p);std::memcpy(&v,&u,8); } return v;
}
template<class T> void write_real(uint8_t* p,T v) {
    if constexpr(sizeof(T)==4) {uint32_t u;std::memcpy(&u,&v,4);wr32(p,u);}
    else {uint64_t u;std::memcpy(&u,&v,8);wr64(p,u);}
}
struct FPGuard {
    fenv_t old;
#if defined(__x86_64__) || defined(__i386__)
    unsigned csr;
#endif
    FPGuard() {
        if(std::fegetenv(&old))fail("cannot save floating point environment");
#if defined(__x86_64__) || defined(__i386__)
        csr=_mm_getcsr();
#endif
        if(std::fesetround(FE_TONEAREST))fail("cannot set nearest-even rounding");
#if defined(__x86_64__) || defined(__i386__)
        _mm_setcsr(_mm_getcsr()&~unsigned((1U<<15)|(1U<<6)));
#endif
    }
    ~FPGuard() {std::fesetenv(&old);
#if defined(__x86_64__) || defined(__i386__)
        _mm_setcsr(csr);
#endif
    }
};
uint32_t validate(const waltz_cpu_config* c,bool encoding) {
    if(!c||!c->nx||!c->ny||!c->nz)fail("nonzero dimensions required");
    if(c->type!=WALTZ_CPU_F32&&c->type!=WALTZ_CPU_F64)fail("type must be FP32 or FP64");
    if(c->backend<WALTZ_CPU_AUTO||c->backend>WALTZ_CPU_AVX2)fail("invalid backend");
    if(c->backend==WALTZ_CPU_AVX2&&!waltz_cpu::transform_has_avx2())fail("AVX2/FMA not available");
    if(encoding&&(!(c->absolute_error>0)||!std::isfinite(c->absolute_error)))fail("positive finite absolute error required");
    size_t n=mul(mul(c->nx,c->ny),c->nz);
    if(n>UINT32_MAX/unsigned(c->type))fail("volume exceeds WLT2 section limits");
    return uint32_t(n);
}
bool use_avx(const waltz_cpu_config& c) {return c.backend!=WALTZ_CPU_SCALAR&&waltz_cpu::transform_has_avx2();}
bool within_bound(double a,double b,double bound) {
    // Compare the exact native difference to the double bound, independently
    // of whether the host long double has 64, 80, or 128 bits. TwoDiff's tail
    // matters only when the rounded high part equals the bound.
    double high=a-b,magnitude=std::fabs(high);
    if(!std::isfinite(high)||magnitude>bound)return false;
    if(magnitude<bound)return true;
    double bv=a-high,av=high+bv,br=bv-b,ar=a-av,low=ar+br;
    return !((high>0&&low>0)||(high<0&&low<0));
}
void disjoint(const void* a,size_t as,const void* b,size_t bs) {
    uintptr_t x=reinterpret_cast<uintptr_t>(a),y=reinterpret_cast<uintptr_t>(b);
    if(as>UINTPTR_MAX-x||bs>UINTPTR_MAX-y)fail("pointer range overflow");
    if(x<y+bs&&y<x+as)fail("input and output must not overlap");
}
uint8_t levels(uint32_t n) {uint8_t l=0;while(n>8&&l<5){n=n/2+n%2;++l;}return l;}
uint32_t zband(Shape d,uint8_t lz,bool dyadic) {
    if(dyadic)return std::min(16U,(d.nz+((1U<<lz)-1U))>>lz);
    if(d.nz<4)return d.nz;
    uint64_t blocks=uint64_t((d.nx+15U)/16)*((d.ny+15U)/16);
    uint64_t full=uint64_t(d.nx/16)*(d.ny/16);
    return (blocks-full)*32>=blocks?4:2;
}
uint16_t morton(uint32_t x,uint32_t y,uint32_t z,uint32_t nx,uint32_t ny,uint32_t nz) {
    uint32_t rank=0;
    while(nx>1||ny>1||nz>1){
        uint32_t sx[2]={nx-nx/2,nx/2},sy[2]={ny-ny/2,ny/2},sz[2]={nz-nz/2,nz/2};
        unsigned cx=x>=sx[0],cy=y>=sy[0],cz=z>=sz[0];
        if(cz)rank+=sz[0]*nx*ny;
        if(cy)rank+=sz[cz]*sy[0]*nx;
        if(cx)rank+=sz[cz]*sy[cy]*sx[0];
        if(cx)x-=sx[0];if(cy)y-=sy[0];if(cz)z-=sz[0];
        nx=sx[cx];ny=sy[cy];nz=sz[cz];
    }return uint16_t(rank);
}
void ranks(waltz_cpu_workspace& w,Shape d,uint32_t bz) {
    if(w.nx==d.nx&&w.ny==d.ny&&w.nz==d.nz&&w.bz==bz)return;
    for(unsigned s=0;s<8;++s){
        uint32_t nx=s&1?d.nx%16:16,ny=s&2?d.ny%16:16,nz=s&4?d.nz%bz:bz;
        for(uint32_t z=0;z<nz;++z)for(uint32_t y=0;y<ny;++y)for(uint32_t x=0;x<nx;++x)
            w.rank[s*8192+(z*ny+y)*nx+x]=morton(x,y,z,nx,ny,nz);
    }
    w.nx=d.nx;w.ny=d.ny;w.nz=d.nz;w.bz=bz;
}
template<class F> void traverse(waltz_cpu_workspace& w,Shape d,uint32_t bz,F fn) {
    ranks(w,d,bz);size_t base=0,plane=size_t(d.nx)*d.ny;
    for(uint32_t z0=0;z0<d.nz;z0+=bz)for(uint32_t y0=0;y0<d.ny;y0+=16)for(uint32_t x0=0;x0<d.nx;x0+=16){
        uint32_t nx=std::min(16U,d.nx-x0),ny=std::min(16U,d.ny-y0),nz=std::min(bz,d.nz-z0);
        unsigned s=unsigned(nx!=16)|(unsigned(ny!=16)<<1)|(unsigned(nz!=bz)<<2);
        const uint16_t* r=w.rank+s*8192;
        for(uint32_t z=0;z<nz;++z)for(uint32_t y=0;y<ny;++y)for(uint32_t x=0;x<nx;++x)
            fn((z0+z)*plane+size_t(y0+y)*d.nx+x0+x,base+r[(z*ny+y)*nx+x]);
        base+=size_t(nx)*ny*nz;
    }
}
template<class T> struct Buffers {
    T *data,*scratch,*errors;
    uint16_t* mag;
    uint32_t *sign,*qidx,*qval,*pidx;
    uint8_t* seen;
};
template<class T> void inverse_reorder(waltz_cpu_workspace& w,Shape d,uint32_t bz,
                                      const uint16_t* mag,const uint32_t* sign,T* output,T q,bool avx) {
    ranks(w,d,bz);size_t base=0,plane=size_t(d.nx)*d.ny;
    for(uint32_t z0=0;z0<d.nz;z0+=bz)for(uint32_t y0=0;y0<d.ny;y0+=16)for(uint32_t x0=0;x0<d.nx;x0+=16){
        const uint32_t nx=std::min(16U,d.nx-x0),ny=std::min(16U,d.ny-y0),nz=std::min(bz,d.nz-z0);
        const unsigned s=unsigned(nx!=16)|(unsigned(ny!=16)<<1)|(unsigned(nz!=bz)<<2);
        const uint16_t* r=w.rank+s*8192;T* tile=output+size_t(z0)*plane+size_t(y0)*d.nx+x0;
        bool done=false;
        if(avx){
            if constexpr(sizeof(T)==4)done=morton_decode_f32(r,mag+base,sign,base,tile,d.nx,plane,nx,ny,nz,q);
            else done=morton_decode_f64(r,mag+base,sign,base,tile,d.nx,plane,nx,ny,nz,q);
        }
        if(!done)for(uint32_t z=0;z<nz;++z)for(uint32_t y=0;y<ny;++y)for(uint32_t x=0;x<nx;++x){
            const size_t packed=base+r[(z*ny+y)*nx+x];T v=T(mag[packed])*q;
            tile[size_t(z)*plane+size_t(y)*d.nx+x]=(sign[packed/32]>>(packed%32)&1U)?-v:v;
        }
        base+=size_t(nx)*ny*nz;
    }
}
template<class T> Buffers<T> buffers(waltz_cpu_workspace& w,uint32_t n,Shape d) {
    size_t sizes[]={mul(n,sizeof(T)),mul(waltz_cpu::workspace_elements(d),sizeof(T)),mul(n,sizeof(T)),
        mul(n,2),mul((size_t(n)+31)/32,4),mul(n,4),mul(n,4),mul(n,4),n};
    size_t total=0;for(size_t s:sizes)total=add(total,align(s,64));
    if(total>w.capacity){void* p=std::aligned_alloc(64,align(total,64));if(!p)throw std::bad_alloc();std::free(w.memory);w.memory=p;w.capacity=total;}
    uint8_t* p=static_cast<uint8_t*>(w.memory);void* starts[9];
    for(unsigned i=0;i<9;++i){starts[i]=p;p+=align(sizes[i],64);}
    return {static_cast<T*>(starts[0]),static_cast<T*>(starts[1]),static_cast<T*>(starts[2]),
        static_cast<uint16_t*>(starts[3]),static_cast<uint32_t*>(starts[4]),static_cast<uint32_t*>(starts[5]),
        static_cast<uint32_t*>(starts[6]),static_cast<uint32_t*>(starts[7]),static_cast<uint8_t*>(starts[8])};
}
template<class T> void transform(T* data,T* tmp,Shape s,uint8_t lx,uint8_t lz,bool dyadic,uint8_t syn,bool avx,bool inverse) {
    if constexpr(sizeof(T)==4){if(inverse)waltz_cpu::inverse_f32(data,tmp,s,lx,lz,dyadic,syn,avx);else waltz_cpu::forward_f32(data,tmp,s,lx,lz,dyadic,syn,avx);}
    else {if(inverse)waltz_cpu::inverse_f64(data,tmp,s,lx,lz,dyadic,syn,avx);else waltz_cpu::forward_f64(data,tmp,s,lx,lz,dyadic,syn,avx);}
}
void sort_pairs(uint32_t* index,uint32_t* value,uint32_t count) {
    auto sift=[&](uint32_t root,uint32_t end){
        while(root<end/2){uint32_t child=root*2+1;
            if(child+1<end&&index[child]<index[child+1])++child;
            if(index[root]>=index[child])break;
            std::swap(index[root],index[child]);std::swap(value[root],value[child]);root=child;}
    };
    for(uint32_t i=count/2;i>0;--i)sift(i-1,count);
    for(uint32_t end=count;end>1;--end){std::swap(index[0],index[end-1]);std::swap(value[0],value[end-1]);sift(0,end-1);}
}
template<class T> size_t encode(waltz_cpu_workspace& w,const waltz_cpu_config& c,const T* input,uint8_t* out,size_t cap,uint32_t n) {
    Shape s{c.nx,c.ny,c.nz};auto b=buffers<T>(w,n,s);bool avx=use_avx(c);
    uint8_t lx=levels(std::min(c.nx,c.ny)),lz=levels(c.nz);bool dyadic=lx==lz;
    uint32_t bz=zband(s,lz,dyadic);double quantum=1.5*c.absolute_error;
    T q=T(quantum),inv=T(1.0/quantum);
    if(!(q>T(0))||!std::isfinite(q)||!std::isfinite(inv))fail("quantum not representable in native type");
    if(avx){
        bool finite;if constexpr(sizeof(T)==4)finite=finite_f32(input,n);else finite=finite_f64(input,n);
        if(!finite)fail("nonfinite input is unsupported");
    } else for(uint32_t i=0;i<n;++i)if(!std::isfinite(input[i]))fail("nonfinite input is unsupported");
    std::memcpy(b.data,input,size_t(n)*sizeof(T));
    transform(b.data,b.scratch,s,lx,lz,dyadic,2,avx,false);
    std::memset(b.sign,0,((size_t(n)+31)/32)*4);
    if(avx){
        auto* temporary=reinterpret_cast<int32_t*>(b.scratch);bool valid;
        if constexpr(sizeof(T)==4)valid=quantize_f32(b.data,n,q,inv,temporary);
        else valid=quantize_f64(b.data,n,q,inv,temporary);
        if(!valid)fail("coefficient exceeds portable integer conversion domain");
    }
    uint32_t nq=0,np=0;
    traverse(w,s,bz,[&](size_t at,size_t packed){
        bool negative;int32_t mag;
        if(avx){
            int32_t v;std::memcpy(&v,reinterpret_cast<const uint8_t*>(b.scratch)+at*4,4);
            negative=v<0;mag=negative?-v:v;
        }else{
            T scaled=b.data[at]*inv;
            if(!std::isfinite(scaled)||std::fabs(static_cast<long double>(scaled))>=9223372036854775808.0L)
                fail("coefficient exceeds portable integer conversion domain");
            long long v=std::llrint(scaled);negative=v<0;
            uint64_t mag64=uint64_t(negative?-v:v);mag=int32_t(std::min<uint64_t>(mag64,INT32_MAX));
            b.data[at]=mag?T(negative?-mag:mag)*q:T(0);
        }
        b.mag[packed]=uint16_t(std::min(mag,65535));
        if(negative&&mag)b.sign[packed/32]|=1U<<(packed%32);
        if(mag>65535){b.qidx[nq]=uint32_t(at);b.qval[nq++]=uint32_t(mag);}
    });
    // Morton traversal is not native index order. Portable side channels are
    // built in row-major order below, with no comparison-sort allocation.
    if(nq<=n/64) {
        // Sparse or absent side channels should not require a full-volume scan.
        sort_pairs(b.qidx,b.qval,nq);
    } else {
    std::memset(b.seen,0,n);for(uint32_t i=0;i<nq;++i){b.seen[b.qidx[i]]=1;}
    // Preserve magnitudes by row-major slot in scratch, whose transform use has
    // ended for now. uint32 accesses use memcpy to avoid strict-aliasing UB.
    for(uint32_t i=0;i<nq;++i)std::memcpy(reinterpret_cast<uint8_t*>(b.scratch)+size_t(b.qidx[i])*4,b.qval+i,4);
    uint32_t qi=0;for(uint32_t i=0;i<n;++i)if(b.seen[i]){b.qidx[qi]=i;std::memcpy(b.qval+qi,reinterpret_cast<uint8_t*>(b.scratch)+size_t(i)*4,4);++qi;}
    }
    transform(b.data,b.scratch,s,lx,lz,dyadic,2,avx,true);
    if(avx){
        bool valid;
        if constexpr(sizeof(T)==4)valid=repairs_f32(input,b.data,n,c.absolute_error,b.pidx,b.errors,&np);
        else valid=repairs_f64(input,b.data,n,c.absolute_error,b.pidx,b.errors,&np);
        if(!valid)fail("native repair cannot meet strict bound or nonfinite reconstruction");
    } else for(uint32_t i=0;i<n;++i){
        T value=b.data[i],e=value-input[i];
        if(!std::isfinite(value)||!std::isfinite(e))fail("nonfinite synthesis/correction unsupported in portable profile");
        // Precisely match Halo3DSynthEmit, including its native FP32 bound.
        bool patch=sizeof(T)==4?std::fabs(e)>T(c.absolute_error):std::fabs(e)>c.absolute_error;
        if(patch){b.pidx[np]=i;b.errors[np++]=e;value=value-e;}
        if(!std::isfinite(value)||!within_bound(double(value),double(input[i]),c.absolute_error))
            fail("native repair cannot meet strict bound for this input; no invalid archive emitted");
    }
    // Match the existing GPU's finite auxiliary-buffer domain explicitly.
    if((n>(1U<<20)&&nq>=(1U<<20))||(n>(1U<<22)&&np>=(1U<<22)))fail("portable GPU side-channel capacity reached");
    if(cap<BlobSize)fail("archive capacity too small");
    size_t wb=wzp_encode(b.mag,b.sign,n,out+BlobSize,cap-BlobSize,avx);
    size_t pi=align(add(BlobSize,wb)),pe=align(add(pi,mul(np,4))),qo=align(add(pe,mul(np,sizeof(T)))),qv=align(add(qo,mul(nq,4))),end=align(add(qv,mul(nq,4)));
    if(end>cap)fail("archive capacity too small");
    out[0]=WaltzBlobHeader::pack_flags(dyadic,sizeof(T),PweEncoding::Native,Synthesis::HaloAllLevels);
    out[1]=lx;out[2]=lz;out[3]=uint8_t(bz);wr32(out+BlobPweBytes,uint32_t(np*sizeof(T)));
    write_real(out+BlobQuantum,quantum);wr64(out+BlobWzpBytes,wb);wr32(out+BlobPweCount,np);wr32(out+BlobQoutCount,nq);
    std::memset(out+BlobSize+wb,0,pi-BlobSize-wb);
    std::memset(out+pi,0,end-pi);
    for(uint32_t i=0;i<np;++i){wr32(out+pi+size_t(i)*4,b.pidx[i]);write_real(out+pe+size_t(i)*sizeof(T),b.errors[i]);}
    for(uint32_t i=0;i<nq;++i){wr32(out+qo+size_t(i)*4,b.qidx[i]);wr32(out+qv+size_t(i)*4,b.qval[i]);}
    return end;
}
unsigned pop(uint32_t x) {unsigned n=0;while(x){x&=x-1;++n;}return n;}
void validate_indices(const uint8_t* indices,uint32_t count,uint32_t n,uint8_t* seen) {
    // Canonical archives are strictly increasing. Validate this common case
    // without clearing/touching a volume-sized duplicate-detection bitmap.
    bool ordered=true;uint32_t previous=0;
    for(uint32_t i=0;i<count;++i){
        const uint32_t index=rd32(indices+size_t(i)*4);
        if(index>=n)fail("outlier index exceeds volume");
        if(i&&index<=previous)ordered=false;
        previous=index;
    }
    if(!ordered){
        std::memset(seen,0,n);
        for(uint32_t i=0;i<count;++i){const uint32_t index=rd32(indices+size_t(i)*4);
            if(seen[index])fail("duplicate outlier index");seen[index]=1;}
    }
}
template<class T> void decode(waltz_cpu_workspace& w,const waltz_cpu_config& c,const uint8_t* in,size_t size,T* out,uint32_t n) {
    if(size<BlobSize||!WaltzBlobHeader::valid_flags(in[0])||WaltzBlobHeader::width(in[0])!=sizeof(T))
        fail("invalid Waltz archive header");
    const auto synthesis=WaltzBlobHeader::synthesis_of(in[0]);
    const bool compact=WaltzBlobHeader::repair_of(in[0])==PweEncoding::Small;
    uint8_t lx=in[1],lz=in[2],bz=in[3];bool dyadic=in[0]&1;
    if(synthesis!=Synthesis::HaloAllLevels)fail("unsupported Waltz synthesis mode");
    if(lx>5||lz>5||bz<1||bz>32||(dyadic&&lx!=lz))fail("invalid levels or z band");
    if(lx!=levels(std::min(c.nx,c.ny))||lz!=levels(c.nz))fail("dimensions/levels mismatch");
    double quantum=read_real<double>(in+BlobQuantum);T q=T(quantum);
    if(!(quantum>0)||!std::isfinite(quantum)||!(q>T(0))||!std::isfinite(q))fail("invalid quantum");
    uint64_t w64=rd64(in+BlobWzpBytes);uint32_t np=rd32(in+BlobPweCount),nq=rd32(in+BlobQoutCount),pb=rd32(in+BlobPweBytes);
    if(w64>SIZE_MAX||np>n||nq>n)fail("invalid section counts");
    size_t wb=size_t(w64),pi=align(add(BlobSize,wb)),pe=align(add(pi,mul(np,4))),qo=align(add(pe,pb)),qv=align(add(qo,mul(nq,4))),end=align(add(qv,mul(nq,4)));
    if(end!=size)fail("truncated archive or trailing data");
    if(!compact&&pb!=size_t(np)*sizeof(T))fail("invalid native correction section");
    if(compact&&(!np||pb<24+size_t((np+31)/32)*8||pb>=size_t(np)*sizeof(T)))fail("invalid compact correction section");
    Shape s{c.nx,c.ny,c.nz};auto b=buffers<T>(w,n,s);const bool avx=use_avx(c);
    wzp_decode(in+BlobSize,wb,n,b.mag,b.sign,avx);
    inverse_reorder(w,s,bz,b.mag,b.sign,b.data,q,avx);
    validate_indices(in+qo,nq,n,b.seen);
    for(uint32_t i=0;i<nq;++i){uint32_t idx=rd32(in+qo+size_t(i)*4),v=rd32(in+qv+size_t(i)*4);
        if(v<=65535||v>INT32_MAX)fail("invalid quantization outlier");
        T x=T(v)*q;b.data[idx]=b.data[idx]<T(0)?-x:x;
    }
    // Validate repairs before synthesis or exposing caller output.
    const uint8_t* repairs=in+pe;size_t prefix=0,payload=0;double step=0;
    if(compact){
        uint32_t groups=(np+31)/32;prefix=24+size_t(groups)*8;
        if(rd32(repairs)!=0x31575057U||rd32(repairs+4)!=groups||rd64(repairs+8)!=pb-prefix)fail("invalid WPW1 directory");
        payload=pb-prefix;step=read_real<double>(repairs+16);
        if(!(step>0)||!std::isfinite(step))fail("invalid compact repair step");
        for(uint32_t g=0;g<groups;++g){uint32_t off=rd32(repairs+24+size_t(g)*8),mask=rd32(repairs+28+size_t(g)*8),count=std::min(32U,np-g*32U);
            if(count<32&&(mask>>count))fail("invalid compact repair mask");
            size_t bytes=(off>>31)?size_t(count)*sizeof(T):align(count,sizeof(T))+pop(mask)*sizeof(T);
            size_t start=off&0x7fffffffU;
            if(start%8||start>payload||bytes>payload-start)fail("invalid compact repair group span");
        }
    }
    validate_indices(in+pi,np,n,b.seen);
    for(uint32_t i=0;i<np;++i){uint32_t idx=rd32(in+pi+size_t(i)*4);
        T correction;
        if(!compact)correction=read_real<T>(repairs+size_t(i)*sizeof(T));
        else {uint32_t g=i/32,lane=i%32,count=std::min(32U,np-g*32U),off=rd32(repairs+24+size_t(g)*8),mask=rd32(repairs+28+size_t(g)*8);
            const uint8_t* p=repairs+prefix+(off&0x7fffffffU);
            if(off>>31)correction=read_real<T>(p+size_t(lane)*sizeof(T));
            else if((mask>>lane)&1U)correction=read_real<T>(p+align(count,sizeof(T))+pop(mask&((1U<<lane)-1U))*sizeof(T));
            else correction=T(double(int8_t(p[lane]))*step);
        }
        if(!std::isfinite(correction))fail("nonfinite correction");b.pidx[i]=idx;b.errors[i]=correction;
    }
    transform(b.data,b.scratch,s,lx,lz,dyadic,uint8_t(synthesis),avx,true);
    for(uint32_t i=0;i<np;++i)b.data[b.pidx[i]]=b.data[b.pidx[i]]-b.errors[i];
    if(avx){
        bool finite;if constexpr(sizeof(T)==4)finite=finite_f32(b.data,n);else finite=finite_f64(b.data,n);
        if(!finite)fail("nonfinite reconstruction");
    } else for(uint32_t i=0;i<n;++i)if(!std::isfinite(b.data[i]))fail("nonfinite reconstruction");
    std::memcpy(out,b.data,size_t(n)*sizeof(T));
}
void error(waltz_cpu_workspace* w,const char* s) {if(w)std::snprintf(w->error,sizeof(w->error),"%s",s);}
}
extern "C" {
waltz_cpu_workspace* waltz_cpu_create(void) {return static_cast<waltz_cpu_workspace*>(std::calloc(1,sizeof(waltz_cpu_workspace)));}
void waltz_cpu_destroy(waltz_cpu_workspace* w) {if(w){std::free(w->memory);std::free(w);}}
const char* waltz_cpu_last_error(const waltz_cpu_workspace* w) {return w?w->error:"null workspace";}
int waltz_cpu_has_avx2(void) {return waltz_cpu::transform_has_avx2()?1:0;}
size_t waltz_cpu_max_compressed_bytes(const waltz_cpu_config* c) {try{uint32_t n=validate(c,true);size_t z=BlobSize;z=add(z,align(wzp_max_bytes(n)));z=add(z,align(mul(n,unsigned(c->type))));for(int i=0;i<3;++i)z=add(z,align(mul(n,4)));return z;}catch(...){return 0;}}
int waltz_cpu_compress(waltz_cpu_workspace* w,const waltz_cpu_config* c,const void* input,void* archive,size_t cap,size_t* written) {
    if(written)*written=0;
    try{if(!w||!input||!archive||!written)fail("null argument");w->error[0]=0;uint32_t n=validate(c,true);if(reinterpret_cast<uintptr_t>(input)%unsigned(c->type))fail("input must have native type alignment");disjoint(input,size_t(n)*unsigned(c->type),archive,cap);FPGuard guard;
        *written=c->type==WALTZ_CPU_F32?encode(*w,*c,static_cast<const float*>(input),static_cast<uint8_t*>(archive),cap,n):encode(*w,*c,static_cast<const double*>(input),static_cast<uint8_t*>(archive),cap,n);return 0;
    }catch(const std::exception& e){error(w,e.what());return -1;}catch(...){error(w,"unknown compression error");return -1;}
}
int waltz_cpu_decompress(waltz_cpu_workspace* w,const waltz_cpu_config* c,const void* archive,size_t bytes,void* output,size_t output_bytes) {
    try{if(!w||!archive||!output)fail("null argument");w->error[0]=0;uint32_t n=validate(c,false);size_t needed=size_t(n)*unsigned(c->type);if(output_bytes<needed)fail("output capacity too small");disjoint(archive,bytes,output,needed);FPGuard guard;
        if(c->type==WALTZ_CPU_F32)decode(*w,*c,static_cast<const uint8_t*>(archive),bytes,static_cast<float*>(output),n);else decode(*w,*c,static_cast<const uint8_t*>(archive),bytes,static_cast<double*>(output),n);return 0;
    }catch(const std::exception& e){error(w,e.what());return -1;}catch(...){error(w,"unknown decompression error");return -1;}
}
}
