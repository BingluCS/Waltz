// Shared CUDA/HIP DataLoader backend. Include after the respective native
// warp/lifting and WZP implementations. All scratch belongs to the caller.
#pragma once
#include "waltz_gpu_batch.h"
#include "utils/Blob.hpp"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <limits>
namespace WALTZ::loader {
#ifdef __HIPCC__
using Stream=hipStream_t;
inline void check(hipError_t e){if(e!=hipSuccess)throw std::runtime_error(hipGetErrorString(e));}
inline void launches(){check(hipGetLastError());}
inline void finish(Stream s){check(hipStreamSynchronize(s));}
#else
using Stream=cudaStream_t;
inline void check(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
inline void launches(){check(cudaGetLastError());}
inline void finish(Stream s){check(cudaStreamSynchronize(s));}
#endif
inline void need(bool yes,const char* msg){if(!yes)throw std::invalid_argument(msg);}
inline size_t mul(size_t a,size_t b) {
    need(!a || b<=std::numeric_limits<size_t>::max()/a,"batch size overflow");return a*b;
}
inline unsigned divup(unsigned n,unsigned k){return 1+(n-1)/k;}
template<class T> struct Intermediate {
    T* out;
    __device__ void set_job(unsigned j,size_t stride){out+=j*stride;}
    __device__ void operator()(T v,size_t at) const{out[at]=v;}
};
template<class T> struct Final {
    T* out; dim3 tile,frame,grid; unsigned samples,sides; unsigned ox=0,oy=0,oz=0;
    __device__ void set_job(unsigned j,size_t){
        ox=(j%grid.x)*tile.x;j/=grid.x;oy=(j%grid.y)*tile.y;j/=grid.y;
        oz=(j%grid.z)*tile.z;j/=grid.z;
        out+=((j%sides)*size_t(samples)+j/sides)*frame.x*frame.y*frame.z;
    }
    __device__ T* ptr(size_t at) const {
        if(tile.x==frame.x && tile.y==frame.y && tile.z==frame.z)return out+at;
        const unsigned x=ox+at%tile.x,y=oy+(at/tile.x)%tile.y,z=oz+at/(size_t(tile.x)*tile.y);
        return x<frame.x && y<frame.y && z<frame.z?out+(size_t(z)*frame.y+y)*frame.x+x:nullptr;
    }
    __device__ void operator()(T v,size_t at)const{if(T* p=ptr(at))*p=v;}
};
template<class T,bool Fill> __global__ void copy_tiles(const T* coeff,size_t stride,unsigned n,
    const uint64_t* spans,Final<T> out,T fill) {
    const unsigned j=blockIdx.y,i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=n || (Fill && spans[2*j+1]))return;
    out.set_job(j,stride);
    out(Fill?fill:coeff[j*stride+i],i);
}
template<class T,bool Quant> __global__ void patches(const unsigned char* src,const uint64_t* spans,
    T* coeff,size_t stride,unsigned n,Final<T> out,unsigned* status) {
    const unsigned j=blockIdx.y,i=blockIdx.x*blockDim.x+threadIdx.x;
    if(!spans[2*j+1])return;
    const unsigned char* a=src+spans[2*j];const auto& h=*reinterpret_cast<const WaltzBlobHeader*>(a);
    if(i>=(Quant?h.n_qo:h.n_pwe))return;
    const auto l=BlobLayout::from_sizes(h.mag_comp_bytes,h.n_pwe,h.pwe_bytes,h.n_qo);
    const auto* indices=reinterpret_cast<const unsigned*>(a+(Quant?l.qi:l.pi));
    const unsigned at=indices[i];
    if(at>=n || (i && indices[i-1]>=at)){atomicOr(status,1u);return;}
    if constexpr(Quant) {
        const int v=reinterpret_cast<const int*>(a+l.qv)[i];
        if(v<=65535){atomicOr(status,1u);return;}
        T& dest=coeff[j*stride+at];T mag=T(v)*T(h.q);dest=dest<T(0)?-mag:mag;
    } else {
        out.set_job(j,stride);
        if(T* dest=out.ptr(at))*dest-=reinterpret_cast<const T*>(a+l.pv)[i];
    }
}
template<class T> void run(const waltz_batch_shape& d,const unsigned char* host,const unsigned char* packed,
    size_t bytes,T* scratch,size_t cap,T* output,size_t outbytes,double fill,Stream stream) {
    need(d.x && d.y && d.z && d.fx && d.fy && d.fz && d.samples && (d.sides==1||d.sides==2),"invalid batch geometry");
    const dim3 tile(d.x,d.y,d.z),frame(d.fx,d.fy,d.fz),grid(divup(d.fx,d.x),divup(d.fy,d.y),divup(d.fz,d.z));
    const size_t n64=mul(mul(d.x,d.y),d.z),jobs64=mul(mul(mul(grid.x,grid.y),grid.z),mul(d.samples,d.sides));
    need(n64<=INT32_MAX/sizeof(T) && jobs64<=65535 && d.y<=65535 && d.z<=65535,"batch launch limit");
    const unsigned n=n64,jobs=jobs64;
    need(outbytes==mul(mul(mul(d.fx,d.fy),d.fz),mul(mul(d.samples,d.sides),sizeof(T))),"incorrect output size");
    const size_t stride=(n64+3)&~size_t(3),bank=stride*jobs;
    const unsigned lx=dwt_levels(std::min(d.x,d.y)),lz=dwt_levels(d.z),bz=lossless::select_block_zband(tile,lz,lx==lz);
    need(cap>=3*bank*sizeof(T)+8*256*bz*4+4 && bytes>=16*jobs64,"short batch buffers");
    const size_t table=bytes-16*jobs64;
    need(!(table%16),"unaligned batch spans");
    const auto* spans=reinterpret_cast<const uint64_t*>(packed+table);
    const unsigned chunks=divup(n,8192);
    unsigned maxq=0,maxp=0;bool holes=false;
    for(unsigned j=0;j<jobs;++j){
        uint64_t pair[2];std::memcpy(pair,host+table+16*j,16);
        const size_t at=pair[0],sz=pair[1];if(!sz){holes=true;continue;}
        need(!(at%16) && at<=table && sz>=BlobSize && sz<=table-at,"invalid batch span");
        WaltzBlobHeader h;std::memcpy(&h,host+at,BlobSize);
        need(h.flags==WaltzBlobHeader::pack_flags(lx==lz,sizeof(T),PweEncoding::Native,Synthesis::HaloAllLevels)
            && h.levels_xy==lx && h.levels_z==lz && h.reorder_bz==bz && T(h.q)>0 && std::isfinite(T(h.q))
            && h.mag_comp_bytes>=3*chunks+4*divup(chunks,16) && h.mag_comp_bytes<=sz-BlobSize
            && h.n_pwe<=n && h.n_qo<=n && h.pwe_bytes==h.n_pwe*sizeof(T)
            && sz==BlobLayout::checked(h).end,
            "invalid canonical batch archive");
        maxq=std::max(maxq,h.n_qo);maxp=std::max(maxp,h.n_pwe);
    }
    auto* ranks=reinterpret_cast<unsigned*>(scratch+3*bank);
    auto* status=ranks+8*256*bz;
    const bool patches_present=maxq||maxp;
#ifdef __HIPCC__
    if(patches_present)check(hipMemsetAsync(status,0,4,stream));
#else
    if(patches_present)check(cudaMemsetAsync(status,0,4,stream));
#endif
    lossless::d_build_block_ioffset<<<dim3(bz,8),256,0,stream>>>(ranks,16,16,bz,d.x,size_t(d.x)*d.y,tile);
#ifdef __HIPCC__
    lossless::decode_kernel<1,T,true><<<dim3(chunks,jobs),512,0,stream>>>(packed,chunks,divup(chunks,16),n*2,
        nullptr,nullptr,scratch,ranks,T(0),d.x,d.y,d.z,bz,spans,stride);
#else
    lossless::wzp_decode_tiles(packed,spans,jobs,n,scratch,stride,ranks,tile,bz,sizeof(T),stream);
#endif
    Final<T> final{output,tile,frame,grid,d.samples,d.sides};
    if(maxq)patches<T,true><<<dim3(divup(maxq,256),jobs),256,0,stream>>>(packed,spans,scratch,stride,n,final,status);
    using Read=allhalo::Read<T>;using Tmp=Intermediate<T>;
    T *a=scratch+bank,*b=a+bank;
    auto xy=[&](Read src,dim3 sz,auto emit){
        allhalo::xy<T,true,decltype(emit),true><<<dim3(divup(sz.x,56)*jobs,divup(sz.y,8),divup(sz.z,4)),128,0,stream>>>(
            src,sz,emit,stride);
    };
    auto zline=[&](Read src,dim3 sz,auto emit){
        constexpr unsigned Z=sizeof(T)==4?16:8;
        allhalo::zline<Z,T,true,decltype(emit),true><<<dim3(divup(sz.x*sz.y,128),divup(sz.z,Z),jobs),128,0,stream>>>(
            src,sz,emit,stride);
    };
    auto active=[&](unsigned level,bool xy_axis,bool z_axis){
        dim3 s=tile;while(level--){if(xy_axis){s.x=(s.x+1)/2;s.y=(s.y+1)/2;}if(z_axis)s.z=(s.z+1)/2;}return s;
    };
    if(lx && lx==lz){
        // For an XYZ inverse, Z feeds XY. Every level sees the previous
        // level's reconstructed low rectangle and the original high bands.
        for(int l=int(lx)-1;l>=0;--l){
            dim3 sz=active(l,true,true);
            zline(Read{scratch,l+1<int(lx)?a:nullptr,tile,active(l+1,true,true)},sz,Tmp{b});
            Read stage{b,nullptr,tile,{}};
            if(l)xy(stage,sz,Tmp{a});else xy(stage,sz,final);
        }
    }else if(lx||lz){
        const T* prior=nullptr;T* next=a;
        for(int l=int(lx)-1;l>=0;--l){
            dim3 sz=active(l,true,false);Read read{scratch,prior,tile,active(l+1,true,false)};
            if(!l && !lz)xy(read,sz,final);else xy(read,sz,Tmp{next});
            prior=next;next=(next==a)?b:a;
        }
        const T* zbase=lx?prior:scratch;prior=nullptr;
        T* alt=lx?scratch:b;
        for(int l=int(lz)-1;l>=0;--l){
            dim3 sz=active(l,false,true);Read read{zbase,prior,tile,active(l+1,false,true)};
            if(!l)zline(read,sz,final);else zline(read,sz,Tmp{next});
            prior=next;std::swap(next,alt);
        }
    }else copy_tiles<T,false><<<dim3(divup(n,256),jobs),256,0,stream>>>(scratch,stride,n,spans,final,T(0));
    if(maxp)patches<T,false><<<dim3(divup(maxp,256),jobs),256,0,stream>>>(packed,spans,scratch,stride,n,final,status);
    if(holes)copy_tiles<T,true><<<dim3(divup(n,256),jobs),256,0,stream>>>(scratch,stride,n,spans,final,T(fill));
    launches();
    unsigned invalid=0;
    if(patches_present){
#ifdef __HIPCC__
        check(hipMemcpyAsync(&invalid,status,4,hipMemcpyDeviceToHost,stream));
#else
        check(cudaMemcpyAsync(&invalid,status,4,cudaMemcpyDeviceToHost,stream));
#endif
    }
    finish(stream);need(!invalid,"invalid batch repair index");
}
} // namespace WALTZ::loader

extern "C" const char* waltz_gpu_decode_batch(const waltz_batch_shape* d,const void* host,const void* packed,
    size_t bytes,void* scratch,size_t cap,void* out,size_t outbytes,double fill,void* raw_stream) {
    using namespace WALTZ::loader;static thread_local char err[256];
    try {
        need(d && host && packed && scratch && out && (d->width==4||d->width==8),"invalid batch arguments");
        auto f=[&](auto* typed){
            using T=std::remove_pointer_t<decltype(typed)>;
            run(*d,static_cast<const unsigned char*>(host),static_cast<const unsigned char*>(packed),bytes,
                static_cast<T*>(scratch),cap,typed,outbytes,fill,reinterpret_cast<Stream>(raw_stream));
        };
        if(d->width==4)f(static_cast<float*>(out));else f(static_cast<double*>(out));
        return nullptr;
    }catch(const std::exception& e){std::snprintf(err,sizeof(err),"%s",e.what());return err;}
}
