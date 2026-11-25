#pragma once
#include <stdexcept>
#include <type_traits>
#include "ICDF97.hpp"
#include "halo3d.cuh"

// All-level native halo/lifting implementation. The dyadic schedule transforms
// XYZ at each level. Plane keeps all Z levels followed by all XY levels (and
// the exact reverse schedule for synthesis).
namespace WALTZ::allhalo {
using namespace warp97;

template<class T> struct Read {
    const T* base;
    const T* low;
    dim3 full, low_dims;
    __device__ __forceinline__ size_t index(int x,int y,int z) const {
        return (size_t(z)*full.y+y)*full.x+x;
    }
    __device__ __forceinline__ T get(int x,int y,int z) const {
        const size_t at=index(x,y,z);
        // The previous inverse level owns only this low-frequency rectangle.
        // Untouched high bands remain in the immutable coefficient array.
        if(low && x<int(low_dims.x) && y<int(low_dims.y) && z<int(low_dims.z)) return low[at];
        return base[at];
    }
};
template<class T> struct Store {
    T* output;
    __device__ __forceinline__ void operator()(T v,size_t i) const {output[i]=v;}
    __device__ __forceinline__ void operator()(T v,size_t i,int,int,int) const {output[i]=v;}
};
template<class T,bool Regular> struct Quant {
    T* low;
    dim3 full, active;
    bool include_z;
    bool last;
    __device__ __forceinline__ void operator()(T v,size_t at,int x,int y,int z) const {
        if(last || x>=int((active.x+1)/2) || y>=int((active.y+1)/2) ||
           (include_z && z>=int((active.z+1)/2)))
            CDF97::quantization<Regular>(v,at,x,y,z,full);
        else low[at]=v;
    }
};
template<class T> struct ZSplit {
    T* bands;
    T* low;
    int low_z;
    bool last;
    __device__ __forceinline__ void operator()(T v,size_t at,int,int,int z) const {
        if(last || z>=low_z) bands[at]=v; else low[at]=v;
    }
};
template<int Z,class T,class Emit> __global__ void forward_xyz(Read<T> src,dim3 n,Emit emit) {
    extern __shared__ __align__(16) unsigned char memory[];
    T* sm=reinterpret_cast<T*>(memory);
    constexpr int P=Z/2+4,XY=56*8;
    const int lane=threadIdx.x%32,warp=threadIdx.x/32;
    const int ix=int(blockIdx.x)*56+2*lane-4,iy=int(blockIdx.y)*8-4,z0=int(blockIdx.z)*Z-4;
    const int x0=halo3d::mirror(ix,n.x),x1=halo3d::mirror(ix+1,n.x);
    for(int k=warp;k<Z+8;k+=blockDim.x/32) {
        const int z=halo3d::mirror(z0+k,n.z);
        T e0[8],o0[8],e1[8],o1[8];
        #pragma unroll
        for(int j=0;j<8;++j) {
            const int y0=halo3d::mirror(iy+2*j,n.y),y1=halo3d::mirror(iy+2*j+1,n.y);
            e0[j]=src.get(x0,y0,z);e1[j]=src.get(x1,y0,z);
            o0[j]=src.get(x0,y1,z);o1[j]=src.get(x1,y1,z);
        }
        analysis_y(e0,o0);analysis_y(e1,o1);
        #pragma unroll
        for(int j=2;j<6;++j) {
            T ll=e0[j]*Constants<T>::K,hl=e1[j]*Constants<T>::K;
            T lh=o0[j]*(-Constants<T>::IK),hh=o1[j]*(-Constants<T>::IK);
            analysis_x(ll,hl);analysis_x(lh,hh);
            if(lane>=2 && lane<30) {
                const int at=k*XY+(j-2)*56+lane-2;
                sm[at]=ll;sm[at+28]=hl;sm[at+4*56]=lh;sm[at+4*56+28]=hh;
            }
        }
    }
    __syncthreads();
    for(int c=threadIdx.x;c<XY;c+=blockDim.x) {
        const int ry=c/56,rx=c%56;
        const int px=int(blockIdx.x)*56+2*(rx%28)+rx/28,py=int(blockIdx.y)*8+2*(ry%4)+ry/4;
        if(px>=n.x || py>=n.y)continue;
        const int x=px/2+(px%2?(n.x+1)/2:0),y=py/2+(py%2?(n.y+1)/2:0);
        T e[P],o[P];
        #pragma unroll
        for(int j=0;j<P;++j){e[j]=sm[2*j*XY+c];o[j]=sm[(2*j+1)*XY+c];}
        halo3d::az(e,o);
        #pragma unroll
        for(int j=2;j<P-2;++j) {
            const int p=z0+2*j,z=p/2,zh=(n.z+1)/2+z;
            if(p<n.z)emit(e[j],src.index(x,y,z),x,y,z);
            if(p+1<n.z)emit(o[j],src.index(x,y,zh),x,y,zh);
        }
    }
}

template<int Z,class T,class Emit,bool SingleFold=false> __global__ void inverse_xyz(Read<T> src,dim3 n,Emit emit) {
    extern __shared__ __align__(16) unsigned char memory[];
    T* sm=reinterpret_cast<T*>(memory);
    constexpr int P=Z/2+4,XY=64*16;
    const int xb=int(blockIdx.x)*56-4,yb=int(blockIdx.y)*8-4,zb=int(blockIdx.z)*Z-4;
    for(int c=threadIdx.x;c<XY;c+=blockDim.x) {
        const int x=halo3d::inverse_coeff<SingleFold>(xb+c%64,n.x);
        const int y=halo3d::inverse_coeff<SingleFold>(yb+c/64,n.y);
        T e[P],o[P];
        #pragma unroll
        for(int j=0;j<P;++j) {
            e[j]=src.get(x,y,halo3d::inverse_coeff<SingleFold>(zb+2*j,n.z));
            o[j]=src.get(x,y,halo3d::inverse_coeff<SingleFold>(zb+2*j+1,n.z));
        }
        halo3d::sz(e,o);
        #pragma unroll
        for(int j=2;j<P-2;++j){sm[(2*j-4)*XY+c]=e[j];sm[(2*j-3)*XY+c]=o[j];}
    }
    __syncthreads();
    const int lane=threadIdx.x%32,warp=threadIdx.x/32,x=xb+2*lane;
    for(int k=warp;k<Z;k+=blockDim.x/32) {
        const int z=zb+4+k;if(z>=n.z)continue;
        T e0[8],o0[8],e1[8],o1[8];
        #pragma unroll
        for(int j=0;j<8;++j) {
            const int at=k*XY+2*j*64+2*lane;
            T el=sm[at],ol=sm[at+1],eh=sm[at+64],oh=sm[at+65];
            synthesis_x(el,ol);synthesis_x(eh,oh);
            e0[j]=el;e1[j]=ol;o0[j]=eh;o1[j]=oh;
        }
        synthesis_y(e0,o0);synthesis_y(e1,o1);
        #pragma unroll
        for(int j=2;j<6;++j) {
            const int y=yb+2*j;
            if(lane>=2 && lane<30 && x<n.x) {
                if(y<n.y){emit(e0[j],src.index(x,y,z));if(x+1<n.x)emit(e1[j],src.index(x+1,y,z));}
                if(y+1<n.y){emit(o0[j],src.index(x,y+1,z));if(x+1<n.x)emit(o1[j],src.index(x+1,y+1,z));}
            }
        }
    }
}

template<class T,bool Inverse,class Emit,bool Batch=false> __global__ void xy(Read<T> src,dim3 n,Emit emit,size_t stride=0) {
    unsigned bx=blockIdx.x;
    if constexpr(Batch) {
        const unsigned cols=(n.x+55)/56,job=bx/cols;
        bx%=cols;src.base+=job*stride;if(src.low)src.low+=job*stride;
        emit.set_job(job,stride);
    }
    const int lane=threadIdx.x%32,warp=threadIdx.x/32,z=int(blockIdx.z)*(blockDim.x/32)+warp;
    if(z>=n.z)return; // Warp-uniform: all shuffles use a complete warp.
    const int xb=int(bx)*56-4,yb=int(blockIdx.y)*8-4,x=xb+2*lane;
    const int x0=Inverse?halo3d::coeff(x,n.x):halo3d::mirror(x,n.x);
    const int x1=Inverse?halo3d::coeff(x+1,n.x):halo3d::mirror(x+1,n.x);
    T e0[8],o0[8],e1[8],o1[8];
    #pragma unroll
    for(int j=0;j<8;++j) {
        const int y0=Inverse?halo3d::coeff(yb+2*j,n.y):halo3d::mirror(yb+2*j,n.y);
        const int y1=Inverse?halo3d::coeff(yb+2*j+1,n.y):halo3d::mirror(yb+2*j+1,n.y);
        e0[j]=src.get(x0,y0,z);e1[j]=src.get(x1,y0,z);
        o0[j]=src.get(x0,y1,z);o1[j]=src.get(x1,y1,z);
        if constexpr(Inverse){synthesis_x(e0[j],e1[j]);synthesis_x(o0[j],o1[j]);}
    }
    if constexpr(Inverse){synthesis_y(e0,o0);synthesis_y(e1,o1);}
    else {analysis_y(e0,o0);analysis_y(e1,o1);}
    #pragma unroll
    for(int j=2;j<6;++j) {
        const int y=yb+2*j;
        if constexpr(Inverse) {
            if(lane>=2 && lane<30 && x<n.x) {
                if(y<n.y){emit(e0[j],src.index(x,y,z));if(x+1<n.x)emit(e1[j],src.index(x+1,y,z));}
                if(y+1<n.y){emit(o0[j],src.index(x,y+1,z));if(x+1<n.x)emit(o1[j],src.index(x+1,y+1,z));}
            }
        } else {
            T ll=e0[j]*Constants<T>::K,hl=e1[j]*Constants<T>::K;
            T lh=o0[j]*(-Constants<T>::IK),hh=o1[j]*(-Constants<T>::IK);
            analysis_x(ll,hl);analysis_x(lh,hh);
            if(lane>=2 && lane<30 && x<n.x) {
                const int ox=x/2,hx=(n.x+1)/2+ox,oy=y/2,hy=(n.y+1)/2+oy;
                if(y<n.y){emit(ll,src.index(ox,oy,z),ox,oy,z);if(x+1<n.x)emit(hl,src.index(hx,oy,z),hx,oy,z);}
                if(y+1<n.y){emit(lh,src.index(ox,hy,z),ox,hy,z);if(x+1<n.x)emit(hh,src.index(hx,hy,z),hx,hy,z);}
            }
        }
    }
}

template<int Z,class T,bool Inverse,class Emit,bool Batch=false> __global__ void zline(Read<T> src,dim3 n,Emit emit,size_t stride=0) {
    if constexpr(Batch) {
        const unsigned job=blockIdx.z;
        src.base+=job*stride;if(src.low)src.low+=job*stride;
        emit.set_job(job,stride);
    }
    const unsigned col=blockIdx.x*blockDim.x+threadIdx.x;
    if(col>=n.x*n.y)return;
    const int x=col%n.x,y=col/n.x,zb=int(blockIdx.y)*Z-4;
    constexpr int P=Z/2+4;
    T e[P],o[P];
    #pragma unroll
    for(int j=0;j<P;++j) {
        const int a=Inverse?halo3d::coeff(zb+2*j,n.z):halo3d::mirror(zb+2*j,n.z);
        const int b=Inverse?halo3d::coeff(zb+2*j+1,n.z):halo3d::mirror(zb+2*j+1,n.z);
        e[j]=src.get(x,y,a);o[j]=src.get(x,y,b);
    }
    if constexpr(Inverse)halo3d::sz(e,o);else halo3d::az(e,o);
    #pragma unroll
    for(int j=2;j<P-2;++j) {
        const int z=zb+2*j;
        if constexpr(Inverse) {
            if(z<n.z)emit(e[j],src.index(x,y,z));
            if(z+1<n.z)emit(o[j],src.index(x,y,z+1));
        } else {
            const int l=z/2,h=(n.z+1)/2+l;
            if(z<n.z)emit(e[j],src.index(x,y,l),x,y,l);
            if(z+1<n.z)emit(o[j],src.index(x,y,h),x,y,h);
        }
    }
}
template<class T,bool Forward,class Emit> __global__ void identity(Read<T> src,dim3 n,Emit emit) {
    const size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(i>=size_t(n.x)*n.y*n.z)return;
    const int x=i%n.x,y=(i/n.x)%n.y,z=i/(size_t(n.x)*n.y);
    if constexpr(Forward)emit(src.get(x,y,z),src.index(x,y,z),x,y,z);
    else emit(src.get(x,y,z),src.index(x,y,z));
}

inline dim3 shrink(dim3 d,int axes){if(axes!=1){d.x=(d.x+1)/2;d.y=(d.y+1)/2;}if(axes!=2)d.z=(d.z+1)/2;return d;}
template<class T> __global__ void merge_low(const T* low,T* coefficients,dim3 full,dim3 n) {
    const size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(i>=size_t(n.x)*n.y*n.z)return;
    const size_t x=i%n.x,y=(i/n.x)%n.y,z=i/(size_t(n.x)*n.y);
    const size_t at=(z*full.y+y)*full.x+x;
    coefficients[at]=low[at];
}
template<class T,class Emit> inline void launch_finest(const T* src,dim3 n,Emit emit,cudaStream_t stream) {
    // Same new halo arithmetic, specialized for contiguous finest-level input.
    constexpr int Z=sizeof(T)==4?16:8;
    static thread_local int configured=-1;
    int device=-1;cudaError_t e=cudaGetDevice(&device);
    if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));
    if(configured!=device) {
        e=cudaFuncSetAttribute(halo3d::inverse<Z,T,Emit>,cudaFuncAttributeMaxDynamicSharedMemorySize,65536);
        if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));
        if constexpr(std::is_same_v<T,double>) {
            e=cudaFuncSetAttribute(halo3d::inverse<Z,T,Emit,true>,cudaFuncAttributeMaxDynamicSharedMemorySize,65536);
            if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));
        }
        configured=device;
    }
    const dim3 grid((n.x+55)/56,(n.y+7)/8,(n.z+Z-1)/Z);
    if constexpr(std::is_same_v<T,double>)if(n.x>=64 && n.y>=64 && n.z>=64) {
        halo3d::inverse<Z,T,Emit,true><<<grid,256,65536,stream>>>(src,n.x,n.y,n.z,emit);return;
    }
    halo3d::inverse<Z,T,Emit><<<grid,256,65536,stream>>>(src,n.x,n.y,n.z,emit);
}
template<class T,class Emit> inline void launch_inverse_xyz(Read<T> src,dim3 n,Emit emit,cudaStream_t stream) {
    constexpr int Z=sizeof(T)==4?16:8;
    static thread_local int configured=-1;
    int device=-1;cudaError_t e=cudaGetDevice(&device);
    if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));
    if(configured!=device) {
        e=cudaFuncSetAttribute(inverse_xyz<Z,T,Emit>,cudaFuncAttributeMaxDynamicSharedMemorySize,65536);
        if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));
        if constexpr(std::is_same_v<T,double>) {
            e=cudaFuncSetAttribute(inverse_xyz<Z,T,Emit,true>,cudaFuncAttributeMaxDynamicSharedMemorySize,65536);
            if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));
        }
        configured=device;
    }
    const dim3 grid((n.x+55)/56,(n.y+7)/8,(n.z+Z-1)/Z);
    if constexpr(std::is_same_v<T,double>) {
        if(n.x>=64 && n.y>=64 && n.z>=64) {
            inverse_xyz<Z,T,Emit,true><<<grid,256,65536,stream>>>(src,n,emit);return;
        }
    }
    inverse_xyz<Z,T,Emit><<<grid,256,65536,stream>>>(src,n,emit);
}

template<class T,bool Regular> inline void forward(const T* input,T* a,T* b,T* coefficients,
    dim3 full,int lxy,int lz,bool dyadic,cudaStream_t stream,bool fused_plane_z=false) {
    const T* current=input;
    if(dyadic) {
        dim3 active=full;
        for(int l=0;l<lxy;++l) {
            T* next=(l&1)?b:a;
            constexpr int Z=sizeof(T)==4?8:4;
            const dim3 grid((active.x+55)/56,(active.y+7)/8,(active.z+Z-1)/Z);
            const auto emit=Quant<T,Regular>{next,full,active,true,l+1==lxy};
            if(l==0)halo3d::forward<Z><<<grid,256,56*8*(Z+8)*sizeof(T),stream>>>(current,active.x,active.y,active.z,emit);
            else forward_xyz<Z><<<grid,256,56*8*(Z+8)*sizeof(T),stream>>>(Read<T>{current,nullptr,full,dim3(0,0,0)},active,emit);
            current=next;active=shrink(active,3);
        }
    } else {
        if constexpr(std::is_same_v<T,float>) {
            if(fused_plane_z) {
                CDF97::d_reset<<<1,1,0,stream>>>();
                CDF97::dwt_launch_z_tiles<CDF97::DWT_Z_WIDE_BYTES>(input,a,full,uint8_t(lz),stream);
            }
        }
        // Persistent Z subbands live in a; two separate low-band workspaces
        // avoid any in-place halo reads/writes between CTAs.
        dim3 active=full;
        for(int l=0;l<lz && !(std::is_same_v<T,float> && fused_plane_z);++l) {
            T* next=(l&1)?coefficients:b;
            constexpr int Z=sizeof(T)==4?16:8;
            zline<Z,T,false><<<dim3((active.x*active.y+255)/256,(active.z+Z-1)/Z),256,0,stream>>>(
                Read<T>{current,nullptr,full,dim3(0,0,0)},active,ZSplit<T>{a,next,int((active.z+1)/2),l+1==lz});
            current=next;active=shrink(active,1);
        }
        current=lz?a:input;active=full;
        for(int l=0;l<lxy;++l) {
            T* next=(l&1)?a:b;
            xy<T,false><<<dim3((active.x+55)/56,(active.y+7)/8,(active.z+3)/4),128,0,stream>>>(
                Read<T>{current,nullptr,full,dim3(0,0,0)},active,Quant<T,Regular>{next,full,active,false,l+1==lxy});
            current=next;active=shrink(active,2);
        }
    }
    if(lxy==0) {
        identity<T,true><<<(size_t(full.x)*full.y*full.z+255)/256,256,0,stream>>>(
            Read<T>{current,nullptr,full,dim3(0,0,0)},full,Quant<T,Regular>{nullptr,full,full,false,true});
    }
}

template<class T,class Final> inline void inverse(T* coefficients,T* a,T* b,dim3 full,
    int lxy,int lz,bool dyadic,Final final,cudaStream_t stream,bool fused_plane_z=false) {
    dim3 sizes[CDF97_MAX_LEVELS];
    dim3 active=full;
    for(int l=0;l<lxy;++l){sizes[l]=active;active=shrink(active,dyadic?3:2);}
    const T* previous=nullptr;
    if(dyadic) {
        for(int l=lxy-1;l>=0;--l) {
            const auto read=Read<T>{coefficients,previous,full,l+1<lxy?sizes[l+1]:dim3(0,0,0)};
            T* dst=(l&1)?a:b;
            if(l==0) {
                if(previous) {
                    const dim3 low=sizes[1];
                    merge_low<<<(size_t(low.x)*low.y*low.z+255)/256,256,0,stream>>>(previous,coefficients,full,low);
                }
                launch_finest(coefficients,sizes[l],final,stream);
            }
            else launch_inverse_xyz(read,sizes[l],Store<T>{dst},stream);
            previous=dst;
        }
    } else {
        for(int l=lxy-1;l>=0;--l) {
            const auto read=Read<T>{coefficients,previous,full,l+1<lxy?sizes[l+1]:dim3(0,0,0)};
            // When Z follows, keep its immutable full-volume input in a.
            T* dst=lz?((l&1)?b:a):((l&1)?a:b);
            const dim3 n=sizes[l],grid((n.x+55)/56,(n.y+7)/8,(n.z+3)/4);
            if(l==0 && !lz)xy<T,true><<<grid,128,0,stream>>>(read,n,final);
            else xy<T,true><<<grid,128,0,stream>>>(read,n,Store<T>{dst});
            previous=dst;
        }
        T* zbase=lxy?a:coefficients;
        if(fused_plane_z) {
            if constexpr(std::is_same_v<T,float> && std::is_same_v<Final,CDF97::Halo3DSynthEmit<T,true>>) {
                CDF97::idwt_plane_z(zbase,b,full,lz,stream,final.original,final.bound,
                    final.count,final.indices,final.errors,final.capacity);
                return;
            } else if constexpr(std::is_same_v<T,float> && std::is_same_v<Final,Store<T>>) {
                CDF97::idwt_plane_z(zbase,final.output,full,lz,stream);
                return;
            }
            throw std::invalid_argument("Waltz: unsupported fused plane-Z output");
        }
        T* aux=lxy?coefficients:a;
        active=full;for(int l=0;l<lz;++l){sizes[l]=active;active=shrink(active,1);}
        previous=nullptr;
        for(int l=lz-1;l>=0;--l) {
            const auto read=Read<T>{zbase,previous,full,l+1<lz?sizes[l+1]:dim3(0,0,0)};
            T* dst=(l&1)?aux:b;const dim3 n=sizes[l];
            constexpr int Z=sizeof(T)==4?16:8;
            const dim3 grid((n.x*n.y+255)/256,(n.z+Z-1)/Z);
            if(l==0)zline<Z,T,true><<<grid,256,0,stream>>>(read,n,final);
            else zline<Z,T,true><<<grid,256,0,stream>>>(read,n,Store<T>{dst});
            previous=dst;
        }
    }
    if(lxy==0 && lz==0)identity<T,false><<<(size_t(full.x)*full.y*full.z+255)/256,256,0,stream>>>(
        Read<T>{coefficients,nullptr,full,dim3(0,0,0)},full,final);
}
} // namespace WALTZ::allhalo
