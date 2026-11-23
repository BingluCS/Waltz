#pragma once
#include "warp97.cuh"

// Finest-level native-precision 3D codec kernels. Same normalization, sign,
// symmetric extension and octant layout as Waltz. Not NVIDIA source.
namespace halo3d {
using namespace warp97;
__host__ __device__ inline int periodic_mirror(int p,int n) {
    if(n<=1) return 0;
    int r=p%(2*n-2); if(r<0) r+=2*n-2;
    return r<n?r:2*n-2-r;
}
__host__ __device__ inline int mirror(int p,int n) {
    if(n<=1) return 0;
    // Uniform fast path: the largest supported tile overreads by <128.
    // For n>=128 at most one reflection suffices; keep periodic tiny axes.
    if(n>=128) {
        const int u=p<0?-p:p;
        return u<n?u:2*n-2-u;
    }
    return periodic_mirror(p,n);
}
__host__ __device__ inline int coeff(int p,int n) {
    // Inverse coefficient addressing deliberately uses the smaller periodic
    // expression. Adding the forward fast branch raises inverse live registers
    // and regresses short-Z cases in the isolated transform measurements.
    const int r=periodic_mirror(p,n); return r/2+(r%2?(n+1)/2:0);
}
template<bool SingleFold> __host__ __device__ inline int inverse_coeff(int p,int n) {
    if constexpr(SingleFold) {
        // Host dispatch proves n >= 64 and the inverse tile overread < 64.
        // Reflection has even period, so it preserves sample parity.
        const int u=p<0?-p:p;
        const int r=u<n?u:2*n-2-u;
        return r/2+((p&1)?(n+1)/2:0);
    } else return coeff(p,n);
}
template<int P, class T> __device__ __forceinline__ void az(T (&e)[P],T (&o)[P]) {
    #pragma unroll
    for(int j=0;j<P-1;++j) o[j]=madd(Constants<T>::A,e[j]+e[j+1],o[j]);
    #pragma unroll
    for(int j=1;j<P-1;++j) e[j]=madd(Constants<T>::B,o[j-1]+o[j],e[j]);
    #pragma unroll
    for(int j=1;j<P-2;++j) o[j]=madd(Constants<T>::G,e[j]+e[j+1],o[j]);
    #pragma unroll
    for(int j=2;j<P-2;++j) e[j]=madd(Constants<T>::D,o[j-1]+o[j],e[j]);
    #pragma unroll
    for(int j=2;j<P-2;++j) {e[j]*=Constants<T>::K; o[j]*=-Constants<T>::IK;}
}
template<int P, class T> __device__ __forceinline__ void sz(T (&e)[P],T (&o)[P]) {
    #pragma unroll
    for(int j=0;j<P;++j) {e[j]*=Constants<T>::IK; o[j]*=-Constants<T>::K;}
    #pragma unroll
    for(int j=1;j<P;++j) e[j]=madd(-Constants<T>::D,o[j-1]+o[j],e[j]);
    #pragma unroll
    for(int j=1;j<P-1;++j) o[j]=madd(-Constants<T>::G,e[j]+e[j+1],o[j]);
    #pragma unroll
    for(int j=2;j<P-1;++j) e[j]=madd(-Constants<T>::B,o[j-1]+o[j],e[j]);
    #pragma unroll
    for(int j=2;j<P-2;++j) o[j]=madd(-Constants<T>::A,e[j]+e[j+1],o[j]);
}

// XY transforms of halo-expanded Z planes -> one CTA barrier -> Z lifting.
// Useful tile: 56 x 8 x Z. Input footprint: 64 x 16 x (Z+8).
// Shared memory holds only useful XY coefficients, not the full raw footprint.
template<int Z, class T, class Emit> __global__ void forward(const T* __restrict__ src,
    int nx,int ny,int nz, Emit emit) {
    static_assert(Z>=2 && Z<=64 && Z%2==0,"bounded even halo tile required");
    extern __shared__ __align__(16) unsigned char halo_storage[];
    T* sm=reinterpret_cast<T*>(halo_storage);
    constexpr int P=Z/2+4, XY=56*8;
    const int lane=threadIdx.x%32, warp=threadIdx.x/32;
    const int ix=int(blockIdx.x)*56+2*lane-4, iy=int(blockIdx.y)*8-4;
    const int z0=int(blockIdx.z)*Z-4;
    const int x0=mirror(ix,nx), x1=mirror(ix+1,nx);
    const size_t plane=size_t(nx)*ny;
    for(int k=warp;k<Z+8;k+=blockDim.x/32) {
        const size_t zo=size_t(mirror(z0+k,nz))*plane;
        T e0[8],o0[8],e1[8],o1[8];
        #pragma unroll
        for(int j=0;j<8;++j) {
            const size_t a=zo+size_t(mirror(iy+2*j,ny))*nx;
            const size_t b=zo+size_t(mirror(iy+2*j+1,ny))*nx;
            e0[j]=src[a+x0]; e1[j]=src[a+x1];
            o0[j]=src[b+x0]; o1[j]=src[b+x1];
        }
        analysis_y(e0,o0); analysis_y(e1,o1);
        #pragma unroll
        for(int j=2;j<6;++j) {
            T ll=e0[j]*Constants<T>::K, hl=e1[j]*Constants<T>::K, lh=o0[j]*(-Constants<T>::IK), hh=o1[j]*(-Constants<T>::IK);
            analysis_x(ll,hl); analysis_x(lh,hh);
            if(lane>=2 && lane<30) {
                const int at=k*XY+(j-2)*56+lane-2;
                sm[at]=ll; sm[at+28]=hl; sm[at+4*56]=lh; sm[at+4*56+28]=hh;
            }
        }
    }
    __syncthreads();
    for(int c=threadIdx.x;c<XY;c+=blockDim.x) {
        const int ry=c/56, rx=c%56;
        const int px=int(blockIdx.x)*56+2*(rx%28)+(rx/28);
        const int py=int(blockIdx.y)*8+2*(ry%4)+(ry/4);
        if(px>=nx || py>=ny) continue;
        const int ox=px/2+(px%2?(nx+1)/2:0), oy=py/2+(py%2?(ny+1)/2:0);
        const size_t out=size_t(oy)*nx+ox;
        T e[P],o[P];
        #pragma unroll
        for(int j=0;j<P;++j) {e[j]=sm[(2*j)*XY+c];o[j]=sm[(2*j+1)*XY+c];}
        az(e,o);
        #pragma unroll
        for(int j=2;j<P-2;++j) {
            const int z=z0+2*j;
            if(z<nz) emit(e[j],size_t(z/2)*plane+out,ox,oy,z/2);
            if(z+1<nz) emit(o[j],size_t((nz+1)/2+z/2)*plane+out,ox,oy,(nz+1)/2+z/2);
        }
    }
}

// Z inverse over the XY coefficient halo -> one CTA barrier -> XY inverse.
// Shared footprint: 64 x 16 x Z; no inter-CTA dependencies.
template<int Z, class T, class Emit, bool SingleFold=false> __global__ void inverse(const T* __restrict__ src,
    int nx,int ny,int nz, Emit emit) {
    static_assert(Z>=2 && Z<=64 && Z%2==0,"bounded even halo tile required");
    extern __shared__ __align__(16) unsigned char halo_storage[];
    T* sm=reinterpret_cast<T*>(halo_storage);
    constexpr int P=Z/2+4, XY=64*16;
    const int xbase=int(blockIdx.x)*56-4, ybase=int(blockIdx.y)*8-4;
    const int z0=int(blockIdx.z)*Z-4;
    const size_t plane=size_t(nx)*ny;
    for(int c=threadIdx.x;c<XY;c+=blockDim.x) {
        const int x=inverse_coeff<SingleFold>(xbase+c%64,nx), y=inverse_coeff<SingleFold>(ybase+c/64,ny);
        const size_t at=size_t(y)*nx+x;
        T e[P],o[P];
        #pragma unroll
        for(int j=0;j<P;++j) {
            e[j]=src[size_t(inverse_coeff<SingleFold>(z0+2*j,nz))*plane+at];
            o[j]=src[size_t(inverse_coeff<SingleFold>(z0+2*j+1,nz))*plane+at];
        }
        sz(e,o);
        #pragma unroll
        for(int j=2;j<P-2;++j) {
            sm[(2*j-4)*XY+c]=e[j]; sm[(2*j-3)*XY+c]=o[j];
        }
    }
    __syncthreads();
    const int lane=threadIdx.x%32, warp=threadIdx.x/32;
    const int ix=xbase+2*lane;
    for(int k=warp;k<Z;k+=blockDim.x/32) {
        if(z0+4+k>=nz) continue; // Warp uniform.
        T e0[8],o0[8],e1[8],o1[8];
        #pragma unroll
        for(int j=0;j<8;++j) {
            const int at=k*XY+2*j*64+2*lane;
            T el=sm[at],ol=sm[at+1],eh=sm[at+64],oh=sm[at+65];
            synthesis_x(el,ol); synthesis_x(eh,oh);
            e0[j]=el;e1[j]=ol;o0[j]=eh;o1[j]=oh;
        }
        synthesis_y(e0,o0); synthesis_y(e1,o1);
        #pragma unroll
        for(int j=2;j<6;++j) {
            const int y=ybase+2*j;
            const size_t at=size_t(z0+4+k)*plane+size_t(y)*nx+ix;
            if(lane>=2 && lane<30 && ix<nx) {
                if(y<ny) {emit(e0[j],at);if(ix+1<nx) emit(e1[j],at+1);}
                if(y+1<ny) {emit(o0[j],at+nx);if(ix+1<nx) emit(o1[j],at+nx+1);}
            }
        }
    }
}
} // namespace halo3d
