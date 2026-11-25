#pragma once
// HIP-specific execution of the same separable XY lifting steps. A 32- or 64-lane
// subgroup owns one XY strip. Four samples of halo at either end are not emitted.
namespace native {
// The caller proves each axis is wide enough that halo coordinates need at
// most one reflection. Reflection preserves parity in the low/high layout.
template<bool Inv> __device__ __forceinline__ int strip_index(int p,int n) {
    const int u=p<0?-p:p, r=u<n?u:2*n-2-u;
    if constexpr(Inv) return (r>>1)+((p&1)?((n+1)>>1):0);
    return r;
}
template<bool Inverse,int Width,class T>
__device__ __forceinline__ void horizontal_pair(T& e,T& o) {
    using C=warp97::Constants<T>;
    if constexpr(Inverse){
        e*=C::IK; o*=-C::K;
        e=warp97::madd(-C::D,o+__shfl_up(o,1,Width),e);
        o=warp97::madd(-C::G,e+__shfl_down(e,1,Width),o);
        e=warp97::madd(-C::B,o+__shfl_up(o,1,Width),e);
        o=warp97::madd(-C::A,e+__shfl_down(e,1,Width),o);
    }else{
        o=warp97::madd(C::A,e+__shfl_down(e,1,Width),o);
        e=warp97::madd(C::B,o+__shfl_up(o,1,Width),e);
        o=warp97::madd(C::G,e+__shfl_down(e,1,Width),o);
        e=warp97::madd(C::D,o+__shfl_up(o,1,Width),e);
        e*=C::K; o*=-C::IK;
    }
}

template<bool Inverse,int Width,int Rows,bool PackRows=false,bool Pairs=false,class T,class Emit>
__global__ void xy_strip(WALTZ::allhalo::Read<T> src,dim3 n,Emit out) {
    constexpr int P=Rows/2+4;
    constexpr int Cols=2*Width-8;
    constexpr int groups=256/Width;
    const int lane=threadIdx.x%Width,group=threadIdx.x/Width;
    const int z=PackRows?int(blockIdx.z):int(groups*blockIdx.z+group);
    const int strip=PackRows?int(groups*blockIdx.y+group):int(blockIdx.y);
    if((!PackRows && z>=n.z) || strip*Rows>=n.y)return;
    // Row-packed blocks own a plane. Share its offset across the lifting taps.
    const unsigned plane=PackRows?unsigned(z)*src.full.x*src.full.y:0u;
    auto xy_index=[&](int col,int row) -> size_t {
        if constexpr(PackRows)return plane+unsigned(row)*src.full.x+unsigned(col);
        else return src.index(col,row,z);
    };
    const int x=int(blockIdx.x)*Cols-4+2*lane,y0=strip*Rows-4;
    const int xl=strip_index<Inverse>(x,n.x);
    const int xr=strip_index<Inverse>(x+1,n.x);
    T e[P],o[P],ee[P],oo[P];
    #pragma unroll
    for(int k=0;k<P;++k){
        const int ya=strip_index<Inverse>(y0+2*k,n.y);
        const int yb=strip_index<Inverse>(y0+2*k+1,n.y);
        const T* a=src.base+xy_index(0,ya), *b=src.base+xy_index(0,yb);
        // Reflection preserves parity. In a multi-level plane inverse only
        // even X / even Y reads the reconstructed LL plane (all z); high
        // bands still come from the original coefficient buffer.
        const T* ll=Inverse&&src.low?src.low:src.base;
        e[k]=ll[xy_index(xl,ya)];ee[k]=a[xr];o[k]=b[xl];oo[k]=b[xr];
        if constexpr(Inverse){horizontal_pair<true,Width>(e[k],ee[k]);horizontal_pair<true,Width>(o[k],oo[k]);}
    }
    if constexpr(Inverse){halo3d::sz(e,o);halo3d::sz(ee,oo);}
    else {halo3d::az(e,o);halo3d::az(ee,oo);}
    #pragma unroll
    for(int k=2;k<P-2;++k){
        const int y=y0+2*k;
        if constexpr(!Inverse){horizontal_pair<false,Width>(e[k],ee[k]);horizontal_pair<false,Width>(o[k],oo[k]);}
        if(lane>=2&&lane<Width-2&&x<n.x&&y<n.y){
            if constexpr(Inverse){
                if constexpr(Pairs){
                    const unsigned at=xy_index(x,y);
                    out.pair(e[k],ee[k],at);
                    if(y+1<n.y)out.pair(o[k],oo[k],at+src.full.x);
                }else{
                    out(e[k],xy_index(x,y));if(x+1<n.x)out(ee[k],xy_index(x+1,y));
                    if(y+1<n.y){out(o[k],xy_index(x,y+1));if(x+1<n.x)out(oo[k],xy_index(x+1,y+1));}
                }
            }else{
                const int lx=x>>1,hx=((n.x+1)>>1)+lx,ly=y>>1,hy=((n.y+1)>>1)+ly;
                out(e[k],xy_index(lx,ly),lx,ly,z);
                if(x+1<n.x)out(ee[k],xy_index(hx,ly),hx,ly,z);
                if(y+1<n.y){out(o[k],xy_index(lx,hy),lx,hy,z);
                    if(x+1<n.x)out(oo[k],xy_index(hx,hy),hx,hy,z);}
            }
        }
    }
}

// Pick the low-band buffer once per XY column, not per halo tap. Mirroring
// preserves even/odd parity for n.z > 1 so odd taps always use base.
template<int Z,class T,class Emit>
__global__ void inverse_zstrip(WALTZ::allhalo::Read<T> src,dim3 n,Emit emit){
    const unsigned col=blockIdx.x*blockDim.x+threadIdx.x;
    if(col>=n.x*n.y)return;
    const unsigned x=col%n.x,y=col/n.x;
    const unsigned pos=n.x==src.full.x?col:y*src.full.x+x;
    const unsigned plane=src.full.x*src.full.y;
    const T* lo=(src.low && x<src.low_dims.x && y<src.low_dims.y ? src.low:src.base)+pos;
    const T* hi=src.base+pos;
    const int z0=int(blockIdx.y)*Z-4;
    constexpr int P=Z/2+4;
    T e[P],o[P];
    #pragma unroll
    for(int j=0;j<P;++j){
        const int v=z0+2*j;
        const unsigned a=unsigned(halo3d::mirror(v,n.z))>>1;
        const unsigned b=(unsigned(halo3d::mirror(v+1,n.z))>>1)+((n.z+1)>>1);
        e[j]=lo[a*plane];o[j]=hi[b*plane];
    }
    halo3d::sz(e,o);
    #pragma unroll
    for(int j=2;j<P-2;++j){
        const unsigned z=unsigned(z0+2*j);
        if(z<n.z)emit(e[j],pos+z*plane);
        if(z+1<n.z)emit(o[j],pos+(z+1)*plane);
    }
}
// Finest dyadic inverse: adjacent x columns share all Z indices. The host
// guarantees an even low/high X split and aligned, tightly packed rows.
__global__ void inverse_zpairs(WALTZ::allhalo::Read<float> src,dim3 n,float* stage){
    const unsigned x=2*(blockIdx.x*blockDim.x+threadIdx.x),y=blockIdx.y;
    if(x>=n.x)return;
    const unsigned pos=(y*n.x+x)/2,stride=n.x*n.y/2;
    const auto* hi=reinterpret_cast<const float2*>(src.base)+pos;
    const auto* lo=src.low && x<src.low_dims.x && y<src.low_dims.y ?
        reinterpret_cast<const float2*>(src.low)+pos:hi;
    auto* out=reinterpret_cast<float2*>(stage)+pos;
    const int z0=int(blockIdx.z)*16-4;
    float e[12],o[12],f[12],p[12];
    #pragma unroll
    for(int j=0;j<12;++j){
        const auto a=lo[unsigned(strip_index<true>(z0+2*j,n.z))*stride];
        const auto b=hi[unsigned(strip_index<true>(z0+2*j+1,n.z))*stride];
        e[j]=a.x;f[j]=a.y;o[j]=b.x;p[j]=b.y;
    }
    halo3d::sz(e,o);halo3d::sz(f,p);
    #pragma unroll
    for(int j=2;j<10;++j){
        const unsigned z=z0+2*j;
        if(z<n.z)out[z*stride]=make_float2(e[j],f[j]);
        if(z+1<n.z)out[(z+1)*stride]=make_float2(o[j],p[j]);
    }
}
} // namespace native
