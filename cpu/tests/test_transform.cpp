#include "transform.hpp"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <type_traits>
#include <vector>

using waltz_cpu::Shape;

namespace {
template<class T> struct C {
    static constexpr T A=T(-1.586134342059924),B=T(-0.052980118572961);
    static constexpr T G=T(0.882911075530934),D=T(0.443506852043971);
    static constexpr T K=T(1.1496043988602418),IK=T(1)/K;
};

unsigned low(unsigned n) { return (n+1)/2; }
int mirror(int p,int n) { if(n<=1)return 0;int r=p%(2*n-2);if(r<0)r+=2*n-2;return r<n?r:2*n-2-r; }
int coeff(int p,int n) { int r=mirror(p,n);return r/2+(r%2?int(low(n)):0); }
std::size_t at(Shape s,unsigned x,unsigned y,unsigned z) { return (std::size_t(z)*s.ny+y)*s.nx+x; }

template<class T> void ay(T* e,T* o,int p) {
    for(int j=0;j<p-1;++j)o[j]=std::fma(C<T>::A,e[j]+e[j+1],o[j]);
    for(int j=1;j<p-1;++j)e[j]=std::fma(C<T>::B,o[j-1]+o[j],e[j]);
    for(int j=1;j<p-2;++j)o[j]=std::fma(C<T>::G,e[j]+e[j+1],o[j]);
    for(int j=2;j<p-2;++j)e[j]=std::fma(C<T>::D,o[j-1]+o[j],e[j]);
}
template<class T> void az(T* e,T* o,int p) {
    ay(e,o,p);
    for(int j=2;j<p-2;++j) { e[j]*=C<T>::K;o[j]*=-C<T>::IK; }
}
template<class T> void sz(T* e,T* o,int p) {
    const T first=o[0],last=o[p-1];
    for(int j=0;j<p;++j) { e[j]*=C<T>::IK;o[j]*=-C<T>::K; }
    // Production CUDA13.1 default-fmad PTX contracts the one-use endpoints'
    // normalization with the subsequent sum; preserve that actual profile.
    for(int j=1;j<p;++j) {
        const T sum=j==1?std::fma(first,-C<T>::K,o[1]):
                    j==p-1?std::fma(last,-C<T>::K,o[p-2]):o[j-1]+o[j];
        e[j]=std::fma(-C<T>::D,sum,e[j]);
    }
    for(int j=1;j<p-1;++j)o[j]=std::fma(-C<T>::G,e[j]+e[j+1],o[j]);
    for(int j=2;j<p-1;++j)e[j]=std::fma(-C<T>::B,o[j-1]+o[j],e[j]);
    for(int j=2;j<p-2;++j)o[j]=std::fma(-C<T>::A,e[j]+e[j+1],o[j]);
}
template<class T> void ax(T* e,T* o) {
    for(int j=0;j<32;++j)o[j]=std::fma(C<T>::A,e[j]+e[j==31?j:j+1],o[j]);
    for(int j=0;j<32;++j)e[j]=std::fma(C<T>::B,o[j]+o[j?j-1:j],e[j]);
    for(int j=0;j<32;++j)o[j]=std::fma(C<T>::G,e[j]+e[j==31?j:j+1],o[j]);
    for(int j=0;j<32;++j)e[j]=std::fma(C<T>::D,o[j]+o[j?j-1:j],e[j]);
    for(int j=0;j<32;++j) { e[j]*=C<T>::K;o[j]*=-C<T>::IK; }
}
template<class T> void sx(T* e,T* o) {
    for(int j=0;j<32;++j) { e[j]*=C<T>::IK;o[j]*=-C<T>::K; }
    for(int j=0;j<32;++j)e[j]=std::fma(-C<T>::D,o[j]+o[j?j-1:j],e[j]);
    for(int j=0;j<32;++j)o[j]=std::fma(-C<T>::G,e[j]+e[j==31?j:j+1],o[j]);
    for(int j=0;j<32;++j)e[j]=std::fma(-C<T>::B,o[j]+o[j?j-1:j],e[j]);
    for(int j=0;j<32;++j)o[j]=std::fma(-C<T>::A,e[j]+e[j==31?j:j+1],o[j]);
}

// Direct CPU emulation of halo3d::{forward,inverse}, preserving its 3D
// coefficient gathers, 32-lane X shuffles, shared-memory layout, and actual
// GPU tile sizes. This is deliberately independent of the line decomposition
// in transform.cpp, and catches reflect-before-synthesis mistakes.
template<class T> void gpu_forward_level(std::vector<T>& data,Shape full,Shape n) {
    const int Z=sizeof(T)==4?8:4,P=Z/2+4;
    const std::vector<T> src=data;
    std::vector<T> shared(std::size_t(Z+8)*448);
    for(unsigned bz=0;bz<n.nz;bz+=Z)for(unsigned by=0;by<n.ny;by+=8)for(unsigned bx=0;bx<n.nx;bx+=56) {
        for(int k=0;k<Z+8;++k) {
            T ye[8][32],yo[8][32],ye1[8][32],yo1[8][32];
            for(int lane=0;lane<32;++lane) {
                const int x0=mirror(int(bx)+2*lane-4,n.nx),x1=mirror(int(bx)+2*lane-3,n.nx),z=mirror(int(bz)+k-4,n.nz);
                T e0[8],o0[8],e1[8],o1[8];
                for(int j=0;j<8;++j) {
                    const int y0=mirror(int(by)+2*j-4,n.ny),y1=mirror(int(by)+2*j-3,n.ny);
                    e0[j]=src[at(full,x0,y0,z)];o0[j]=src[at(full,x0,y1,z)];
                    e1[j]=src[at(full,x1,y0,z)];o1[j]=src[at(full,x1,y1,z)];
                }
                ay(e0,o0,8);ay(e1,o1,8);
                for(int j=2;j<6;++j) {
                    ye[j][lane]=e0[j]*C<T>::K;ye1[j][lane]=e1[j]*C<T>::K;
                    yo[j][lane]=o0[j]*(-C<T>::IK);yo1[j][lane]=o1[j]*(-C<T>::IK);
                }
            }
            for(int j=2;j<6;++j) {
                ax(ye[j],ye1[j]);ax(yo[j],yo1[j]);
                for(int lane=2;lane<30;++lane) {
                    const std::size_t i=std::size_t(k)*448+(j-2)*56+lane-2;
                    shared[i]=ye[j][lane];shared[i+28]=ye1[j][lane];
                    shared[i+4*56]=yo[j][lane];shared[i+4*56+28]=yo1[j][lane];
                }
            }
        }
        for(int c=0;c<448;++c) {
            const int ry=c/56,rx=c%56,px=bx+2*(rx%28)+rx/28,py=by+2*(ry%4)+ry/4;
            if(px>=int(n.nx)||py>=int(n.ny))continue;
            const int x=px/2+(px%2?low(n.nx):0),y=py/2+(py%2?low(n.ny):0);
            T e[12],o[12];
            for(int j=0;j<P;++j) { e[j]=shared[2*j*448+c];o[j]=shared[(2*j+1)*448+c]; }
            az(e,o,P);
            for(int j=2;j<P-2;++j) {
                const int z=bz+2*j-4;
                if(z<int(n.nz))data[at(full,x,y,z/2)]=e[j];
                if(z+1<int(n.nz))data[at(full,x,y,low(n.nz)+z/2)]=o[j];
            }
        }
    }
}

template<class T> void gpu_inverse_level(std::vector<T>& data,Shape full,Shape n,int Z) {
    const int P=Z/2+4;
    const std::vector<T> src=data;
    std::vector<T> shared(std::size_t(Z)*1024);
    for(unsigned bz=0;bz<n.nz;bz+=Z)for(unsigned by=0;by<n.ny;by+=8)for(unsigned bx=0;bx<n.nx;bx+=56) {
        for(int c=0;c<1024;++c) {
            const int x=coeff(int(bx)+c%64-4,n.nx),y=coeff(int(by)+c/64-4,n.ny);
            T e[12],o[12];
            for(int j=0;j<P;++j) {
                e[j]=src[at(full,x,y,coeff(int(bz)+2*j-4,n.nz))];
                o[j]=src[at(full,x,y,coeff(int(bz)+2*j-3,n.nz))];
            }
            sz(e,o,P);
            for(int j=2;j<P-2;++j) { shared[(2*j-4)*1024+c]=e[j];shared[(2*j-3)*1024+c]=o[j]; }
        }
        for(int k=0;k<Z&&bz+k<n.nz;++k) {
            T xy[16][64];
            for(int row=0;row<16;++row) {
                T e[32],o[32];
                for(int lane=0;lane<32;++lane) { e[lane]=shared[k*1024+row*64+2*lane];o[lane]=shared[k*1024+row*64+2*lane+1]; }
                sx(e,o);
                for(int lane=0;lane<32;++lane) { xy[row][2*lane]=e[lane];xy[row][2*lane+1]=o[lane]; }
            }
            for(int col=4;col<60&&int(bx)+col-4<int(n.nx);++col) {
                T e[8],o[8];
                for(int j=0;j<8;++j) { e[j]=xy[2*j][col];o[j]=xy[2*j+1][col]; }
                sz(e,o,8);
                for(int j=2;j<6;++j) {
                    const unsigned y=by+2*j-4,x=bx+col-4;
                    if(y<n.ny)data[at(full,x,y,bz+k)]=e[j];
                    if(y+1<n.ny)data[at(full,x,y+1,bz+k)]=o[j];
                }
            }
        }
    }
}

template<class T> void forward(std::vector<T>& data,std::vector<T>& scratch,Shape s,unsigned xy,unsigned z,bool dyadic,unsigned mode,bool simd) {
    if constexpr(std::is_same_v<T,float>)waltz_cpu::forward_f32(data.data(),scratch.data(),s,xy,z,dyadic,mode,simd);
    else waltz_cpu::forward_f64(data.data(),scratch.data(),s,xy,z,dyadic,mode,simd);
}
template<class T> void inverse(std::vector<T>& data,std::vector<T>& scratch,Shape s,unsigned xy,unsigned z,bool dyadic,unsigned mode,bool simd) {
    if constexpr(std::is_same_v<T,float>)waltz_cpu::inverse_f32(data.data(),scratch.data(),s,xy,z,dyadic,mode,simd);
    else waltz_cpu::inverse_f64(data.data(),scratch.data(),s,xy,z,dyadic,mode,simd);
}

template<class T> void equal(const std::vector<T>& a,const std::vector<T>& b,const char* label,Shape s,unsigned xy,unsigned z,unsigned mode) {
    if(!std::memcmp(a.data(),b.data(),a.size()*sizeof(T)))return;
    for(std::size_t i=0;i<a.size();++i)if(std::memcmp(&a[i],&b[i],sizeof(T))) {
        std::fprintf(stderr,"%s f%u %ux%ux%u xy%u z%u mode%u mismatch[%zu]: %a != %a\n",label,unsigned(sizeof(T)*8),s.nx,s.ny,s.nz,xy,z,mode,i,double(a[i]),double(b[i]));
        throw std::runtime_error("transform bitwise comparison failed");
    }
}

template<class T> std::vector<T> samples(Shape s) {
    std::vector<T> result(waltz_cpu::workspace_elements(s));std::uint32_t rng=727;
    for(auto& v:result) { rng=rng*1664525U+1013904223U;v=T(std::int32_t(rng))/T(1048576); }
    if(result.size()>7) { result[0]=T(0);result[1]=-T(0);result[2]=T(1);result[3]=T(-1); }
    return result;
}

template<class T> void test_type() {
    const Shape shapes[]={{1,1,1},{2,2,2},{2,3,4},{3,2,1},{7,9,5},
        {9,11,13},{15,17,19},{16,8,8},{8,16,9},
        {17,5,33},{57,9,17},{65,17,19},{11,13,257},{64,64,64},{65,67,69}};
    for(Shape s:shapes) {
        const auto original=samples<T>(s);std::vector<T> scratch(original.size());
        for(unsigned levels=0;levels<=5;++levels) {
            auto scalar=original,simd=original;
            forward(scalar,scratch,s,levels,levels,true,2,false);
            forward(simd,scratch,s,levels,levels,true,2,true);
            equal(scalar,simd,"scalar/SIMD forward",s,levels,levels,2);
            inverse(scalar,scratch,s,levels,levels,true,2,false);
            inverse(simd,scratch,s,levels,levels,true,2,true);
            equal(scalar,simd,"scalar/SIMD inverse",s,levels,levels,2);
            if(levels<=2 && s.nx<=65 && s.ny<=17) {
                auto reference=original;Shape n=s;
                for(unsigned l=0;l<levels;++l) { gpu_forward_level(reference,s,n);n={low(n.nx),low(n.ny),low(n.nz)}; }
                auto actual=original;forward(actual,scratch,s,levels,levels,true,2,false);
                equal(actual,reference,"GPU tiled forward",s,levels,levels,2);
                Shape sizes[5];n=s;
                for(unsigned l=0;l<levels;++l) { sizes[l]=n;n={low(n.nx),low(n.ny),low(n.nz)}; }
                reference=original;
                for(int l=int(levels)-1;l>=0;--l)gpu_inverse_level(reference,s,sizes[l],sizeof(T)==4?16:8);
                actual=original;inverse(actual,scratch,s,levels,levels,true,2,false);
                equal(actual,reference,"GPU tiled inverse",s,levels,levels,2);
            }
        }
        const unsigned schedules[][2]={{0,0},{0,3},{2,0},{1,2},{3,1},{5,5}};
        for(const auto& schedule:schedules) {
            const unsigned xy=schedule[0],z=schedule[1];auto scalar=original,simd=original;
            forward(scalar,scratch,s,xy,z,false,2,false);forward(simd,scratch,s,xy,z,false,2,true);
            equal(scalar,simd,"scalar/SIMD plane forward",s,xy,z,2);
            scalar=original;simd=original;
            inverse(scalar,scratch,s,xy,z,false,2,false);inverse(simd,scratch,s,xy,z,false,2,true);
            equal(scalar,simd,"scalar/SIMD plane inverse",s,xy,z,2);
        }
    }
}

template<class T> void bench_type(Shape s,unsigned xy,unsigned z) {
    const std::size_t n=waltz_cpu::workspace_elements(s),bytes=n*sizeof(T);
    auto* data=static_cast<T*>(std::aligned_alloc(64,(bytes+63)&~std::size_t(63)));
    auto* scratch=static_cast<T*>(std::aligned_alloc(64,(bytes+63)&~std::size_t(63)));
    if(!data||!scratch)throw std::bad_alloc();
    const auto original=samples<T>(s);
    constexpr unsigned repeats=5;
    for(bool inv:{false,true}) {
        double total=0;std::uint64_t hash=1469598103934665603ULL;
        for(unsigned run=0;run<=repeats;++run) {
            std::memcpy(data,original.data(),bytes);
            const auto begin=std::chrono::steady_clock::now();
            if constexpr(std::is_same_v<T,float>) {
                if(inv)waltz_cpu::inverse_f32(data,scratch,s,xy,z,xy==z,2,true);
                else waltz_cpu::forward_f32(data,scratch,s,xy,z,xy==z,2,true);
            } else {
                if(inv)waltz_cpu::inverse_f64(data,scratch,s,xy,z,xy==z,2,true);
                else waltz_cpu::forward_f64(data,scratch,s,xy,z,xy==z,2,true);
            }
            const auto end=std::chrono::steady_clock::now();
            if(run)total+=std::chrono::duration<double>(end-begin).count();
        }
        const auto* p=reinterpret_cast<const unsigned char*>(data);
        for(std::size_t i=0;i<bytes;++i) { hash^=p[i];hash*=1099511628211ULL; }
        std::printf("RAW_BENCH f%zu %ux%ux%u %s ms=%.6f MiB_s=%.3f hash=%016llx\n",
                    sizeof(T)*8,s.nx,s.ny,s.nz,inv?"inverse":"forward",total*1000/repeats,
                    double(bytes)*repeats/(1048576*total),static_cast<unsigned long long>(hash));
    }
    std::free(data);std::free(scratch);
}
} // namespace

int main(int argc,char** argv) {
    try {
        if(argc==2&&!std::strcmp(argv[1],"--bench")) {
            if(!waltz_cpu::transform_has_avx2())throw std::runtime_error("AVX2/FMA unavailable");
            bench_type<float>({64,64,64},3,3);bench_type<double>({64,64,64},3,3);
            bench_type<float>({128,128,8},4,0);bench_type<double>({128,128,8},4,0);
            bench_type<float>({128,128,128},4,4);bench_type<double>({128,128,128},4,4);
            bench_type<float>({512,512,8},5,0);bench_type<double>({512,512,8},5,0);
            return 0;
        }
        test_type<float>();test_type<double>();
        std::printf("transform tests passed (AVX2/FMA available: %s)\n",waltz_cpu::transform_has_avx2()?"yes":"no");
    } catch(const std::exception& e) { std::fprintf(stderr,"%s\n",e.what());return 1; }
}
