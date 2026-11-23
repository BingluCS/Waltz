#include "transform.hpp"

#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <type_traits>

#if (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__))
#include <immintrin.h>
#define WALTZ_CPU_X86_TARGET 1
#define WALTZ_CPU_AVX2 __attribute__((target("avx2,fma")))
#else
#define WALTZ_CPU_X86_TARGET 0
#define WALTZ_CPU_AVX2
#endif

#if defined(__clang__)
#define WALTZ_CPU_UNROLL _Pragma("clang loop unroll(full)")
#elif defined(__GNUC__)
#define WALTZ_CPU_UNROLL _Pragma("GCC unroll 16")
#else
#define WALTZ_CPU_UNROLL
#endif

// This arithmetic profile matches the production CUDA 13.1 default-fmad PTX.
// Besides warp97's explicit madd, NVCC contracts two one-use halo endpoint
// normalizations into their following sum (see halo's folded_endpoints).
// Keep those contractions explicit here; build without implicit contraction
// or fast-math. std::fma keeps the scalar path correctly rounded even when
// the baseline architecture does not have a hardware FMA.
#if defined(__clang__)
#pragma clang fp contract(off)
#elif defined(__GNUC__)
#pragma GCC optimize ("fp-contract=off")
#endif

namespace waltz_cpu {
namespace {

template<class T> struct Constants {
    static constexpr T a = T(-1.586134342059924);
    static constexpr T b = T(-0.052980118572961);
    static constexpr T g = T(0.882911075530934);
    static constexpr T d = T(0.443506852043971);
    static constexpr T k = T(1.1496043988602418);
    static constexpr T ik = T(1) / k;
};

enum Axis { X, Y, Z };

std::uint32_t half(std::uint32_t n) { return n / 2 + n % 2; }

Shape shrink(Shape n, bool xy, bool z) {
    if (xy) { n.nx = half(n.nx); n.ny = half(n.ny); }
    if (z) n.nz = half(n.nz);
    return n;
}

std::uint32_t reflect(std::int64_t p, std::uint32_t n) {
    if (n <= 1) return 0;
    const std::int64_t period = 2 * std::int64_t(n) - 2;
    std::int64_t r = p % period;
    if (r < 0) r += period;
    return std::uint32_t(r < n ? r : period - r);
}

std::uint32_t coefficient(std::int64_t p, std::uint32_t n) {
    const std::uint32_t r = reflect(p, n);
    return r / 2 + (r % 2 ? half(n) : 0);
}

// SIMD tiles reach at most 59 samples beyond an axis. Most reads are interior;
// axes >=64 need at most one reflection, avoiding per-vector integer division.
// Keep the scalar reference's periodic address calculation independent.
template<bool Inverse>
std::uint32_t simd_address(std::int64_t p,std::uint32_t n) {
    std::uint32_t r;
    if(std::uint64_t(p)<n)r=std::uint32_t(p);
    else if(n>=64) {
        const std::int64_t u=p<0?-p:p;
        r=std::uint32_t(u<n?u:2*std::int64_t(n)-2-u);
    } else r=reflect(p,n);
    if constexpr(Inverse)return r/2+(r%2?half(n):0);
    else return r;
}

std::uint32_t length(Shape n, Axis axis) {
    return axis == X ? n.nx : axis == Y ? n.ny : n.nz;
}

std::size_t line_count(Shape n, Axis axis) {
    return axis == X ? std::size_t(n.ny) * n.nz
         : axis == Y ? std::size_t(n.nx) * n.nz
                     : std::size_t(n.nx) * n.ny;
}

std::size_t line_stride(Shape full, Axis axis) {
    return axis == X ? 1 : axis == Y ? full.nx : std::size_t(full.nx) * full.ny;
}

std::size_t line_offset(Shape full, Shape active, Axis axis, std::size_t line) {
    if (axis == X)
        return (line / active.ny * full.ny + line % active.ny) * full.nx;
    if (axis == Y)
        return line / active.nx * std::size_t(full.nx) * full.ny + line % active.nx;
    return line / active.nx * full.nx + line % active.nx;
}

// These restricted loop bounds intentionally match the *local halo* lifting
// schedules, rather than a conventional whole-line CDF boundary treatment.
// In particular an odd or length-one axis must reflect the original spatial
// coordinate before looking up its split low/high coefficient.
template<class T, bool Inverse>
void halo(T* e, T* o, int pairs, bool folded_endpoints) {
    using C = Constants<T>;
    if constexpr (Inverse) {
        const T first_odd=o[0],last_odd=o[pairs-1];
        for (int j = 0; j < pairs; ++j) { e[j] *= C::ik; o[j] *= -C::k; }
        for (int j = 1; j < pairs; ++j) {
            // In the unrolled CUDA Y/Z halo, o[0] and o[pairs-1] are each
            // consumed once. Production PTX therefore uses fma(raw,-K,
            // scaled_neighbor), not add(mul(raw,-K),scaled_neighbor).
            // Warp-shuffled X has no such one-use values and must not fold.
            const T sum=folded_endpoints&&j==1?std::fma(first_odd,-C::k,o[1]):
                        folded_endpoints&&j==pairs-1?std::fma(last_odd,-C::k,o[pairs-2]):
                        o[j-1]+o[j];
            e[j] = std::fma(-C::d, sum, e[j]);
        }
        for (int j = 1; j < pairs-1; ++j)
            o[j] = std::fma(-C::g, e[j] + e[j+1], o[j]);
        for (int j = 2; j < pairs-1; ++j)
            e[j] = std::fma(-C::b, o[j-1] + o[j], e[j]);
        for (int j = 2; j < pairs-2; ++j)
            o[j] = std::fma(-C::a, e[j] + e[j+1], o[j]);
    } else {
        for (int j = 0; j < pairs-1; ++j)
            o[j] = std::fma(C::a, e[j] + e[j+1], o[j]);
        for (int j = 1; j < pairs-1; ++j)
            e[j] = std::fma(C::b, o[j-1] + o[j], e[j]);
        for (int j = 1; j < pairs-2; ++j)
            o[j] = std::fma(C::g, e[j] + e[j+1], o[j]);
        for (int j = 2; j < pairs-2; ++j)
            e[j] = std::fma(C::d, o[j-1] + o[j], e[j]);
        for (int j = 2; j < pairs-2; ++j) { e[j] *= C::k; o[j] *= -C::ik; }
    }
}

template<class T, bool Inverse>
void axis_scalar(T* data, T* scratch, Shape full, Shape active, Axis axis, int tile) {
    const std::uint32_t n = length(active, axis);
    const std::size_t count = line_count(active, axis), stride = line_stride(full, axis);
    const int pairs = tile / 2 + 4;
    for (std::size_t line = 0; line < count; ++line) {
        T* const destination = data + line_offset(full, active, axis, line);
        for (std::uint32_t p = 0; p < n; ++p) scratch[p] = destination[std::size_t(p) * stride];
        for (std::uint64_t start = 0; start < n; start += unsigned(tile)) {
            T e[32], o[32];
            for (int j = 0; j < pairs; ++j) {
                const std::int64_t p = std::int64_t(start) - 4 + 2*j;
                const auto a = Inverse ? coefficient(p, n) : reflect(p, n);
                const auto b = Inverse ? coefficient(p+1, n) : reflect(p+1, n);
                e[j] = scratch[a]; o[j] = scratch[b];
            }
            halo<T, Inverse>(e, o, pairs, axis!=X);
            for (int j = 2; j < pairs-2; ++j) {
                const std::uint64_t p = start + 2*unsigned(j-2);
                if (p < n) {
                    const std::size_t at = Inverse ? std::size_t(p) : std::size_t(p/2);
                    destination[at*stride] = e[j];
                }
                if (p+1 < n) {
                    const std::size_t at = Inverse ? std::size_t(p+1) : std::size_t(half(n)+p/2);
                    destination[at*stride] = o[j];
                }
            }
        }
    }
}

#if WALTZ_CPU_X86_TARGET
template<class T> struct Simd;

template<> struct Simd<float> {
    using V = __m256;
    static constexpr unsigned width = 8;
    WALTZ_CPU_AVX2 static V load(const float* p) { return _mm256_loadu_ps(p); }
    WALTZ_CPU_AVX2 static void store(float* p, V x) { _mm256_storeu_ps(p, x); }
    WALTZ_CPU_AVX2 static V set(float x) { return _mm256_set1_ps(x); }
    WALTZ_CPU_AVX2 static V add(V x, V y) { return _mm256_add_ps(x,y); }
    WALTZ_CPU_AVX2 static V mul(V x, V y) { return _mm256_mul_ps(x,y); }
    WALTZ_CPU_AVX2 static V fma(V x, V y, V z) { return _mm256_fmadd_ps(x,y,z); }
};

template<> struct Simd<double> {
    using V = __m256d;
    static constexpr unsigned width = 4;
    WALTZ_CPU_AVX2 static V load(const double* p) { return _mm256_loadu_pd(p); }
    WALTZ_CPU_AVX2 static void store(double* p, V x) { _mm256_storeu_pd(p, x); }
    WALTZ_CPU_AVX2 static V set(double x) { return _mm256_set1_pd(x); }
    WALTZ_CPU_AVX2 static V add(V x, V y) { return _mm256_add_pd(x,y); }
    WALTZ_CPU_AVX2 static V mul(V x, V y) { return _mm256_mul_pd(x,y); }
    WALTZ_CPU_AVX2 static V fma(V x, V y, V z) { return _mm256_fmadd_pd(x,y,z); }
};

// Register-only square transpose: no arithmetic, no rounding changes.
WALTZ_CPU_AVX2 inline void square_transpose(__m256* v) {
    __m256 t[8],u[8];
    for(unsigned i=0;i<4;++i){t[2*i]=_mm256_unpacklo_ps(v[2*i],v[2*i+1]);t[2*i+1]=_mm256_unpackhi_ps(v[2*i],v[2*i+1]);}
    u[0]=_mm256_shuffle_ps(t[0],t[2],0x44);u[1]=_mm256_shuffle_ps(t[0],t[2],0xee);
    u[2]=_mm256_shuffle_ps(t[1],t[3],0x44);u[3]=_mm256_shuffle_ps(t[1],t[3],0xee);
    u[4]=_mm256_shuffle_ps(t[4],t[6],0x44);u[5]=_mm256_shuffle_ps(t[4],t[6],0xee);
    u[6]=_mm256_shuffle_ps(t[5],t[7],0x44);u[7]=_mm256_shuffle_ps(t[5],t[7],0xee);
    for(unsigned i=0;i<4;++i){v[i]=_mm256_permute2f128_ps(u[i],u[i+4],0x20);v[i+4]=_mm256_permute2f128_ps(u[i],u[i+4],0x31);}
}
WALTZ_CPU_AVX2 inline void square_transpose(__m256d* v) {
    const auto a=_mm256_unpacklo_pd(v[0],v[1]),b=_mm256_unpackhi_pd(v[0],v[1]);
    const auto c=_mm256_unpacklo_pd(v[2],v[3]),d=_mm256_unpackhi_pd(v[2],v[3]);
    v[0]=_mm256_permute2f128_pd(a,c,0x20);v[1]=_mm256_permute2f128_pd(b,d,0x20);
    v[2]=_mm256_permute2f128_pd(a,c,0x31);v[3]=_mm256_permute2f128_pd(b,d,0x31);
}
template<class T>
WALTZ_CPU_AVX2 inline void scatter_x(const typename Simd<T>::V* e,const typename Simd<T>::V* o,
                                    bool interleave,unsigned count,T* const* rows,size_t at) {
    using S=Simd<T>;using V=typename S::V;constexpr unsigned W=S::width;
    unsigned j=0;
    for(;j+W<=count;j+=W){V v[W];
        for(unsigned k=0;k<W;++k){unsigned p=j+k;v[k]=interleave?(p&1?o[p/2]:e[p/2]):e[p];}
        square_transpose(v);for(unsigned k=0;k<W;++k)S::store(rows[k]+at+j,v[k]);
    }
    for(;j<count;++j){T v[W];S::store(v,interleave?(j&1?o[j/2]:e[j/2]):e[j]);for(unsigned k=0;k<W;++k)rows[k][at+j]=v[k];}
}

template<class T, bool Inverse,Axis axis,int tile>
WALTZ_CPU_AVX2 void axis_simd(T* data, T* scratch, Shape full, Shape active) {
    using S = Simd<T>;
    using V = typename S::V;
    using C = Constants<T>;
    constexpr unsigned width = S::width;
    const std::uint32_t n = length(active, axis);
    const std::size_t count = line_count(active, axis), stride = line_stride(full, axis);
    constexpr int pairs = tile / 2 + 4;
    const V a = S::set(Inverse ? -C::a : C::a), b = S::set(Inverse ? -C::b : C::b);
    const V g = S::set(Inverse ? -C::g : C::g), d = S::set(Inverse ? -C::d : C::d);
    const V kl = S::set(Inverse ? C::ik : C::k), kh = S::set(Inverse ? -C::k : -C::ik);
    std::size_t line = 0;
    for (; count-line >= width; line += width) {
        T* destination[width];
        for (unsigned lane = 0; lane < width; ++lane)
            destination[lane] = data + line_offset(full, active, axis, line+lane);
        const bool contiguous=axis!=X && line%active.nx+width<=active.nx;
        // Staging a complete group makes all halo reads immutable and all SIMD
        // loads contiguous. At most width*n <= full volume elements are used.
        if(contiguous) {
            for(std::uint32_t p=0;p<n;++p)
                S::store(scratch+std::size_t(p)*width,S::load(destination[0]+std::size_t(p)*stride));
        } else if constexpr(axis==X) {
            std::uint32_t p=0;
            for(;p+width<=n;p+=width){V v[width];
                for(unsigned k=0;k<width;++k)v[k]=S::load(destination[k]+p);
                square_transpose(v);for(unsigned k=0;k<width;++k)S::store(scratch+std::size_t(p+k)*width,v[k]);
            }
            for(;p<n;++p)for(unsigned k=0;k<width;++k)scratch[std::size_t(p)*width+k]=destination[k][p];
        } else {
            for (std::uint32_t p = 0; p < n; ++p)
                for (unsigned lane = 0; lane < width; ++lane)
                    scratch[std::size_t(p)*width+lane] = destination[lane][std::size_t(p)*stride];
        }
        for (std::uint64_t start = 0; start < n; start += unsigned(tile)) {
            V e[32], o[32];
            WALTZ_CPU_UNROLL
            for (int j = 0; j < pairs; ++j) {
                const std::int64_t p = std::int64_t(start)-4+2*j;
                const auto ia = simd_address<Inverse>(p,n);
                const auto ib = simd_address<Inverse>(p+1,n);
                e[j] = S::load(scratch+std::size_t(ia)*width);
                o[j] = S::load(scratch+std::size_t(ib)*width);
            }
            if constexpr (Inverse) {
                const V first_odd=o[0],last_odd=o[pairs-1];
                WALTZ_CPU_UNROLL
                for (int j=0;j<pairs;++j) { e[j]=S::mul(e[j],kl); o[j]=S::mul(o[j],kh); }
                WALTZ_CPU_UNROLL
                for (int j=1;j<pairs;++j) {
                    const V sum=axis!=X&&j==1?S::fma(first_odd,kh,o[1]):
                                axis!=X&&j==pairs-1?S::fma(last_odd,kh,o[pairs-2]):
                                S::add(o[j-1],o[j]);
                    e[j]=S::fma(d,sum,e[j]);
                }
                WALTZ_CPU_UNROLL
                for (int j=1;j<pairs-1;++j) o[j]=S::fma(g,S::add(e[j],e[j+1]),o[j]);
                WALTZ_CPU_UNROLL
                for (int j=2;j<pairs-1;++j) e[j]=S::fma(b,S::add(o[j-1],o[j]),e[j]);
                WALTZ_CPU_UNROLL
                for (int j=2;j<pairs-2;++j) o[j]=S::fma(a,S::add(e[j],e[j+1]),o[j]);
            } else {
                WALTZ_CPU_UNROLL
                for (int j=0;j<pairs-1;++j) o[j]=S::fma(a,S::add(e[j],e[j+1]),o[j]);
                WALTZ_CPU_UNROLL
                for (int j=1;j<pairs-1;++j) e[j]=S::fma(b,S::add(o[j-1],o[j]),e[j]);
                WALTZ_CPU_UNROLL
                for (int j=1;j<pairs-2;++j) o[j]=S::fma(g,S::add(e[j],e[j+1]),o[j]);
                WALTZ_CPU_UNROLL
                for (int j=2;j<pairs-2;++j) e[j]=S::fma(d,S::add(o[j-1],o[j]),e[j]);
                WALTZ_CPU_UNROLL
                for (int j=2;j<pairs-2;++j) { e[j]=S::mul(e[j],kl); o[j]=S::mul(o[j],kh); }
            }
            if constexpr(axis==X) {
                const unsigned remaining=unsigned(std::min<uint64_t>(tile,n-start));
                if constexpr(Inverse)scatter_x<T>(e+2,o+2,true,remaining,destination,size_t(start));
                else {scatter_x<T>(e+2,o+2,false,(remaining+1)/2,destination,size_t(start/2));
                      scatter_x<T>(o+2,e+2,false,remaining/2,destination,size_t(half(n)+start/2));}
                continue;
            }
            T values[width];
            WALTZ_CPU_UNROLL
            for (int j=2;j<pairs-2;++j) {
                const std::uint64_t p=start+2*unsigned(j-2);
                if (p<n) {
                    const std::size_t at=Inverse?std::size_t(p):std::size_t(p/2);
                    if(contiguous)S::store(destination[0]+at*stride,e[j]);
                    else {
                        S::store(values,e[j]);
                        for (unsigned lane=0;lane<width;++lane) destination[lane][at*stride]=values[lane];
                    }
                }
                if (p+1<n) {
                    const std::size_t at=Inverse?std::size_t(p+1):std::size_t(half(n)+p/2);
                    if(contiguous)S::store(destination[0]+at*stride,o[j]);
                    else {
                        S::store(values,o[j]);
                        for (unsigned lane=0;lane<width;++lane) destination[lane][at*stride]=values[lane];
                    }
                }
            }
        }
    }
    // The last incomplete group uses exactly the same scalar arithmetic.
    for (;line<count;++line) {
        T* const destination=data+line_offset(full,active,axis,line);
        for (std::uint32_t p=0;p<n;++p) scratch[p]=destination[std::size_t(p)*stride];
        for (std::uint64_t start=0;start<n;start+=unsigned(tile)) {
            T e[32],o[32];
            for (int j=0;j<pairs;++j) {
                const std::int64_t p=std::int64_t(start)-4+2*j;
                e[j]=scratch[Inverse?coefficient(p,n):reflect(p,n)];
                o[j]=scratch[Inverse?coefficient(p+1,n):reflect(p+1,n)];
            }
            halo<T,Inverse>(e,o,pairs,axis!=X);
            for (int j=2;j<pairs-2;++j) {
                const std::uint64_t p=start+2*unsigned(j-2);
                if(p<n)destination[(Inverse?std::size_t(p):std::size_t(p/2))*stride]=e[j];
                if(p+1<n)destination[(Inverse?std::size_t(p+1):std::size_t(half(n)+p/2))*stride]=o[j];
            }
        }
    }
}
#endif

// Instantiate the same line schedule twice: genuinely baseline scalar code
// (including its lambdas), and separately runtime-dispatched AVX2/FMA code.
#define WALTZ_LINE_SCALAR 1
#define WALTZ_LINE_TARGET
#define whole_line whole_line_scalar
#define compatibility_edges compatibility_edges_scalar
#include "transform_nohalo.inc"
#undef compatibility_edges
#undef whole_line
#undef WALTZ_LINE_TARGET
#undef WALTZ_LINE_SCALAR
#if WALTZ_CPU_X86_TARGET
#define WALTZ_LINE_SCALAR 0
#define WALTZ_LINE_TARGET WALTZ_CPU_AVX2
#include "transform_nohalo.inc"
#undef WALTZ_LINE_TARGET
#undef WALTZ_LINE_SCALAR
#endif

template<class T, bool Inverse>
void axis_transform(T* data,T* scratch,Shape full,Shape active,Axis axis,int tile,bool avx2) {
#if WALTZ_CPU_NOHALO
#if WALTZ_CPU_X86_TARGET
    if(avx2){axis_whole_simd<T,Inverse>(data,scratch,full,active,axis,tile);return;}
#endif
    axis_whole_scalar<T,Inverse>(data,scratch,full,active,axis,tile);return;
#endif
#if WALTZ_CPU_X86_TARGET
    if(avx2) {
        if(axis==X)axis_simd<T,Inverse,X,56>(data,scratch,full,active);
        else if(axis==Y)axis_simd<T,Inverse,Y,8>(data,scratch,full,active);
        else if(tile==16)axis_simd<T,Inverse,Z,16>(data,scratch,full,active);
        else if(tile==8)axis_simd<T,Inverse,Z,8>(data,scratch,full,active);
        else axis_simd<T,Inverse,Z,4>(data,scratch,full,active);
        return;
    }
#else
    (void)avx2;
#endif
    axis_scalar<T,Inverse>(data,scratch,full,active,axis,tile);
}

template<class T>
void validate(T* data,T* scratch,Shape shape,unsigned lxy,unsigned lz,bool dyadic,unsigned synthesis) {
    (void)workspace_elements(shape);
    if(!data || !scratch || data==scratch)throw std::invalid_argument("Waltz CPU: nonoverlapping data and scratch are required");
    if(lxy>5 || lz>5 || (dyadic && lxy!=lz))throw std::invalid_argument("Waltz CPU: invalid transform levels");
    if(synthesis==0 || synthesis==1)throw std::invalid_argument("Waltz CPU: convolution synthesis modes 0/1 are not implemented");
    if(synthesis!=2)throw std::invalid_argument("Waltz CPU: unsupported synthesis mode");
}

template<class T,bool Inverse>
void transform(T* data,T* scratch,Shape full,unsigned lxy,unsigned lz,bool dyadic,unsigned synthesis,bool avx2) {
    validate(data,scratch,full,lxy,lz,dyadic,synthesis);
    avx2=avx2&&transform_has_avx2();
    constexpr int ztile=sizeof(T)==4?16:8;
    if(dyadic) {
        Shape sizes[5],active=full;
        for(unsigned l=0;l<lxy;++l) { sizes[l]=active;active=shrink(active,true,true); }
        for(unsigned step=0;step<lxy;++step) {
            const unsigned level=Inverse?lxy-1-step:step;
            active=sizes[level];
            if constexpr(Inverse) {
                // Reproduce the native GPU Z tile's endpoint arithmetic.
                axis_transform<T,true>(data,scratch,full,active,Z,ztile,avx2);
                axis_transform<T,true>(data,scratch,full,active,X,56,avx2);
                axis_transform<T,true>(data,scratch,full,active,Y,8,avx2);
            } else {
                axis_transform<T,false>(data,scratch,full,active,Y,8,avx2);
                axis_transform<T,false>(data,scratch,full,active,X,56,avx2);
                constexpr int forward_ztile=sizeof(T)==4?8:4;
                axis_transform<T,false>(data,scratch,full,active,Z,forward_ztile,avx2);
            }
        }
        return;
    }
    const auto xy_levels=[&]() {
        Shape sizes[5],active=full;
        for(unsigned l=0;l<lxy;++l) { sizes[l]=active;active=shrink(active,true,false); }
        for(unsigned step=0;step<lxy;++step) {
            active=sizes[Inverse?lxy-1-step:step];
            if constexpr(Inverse) {
                axis_transform<T,true>(data,scratch,full,active,X,56,avx2);
                axis_transform<T,true>(data,scratch,full,active,Y,8,avx2);
            } else {
                axis_transform<T,false>(data,scratch,full,active,Y,8,avx2);
                axis_transform<T,false>(data,scratch,full,active,X,56,avx2);
            }
        }
    };
    const auto z_levels=[&]() {
        Shape sizes[5],active=full;
        for(unsigned l=0;l<lz;++l) { sizes[l]=active;active=shrink(active,false,true); }
        for(unsigned step=0;step<lz;++step)
            axis_transform<T,Inverse>(data,scratch,full,sizes[Inverse?lz-1-step:step],Z,ztile,avx2);
    };
    if constexpr(Inverse) { xy_levels();z_levels(); }
    else { z_levels();xy_levels(); }
}

} // namespace

std::size_t workspace_elements(Shape shape) {
    if(!shape.nx || !shape.ny || !shape.nz)throw std::invalid_argument("Waltz CPU: dimensions must be positive");
    const std::size_t maximum=std::numeric_limits<std::size_t>::max();
    if(std::size_t(shape.nx)>maximum/shape.ny || std::size_t(shape.nx)*shape.ny>maximum/shape.nz)
        throw std::overflow_error("Waltz CPU: transform shape overflow");
    const size_t volume=std::size_t(shape.nx)*shape.ny*shape.nz;
    const size_t longest=std::max(shape.nx,std::max(shape.ny,shape.nz));
    if(longest>maximum/32)throw std::overflow_error("Waltz CPU: line scratch overflow");
    return std::max(volume,longest*32);
}

bool transform_has_avx2() noexcept {
#if WALTZ_CPU_X86_TARGET
    // GCC/Clang's builtin also checks OS XSAVE support before reporting AVX.
    __builtin_cpu_init();
    return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
#else
    return false;
#endif
}

void forward_f32(float* data,float* scratch,Shape s,std::uint8_t xy,std::uint8_t z,bool d,std::uint8_t mode,bool avx2) {
    transform<float,false>(data,scratch,s,xy,z,d,mode,avx2);
}
void forward_f64(double* data,double* scratch,Shape s,std::uint8_t xy,std::uint8_t z,bool d,std::uint8_t mode,bool avx2) {
    transform<double,false>(data,scratch,s,xy,z,d,mode,avx2);
}
void inverse_f32(float* data,float* scratch,Shape s,std::uint8_t xy,std::uint8_t z,bool d,std::uint8_t mode,bool avx2) {
    transform<float,true>(data,scratch,s,xy,z,d,mode,avx2);
}
void inverse_f64(double* data,double* scratch,Shape s,std::uint8_t xy,std::uint8_t z,bool d,std::uint8_t mode,bool avx2) {
    transform<double,true>(data,scratch,s,xy,z,d,mode,avx2);
}

} // namespace waltz_cpu
