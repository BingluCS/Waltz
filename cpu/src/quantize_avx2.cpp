#include "quantize_avx2.hpp"

#include <cfenv>
#include <cmath>
#include <cstdint>
#include <limits>

#if (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__))
#include <immintrin.h>
#define WALTZ_QUANT_X86 1
#define WALTZ_QUANT_AVX2 __attribute__((target("avx2,fma")))
#else
#define WALTZ_QUANT_X86 0
#define WALTZ_QUANT_AVX2
#endif

#if defined(__clang__)
#pragma clang fp contract(off)
#elif defined(__GNUC__)
#pragma GCC optimize ("fp-contract=off")
#endif

namespace waltz_cpu_detail {
namespace {

bool environment_ok() noexcept {
    if (std::fegetround() != FE_TONEAREST) return false;
#if WALTZ_QUANT_X86
    // Inspect only rounding, FTZ and DAZ, not sticky exception flags. The
    // enclosing codec restores all of its caller's floating-point state.
    if (_mm_getcsr() & ((3U << 13) | (1U << 15) | (1U << 6))) return false;
#endif
    return true;
}

template <class T>
bool valid_arguments(T* data, std::uint32_t n, T q, T inv, std::int32_t* codes) noexcept {
    if (!n) return true;
    return data && codes && q > T(0) && std::isfinite(q) && inv >= T(0) &&
           std::isfinite(inv) && environment_ok();
}

template <class T>
bool scalar(T* data, std::uint32_t begin, std::uint32_t n, T q, T inv,
              std::int32_t* codes) noexcept {
    for (std::uint32_t i = begin; i < n; ++i) {
        const T scaled = data[i] * inv;
        // 2^63 is exact in both native types. The strict comparison excludes
        // LLONG_MIN as well as positive overflow before integer conversion.
        if (!std::isfinite(scaled) || !(std::fabs(scaled) < T(0x1p63))) return false;
        const long long rounded = std::llrint(scaled);
        const bool negative = rounded < 0;
        const std::uint64_t magnitude = std::uint64_t(negative ? -rounded : rounded);
        const std::int32_t clipped = magnitude > std::uint64_t(INT32_MAX)
                                   ? INT32_MAX : std::int32_t(magnitude);
        const std::int32_t code = negative ? -clipped : clipped;
        codes[i] = code;
        data[i] = code ? T(code) * q : T(0);
    }
    return true;
}

#if WALTZ_QUANT_X86
WALTZ_QUANT_AVX2
bool vector_f32(float* data, std::uint32_t n, float q, float inv, std::int32_t* codes) noexcept {
    const __m256 reciprocal = _mm256_set1_ps(inv), quantum = _mm256_set1_ps(q);
    const __m256 sign_clear = _mm256_castsi256_ps(_mm256_set1_epi32(INT32_MAX));
    const __m256 int64_limit = _mm256_set1_ps(0x1p63f);
    // INT32_MAX rounds to 2^31 in FP32, which cvtps_epi32 cannot represent.
    // Clip to the largest *safe representable float* before conversion, then
    // replace larger magnitudes with exact int32 saturation in integer lanes.
    const __m256 safe_int32 = _mm256_set1_ps(0x1.fffffep30f); // 2147483520
    const __m256i saturation = _mm256_set1_epi32(INT32_MAX);
    const std::uint32_t end = n & ~std::uint32_t(7);
    std::uint32_t i = 0;
    for (; i < end; i += 8) {
        const __m256 scaled = _mm256_mul_ps(_mm256_loadu_ps(data + i), reciprocal);
        const __m256 magnitude = _mm256_and_ps(scaled, sign_clear);
        if (_mm256_movemask_ps(_mm256_cmp_ps(magnitude, int64_limit, _CMP_LT_OQ)) != 255) return false;
        const __m256 large = _mm256_cmp_ps(magnitude, safe_int32, _CMP_GT_OQ);
        const __m256 clipped = _mm256_min_ps(magnitude, safe_int32);
        // MXCSR nearest-even matches llrint of the already native-rounded
        // product. All converted lanes are <=2147483520, so none overflow.
        __m256i mag = _mm256_cvtps_epi32(clipped);
        mag = _mm256_blendv_epi8(mag, saturation, _mm256_castps_si256(large));
        const __m256i negative = _mm256_srai_epi32(_mm256_castps_si256(scaled), 31);
        const __m256i code = _mm256_sub_epi32(_mm256_xor_si256(mag, negative), negative);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(codes + i), code);
        // Integer zero converts to +0 even when the source was -0 or rounded
        // from a negative half tie. q is positive, so the multiply retains +0.
        _mm256_storeu_ps(data + i, _mm256_mul_ps(_mm256_cvtepi32_ps(code), quantum));
    }
    return scalar(data, i, n, q, inv, codes);
}

WALTZ_QUANT_AVX2
bool vector_f64(double* data, std::uint32_t n, double q, double inv, std::int32_t* codes) noexcept {
    const __m256d reciprocal = _mm256_set1_pd(inv), quantum = _mm256_set1_pd(q);
    const __m256d sign_clear = _mm256_castsi256_pd(_mm256_set1_epi64x(INT64_MAX));
    const __m256d int64_limit = _mm256_set1_pd(0x1p63);
    const __m256d low = _mm256_set1_pd(-double(INT32_MAX));
    const __m256d high = _mm256_set1_pd(double(INT32_MAX));
    const std::uint32_t end = n & ~std::uint32_t(3);
    std::uint32_t i = 0;
    for (; i < end; i += 4) {
        const __m256d scaled = _mm256_mul_pd(_mm256_loadu_pd(data + i), reciprocal);
        const __m256d magnitude = _mm256_and_pd(scaled, sign_clear);
        if (_mm256_movemask_pd(_mm256_cmp_pd(magnitude, int64_limit, _CMP_LT_OQ)) != 15) return false;
        // +/-INT32_MAX are exactly representable as doubles. Clipping before
        // nearest-even conversion is equivalent to rounding then saturating,
        // and excludes both positive int32 overflow and the unwanted INT_MIN.
        const __m256d clipped = _mm256_max_pd(low, _mm256_min_pd(scaled, high));
        const __m128i code = _mm256_cvtpd_epi32(clipped);
        _mm_storeu_si128(reinterpret_cast<__m128i*>(codes + i), code);
        _mm256_storeu_pd(data + i, _mm256_mul_pd(_mm256_cvtepi32_pd(code), quantum));
    }
    return scalar(data, i, n, q, inv, codes);
}
#endif
} // namespace

bool quantize_has_avx2() noexcept {
#if WALTZ_QUANT_X86
    __builtin_cpu_init();
    return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
#else
    return false;
#endif
}

bool quantize_f32(float* data, std::uint32_t n, float q, float inv, std::int32_t* codes) noexcept {
    if (!valid_arguments(data, n, q, inv, codes)) return false;
    if (!n) return true;
#if WALTZ_QUANT_X86
    if (quantize_has_avx2()) return vector_f32(data, n, q, inv, codes);
#endif
    return scalar(data, 0, n, q, inv, codes);
}

bool quantize_f64(double* data, std::uint32_t n, double q, double inv, std::int32_t* codes) noexcept {
    if (!valid_arguments(data, n, q, inv, codes)) return false;
    if (!n) return true;
#if WALTZ_QUANT_X86
    if (quantize_has_avx2()) return vector_f64(data, n, q, inv, codes);
#endif
    return scalar(data, 0, n, q, inv, codes);
}
template<class T> bool quantize_morton(T* data,uint32_t row,size_t plane,
 uint32_t nx,uint32_t ny,uint32_t nz,const uint16_t* rank,size_t base,
 T q,T inv,uint16_t* mag,uint32_t* sign,uint32_t* qi,uint32_t* qv,
 uint32_t& count,size_t origin) noexcept {
    alignas(32) int32_t codes[16];
    if(nx>16||!valid_arguments(data,nx,q,inv,codes))return false;
    const bool avx=quantize_has_avx2();
    for(uint32_t z=0;z<nz;++z)for(uint32_t y=0;y<ny;++y){
        const size_t at=size_t(z)*plane+size_t(y)*row;
        bool ok;
#if WALTZ_QUANT_X86
        if(avx){
            if constexpr(sizeof(T)==4)ok=vector_f32(data+at,nx,q,inv,codes);
            else ok=vector_f64(data+at,nx,q,inv,codes);
        }else
#endif
        ok=scalar(data+at,0,nx,q,inv,codes);
        if(!ok)return false;
        for(uint32_t x=0;x<nx;++x){
            const int32_t v=codes[x];const uint32_t m=v<0?uint32_t(-v):uint32_t(v);
            const size_t packed=base+rank[(z*ny+y)*nx+x];
            mag[packed]=uint16_t(m>65535?65535:m);
            if(v<0)sign[packed/32]|=1U<<(packed%32);
            if(m>65535){qi[count]=uint32_t(origin+at+x);qv[count++]=m;}
        }
    }
    return true;
}
#define MORTON_WRAPPER(S,T) \
bool quantize_morton_##S(T* d,uint32_t row,size_t plane,uint32_t nx,uint32_t ny,uint32_t nz, \
 const uint16_t* r,size_t base,T q,T inv,uint16_t* m,uint32_t* s,uint32_t* qi,uint32_t* qv, \
 uint32_t& count,size_t origin) noexcept { \
 return quantize_morton(d,row,plane,nx,ny,nz,r,base,q,inv,m,s,qi,qv,count,origin); }
MORTON_WRAPPER(f32,float)
MORTON_WRAPPER(f64,double)
#undef MORTON_WRAPPER
} // namespace waltz_cpu_detail
