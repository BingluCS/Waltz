#include "quantize_avx2.hpp"

#include <cfenv>
#include <cmath>
#include <cstdint>
#include <cstring>
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

#if WALTZ_QUANT_X86
namespace {
// Full Morton rows are 16 samples wide. At ties, nearest-even produces +0.
WALTZ_QUANT_AVX2
bool clear_zero_row(float* data, float inv) noexcept {
    const auto scale = _mm256_set1_ps(inv), half = _mm256_set1_ps(.5f);
    const auto mask = _mm256_castsi256_ps(_mm256_set1_epi32(INT32_MAX));
    auto a = _mm256_and_ps(mask, _mm256_mul_ps(_mm256_loadu_ps(data), scale));
    auto b = _mm256_and_ps(mask, _mm256_mul_ps(_mm256_loadu_ps(data+8), scale));
    auto ok = _mm256_and_ps(_mm256_cmp_ps(a, half, _CMP_LE_OQ), _mm256_cmp_ps(b, half, _CMP_LE_OQ));
    if (_mm256_movemask_ps(ok) != 255) return false;
    _mm256_storeu_ps(data, _mm256_setzero_ps());
    _mm256_storeu_ps(data+8, _mm256_setzero_ps());
    return true;
}

// Return false for the uncommon wide-magnitude row; its exceptions are
// recorded by the ordinary loop. No exception or sign is emitted for zero.
WALTZ_QUANT_AVX2
bool scatter_narrow(const int32_t* codes,const uint16_t* rank,uint16_t* mag,
                    uint32_t* sign,size_t base) noexcept {
    const __m256i a=_mm256_loadu_si256(reinterpret_cast<const __m256i*>(codes));
    const __m256i b=_mm256_loadu_si256(reinterpret_cast<const __m256i*>(codes+8));
    const __m256i aa=_mm256_abs_epi32(a),ab=_mm256_abs_epi32(b);
    const __m256i limit=_mm256_set1_epi32(65535);
    if(!_mm256_testz_si256(_mm256_or_si256(aa,ab),_mm256_xor_si256(limit,_mm256_set1_epi32(-1))))return false;
    const __m256i zero=_mm256_setzero_si256();
    unsigned nz=unsigned(_mm256_movemask_ps(_mm256_castsi256_ps(_mm256_cmpgt_epi32(aa,zero))))
               |(unsigned(_mm256_movemask_ps(_mm256_castsi256_ps(_mm256_cmpgt_epi32(ab,zero))))<<8);
    unsigned neg=unsigned(_mm256_movemask_ps(_mm256_castsi256_ps(a)))
               |(unsigned(_mm256_movemask_ps(_mm256_castsi256_ps(b)))<<8);
    for(;nz;nz&=nz-1){unsigned x=__builtin_ctz(nz);int32_t v=codes[x];mag[rank[x]]=uint16_t(v<0?-v:v);}
    for(;neg;neg&=neg-1){unsigned x=__builtin_ctz(neg);size_t bit=base+rank[x];sign[bit/32]|=1U<<(bit%32);}
    return true;
}
} // namespace
#endif

// Dispatch is performed once by the outer encoder; scalar and FP64 keep the
// original traversal. Partial brick rows use the same quantizer with its tail.
WALTZ_QUANT_AVX2
bool quantize_morton_f32(float* data,waltz_cpu::Shape field,waltz_cpu::Shape b,
                         const uint16_t* rank,uint16_t* mag,uint32_t* sign,
                         size_t origin,size_t base,float q,float inv,
                         uint32_t* indices,uint32_t* values,uint32_t& count) noexcept {
#if WALTZ_QUANT_X86
    const size_t plane=size_t(field.nx)*field.ny;
    std::memset(mag,0,size_t(b.nx)*b.ny*b.nz*sizeof(*mag));
    for(uint32_t z=0;z<b.nz;++z)for(uint32_t y=0;y<b.ny;++y,rank+=b.nx){
        const size_t at=z*plane+size_t(y)*field.nx;
        float* row=data+at;
        if(b.nx==16 && clear_zero_row(row,inv))continue;
        int32_t codes[16];
        if(!vector_f32(row,b.nx,q,inv,codes))return false;
        if(b.nx==16 && scatter_narrow(codes,rank,mag,sign,base))continue;
        for(uint32_t x=0;x<b.nx;++x){
            const int32_t code=codes[x];
            if(!code)continue;
            const uint32_t magnitude=code<0?-code:code, r=rank[x];
            mag[r]=uint16_t(magnitude>65535?65535:magnitude);
            if(code<0){const size_t bit=base+r;sign[bit/32]|=1U<<(bit%32);}
            if(magnitude>65535){indices[count]=uint32_t(origin+at+x);values[count++]=magnitude;}
        }
    }
    return true;
#else
    return false; // Unreachable: quantize_has_avx2() is false on this build.
#endif
}

} // namespace waltz_cpu_detail
