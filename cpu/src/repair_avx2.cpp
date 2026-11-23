#include "repair_avx2.hpp"

#include <cfenv>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>

#if (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__))
#include <immintrin.h>
#define WALTZ_REPAIR_X86 1
#define WALTZ_REPAIR_AVX2 __attribute__((target("avx2,fma")))
#else
#define WALTZ_REPAIR_X86 0
#define WALTZ_REPAIR_AVX2
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
#if WALTZ_REPAIR_X86
    if (_mm_getcsr() & ((3U << 13) | (1U << 15) | (1U << 6))) return false;
#endif
    return true;
}

template <class T>
bool finite_one(T value) noexcept {
    if constexpr (sizeof(T) == 4) {
        std::uint32_t bits;
        std::memcpy(&bits, &value, sizeof(bits));
        return (bits & UINT32_C(0x7f800000)) != UINT32_C(0x7f800000);
    } else {
        std::uint64_t bits;
        std::memcpy(&bits, &value, sizeof(bits));
        return (bits & UINT64_C(0x7ff0000000000000)) != UINT64_C(0x7ff0000000000000);
    }
}

bool within_bound(double value, double original, double bound) noexcept {
    const double high = value - original, magnitude = std::fabs(high);
    if (!finite_one(high) || magnitude > bound) return false;
    if (magnitude < bound) return true;
    const double bv = value - high, av = high + bv;
    const double br = bv - original, ar = value - av, low = ar + br;
    return !((high > 0 && low > 0) || (high < 0 && low < 0));
}

template <class T>
bool scalar_repairs(const T* original, const T* reconstructed, std::uint32_t begin,
                      std::uint32_t n, double bound, std::uint32_t* indices,
                      T* errors, std::uint32_t& count) noexcept {
    const T native_bound = T(bound);
    for (std::uint32_t i = begin; i < n; ++i) {
        T value = reconstructed[i];
        const T error = value - original[i];
        if (!finite_one(value) || !finite_one(error)) return false;
        if (std::fabs(error) > native_bound) {
            indices[count] = i;
            errors[count++] = error;
            value = value - error;
        }
        if (!finite_one(value) || !within_bound(double(value), double(original[i]), bound)) return false;
    }
    return true;
}

template <class T>
bool scalar_finite(const T* values, std::uint32_t begin, std::uint32_t n) noexcept {
    for (std::uint32_t i = begin; i < n; ++i) if (!finite_one(values[i])) return false;
    return true;
}

#if WALTZ_REPAIR_X86
struct alignas(32) Mask8 {
    std::int32_t lanes[256][8]{};
    std::uint8_t count[256]{};
};
constexpr Mask8 make_mask8() {
    Mask8 table{};
    for (unsigned mask = 0; mask < 256; ++mask) {
        unsigned count = 0;
        for (unsigned lane = 0; lane < 8; ++lane)
            if (mask & (1U << lane)) table.lanes[mask][count++] = std::int32_t(lane);
        table.count[mask] = std::uint8_t(count);
    }
    return table;
}
struct alignas(32) Mask4 {
    std::int32_t lanes[16][4]{};
    std::int32_t words[16][8]{};
    std::uint8_t count[16]{};
};
constexpr Mask4 make_mask4() {
    Mask4 table{};
    for (unsigned mask = 0; mask < 16; ++mask) {
        unsigned count = 0;
        for (unsigned lane = 0; lane < 4; ++lane) if (mask & (1U << lane)) {
            table.lanes[mask][count] = std::int32_t(lane);
            table.words[mask][2 * count] = std::int32_t(2 * lane);
            table.words[mask][2 * count + 1] = std::int32_t(2 * lane + 1);
            ++count;
        }
        table.count[mask] = std::uint8_t(count);
    }
    return table;
}
constexpr Mask8 masks8 = make_mask8();
constexpr Mask4 masks4 = make_mask4();

WALTZ_REPAIR_AVX2 inline bool finite8(__m256 values) noexcept {
    const __m256i exponent = _mm256_set1_epi32(0x7f800000);
    const __m256i bad = _mm256_cmpeq_epi32(_mm256_and_si256(_mm256_castps_si256(values), exponent), exponent);
    return !_mm256_movemask_ps(_mm256_castsi256_ps(bad));
}
WALTZ_REPAIR_AVX2 inline bool finite4(__m256d values) noexcept {
    const __m256i exponent = _mm256_set1_epi64x(INT64_C(0x7ff0000000000000));
    const __m256i bad = _mm256_cmpeq_epi64(_mm256_and_si256(_mm256_castpd_si256(values), exponent), exponent);
    return !_mm256_movemask_pd(_mm256_castsi256_pd(bad));
}

WALTZ_REPAIR_AVX2
bool vector_repairs_f32(const float* original, const float* reconstructed, std::uint32_t n,
                          double bound, std::uint32_t* indices, float* errors,
                          std::uint32_t& count) noexcept {
    const float native_bound = float(bound);
    const __m256 patch_bound = _mm256_set1_ps(native_bound);
    const __m256 safe_bound = _mm256_set1_ps(native_bound > 0 ? native_bound : std::numeric_limits<float>::denorm_min());
    const __m256 sign_clear = _mm256_castsi256_ps(_mm256_set1_epi32(INT32_MAX));
    const __m256i lane_indices = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
    const std::uint32_t end = n & ~std::uint32_t(7);
    std::uint32_t i = 0;
    for (; i < end; i += 8) {
        const __m256 source = _mm256_loadu_ps(original + i);
        const __m256 reconstructed_value = _mm256_loadu_ps(reconstructed + i);
        const __m256 error = _mm256_sub_ps(reconstructed_value, source);
        if (!finite8(reconstructed_value) || !finite8(error)) return false;
        const __m256 magnitude = _mm256_and_ps(error, sign_clear);
        const __m256 patch = _mm256_cmp_ps(magnitude, patch_bound, _CMP_GT_OQ);
        const unsigned patch_mask = unsigned(_mm256_movemask_ps(patch));
        __m256 value = reconstructed_value, check_error = error;
        if (patch_mask) {
            // Unpatched lanes subtract +0, which preserves even -0 under the
            // required nearest-even mode; patched lanes perform native rec-e.
            value = _mm256_sub_ps(value, _mm256_and_ps(error, patch));
            if (!finite8(value)) return false;
            check_error = _mm256_sub_ps(value, source);
        }
        const __m256 check_magnitude = _mm256_and_ps(check_error, sign_clear);
        unsigned uncertain = unsigned(_mm256_movemask_ps(_mm256_cmp_ps(check_magnitude, safe_bound, _CMP_NLT_UQ)));
        if (uncertain) {
            // Under nearest-even, native |difference| < float(bound) implies
            // exact |difference| <= bound: the previous float's rounding cell
            // ends no later than the lower cell edge of float(bound). For an
            // underflowed native bound only exact native zero is accepted.
            // All other lanes use the scalar TwoDiff test, not an epsilon.
            float values[8];
            _mm256_storeu_ps(values, value);
            do {
                const unsigned lane = unsigned(__builtin_ctz(uncertain));
                if (!within_bound(double(values[lane]), double(original[i + lane]), bound)) return false;
                uncertain &= uncertain - 1;
            } while (uncertain);
        }
        if (patch_mask) {
            const __m256i at = _mm256_add_epi32(_mm256_set1_epi32(std::int32_t(i)), lane_indices);
            const __m256i permutation = _mm256_load_si256(reinterpret_cast<const __m256i*>(masks8.lanes[patch_mask]));
            const __m256i packed_at = _mm256_permutevar8x32_epi32(at, permutation);
            const __m256 packed_error = _mm256_permutevar8x32_ps(error, permutation);
            // count<=i and i+8<=n: complete stores remain within n-entry
            // capacity, including spare lanes after the logical count.
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(indices + count), packed_at);
            _mm256_storeu_ps(errors + count, packed_error);
            count += masks8.count[patch_mask];
        }
    }
    return scalar_repairs(original, reconstructed, i, n, bound, indices, errors, count);
}

WALTZ_REPAIR_AVX2
bool vector_repairs_f64(const double* original, const double* reconstructed, std::uint32_t n,
                          double bound, std::uint32_t* indices, double* errors,
                          std::uint32_t& count) noexcept {
    const __m256d patch_bound = _mm256_set1_pd(bound);
    const __m256d sign_clear = _mm256_castsi256_pd(_mm256_set1_epi64x(INT64_MAX));
    const __m128i lane_indices = _mm_setr_epi32(0, 1, 2, 3);
    const std::uint32_t end = n & ~std::uint32_t(3);
    std::uint32_t i = 0;
    for (; i < end; i += 4) {
        const __m256d source = _mm256_loadu_pd(original + i);
        const __m256d reconstructed_value = _mm256_loadu_pd(reconstructed + i);
        const __m256d error = _mm256_sub_pd(reconstructed_value, source);
        if (!finite4(reconstructed_value) || !finite4(error)) return false;
        const __m256d magnitude = _mm256_and_pd(error, sign_clear);
        const __m256d patch = _mm256_cmp_pd(magnitude, patch_bound, _CMP_GT_OQ);
        const unsigned patch_mask = unsigned(_mm256_movemask_pd(patch));
        __m256d value = reconstructed_value, check_error = error;
        if (patch_mask) {
            value = _mm256_sub_pd(value, _mm256_and_pd(error, patch));
            if (!finite4(value)) return false;
            check_error = _mm256_sub_pd(value, source);
        }
        const __m256d check_magnitude = _mm256_and_pd(check_error, sign_clear);
        unsigned uncertain = unsigned(_mm256_movemask_pd(_mm256_cmp_pd(check_magnitude, patch_bound, _CMP_NLT_UQ)));
        if (uncertain) {
            double values[4];
            _mm256_storeu_pd(values, value);
            do {
                const unsigned lane = unsigned(__builtin_ctz(uncertain));
                if (!within_bound(values[lane], original[i + lane], bound)) return false;
                uncertain &= uncertain - 1;
            } while (uncertain);
        }
        if (patch_mask) {
            const __m128i at = _mm_add_epi32(_mm_set1_epi32(std::int32_t(i)), lane_indices);
            const __m128i permutation = _mm_load_si128(reinterpret_cast<const __m128i*>(masks4.lanes[patch_mask]));
            const __m256i words = _mm256_load_si256(reinterpret_cast<const __m256i*>(masks4.words[patch_mask]));
            const __m128i packed_at = _mm_castps_si128(_mm_permutevar_ps(_mm_castsi128_ps(at), permutation));
            const __m256i packed_error = _mm256_permutevar8x32_epi32(_mm256_castpd_si256(error), words);
            _mm_storeu_si128(reinterpret_cast<__m128i*>(indices + count), packed_at);
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(errors + count), packed_error);
            count += masks4.count[patch_mask];
        }
    }
    return scalar_repairs(original, reconstructed, i, n, bound, indices, errors, count);
}

WALTZ_REPAIR_AVX2
bool vector_finite_f32(const float* values, std::uint32_t n) noexcept {
    const std::uint32_t end = n & ~std::uint32_t(7);
    std::uint32_t i = 0;
    for (; i < end; i += 8) if (!finite8(_mm256_loadu_ps(values + i))) return false;
    return scalar_finite(values, i, n);
}
WALTZ_REPAIR_AVX2
bool vector_finite_f64(const double* values, std::uint32_t n) noexcept {
    const std::uint32_t end = n & ~std::uint32_t(3);
    std::uint32_t i = 0;
    for (; i < end; i += 4) if (!finite4(_mm256_loadu_pd(values + i))) return false;
    return scalar_finite(values, i, n);
}
#endif

template <class T>
bool valid_arguments(const T* original, const T* reconstructed, std::uint32_t n,
                       double bound, std::uint32_t* indices, T* errors,
                       std::uint32_t* count) noexcept {
    if (!count) return false;
    *count = 0;
    if (!n) return true;
    return original && reconstructed && indices && errors && bound > 0 &&
           finite_one(bound) && environment_ok();
}
} // namespace

bool repair_has_avx2() noexcept {
#if WALTZ_REPAIR_X86
    __builtin_cpu_init();
    return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
#else
    return false;
#endif
}

bool repairs_f32(const float* original, const float* reconstructed, std::uint32_t n,
                   double bound, std::uint32_t* indices, float* errors,
                   std::uint32_t* count) noexcept {
    if (!valid_arguments(original, reconstructed, n, bound, indices, errors, count)) return false;
    if (!n) return true;
#if WALTZ_REPAIR_X86
    if (repair_has_avx2()) return vector_repairs_f32(original, reconstructed, n, bound, indices, errors, *count);
#endif
    return scalar_repairs(original, reconstructed, 0, n, bound, indices, errors, *count);
}
bool repairs_f64(const double* original, const double* reconstructed, std::uint32_t n,
                   double bound, std::uint32_t* indices, double* errors,
                   std::uint32_t* count) noexcept {
    if (!valid_arguments(original, reconstructed, n, bound, indices, errors, count)) return false;
    if (!n) return true;
#if WALTZ_REPAIR_X86
    if (repair_has_avx2()) return vector_repairs_f64(original, reconstructed, n, bound, indices, errors, *count);
#endif
    return scalar_repairs(original, reconstructed, 0, n, bound, indices, errors, *count);
}
bool finite_f32(const float* values, std::uint32_t n) noexcept {
    if (!n) return true;
    if (!values) return false;
#if WALTZ_REPAIR_X86
    if (repair_has_avx2()) return vector_finite_f32(values, n);
#endif
    return scalar_finite(values, 0, n);
}
bool finite_f64(const double* values, std::uint32_t n) noexcept {
    if (!n) return true;
    if (!values) return false;
#if WALTZ_REPAIR_X86
    if (repair_has_avx2()) return vector_finite_f64(values, n);
#endif
    return scalar_finite(values, 0, n);
}
} // namespace waltz_cpu_detail
