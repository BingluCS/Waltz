#include "morton_decode_avx2.hpp"

#include <limits>

#if (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__))
#include <immintrin.h>
#define WALTZ_MORTON_X86 1
#define WALTZ_MORTON_AVX2 __attribute__((target("avx2")))
#else
#define WALTZ_MORTON_X86 0
#endif

namespace waltz_cpu_detail {
namespace {

bool available() {
#if WALTZ_MORTON_X86
    __builtin_cpu_init();
    return __builtin_cpu_supports("avx2");
#else
    return false;
#endif
}

bool valid(const std::uint16_t* rank, const std::uint16_t* mag,
           const std::uint32_t* sign, std::size_t sign_base, const void* output,
           std::size_t row_stride, std::size_t plane_stride,
           std::uint32_t nx, std::uint32_t ny, std::uint32_t nz,
           std::uint32_t width, bool avx2) {
    if (!avx2 || rank == nullptr || mag == nullptr || sign == nullptr || output == nullptr ||
        nx < width || nx > 16 || ny == 0 || ny > 16 || nz == 0 || nz > 32 || row_stride < nx)
        return false;
    const std::size_t maximum = std::numeric_limits<std::size_t>::max();
    if (row_stride > maximum / ny || plane_stride < row_stride * ny ||
        plane_stride > maximum / nz || sign_base > maximum - std::size_t(nx) * ny * nz)
        return false;
    return available();
}

template<class T>
void tail(const std::uint16_t* rank, const std::uint16_t* mag,
          const std::uint32_t* sign, std::size_t sign_base, T* output,
          std::uint32_t first, std::uint32_t nx, T q) {
    for (std::uint32_t x = first; x != nx; ++x) {
        const std::size_t index = sign_base + rank[x];
        const T value = T(mag[rank[x]]) * q;
        output[x] = ((sign[index >> 5u] >> (index & 31u)) & 1u) ? -value : value;
    }
}

#if WALTZ_MORTON_X86
WALTZ_MORTON_AVX2 void decode_vectors(const std::uint16_t* rank, const std::uint16_t* mag,
                                 const std::uint32_t* sign, std::size_t sign_base,
                                 float* output, std::size_t row_stride, std::size_t plane_stride,
                                 std::uint32_t nx, std::uint32_t ny, std::uint32_t nz, float q) {
    const __m256 scale = _mm256_set1_ps(q);
    const __m256i last = _mm256_set1_epi32(static_cast<int>(nx * ny * nz - 1u));
    const __m256i safe_last = _mm256_sub_epi32(last, _mm256_set1_epi32(1));
    const __m256i low16 = _mm256_set1_epi32(0xffff);
    const __m256i bit_base = _mm256_set1_epi32(static_cast<int>(sign_base & 31u));
    const __m256i low5 = _mm256_set1_epi32(31);
    const __m256i high16 = _mm256_set1_epi32(16);
    const std::uint32_t* const signs = sign + (sign_base >> 5u);
    for (std::uint32_t z = 0; z != nz; ++z) {
        for (std::uint32_t y = 0; y != ny; ++y) {
            const std::uint16_t* const row = rank + (z * ny + y) * nx;
            float* const destination = output + std::size_t(z) * plane_stride + std::size_t(y) * row_stride;
            std::uint32_t x = 0;
            for (; x + 8u <= nx; x += 8u) {
                const __m256i indices = _mm256_cvtepu16_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(row + x)));
                // AVX2 has no u16 gather. Clamp the final element to the
                // preceding u16, load that safe pair, then select its high
                // half. Every 32-bit load stays inside the exact mag span.
                const __m256i safe = _mm256_min_epu32(indices, safe_last);
                __m256i values = _mm256_i32gather_epi32(reinterpret_cast<const int*>(mag), safe, 2);
                const __m256i shift = _mm256_and_si256(_mm256_cmpeq_epi32(indices, last), high16);
                values = _mm256_and_si256(_mm256_srlv_epi32(values, shift), low16);
                const __m256i positions = _mm256_add_epi32(indices, bit_base);
                const __m256i words = _mm256_i32gather_epi32(reinterpret_cast<const int*>(signs),
                                                            _mm256_srli_epi32(positions, 5), 4);
                const __m256i bits = _mm256_srlv_epi32(words, _mm256_and_si256(positions, low5));
                const __m256i negative = _mm256_slli_epi32(bits, 31);
                const __m256 value = _mm256_mul_ps(_mm256_cvtepi32_ps(values), scale);
                _mm256_storeu_ps(destination + x, _mm256_xor_ps(value, _mm256_castsi256_ps(negative)));
            }
            tail(row, mag, sign, sign_base, destination, x, nx, q);
        }
    }
}

WALTZ_MORTON_AVX2 void decode_vectors(const std::uint16_t* rank, const std::uint16_t* mag,
                                 const std::uint32_t* sign, std::size_t sign_base,
                                 double* output, std::size_t row_stride, std::size_t plane_stride,
                                 std::uint32_t nx, std::uint32_t ny, std::uint32_t nz, double q) {
    const __m256d scale = _mm256_set1_pd(q);
    const __m128i last = _mm_set1_epi32(static_cast<int>(nx * ny * nz - 1u));
    const __m128i safe_last = _mm_sub_epi32(last, _mm_set1_epi32(1));
    const __m128i low16 = _mm_set1_epi32(0xffff);
    const __m128i bit_base = _mm_set1_epi32(static_cast<int>(sign_base & 31u));
    const __m128i low5 = _mm_set1_epi32(31);
    const __m128i high16 = _mm_set1_epi32(16);
    const __m128i one = _mm_set1_epi32(1);
    const std::uint32_t* const signs = sign + (sign_base >> 5u);
    for (std::uint32_t z = 0; z != nz; ++z) {
        for (std::uint32_t y = 0; y != ny; ++y) {
            const std::uint16_t* const row = rank + (z * ny + y) * nx;
            double* const destination = output + std::size_t(z) * plane_stride + std::size_t(y) * row_stride;
            std::uint32_t x = 0;
            for (; x + 4u <= nx; x += 4u) {
                const __m128i indices = _mm_cvtepu16_epi32(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(row + x)));
                const __m128i safe = _mm_min_epu32(indices, safe_last);
                __m128i values = _mm_i32gather_epi32(reinterpret_cast<const int*>(mag), safe, 2);
                const __m128i shift = _mm_and_si128(_mm_cmpeq_epi32(indices, last), high16);
                values = _mm_and_si128(_mm_srlv_epi32(values, shift), low16);
                const __m128i positions = _mm_add_epi32(indices, bit_base);
                const __m128i words = _mm_i32gather_epi32(reinterpret_cast<const int*>(signs),
                                                        _mm_srli_epi32(positions, 5), 4);
                const __m128i bits = _mm_and_si128(_mm_srlv_epi32(words, _mm_and_si128(positions, low5)), one);
                const __m256i negative = _mm256_slli_epi64(_mm256_cvtepi32_epi64(bits), 63);
                const __m256d value = _mm256_mul_pd(_mm256_cvtepi32_pd(values), scale);
                _mm256_storeu_pd(destination + x, _mm256_xor_pd(value, _mm256_castsi256_pd(negative)));
            }
            tail(row, mag, sign, sign_base, destination, x, nx, q);
        }
    }
}
#endif

} // namespace

template<class T>
bool morton_decode(const std::uint16_t* rank, const std::uint16_t* mag,
                   const std::uint32_t* sign, std::size_t sign_base, T* output,
                   std::size_t row_stride, std::size_t plane_stride,
                   std::uint32_t nx, std::uint32_t ny, std::uint32_t nz, T q, bool avx2) {
    constexpr std::uint32_t lanes = 32 / sizeof(T); // one AVX2 register
    if (!valid(rank, mag, sign, sign_base, output, row_stride, plane_stride, nx, ny, nz, lanes, avx2))
        return false;
#if WALTZ_MORTON_X86
    decode_vectors(rank, mag, sign, sign_base, output, row_stride, plane_stride, nx, ny, nz, q);
    return true;
#else
    (void)q;
    return false;
#endif
}

template bool morton_decode<float>(const std::uint16_t*, const std::uint16_t*, const std::uint32_t*, std::size_t,
    float*, std::size_t, std::size_t, std::uint32_t, std::uint32_t, std::uint32_t, float, bool);
template bool morton_decode<double>(const std::uint16_t*, const std::uint16_t*, const std::uint32_t*, std::size_t,
    double*, std::size_t, std::size_t, std::uint32_t, std::uint32_t, std::uint32_t, double, bool);

} // namespace waltz_cpu_detail
