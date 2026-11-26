#include "repair_avx2.hpp"

#include <cfenv>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <initializer_list>
#include <limits>
#if defined(__x86_64__) || defined(__i386__)
#include <xmmintrin.h>
#endif

#if defined(__clang__)
#pragma clang fp contract(off)
#elif defined(__GNUC__)
#pragma GCC optimize ("fp-contract=off")
#endif

namespace {
unsigned cases = 0, checks = 0, failures = 0;
void check(bool good, const char* label) {
    ++checks;
    if (!good) {
        ++failures;
        if (failures < 20) std::fprintf(stderr, "FAIL: %s\n", label);
    }
}
struct Buffer {
    unsigned char* base;
    unsigned char* data;
    size_t size, prefix;
    explicit Buffer(size_t n, size_t offset = 0)
        : base(static_cast<unsigned char*>(std::malloc(n + 128 + offset))),
          data(base ? base + 64 + offset : nullptr), size(n), prefix(64 + offset) {
        if (!base) std::exit(2);
        std::memset(base, 0xa5, n + prefix + 64);
    }
    ~Buffer() { std::free(base); }
    bool guards() const {
        for (size_t i = 0; i < prefix; ++i) if (base[i] != 0xa5) return false;
        for (size_t i = 0; i < 64; ++i) if (data[size + i] != 0xa5) return false;
        return true;
    }
};
template <class T>
bool finite(T value) {
    if constexpr (sizeof(T) == 4) {
        uint32_t bits; std::memcpy(&bits, &value, 4);
        return (bits & 0x7f800000U) != 0x7f800000U;
    } else {
        uint64_t bits; std::memcpy(&bits, &value, 8);
        return (bits & UINT64_C(0x7ff0000000000000)) != UINT64_C(0x7ff0000000000000);
    }
}
bool strict(double value, double input, double bound) {
    const double high = value - input, magnitude = std::fabs(high);
    if (!finite(high) || magnitude > bound) return false;
    if (magnitude < bound) return true;
    const double bv = value - high;
    const double av = high + bv;
    const double br = bv - input;
    const double ar = value - av;
    const double tail = ar + br;
    return !((high > 0 && tail > 0) || (high < 0 && tail < 0));
}
template <class T>
bool oracle(const T* input, const T* reconstructed, uint32_t n, double bound,
              uint32_t* indices, T* errors, uint32_t* count) {
    *count = 0;
    for (uint32_t i = 0; i < n; ++i) {
        T value = reconstructed[i];
        const T error = value - input[i];
        if (!finite(value) || !finite(error)) return false;
        if (std::fabs(error) > T(bound)) {
            indices[*count] = i;
            errors[(*count)++] = error;
            value = value - error;
        }
        if (!finite(value) || !strict(double(value), double(input[i]), bound)) return false;
    }
    return true;
}
template <class T>
bool invoke(const T* input, const T* reconstructed, uint32_t n, double bound,
              uint32_t* indices, T* errors, uint32_t* count) {
    return waltz_cpu_detail::collect_repairs(input,reconstructed,n,bound,indices,errors,count,true);
}
template <class T>
bool scan(const T* input, uint32_t n) {
    if constexpr (sizeof(T) == 4) return waltz_cpu_detail::finite_f32(input, n);
    else return waltz_cpu_detail::finite_f64(input, n);
}

template <class T>
bool compare(const T* input, const T* reconstructed, uint32_t n, double bound, bool offset) {
    ++cases;
    Buffer source(size_t(n) * sizeof(T), offset ? sizeof(T) : 0), saved_source(source.size);
    Buffer values(source.size, offset ? sizeof(T) : 0), saved_values(source.size);
    Buffer indices(size_t(n) * 4, offset ? 4 : 0), expected_indices(size_t(n) * 4);
    Buffer errors(source.size, offset ? sizeof(T) : 0), expected_errors(source.size);
    if (n) {
        std::memcpy(source.data, input, source.size); std::memcpy(saved_source.data, input, source.size);
        std::memcpy(values.data, reconstructed, source.size); std::memcpy(saved_values.data, reconstructed, source.size);
    }
    uint32_t expected_count = 0, count = UINT32_MAX;
    const bool expected = oracle(reinterpret_cast<T*>(source.data), reinterpret_cast<T*>(values.data), n, bound,
                                  reinterpret_cast<uint32_t*>(expected_indices.data),
                                  reinterpret_cast<T*>(expected_errors.data), &expected_count);
    const bool got = invoke(reinterpret_cast<T*>(source.data), reinterpret_cast<T*>(values.data), n, bound,
                             reinterpret_cast<uint32_t*>(indices.data), reinterpret_cast<T*>(errors.data), &count);
    check(expected == got, "repair status matches native scalar/TwoDiff oracle");
    check(count <= n, "reported repair count within caller capacity");
    if (expected && got) {
        check(count == expected_count, "native patch predicate produces identical count");
        check(count == expected_count && !std::memcmp(indices.data, expected_indices.data, size_t(count) * 4),
              "stable compressed original indices match scalar order");
        check(count == expected_count && !std::memcmp(errors.data, expected_errors.data, size_t(count) * sizeof(T)),
              "recorded native residual bits match scalar order");
    }
    bool ascending = true;
    const auto* at = reinterpret_cast<uint32_t*>(indices.data);
    for (uint32_t i = 0; i < count && count <= n; ++i)
        if (at[i] >= n || (i && at[i - 1] >= at[i])) ascending = false;
    check(ascending, "partial and successful records use ascending valid indices");
    check(!std::memcmp(source.data, saved_source.data, source.size) &&
          !std::memcmp(values.data, saved_values.data, values.size), "original and reconstruction arrays unchanged");
    check(source.guards() && values.guards() && indices.guards() && errors.guards() &&
          expected_indices.guards() && expected_errors.guards(), "all output capacities and native-aligned SIMD tails guarded");
    bool input_finite = true, values_finite = true;
    for (uint32_t i = 0; i < n; ++i) { input_finite &= finite(input[i]); values_finite &= finite(reconstructed[i]); }
    check(scan(reinterpret_cast<T*>(source.data), n) == input_finite, "input finite scan equals bitwise oracle");
    for (bool avx : {false, true}) {
        Buffer copy(source.size, offset ? sizeof(T) : 0);
        const bool ok=waltz_cpu_detail::copy_finite(input,reinterpret_cast<T*>(copy.data),n,avx);
        check(ok==input_finite && (!ok || !n || std::memcmp(copy.data,input,source.size)==0),
              "encoder finite copy preserves input bits or rejects");
        check(copy.guards(), "encoder finite copy stays within destination");
    }
    check(scan(reinterpret_cast<T*>(values.data), n) == values_finite, "reconstruction finite scan equals bitwise oracle");
    return got;
}

template <class T>
void explicit_pair(T input, T value, double bound, bool expected, const char* label) {
    T originals[17], values[17];
    for (uint32_t i = 0; i < 17; ++i) { originals[i] = input; values[i] = value; }
    for (uint32_t n = 1; n <= 17; ++n) check(compare(originals, values, n, bound, bool(n & 1)) == expected, label);
}

template <class T>
void type_suite() {
    T source[67], values[67];
    // Exhaust all masks, positions and short tails; interleave complementary
    // masks so compaction crosses SIMD blocks with every possible prefix.
    const unsigned masks = sizeof(T) == 4 ? 256 : 16, lanes = sizeof(T) == 4 ? 8 : 4;
    for (unsigned mask = 0; mask < masks; ++mask) {
        for (unsigned i = 0; i < 67; ++i) {
            source[i] = (i & 1) ? T(-0.0) : T(0);
            const unsigned block_mask = (i / lanes) & 1 ? (mask ^ (masks - 1)) : mask;
            values[i] = (block_mask & (1U << (i % lanes))) ? T(0.25) : T(0.0625);
            if (i & 2) values[i] = -values[i];
        }
        for (uint32_t n = 1; n <= 2 * lanes + 3; ++n) compare(source, values, n, 0.125, bool(mask & 1));
        compare(source, values, 67, 0.125, true);
    }
    const T tiny = std::numeric_limits<T>::denorm_min(), maximum = std::numeric_limits<T>::max();
    explicit_pair(T(0), T(0.5), 0.5, true, "exact unpatched bound accepted");
    explicit_pair(-tiny, T(1), 1, false, "positive TwoDiff tail beyond bound rejects");
    explicit_pair(tiny, T(1), 1, true, "negative TwoDiff tail within bound accepts");
    explicit_pair(tiny, T(-1), 1, false, "negative exact difference beyond bound rejects");
    explicit_pair(-tiny, T(-1), 1, true, "positive tail reduces negative error magnitude");
    explicit_pair(T(1), T(1.25), 0.001, true, "ordinary native correction succeeds");
    explicit_pair(T(1), T(1e30), 0.001, false, "native catastrophic cancellation fails strict verification");
    explicit_pair(-maximum, maximum, std::numeric_limits<double>::max(), false, "nonfinite native initial error rejects");
    explicit_pair(T(0), T(-0.0), 0.001, true, "unpatched negative zero is unchanged");
    explicit_pair(T(-0.0), T(0), 0.001, true, "zero-sign difference emits no repair");
    explicit_pair(tiny, T(0), std::numeric_limits<double>::denorm_min(), true, "subnormal native correction survives");
    if constexpr (sizeof(T) == 4) {
        explicit_pair(T(0), T(1), std::nextafter(1.0, 0.0), false, "FP32 bound rounding does not relax strict double bound");
        explicit_pair(T(0), T(1), std::nextafter(1.0, 2.0), true, "FP32 native bound may conservatively pass exact input");
        explicit_pair(T(0), tiny, double(tiny) / 4, true, "underflowed FP32 native bound still repairs");
        explicit_pair(maximum, T(0), std::numeric_limits<double>::max(), true, "overflowed native bound still uses strict finite double bound");
    }

    const uint32_t counts[] = {0, 1, 3, 7, 8, 9, 31, 32, 33, 129, 8191, 8192, 8193, 65549};
    const double bounds[] = {0.001, 0.125, 1e-7, double(std::numeric_limits<T>::min()),
                             double(tiny), std::numeric_limits<double>::denorm_min()};
    for (uint32_t n : counts) {
        Buffer original(size_t(n) * sizeof(T)), reconstructed(original.size);
        auto* a = reinterpret_cast<T*>(original.data);
        auto* b = reinterpret_cast<T*>(reconstructed.data);
        for (double bound : bounds) for (unsigned mode = 0; mode < 4; ++mode) {
            uint32_t rng = 0x51784ef1U;
            for (uint32_t i = 0; i < n; ++i) {
                rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5;
                a[i] = T(double(int64_t(rng) - INT64_C(2147483648)) / (1U << 21));
                double error = 0;
                if (mode == 1 && i % 17 == 0) error = bound * ((i & 1) ? -2.5 : 2.5);
                if (mode == 2) error = bound * ((i & 1) ? -2.5 : 2.5);
                if (mode == 3) error = bound * (double(int(rng & 31) - 16) / 8);
                b[i] = T(double(a[i]) + error);
            }
            compare(a, b, n, bound, bool(mode & 1));
        }
    }
    const T invalid[] = {std::numeric_limits<T>::quiet_NaN(), std::numeric_limits<T>::infinity(),
                          -std::numeric_limits<T>::infinity()};
    for (T bad : invalid) for (uint32_t at = 0; at < 19; ++at) {
        for (unsigned i = 0; i < 67; ++i) source[i] = values[i] = T(1);
        values[at] = bad; compare(source, values, 19, 0.001, true);
        values[at] = T(1); source[at] = bad; compare(source, values, 19, 0.001, false);
    }

    uint32_t indices[67], count = 123;
    T errors[67];
    for (unsigned i = 0; i < 67; ++i) source[i] = values[i] = T(1);
    check(invoke<T>(nullptr, nullptr, 0, -1, nullptr, nullptr, &count) && count == 0, "empty collection resets count and accepts null arrays");
    check(!invoke(source, values, 1, 1, indices, errors, nullptr), "null count rejects");
    check(!invoke<T>(nullptr, values, 1, 1, indices, errors, &count) && count == 0, "nonempty null original rejects");
    check(!invoke<T>(source, nullptr, 1, 1, indices, errors, &count), "nonempty null reconstruction rejects");
    check(!invoke(source, values, 1, 1, nullptr, errors, &count), "nonempty null indices rejects");
    check(!invoke<T>(source, values, 1, 1, indices, nullptr, &count), "nonempty null errors rejects");
    const double bad_bounds[] = {0, -1, std::numeric_limits<double>::infinity(), std::numeric_limits<double>::quiet_NaN()};
    for (double bad : bad_bounds) check(!invoke(source, values, 1, bad, indices, errors, &count), "invalid bound rejects");
    check(scan<T>(nullptr, 0) && !scan<T>(nullptr, 1), "finite null/empty input contract");
    std::fesetround(FE_UPWARD);
    check(!invoke(source, values, 1, 1, indices, errors, &count), "non-nearest collection environment rejects");
    check(std::fegetround() == FE_UPWARD && scan(source, 67), "finite scan does not require or change rounding mode");
    std::fesetround(FE_TONEAREST);
#if defined(__x86_64__) || defined(__i386__)
    const unsigned csr = _mm_getcsr();
    _mm_setcsr(csr | (1U << 15) | (1U << 6));
    check(!invoke(source, values, 1, 1, indices, errors, &count), "FTZ/DAZ collection environment rejects");
    check(scan(source, 67) && (_mm_getcsr() & ((1U << 15) | (1U << 6))) == ((1U << 15) | (1U << 6)),
          "finite classifier leaves FTZ/DAZ environment unchanged");
    _mm_setcsr(csr);
#endif
    // Verify bitwise classification, including signaling NaNs in every lane
    // and tail, without setting the floating-point invalid exception.
    for (uint32_t at = 0; at < 19; ++at) {
        for (unsigned i = 0; i < 67; ++i) source[i] = T(1);
        if constexpr (sizeof(T) == 4) {
            const uint32_t bits = 0x7f800001U; std::memcpy(source + at, &bits, 4);
        } else {
            const uint64_t bits = UINT64_C(0x7ff0000000000001); std::memcpy(source + at, &bits, 8);
        }
        std::feclearexcept(FE_ALL_EXCEPT);
        check(!scan(source, 19), "signaling NaN exponent rejects");
        for (bool avx : {false, true})
            check(!waltz_cpu_detail::copy_finite(source,values,19,avx), "finite copy rejects signaling NaN");
        check(!std::fetestexcept(FE_INVALID), "finite classification does not raise invalid for signaling NaN");
    }
}
} // namespace

int main() {
    std::fenv_t old; std::fegetenv(&old);
#if defined(__x86_64__) || defined(__i386__)
    const unsigned csr = _mm_getcsr();
#endif
    std::fesetround(FE_TONEAREST);
#if defined(__x86_64__) || defined(__i386__)
    _mm_setcsr(_mm_getcsr() & ~unsigned((1U << 15) | (1U << 6)));
#endif
    std::printf("Waltz repair AVX2/FMA dispatch=%u\n", unsigned(waltz_cpu_detail::repair_has_avx2()));
    type_suite<float>(); type_suite<double>();
    std::fesetenv(&old);
#if defined(__x86_64__) || defined(__i386__)
    _mm_setcsr(csr);
#endif
    std::printf("SUMMARY cases=%u checks=%u failures=%u\n", cases, checks, failures);
    return failures ? 1 : 0;
}
