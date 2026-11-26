#include "quantize_avx2.hpp"

#include <algorithm>
#include <cfenv>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#if defined(__x86_64__) || defined(__i386__)
#include <xmmintrin.h>
#endif

// Independent scalar llrint oracle; caller storage and scratch are malloc-owned.
// Compile without fast-math or implicit contraction, as for the codec itself.
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
        if (failures < 16) std::fprintf(stderr, "FAIL: %s\n", label);
    }
}
struct Buffer {
    unsigned char* base;
    unsigned char* data;
    size_t size, prefix;
    explicit Buffer(size_t n, size_t misalignment = 0)
        : base(static_cast<unsigned char*>(std::malloc(n + 128 + misalignment))),
          data(base ? base + 64 + misalignment : nullptr), size(n), prefix(64 + misalignment) {
        if (!base) { std::fprintf(stderr, "test allocation failed\n"); std::exit(2); }
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
bool oracle(T* values, uint32_t n, T q, T inv, int32_t* codes) {
    for (uint32_t i = 0; i < n; ++i) {
        const T scaled = values[i] * inv;
        if (!std::isfinite(scaled) || std::fabs(scaled) >= T(0x1p63)) return false;
        const long long v = std::llrint(scaled);
        const bool negative = v < 0;
        const uint64_t magnitude = uint64_t(negative ? -v : v);
        const int32_t bounded = magnitude < uint64_t(INT32_MAX) ? int32_t(magnitude) : INT32_MAX;
        codes[i] = negative ? -bounded : bounded;
        values[i] = bounded ? T(codes[i]) * q : T(0);
    }
    return true;
}

template <class T>
bool invoke(T* values, uint32_t n, T q, T inv, int32_t* codes) {
    if constexpr (sizeof(T) == 4) return waltz_cpu_detail::quantize_f32(values, n, q, inv, codes);
    else return waltz_cpu_detail::quantize_f64(values, n, q, inv, codes);
}

template <class T>
void compare(const T* source, uint32_t n, T q, T inv, bool unaligned) {
    ++cases;
    Buffer actual(size_t(n) * sizeof(T), unaligned ? sizeof(T) : 0), expected(actual.size);
    Buffer actual_codes(size_t(n) * sizeof(int32_t), unaligned ? sizeof(int32_t) : 0), expected_codes(actual_codes.size);
    if (n) {
        std::memcpy(actual.data, source, actual.size);
        std::memcpy(expected.data, source, actual.size);
    }
    const bool want = oracle(reinterpret_cast<T*>(expected.data), n, q, inv, reinterpret_cast<int32_t*>(expected_codes.data));
    const bool got = invoke(reinterpret_cast<T*>(actual.data), n, q, inv, reinterpret_cast<int32_t*>(actual_codes.data));
    check(want == got, "domain status equals scalar llrint oracle");
    if (want && got) {
        const bool integers = !std::memcmp(expected_codes.data, actual_codes.data, actual_codes.size);
        const bool decoded = !std::memcmp(expected.data, actual.data, actual.size);
        check(integers, "signed int32 saturation equals scalar oracle");
        check(decoded, "native dequantized bits equal scalar oracle");
        if ((!integers || !decoded) && failures < 16) {
            for (uint32_t i = 0; i < n; ++i) {
                const int32_t a = reinterpret_cast<int32_t*>(actual_codes.data)[i];
                const int32_t b = reinterpret_cast<int32_t*>(expected_codes.data)[i];
                if (a != b || std::memcmp(actual.data + size_t(i) * sizeof(T), expected.data + size_t(i) * sizeof(T), sizeof(T))) {
                    std::fprintf(stderr, "  f%zu n=%u first=%u src=%.17g q=%.17g inv=%.17g got=%d want=%d\n",
                                 sizeof(T) * 8, n, i, double(source[i]), double(q), double(inv), a, b);
                    break;
                }
            }
        }
        const T zero = T(0);
        bool canonical_zero = true;
        for (uint32_t i = 0; i < n; ++i)
            if (!reinterpret_cast<int32_t*>(actual_codes.data)[i] &&
                std::memcmp(actual.data + size_t(i) * sizeof(T), &zero, sizeof(T))) canonical_zero = false;
        check(canonical_zero, "integer zero always dequantizes to positive zero");
    }
    check(actual.guards() && actual_codes.guards() && expected.guards() && expected_codes.guards(),
          "unaligned vector and scalar tails preserve buffer guards");
}

template <class T>
void type_suite() {
    const T integer_limit = T(0x1p31), wide_limit = T(0x1p63);
    const T threshold[] = {
        T(0), -T(0), T(0.25), T(-0.25), T(0.5), T(-0.5), T(1.5), T(-1.5), T(2.5), T(-2.5), T(3.5), T(-3.5),
        std::nextafter(T(0.5), T(0)), std::nextafter(T(0.5), T(1)),
        std::nextafter(T(-0.5), T(0)), std::nextafter(T(-0.5), T(-1)),
        T(65534.5), T(65535), T(65535.5), T(65536.5), T(-65534.5), T(-65535.5),
        T(2147483645.5), T(2147483646), T(2147483646.5), T(2147483647), T(2147483647.5),
        T(-2147483645.5), T(-2147483646), T(-2147483646.5), T(-2147483647), T(-2147483647.5),
        std::nextafter(integer_limit, T(0)), integer_limit, std::nextafter(integer_limit, wide_limit),
        -std::nextafter(integer_limit, T(0)), -integer_limit, -std::nextafter(integer_limit, wide_limit),
        std::nextafter(wide_limit, T(0)), -std::nextafter(wide_limit, T(0)),
        std::numeric_limits<T>::denorm_min(), -std::numeric_limits<T>::denorm_min(),
        std::numeric_limits<T>::min(), -std::numeric_limits<T>::min()
    };
    constexpr uint32_t threshold_count = sizeof(threshold) / sizeof(threshold[0]);
    for (uint32_t n = 0; n <= threshold_count; ++n) {
        compare(threshold, n, T(1), T(1), false);
        compare(threshold, n, T(0.0015), T(1), true);
    }
    // Rotate boundary values across every SIMD lane and every possible tail.
    T rotated[67];
    for (uint32_t offset = 0; offset < threshold_count; ++offset) {
        for (uint32_t i = 0; i < 67; ++i) rotated[i] = threshold[(i + offset) % threshold_count];
        for (uint32_t n = 1; n <= 19; ++n) compare(rotated, n, T(0.25), T(1), bool(offset & 1));
    }
    const uint32_t counts[] = {31, 32, 33, 127, 128, 129, 8191, 8192, 8193, 65549};
    struct Parameters { T q, inv; };
    const Parameters params[] = {{T(1), T(1)}, {T(0.0015), T(1.0 / 0.0015)},
                                 {T(1e-30), T(1e30)}, {T(1e30), T(1e-30)},
                                 {std::numeric_limits<T>::denorm_min(), T(1)},
                                 {std::numeric_limits<T>::max(), T(0)}};
    for (uint32_t n : counts) {
        Buffer source(size_t(n) * sizeof(T));
        auto* values = reinterpret_cast<T*>(source.data);
        for (const auto& p : params) {
            uint32_t rng = 0x15f32ca9;
            for (uint32_t i = 0; i < n; ++i) {
                rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5;
                double desired = double(int64_t(rng) - INT64_C(2147483648)) / 8;
                if (i % 17 == 0) desired *= 65536; // safely inside int64, above int32
                values[i] = p.inv == T(0) ? T(desired) : T(T(desired) / p.inv);
            }
            compare(values, n, p.q, p.inv, false);
            compare(values, n, p.q, p.inv, true);
        }
    }
    const T rejected[] = {wide_limit, -wide_limit, std::nextafter(wide_limit, std::numeric_limits<T>::infinity()),
                           std::numeric_limits<T>::infinity(), -std::numeric_limits<T>::infinity(),
                           std::numeric_limits<T>::quiet_NaN()};
    const uint32_t positions[] = {0, 3, 7, 8, 15, 16, 22};
    T source[23];
    for (T invalid : rejected) for (uint32_t position : positions) {
        for (T& value : source) value = T(1.5);
        source[position] = invalid;
        compare(source, 23, T(1), T(1), bool(position & 1));
    }
    for (T& value : source) value = T(1);
    source[7] = std::numeric_limits<T>::max();
    compare(source, 23, T(1), T(2), false); // overflow occurs in native multiplication

    int32_t codes[23];
    check(invoke<T>(nullptr, 0, T(0), T(-1), nullptr), "empty input performs no pointer/parameter access");
    check(!invoke<T>(nullptr, 1, T(1), T(1), codes), "nonempty null input rejects");
    check(!invoke<T>(source, 1, T(1), T(1), nullptr), "nonempty null codes rejects");
    check(!invoke<T>(source, 1, T(0), T(1), codes), "zero quantum rejects");
    check(!invoke<T>(source, 1, T(-1), T(1), codes), "negative quantum rejects");
    check(!invoke<T>(source, 1, std::numeric_limits<T>::infinity(), T(1), codes), "nonfinite quantum rejects");
    check(!invoke<T>(source, 1, T(1), T(-1), codes), "negative reciprocal rejects");
    check(!invoke<T>(source, 1, T(1), std::numeric_limits<T>::quiet_NaN(), codes), "nonfinite reciprocal rejects");
    std::fesetround(FE_DOWNWARD);
    check(!invoke<T>(source, 1, T(1), T(1), codes), "non-nearest floating-point environment rejects");
    check(std::fegetround() == FE_DOWNWARD, "rejected call leaves caller rounding unchanged");
    std::fesetround(FE_TONEAREST);
#if defined(__x86_64__) || defined(__i386__)
    const unsigned csr = _mm_getcsr();
    _mm_setcsr(csr | (1U << 15) | (1U << 6));
    check(!invoke<T>(source, 1, T(1), T(1), codes), "FTZ/DAZ environment rejects");
    check((_mm_getcsr() & ((1U << 15) | (1U << 6))) == ((1U << 15) | (1U << 6)),
          "rejected call leaves caller FTZ/DAZ unchanged");
    _mm_setcsr(csr);
#endif
}
// Feed the fused path exact half ties, an adjacent nonzero, and a wide code.
// Reverse ranks and an unaligned sign base catch accidental row-order stores.
void brick_rows() {
    using T=float;
    for (uint32_t width : {15u,16u}) {
        constexpr uint32_t stride=19, base=11, origin=37;
        T data[3*stride]{}, ref[3*stride]{};
        uint16_t rank[48]{}, mag[48];
        uint32_t signs[2]={1,0}, want_signs[2]={1,0}, indices[48]{}, values[48]{}, count=0;
        int32_t codes[48]{};
        for(uint32_t x=0;x<width;++x){
            data[x]=x%2 ? T(-0.5) : T(0.5);
            data[stride+x]=x==3 ? T(-70000) : x==5 ? std::nextafter(T(0.5),T(1)) : T(-1.5);
            data[2*stride+x]=x%3==0 ? T(-65535) : x%3==1 ? T(25) : T(-0.25);
        }
        std::memcpy(ref,data,sizeof(data)); std::memset(mag,0xab,sizeof(mag));
        for(uint32_t i=0;i<3*width;++i)rank[i]=uint16_t(3*width-1-i);
        for(uint32_t y=0;y<3;++y)oracle(ref+y*stride,width,T(1),T(1),codes+y*width);
        check(waltz_cpu_detail::quantize_morton_f32(data,{stride,3,1},{width,3,1},rank,mag,
              signs,origin,base,T(1),T(1),indices,values,count),"fused brick status");
        for(uint32_t i=0;i<3*width;++i){
            int32_t v=codes[i];const auto bit=base+rank[i];
            if(v<0)want_signs[bit/32]|=1U<<(bit%32);
            check(mag[rank[i]]==uint16_t(std::min(std::abs(v),65535)),"fused Morton magnitude");
        }
        check(!std::memcmp(ref,data,sizeof(data)),"fused native values, including +0 and padding");
        check(!std::memcmp(signs,want_signs,sizeof(signs)),"fused sign bits");
        check(count==1 && indices[0]==origin+stride+3 && values[0]==70000,"fused wide index/value");
    }
}
} // namespace

int main() {
    std::fenv_t old;
    std::fegetenv(&old);
#if defined(__x86_64__) || defined(__i386__)
    const unsigned csr = _mm_getcsr();
#endif
    std::fesetround(FE_TONEAREST);
#if defined(__x86_64__) || defined(__i386__)
    _mm_setcsr(_mm_getcsr() & ~unsigned((1U << 15) | (1U << 6)));
#endif
    std::printf("Waltz quantizer AVX2/FMA dispatch=%u\n", unsigned(waltz_cpu_detail::quantize_has_avx2()));
    type_suite<float>();
    type_suite<double>();
    if(waltz_cpu_detail::quantize_has_avx2())brick_rows();
    std::fesetenv(&old);
#if defined(__x86_64__) || defined(__i386__)
    _mm_setcsr(csr);
#endif
    std::printf("SUMMARY cases=%u checks=%u failures=%u\n", cases, checks, failures);
    return failures ? 1 : 0;
}
