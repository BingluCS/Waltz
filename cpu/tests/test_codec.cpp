#include "waltz_cpu.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>

// Deliberately use caller-owned malloc storage, not STL containers. The public
// C API is the only codec interface used by this acceptance test.
namespace {
unsigned checks = 0, failures = 0, cases = 0, equal_archives = 0;

void check(bool condition, const char* what) {
    ++checks;
    if (!condition) {
        ++failures;
        std::fprintf(stderr, "FAIL: %s\n", what);
    }
}

struct Buffer {
    unsigned char* base;
    unsigned char* data;
    size_t size;
    explicit Buffer(size_t n) : base(static_cast<unsigned char*>(std::malloc(n + 128))),
                                data(base ? base + 64 : nullptr), size(n) {
        if (!base) { std::fprintf(stderr, "test allocation failed\n"); std::exit(2); }
        std::memset(base, 0xa5, n + 128);
    }
    ~Buffer() { std::free(base); }
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
    bool guards() const {
        for (size_t i = 0; i < 64; ++i)
            if (base[i] != 0xa5 || data[size + i] != 0xa5) return false;
        return true;
    }
};

struct Workspace {
    waltz_cpu_workspace* value;
    Workspace() : value(waltz_cpu_create()) {
        if (!value) { std::fprintf(stderr, "workspace allocation failed\n"); std::exit(2); }
    }
    ~Workspace() { waltz_cpu_destroy(value); }
};

uint32_t u32(const unsigned char* p) {
    return uint32_t(p[0]) | uint32_t(p[1]) << 8 | uint32_t(p[2]) << 16 | uint32_t(p[3]) << 24;
}
uint64_t u64(const unsigned char* p) { return u32(p) | uint64_t(u32(p + 4)) << 32; }
void put32(unsigned char* p, uint32_t v) {
    for (unsigned i = 0; i < 4; ++i) p[i] = static_cast<unsigned char>(v >> (8 * i));
}
void put64(unsigned char* p, uint64_t v) {
    for (unsigned i = 0; i < 8; ++i) p[i] = static_cast<unsigned char>(v >> (8 * i));
}
void put_double(unsigned char* p, double value) {
    uint64_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    put64(p, bits);
}
size_t align8(size_t n) { return (n + 7) & ~size_t(7); }
unsigned levels(uint32_t n) {
    unsigned l = 0;
    while (n > 8 && l < 5) { n = (n + 1) / 2; ++l; }
    return l;
}

template <typename T>
void fill(T* data, const waltz_cpu_config& c, unsigned pattern) {
    uint32_t rng = 0x1879b52d;
    for (uint32_t z = 0; z < c.nz; ++z)
        for (uint32_t y = 0; y < c.ny; ++y)
            for (uint32_t x = 0; x < c.nx; ++x) {
                const size_t i = (size_t(z) * c.ny + y) * c.nx + x;
                rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5;
                double v;
                switch (pattern) {
                case 0: v = 0; break;
                case 1: v = 1.25; break;
                case 2: v = std::sin(0.17 * x) + std::cos(0.21 * y) - 0.03 * z;
                        v += ((x + 3 * y + 7 * z) % 19 == 0) ? 0.125 : 0; break;
                case 3: v = (double(rng & 0x00ffffffU) / 0x01000000U - 0.5) * 2; break;
                case 4: v = ((x + y + z) & 1) ? -4 : 4;
                        if (i == 0) v = 32;
                        if (i + 1 == size_t(c.nx) * c.ny * c.nz) v = -32; break;
                case 5: v = (i % 7 == 0) ? -0.0 : (double(int(i % 9) - 4) * c.absolute_error * 0.499); break;
                default: v = 1048576.0 + (int(i % 33) - 16) * 0.125; break;
                }
                data[i] = static_cast<T>(v);
            }
}

template <typename T>
bool within_bound(const T* input, const T* output, size_t n, double bound, double* max_error) {
    // Compare the actual represented values in long double. No epsilon slack:
    // RMSE and a T-rounded subtraction can both conceal pointwise violations.
    long double maximum = 0;
    bool good = true;
    for (size_t i = 0; i < n; ++i) {
        const long double error = std::fabs(static_cast<long double>(input[i]) - output[i]);
        if (!std::isfinite(output[i]) || !(error <= static_cast<long double>(bound))) {
            if (good) std::fprintf(stderr, "  bound violation index=%zu input=%.17g output=%.17g error=%.21Lg bound=%.17g\n",
                                   i, double(input[i]), double(output[i]), error, bound);
            good = false;
        }
        if (error > maximum) maximum = error;
    }
    *max_error = double(maximum);
    return good;
}

bool decode(Workspace& w, const waltz_cpu_config& c, const Buffer& archive,
            size_t written, Buffer& output, const char* what) {
    const int rc = waltz_cpu_decompress(w.value, &c, archive.data, written, output.data, output.size);
    if (rc) std::fprintf(stderr, "  %s: %s\n", what, waltz_cpu_last_error(w.value));
    check(rc == 0, what);
    check(output.guards(), "decode output guards");
    return rc == 0;
}

template <typename T>
void roundtrip(Workspace& w, uint32_t nx, uint32_t ny, uint32_t nz, unsigned pattern, double bound) {
    ++cases;
    waltz_cpu_config c{nx, ny, nz, sizeof(T) == 4 ? WALTZ_CPU_F32 : WALTZ_CPU_F64,
                       bound, WALTZ_CPU_SCALAR};
    const size_t n = size_t(nx) * ny * nz, bytes = n * sizeof(T);
    const size_t capacity = waltz_cpu_max_compressed_bytes(&c);
    check(capacity >= 32, "nonzero archive capacity");
    if (capacity < 32) return;
    Buffer input(bytes), saved(bytes), scalar(capacity), other(capacity), output(bytes), output2(bytes);
    fill(reinterpret_cast<T*>(input.data), c, pattern);
    std::memcpy(saved.data, input.data, bytes);
    size_t written = 0, written2 = 0;
    int rc = waltz_cpu_compress(w.value, &c, input.data, scalar.data, scalar.size, &written);
    if (rc) std::fprintf(stderr, "  case f%zu %ux%ux%u pattern=%u eb=%.17g: %s\n",
                         8 * sizeof(T), nx, ny, nz, pattern, bound, waltz_cpu_last_error(w.value));
    check(rc == 0, "scalar compression");
    if (rc) return;
    check(written >= 32 && written <= capacity, "archive written length within capacity");
    check(scalar.guards() && input.guards(), "compress buffer guards");
    check(std::memcmp(input.data, saved.data, bytes) == 0, "compression preserves caller input");
    check((scalar.data[0] & ~1U) == (sizeof(T)==8 ? 28U : 24U), "32-byte header flags");
    const unsigned lxy = levels(nx < ny ? nx : ny), lz = levels(nz);
    check(scalar.data[1] == lxy && scalar.data[2] == lz && (scalar.data[0] & 1U) == (lxy == lz),
          "dimension-only dyadic/plane schedule");
    double max_error = 0;
    if (!decode(w, c, scalar, written, output, "scalar decode")) return;
    check(within_bound(reinterpret_cast<T*>(input.data), reinterpret_cast<T*>(output.data), n, bound, &max_error),
          "strict pointwise absolute error");
    rc = waltz_cpu_compress(w.value, &c, input.data, other.data, other.size, &written2);
    check(rc == 0 && written == written2 && std::memcmp(scalar.data, other.data, written) == 0,
          "reused workspace deterministic scalar archive");
    c.backend = WALTZ_CPU_AUTO;
    if (decode(w, c, scalar, written, output2, "AUTO decode"))
        check(std::memcmp(output.data, output2.data, bytes) == 0, "AUTO/scalar same-archive decoded bits");
    if (waltz_cpu_has_avx2()) {
        c.backend = WALTZ_CPU_AVX2;
        if (decode(w, c, scalar, written, output2, "AVX2 decode scalar archive"))
            check(std::memcmp(output.data, output2.data, bytes) == 0, "AVX2/scalar same-archive decoded bits");
        rc = waltz_cpu_compress(w.value, &c, input.data, other.data, other.size, &written2);
        check(rc == 0, "AVX2 compression");
        if (!rc) {
            const bool equal = written == written2 && std::memcmp(scalar.data, other.data, written) == 0;
            if (equal) ++equal_archives;
            check(equal, "scalar/AVX2 complete encoded archive bytes");
            if (decode(w, c, other, written2, output2, "AVX2 decode AVX2 archive")) {
                check(std::memcmp(output.data, output2.data, bytes) == 0, "scalar/AVX2 encoder reconstruction bits");
                check(within_bound(reinterpret_cast<T*>(input.data), reinterpret_cast<T*>(output2.data), n, bound, &max_error),
                      "AVX2 strict pointwise absolute error");
            }
            c.backend = WALTZ_CPU_SCALAR;
            if (decode(w, c, other, written2, output2, "scalar decode AVX2 archive"))
                check(std::memcmp(output.data, output2.data, bytes) == 0, "scalar accepts AVX2 archive bits");
        }
    }
    std::printf("PASS f%zu %ux%ux%u %s pattern=%u eb=%.3g bytes=%zu max_error=%.9g\n",
                8 * sizeof(T), nx, ny, nz, lxy == lz ? "dyadic" : "plane", pattern, bound, written, max_error);
}

void rejected(Workspace& w, const waltz_cpu_config& c, const Buffer& archive, size_t bytes,
              Buffer& output, const char* what) {
    check(waltz_cpu_decompress(w.value, &c, archive.data, bytes, output.data, output.size) != 0, what);
    check(output.guards(), "malformed decode output guards");
}

template <typename T>
void invalid_inputs(Workspace& w) {
    waltz_cpu_config c{9, 11, 17, sizeof(T) == 4 ? WALTZ_CPU_F32 : WALTZ_CPU_F64, 0.001, WALTZ_CPU_SCALAR};
    const size_t bytes = size_t(c.nx) * c.ny * c.nz * sizeof(T);
    Buffer input(bytes), archive(waltz_cpu_max_compressed_bytes(&c));
    fill(reinterpret_cast<T*>(input.data), c, 2);
    T* values = reinterpret_cast<T*>(input.data);
    const T original = values[7];
    const T invalid[] = {std::numeric_limits<T>::quiet_NaN(), std::numeric_limits<T>::infinity(),
                         -std::numeric_limits<T>::infinity()};
    size_t written = 0;
    for (T v : invalid) {
        values[7] = v;
        check(waltz_cpu_compress(w.value, &c, input.data, archive.data, archive.size, &written) != 0,
              "reject NaN and infinity input");
        check(waltz_cpu_last_error(w.value) && *waltz_cpu_last_error(w.value), "diagnostic on rejected input");
    }
    values[7] = original;
    const double invalid_bounds[] = {0, -1, std::numeric_limits<double>::quiet_NaN(),
                                     std::numeric_limits<double>::infinity()};
    for (double eb : invalid_bounds) {
        c.absolute_error = eb;
        check(waltz_cpu_compress(w.value, &c, input.data, archive.data, archive.size, &written) != 0,
              "reject nonpositive/nonfinite error bound");
    }
    c.absolute_error = 0.001;
    c.nx = 0;
    check(waltz_cpu_max_compressed_bytes(&c) == 0, "zero dimension capacity rejection");
    check(waltz_cpu_compress(w.value, &c, input.data, archive.data, archive.size, &written) != 0, "reject zero dimension");
    c.nx = 9;
    check(waltz_cpu_compress(w.value, &c, input.data, archive.data, 1, &written) != 0,
          "reject undersized archive capacity");
    if (!waltz_cpu_has_avx2()) {
        c.backend = WALTZ_CPU_AVX2;
        check(waltz_cpu_compress(w.value, &c, input.data, archive.data, archive.size, &written) != 0,
              "unsupported explicit AVX2 rejects");
    }
    check(archive.guards() && input.guards(), "rejected input buffer guards");
}

void malformed_archives(Workspace& w) {
    waltz_cpu_config c{17, 19, 23, WALTZ_CPU_F64, 0.00001, WALTZ_CPU_SCALAR};
    const size_t n = size_t(c.nx) * c.ny * c.nz;
    Buffer input(n * sizeof(double)), output(input.size), archive(waltz_cpu_max_compressed_bytes(&c)), copy(archive.size);
    fill(reinterpret_cast<double*>(input.data), c, 3);
    size_t written = 0;
    const int rc = waltz_cpu_compress(w.value, &c, input.data, archive.data, archive.size, &written);
    check(rc == 0, "malformation fixture compression");
    if (rc) return;
    for (size_t length = 0; length < written; length += written < 8192 ? 1 : (length < 64 ? 1 : 127))
        rejected(w, c, archive, length, output, "reject truncated archive");
    rejected(w, c, archive, written - 1, output, "reject final alignment-byte truncation");
    check(waltz_cpu_decompress(w.value, &c, archive.data, written, output.data, output.size - 1) != 0,
          "reject undersized output");
    // Header offsets are wire-format constants, not C++ struct layout assumptions.
    struct Mutation { unsigned offset, width; uint64_t value; const char* what; };
    const Mutation mutations[] = {
        {0, 1, 0, "reject unset synthesis"}, {0, 1, 0x57, "reject obsolete 40-byte header flags"},
        {0, 1, 24, "reject mismatched element width"}, {0, 1, 0x3c, "reject reserved flag bits"},
        {0, 1, 12, "reject unsupported convolution synthesis"},
        {1, 1, 255, "reject excess XY levels"}, {2, 1, 255, "reject excess Z levels"},
        {3, 1, 0, "reject zero reorder band"}, {3, 1, 33, "reject excessive reorder band"},
        {4, 4, UINT32_MAX, "reject impossible PWE section bytes"},
        {16, 8, UINT64_MAX, "reject overflowing magnitude length"},
        {16, 8, 0, "reject missing magnitude stream"},
        {24, 4, uint64_t(n) + 1, "reject excessive repair count"},
        {28, 4, uint64_t(n) + 1, "reject excessive quantization outlier count"}
    };
    for (const auto& m : mutations) {
        std::memcpy(copy.data, archive.data, written);
        if (m.width == 1) copy.data[m.offset] = static_cast<unsigned char>(m.value);
        else if (m.width == 4) put32(copy.data + m.offset, uint32_t(m.value));
        else put64(copy.data + m.offset, m.value);
        rejected(w, c, copy, written, output, m.what);
    }
    const double invalid_q[] = {0, -1, std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::infinity()};
    for (double q : invalid_q) {
        std::memcpy(copy.data, archive.data, written);
        put_double(copy.data + 8, q);
        rejected(w, c, copy, written, output, "reject invalid quantization step");
    }
    const size_t magnitude_bytes = size_t(u64(archive.data + 16));
    const size_t pwe_indices = align8(32 + magnitude_bytes);
    const uint32_t repairs = u32(archive.data + 24), outliers = u32(archive.data + 28);
    Buffer expected(output.size);
    decode(w,c,archive,written,expected,"reference decode for side-channel permutations");
    const size_t pvalues=align8(pwe_indices+size_t(repairs)*4);
    const size_t qindices=align8(pvalues+u32(archive.data+4));
    const size_t qvalues=align8(qindices+size_t(outliers)*4);
    const auto swapped=[&](size_t start,size_t width,uint32_t count){
        for(size_t j=0;j<width;++j){unsigned char t=copy.data[start+j];
            copy.data[start+j]=copy.data[start+size_t(count-1)*width+j];
            copy.data[start+size_t(count-1)*width+j]=t;}
    };
    for(unsigned kind=0;kind<2;++kind){
        const uint32_t count=kind?outliers:repairs;if(count<2)continue;
        const size_t indices=kind?qindices:pwe_indices,values=kind?qvalues:pvalues;
        std::memcpy(copy.data,archive.data,written);
        swapped(indices,4,count);swapped(values,kind?4:sizeof(double),count);
        if(decode(w,c,copy,written,output,"accept unsorted valid side-channel pairs"))
            check(!std::memcmp(output.data,expected.data,output.size),"side-channel permutation preserves decoded bits");
        put32(copy.data+indices+4,u32(copy.data+indices));
        rejected(w,c,copy,written,output,"reject duplicate side-channel indices");
    }
    if (repairs) {
        std::memcpy(copy.data, archive.data, written);
        put32(copy.data + pwe_indices, uint32_t(n));
        rejected(w, c, copy, written, output, "reject repair index outside output");
    }
    if (outliers) {
        const size_t qindices = align8(align8(pwe_indices + size_t(repairs) * 4) + u32(archive.data + 4));
        std::memcpy(copy.data, archive.data, written);
        put32(copy.data + qindices, uint32_t(n));
        rejected(w, c, copy, written, output, "reject coefficient outlier index outside output");
    }
    check(archive.guards() && copy.guards() && output.guards(), "malformation buffer guards");
    // A rejected decode must not poison subsequent workspace reuse.
    decode(w, c, archive, written, output, "valid decode after malformed archives");
}

template <typename T>
void tight_bound_fail_closed(Workspace& w) {
    waltz_cpu_config c{17, 19, 23, sizeof(T) == 4 ? WALTZ_CPU_F32 : WALTZ_CPU_F64,
                       1e-12, WALTZ_CPU_SCALAR};
    const size_t n = size_t(c.nx) * c.ny * c.nz;
    Buffer input(n * sizeof(T)), output(input.size), archive(waltz_cpu_max_compressed_bytes(&c));
    fill(reinterpret_cast<T*>(input.data), c, 2);
    size_t written = 0;
    const int rc = waltz_cpu_compress(w.value, &c, input.data, archive.data, archive.size, &written);
    if (rc) {
        check(waltz_cpu_last_error(w.value) && *waltz_cpu_last_error(w.value), "tight-bound fail-closed diagnostic");
        std::printf("TIGHT_BOUND f%zu rejected explicitly: %s\n", 8 * sizeof(T), waltz_cpu_last_error(w.value));
    } else if (decode(w, c, archive, written, output, "tight-bound decode")) {
        double error = 0;
        check(within_bound(reinterpret_cast<T*>(input.data), reinterpret_cast<T*>(output.data), n, c.absolute_error, &error),
              "tight bound either strictly satisfied or encoding rejected");
    }
    check(input.guards() && output.guards() && archive.guards(), "tight-bound guards");
}

template <typename T>
void type_suite(Workspace& w) {
    struct Shape { uint32_t x, y, z; };
    const Shape shapes[] = {{1, 1, 1}, {2, 3, 5}, {8, 8, 8}, {16, 16, 16}, {17, 19, 23},
                            {48, 32, 9}, {9, 11, 65}, {1, 17, 33}, {33, 31, 17}, {64, 64, 64}};
    for (const auto& s : shapes)
        for (unsigned pattern = 0; pattern < 5; ++pattern)
            roundtrip<T>(w, s.x, s.y, s.z, pattern, 0.001);
    roundtrip<T>(w, 17, 19, 23, 5, 0.00001);
    roundtrip<T>(w, 9, 11, 17, 6, 0.00001);
    roundtrip<T>(w, 17, 19, 23, 2, 1e-7);
    roundtrip<T>(w, 64, 64, 64, 1, 1e-5); // Sparse nonzero quantization outlier list.
    tight_bound_fail_closed<T>(w);
    invalid_inputs<T>(w);
}
} // namespace

int main() {
    Workspace w;
    std::printf("Waltz CPU C-API acceptance: AVX2=%d\n", waltz_cpu_has_avx2());
    type_suite<float>(w);
    type_suite<double>(w);
    malformed_archives(w);
    std::printf("SUMMARY cases=%u checks=%u failures=%u scalar/AVX2_equal_archives=%u\n",
                cases, checks, failures, equal_archives);
    return failures ? 1 : 0;
}
