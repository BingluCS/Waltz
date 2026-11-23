#include "../src/morton_decode_avx2.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <vector>

#if defined(__unix__)
#include <sys/mman.h>
#include <unistd.h>
#endif

namespace {

void require(bool good, const char* message) {
    if (!good) throw std::runtime_error(message);
}

std::uint32_t random_word(std::uint32_t& state) {
    state ^= state << 13u;
    state ^= state >> 17u;
    state ^= state << 5u;
    return state;
}

std::uint16_t morton(std::uint32_t x, std::uint32_t y, std::uint32_t z,
                     std::uint32_t nx, std::uint32_t ny, std::uint32_t nz) {
    std::uint32_t rank = 0;
    while (nx > 1 || ny > 1 || nz > 1) {
        const std::uint32_t sx[] = {nx - nx / 2u, nx / 2u};
        const std::uint32_t sy[] = {ny - ny / 2u, ny / 2u};
        const std::uint32_t sz[] = {nz - nz / 2u, nz / 2u};
        const unsigned cx = x >= sx[0], cy = y >= sy[0], cz = z >= sz[0];
        if (cz) rank += sz[0] * nx * ny;
        if (cy) rank += sz[cz] * sy[0] * nx;
        if (cx) rank += sz[cz] * sy[cy] * sx[0];
        if (cx) x -= sx[0];
        if (cy) y -= sy[0];
        if (cz) z -= sz[0];
        nx = sx[cx]; ny = sy[cy]; nz = sz[cz];
    }
    return static_cast<std::uint16_t>(rank);
}

void make_rank(std::uint16_t* rank, std::uint32_t nx, std::uint32_t ny, std::uint32_t nz) {
    for (std::uint32_t z = 0; z != nz; ++z)
        for (std::uint32_t y = 0; y != ny; ++y)
            for (std::uint32_t x = 0; x != nx; ++x)
                rank[(z * ny + y) * nx + x] = morton(x, y, z, nx, ny, nz);
}

template<class T>
void scalar(const std::uint16_t* rank, const std::uint16_t* mag,
            const std::uint32_t* sign, std::size_t sign_base,
            T* output, std::size_t row_stride, std::size_t plane_stride,
            std::uint32_t nx, std::uint32_t ny, std::uint32_t nz, T q) {
    for (std::uint32_t z = 0; z != nz; ++z)
        for (std::uint32_t y = 0; y != ny; ++y)
            for (std::uint32_t x = 0; x != nx; ++x) {
                const std::size_t r = rank[(z * ny + y) * nx + x];
                const std::size_t index = sign_base + r;
                const T value = T(mag[r]) * q;
                output[std::size_t(z) * plane_stride + std::size_t(y) * row_stride + x] =
                    ((sign[index >> 5u] >> (index & 31u)) & 1u) ? -value : value;
            }
}

template<class T>
bool simd(const std::uint16_t* rank, const std::uint16_t* mag,
          const std::uint32_t* sign, std::size_t sign_base,
          T* output, std::size_t row_stride, std::size_t plane_stride,
          std::uint32_t nx, std::uint32_t ny, std::uint32_t nz, T q, bool avx2 = true) {
    if constexpr (std::is_same_v<T, float>)
        return waltz_cpu_detail::morton_decode_f32(rank, mag, sign, sign_base, output,
                                                  row_stride, plane_stride, nx, ny, nz, q, avx2);
    else
        return waltz_cpu_detail::morton_decode_f64(rank, mag, sign, sign_base, output,
                                                  row_stride, plane_stride, nx, ny, nz, q, avx2);
}

template<class T>
void round_trips() {
    const std::uint32_t shapes[][3] = {{1,1,1}, {3,5,7}, {4,3,5}, {7,7,3}, {8,1,1},
        {8,3,5}, {9,11,7}, {13,15,17}, {15,7,31}, {16,16,1}, {16,16,2},
        {16,16,8}, {16,16,16}, {16,16,32}};
    const std::size_t bases[] = {0, 1, 15, 31, 32, 63, 8191};
    const T scales[] = {T(0), -T(0), T(1), T(0.0015), T(-0.00375),
                        std::numeric_limits<T>::min(), std::numeric_limits<T>::denorm_min()};
    std::uint32_t state = 0x924fb327u;
    unsigned cases = 0, vector_cases = 0;
    for (const auto& shape : shapes) {
        const std::uint32_t nx = shape[0], ny = shape[1], nz = shape[2], n = nx * ny * nz;
        // +1 then offset1 tests non-vector-aligned uint16 views without any
        // readable trailing padding after their exact final elements.
        std::vector<std::uint16_t> rank_storage(n + 1u), mag_storage(n + 1u);
        auto* rank = rank_storage.data() + 1u;
        auto* mag = mag_storage.data() + 1u;
        const std::size_t row = nx + 3u, plane = row * (ny + 2u);
        const std::size_t output_size = plane * nz + 2u;
        std::vector<T> expected(output_size, T(-123.5)), output(expected);
        for (unsigned permutation = 0; permutation != 2; ++permutation) {
            if (permutation == 0)
                make_rank(rank, nx, ny, nz);
            else for (std::uint32_t i = 0; i != n; ++i)
                rank[i] = static_cast<std::uint16_t>(n - i - 1u);
            for (const std::size_t base : bases) {
                std::vector<std::uint32_t> signs((base + n + 31u) / 32u);
                for (auto& word : signs) word = random_word(state);
                for (unsigned pattern = 0; pattern != 4; ++pattern) {
                    for (std::uint32_t i = 0; i != n; ++i) {
                        const std::uint32_t r = random_word(state);
                        mag[i] = pattern == 0 ? 0 : pattern == 1 ? 65535u :
                                 pattern == 2 ? static_cast<std::uint16_t>(r) :
                                 r % 7u ? 0 : static_cast<std::uint16_t>(r & 31u);
                    }
                    for (const T q : scales) {
                        std::fill(expected.begin(), expected.end(), T(-123.5));
                        std::fill(output.begin(), output.end(), T(-123.5));
                        require(!simd(rank, mag, signs.data(), base, output.data() + 1u,
                                      row, plane, nx, ny, nz, q, false), "disabled AVX path handled a brick");
                        require(std::memcmp(output.data(), expected.data(), output_size * sizeof(T)) == 0,
                                "disabled AVX path wrote output");
                        scalar(rank, mag, signs.data(), base, expected.data() + 1u,
                               row, plane, nx, ny, nz, q);
                        const bool used = simd(rank, mag, signs.data(), base, output.data() + 1u,
                                               row, plane, nx, ny, nz, q);
                        if (!used)
                            scalar(rank, mag, signs.data(), base, output.data() + 1u,
                                   row, plane, nx, ny, nz, q);
                        else ++vector_cases;
                        require(std::memcmp(output.data(), expected.data(), output_size * sizeof(T)) == 0,
                                "Morton decode scalar/AVX2 bit or output-padding mismatch");
                        ++cases;
                    }
                }
            }
        }
    }
    std::printf("Morton %s: %u bit-identical shape/base/permutation/pattern/scale cases (%u vectorized)\n",
                sizeof(T) == 4 ? "f32" : "f64", cases, vector_cases);
}

void protected_boundary_tests() {
#if defined(__unix__)
    struct Guarded {
        void* allocation;
        void* data;
        std::size_t page;
        explicit Guarded(std::size_t bytes) {
            const long size = sysconf(_SC_PAGESIZE);
            require(size > 0 && bytes < static_cast<std::size_t>(size), "invalid guard-page size");
            page = static_cast<std::size_t>(size);
            allocation = mmap(nullptr, page * 2u, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
            require(allocation != MAP_FAILED, "cannot allocate guard pages");
            if (mprotect(static_cast<std::uint8_t*>(allocation) + page, page, PROT_NONE) != 0) {
                munmap(allocation, page * 2u);
                throw std::runtime_error("cannot protect guard page");
            }
            data = static_cast<std::uint8_t*>(allocation) + page - bytes;
        }
        ~Guarded() { munmap(allocation, page * 2u); }
    };
    constexpr unsigned n = 9;
    Guarded mag_mem(n * sizeof(std::uint16_t)), rank_mem(n * sizeof(std::uint16_t));
    Guarded sign_mem(2u * sizeof(std::uint32_t)), float_mem(n * sizeof(float)), double_mem(n * sizeof(double));
    auto* mag = static_cast<std::uint16_t*>(mag_mem.data);
    auto* rank = static_cast<std::uint16_t*>(rank_mem.data);
    auto* sign = static_cast<std::uint32_t*>(sign_mem.data);
    for (unsigned i = 0; i != n; ++i) {
        rank[i] = static_cast<std::uint16_t>(n - 1u - i);
        mag[i] = static_cast<std::uint16_t>(i * 719u);
    }
    sign[0] = 0x938fac42u;
    sign[1] = 0x9eabe248u;
    float expected_f[n]; double expected_d[n];
    scalar(rank, mag, sign, 31, expected_f, n, n, n, 1, 1, 0.0015f);
    scalar(rank, mag, sign, 31, expected_d, n, n, n, 1, 1, 0.0015);
    auto* f = static_cast<float*>(float_mem.data);
    auto* d = static_cast<double*>(double_mem.data);
    if (!simd(rank, mag, sign, 31, f, n, n, n, 1, 1, 0.0015f))
        scalar(rank, mag, sign, 31, f, n, n, n, 1, 1, 0.0015f);
    if (!simd(rank, mag, sign, 31, d, n, n, n, 1, 1, 0.0015))
        scalar(rank, mag, sign, 31, d, n, n, n, 1, 1, 0.0015);
    require(std::memcmp(f, expected_f, sizeof(expected_f)) == 0 &&
            std::memcmp(d, expected_d, sizeof(expected_d)) == 0, "protected-boundary output mismatch");
    std::printf("Morton: exact mag/rank/sign/output spans before protected pages passed\n");
#endif
}

struct Volume { std::uint32_t nx, ny, nz, bz; };

template<class T>
void volume_decode(Volume v, const std::uint16_t* const* tables,
                   const std::uint16_t* mag, const std::uint32_t* sign, T* out, bool avx2) {
    std::size_t base = 0;
    const std::size_t plane = std::size_t(v.nx) * v.ny;
    const T q = T(0.0015);
    for (std::uint32_t z = 0; z < v.nz; z += v.bz)
        for (std::uint32_t y = 0; y < v.ny; y += 16u)
            for (std::uint32_t x = 0; x < v.nx; x += 16u) {
                const std::uint32_t nx = std::min(16u, v.nx - x);
                const std::uint32_t ny = std::min(16u, v.ny - y);
                const std::uint32_t nz = std::min(v.bz, v.nz - z);
                const unsigned shape = unsigned(nx != 16u) | (unsigned(ny != 16u) << 1u) |
                                       (unsigned(nz != v.bz) << 2u);
                T* const output = out + std::size_t(z) * plane + std::size_t(y) * v.nx + x;
                if (!avx2 || !simd(tables[shape], mag + base, sign, base, output,
                                   v.nx, plane, nx, ny, nz, q))
                    scalar(tables[shape], mag + base, sign, base, output,
                           v.nx, plane, nx, ny, nz, q);
                base += std::size_t(nx) * ny * nz;
            }
}

template<class T>
void benchmark() {
    const Volume volumes[] = {{128,128,128,8}, {128,128,128,2}, {129,131,65,5}, {33,31,17,5}, {9,11,65,4}};
    std::uint32_t state = 0x623a98b1u;
    for (const Volume v : volumes) {
        const std::size_t n = std::size_t(v.nx) * v.ny * v.nz;
        std::vector<std::uint16_t> mag(n);
        std::vector<std::uint32_t> sign((n + 31u) / 32u);
        std::vector<T> expected(n), output(n);
        for (auto& value : mag) value = static_cast<std::uint16_t>(random_word(state));
        for (auto& value : sign) value = random_word(state);
        std::uint16_t rank_tables[8][8192];
        const std::uint16_t* tables[8];
        for (unsigned shape = 0; shape != 8; ++shape) {
            tables[shape] = rank_tables[shape];
            const auto nx = shape & 1u ? v.nx % 16u : 16u;
            const auto ny = shape & 2u ? v.ny % 16u : 16u;
            const auto nz = shape & 4u ? v.nz % v.bz : v.bz;
            if (nx && ny && nz) make_rank(rank_tables[shape], nx, ny, nz);
        }
        volume_decode(v, tables, mag.data(), sign.data(), expected.data(), false);
        volume_decode(v, tables, mag.data(), sign.data(), output.data(), true);
        require(std::memcmp(output.data(), expected.data(), n * sizeof(T)) == 0,
                "whole-volume Morton decode mismatch");
        constexpr unsigned repeats = 12;
        double best[2] = {1e30, 1e30};
        for (unsigned run = 0; run != 4; ++run) {
            for (unsigned mode = 0; mode != 2; ++mode) {
                const auto start = std::chrono::steady_clock::now();
                for (unsigned repeat = 0; repeat != repeats; ++repeat)
                    volume_decode(v, tables, mag.data(), sign.data(), output.data(), mode != 0);
                const auto end = std::chrono::steady_clock::now();
                best[mode] = std::min(best[mode], std::chrono::duration<double>(end - start).count());
            }
        }
        const double gigabytes = double(n) * sizeof(T) * repeats / 1e9;
        std::printf("Morton %s %ux%ux%u bz=%u scalar=%.3f GB/s AVX2=%.3f GB/s speedup=%.2fx\n",
                    sizeof(T) == 4 ? "f32" : "f64", v.nx, v.ny, v.nz, v.bz,
                    gigabytes / best[0], gigabytes / best[1], best[0] / best[1]);
    }
}

} // namespace

int main(int argc, char** argv) {
    try {
        if (argc == 2 && std::strcmp(argv[1], "--benchmark") == 0) {
            benchmark<float>(); benchmark<double>();
        } else {
            round_trips<float>(); round_trips<double>();
            protected_boundary_tests();
        }
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "Morton decode failure: %s\n", error.what());
        return EXIT_FAILURE;
    }
}
