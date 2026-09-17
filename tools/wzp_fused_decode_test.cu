// Check fused inverse reorder/dequant against an independent CPU octree traversal.
#include "lossless/wzp.hpp"
#include "lossless/reorder.hpp"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <type_traits>
#include <vector>

static void check(cudaError_t e) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "%s\n", cudaGetErrorString(e));
        std::exit(2);
    }
}

template <typename T>
static unsigned run(bool small) {
    using namespace WALTZ::lossless;
    unsigned cases = 0;
    for (dim3 dims : {dim3(1, 1, 1), dim3(7, 9, 3), dim3(33, 17, 5),
                      dim3(65, 49, 37), dim3(128, 128, 32), dim3(129, 113, 67)}) {
        const uint32_t n = dims.x * dims.y * dims.z;
        // Optional short run for sanitizer checks; retain multi-chunk edge cases.
        if (small && n > 120000U) continue;
        for (uint32_t bz : {1U, 2U, 3U, 4U, 5U, 8U, 16U, 32U}) {
            std::vector<uint32_t> permutation;
            std::function<void(uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t)> visit;
            visit = [&](uint32_t x, uint32_t y, uint32_t z,
                        uint32_t nx, uint32_t ny, uint32_t nz) {
                if (!nx || !ny || !nz) return;
                if (nx == 1 && ny == 1 && nz == 1) {
                    permutation.push_back(x + dims.x * (y + dims.y * z));
                    return;
                }
                const uint32_t ax = (nx + 1) / 2, ay = (ny + 1) / 2, az = (nz + 1) / 2;
                for (unsigned child = 0; child < 8; ++child)
                    visit(x + ((child & 1) ? ax : 0), y + ((child & 2) ? ay : 0),
                          z + ((child & 4) ? az : 0), (child & 1) ? nx - ax : ax,
                          (child & 2) ? ny - ay : ay, (child & 4) ? nz - az : az);
            };
            for (uint32_t z = 0; z < dims.z; z += bz)
                for (uint32_t y = 0; y < dims.y; y += 16)
                    for (uint32_t x = 0; x < dims.x; x += 16)
                        visit(x, y, z, std::min(16U, dims.x - x),
                              std::min(16U, dims.y - y), std::min(bz, dims.z - z));
            if (permutation.size() != n) std::abort();
            uint16_t* input;
            uint32_t *signs, *table;
            T* output;
            const size_t sign_words = ((n + 8191ULL) / 8192) * 256;
            check(cudaMalloc(&input, n * sizeof(uint16_t)));
            check(cudaMalloc(&signs, sign_words * sizeof(uint32_t)));
            check(cudaMalloc(&table, 8U * 256U * bz * sizeof(uint32_t)));
            check(cudaMalloc(&output, n * sizeof(T)));
            WALTZ::lossless::d_build_block_ioffset<<<dim3(bz, 8), 256>>>(
                table, 16, 16, bz, dims.x, static_cast<size_t>(dims.x) * dims.y, dims);
            check(cudaGetLastError());
            std::vector<uint16_t> values(n);
            std::vector<uint32_t> bits(sign_words);
            std::vector<T> expected(n), actual(n);
            for (unsigned pattern = 0; pattern < 4; ++pattern) {
                std::fill(bits.begin(), bits.end(), 0U);
                uint32_t rng = 123456789;
                const T q = static_cast<T>(0.0137);
                for (uint32_t i = 0; i < n; ++i) {
                    rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5;
                    values[i] = pattern == 0 ? 0 : pattern == 1 ?
                        (i % 101 == 0 ? rng & 255U : 0) : pattern == 2 ? rng :
                        ((i / 8192U) % 2U == 0U ? 0U : rng);
                    const bool negative = values[i] && i % 3 == 1;
                    if (negative) bits[i / 32] |= 1U << (i % 32);
                    const T value = static_cast<T>(values[i]) * q;
                    expected[permutation[i]] = negative ? -value : value;
                }
                check(cudaMemcpy(input, values.data(), n * sizeof(uint16_t), cudaMemcpyHostToDevice));
                check(cudaMemcpy(signs, bits.data(), sign_words * sizeof(uint32_t), cudaMemcpyHostToDevice));
                wzp_encode_with_sign(input, signs, n, nullptr, nullptr);
                check(cudaMemset(output, 0xff, n * sizeof(T)));
                if constexpr (std::is_same_v<T, float>)
                    wzp_decode_reordered_float(wzp_encoded_buf(), n, output, table, q, dims, bz, nullptr);
                else
                    wzp_decode_reordered_double(wzp_encoded_buf(), n, output, table, q, dims, bz, nullptr);
                check(cudaMemcpy(actual.data(), output, n * sizeof(T), cudaMemcpyDeviceToHost));
                for (uint32_t i = 0; i < n; ++i) {
                    if (actual[i] != expected[i]) {
                        std::fprintf(stderr, "FAIL sizeof(T)=%zu dims=%u,%u,%u bz=%u pattern=%u i=%u\n",
                                     sizeof(T), dims.x, dims.y, dims.z, bz, pattern, i);
                        std::exit(1);
                    }
                }
                ++cases;
            }
            check(cudaFree(input)); check(cudaFree(signs));
            check(cudaFree(table)); check(cudaFree(output));
        }
    }
    return cases;
}

int main(int argc, char**) {
    const unsigned count = run<float>(argc > 1) + run<double>(argc > 1);
    std::printf("PASS %u fused decode cases (float/double, regular/edges, zero/sparse/raw/mixed chunks)\n", count);
}
