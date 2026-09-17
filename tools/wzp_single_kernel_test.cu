// Public-API regression for fused encode/pack: counter reuse, capacity growth,
// split/convenience API interleaving, raw tails, and unaligned input pointers.
#include "lossless/wzp.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>

static void check(cudaError_t error) {
    if (error != cudaSuccess) {
        std::fprintf(stderr, "%s\n", cudaGetErrorString(error));
        std::exit(2);
    }
}

int main(int argc, char**) {
    // Optional argument: smaller racecheck run, retaining multi-group tails.
    const unsigned repeats = argc > 1 ? 3U : 9U;
    cudaStream_t stream;
    check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    unsigned cases = 0;
    for (uint32_t n : {1U, 17U, 8191U, 8192U, 8193U, 139387U, 1048576U, 8193U, 17U}) {
        if (argc > 1 && n == 1048576U) continue;
        for (unsigned pattern = 0; pattern < 3; ++pattern) {
            std::vector<uint16_t> input(n), output(n);
            std::vector<uint32_t> signs((n + 31U) / 32U), restored(signs.size());
            uint32_t rng = 123456789U, expected_nnz = 0;
            for (uint32_t i = 0; i < n; ++i) {
                rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5;
                input[i] = pattern == 0 ? 0U : pattern == 1 ? (i % 101U == 0U ? rng & 255U : 0U) : rng;
                if (input[i]) {
                    ++expected_nnz;
                    if (i % 3U == 1U) signs[i / 32U] |= 1U << (i % 32U);
                }
            }
            for (unsigned offset : {0U, 1U}) {
                uint16_t *allocation, *decoded;
                uint32_t *sign, *decoded_sign;
                check(cudaMalloc(&allocation, (n + 8ULL) * sizeof(uint16_t)));
                check(cudaMalloc(&decoded, n * sizeof(uint16_t)));
                check(cudaMalloc(&sign, signs.size() * sizeof(uint32_t)));
                check(cudaMalloc(&decoded_sign, signs.size() * sizeof(uint32_t)));
                uint16_t* src = allocation + offset;
                check(cudaMemcpyAsync(src, input.data(), n * sizeof(uint16_t), cudaMemcpyHostToDevice, stream));
                check(cudaMemcpyAsync(sign, signs.data(), signs.size() * sizeof(uint32_t), cudaMemcpyHostToDevice, stream));
                size_t signed_bytes = 0;
                for (unsigned rep = 0; rep < repeats; ++rep) {
                    uint32_t nnz = 0;
                    size_t bytes;
                    if (rep % 2 == 0) {
                        WALTZ::lossless::wzp_encode(src, sign, n, stream);
                        bytes = WALTZ::lossless::wzp_encode_with_sign_finish(&nnz, nullptr);
                    } else {
                        bytes = WALTZ::lossless::wzp_encode_with_sign(src, sign, n, &nnz, nullptr, stream);
                    }
                    WALTZ::lossless::wzp_decode_with_sign(WALTZ::lossless::wzp_encoded_buf(), n, decoded, decoded_sign, nullptr, stream);
                    check(cudaMemcpyAsync(restored.data(), decoded_sign, signs.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost, stream));
                    check(cudaMemcpyAsync(output.data(), decoded, n * sizeof(uint16_t), cudaMemcpyDeviceToHost, stream));
                    check(cudaStreamSynchronize(stream));
                    if (output != input || restored != signs || nnz != expected_nnz ||
                        bytes > WALTZ::lossless::wzp_maxsize(n) || (signed_bytes && signed_bytes != bytes)) {
                        std::fprintf(stderr, "FAIL n=%u pattern=%u offset=%u rep=%u\n", n, pattern, offset, rep);
                        return 3;
                    }
                    signed_bytes = bytes;
                }
                check(cudaFree(allocation)); check(cudaFree(decoded));
                check(cudaFree(sign)); check(cudaFree(decoded_sign));
                ++cases;
            }
        }
    }
    check(cudaStreamDestroy(stream));
    std::printf("PASS: %u cases, %u interleaved encode/decode calls each\n", cases, repeats);
}
