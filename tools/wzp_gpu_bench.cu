#include "lossless/wzp.hpp"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <fstream>
#include <vector>

namespace {
void check(cudaError_t error, const char* what) {
    if (error != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(error));
        std::exit(2);
    }
}
} // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: %s INPUT.qmag\n", argv[0]);
        return 2;
    }
    std::ifstream file(argv[1], std::ios::binary | std::ios::ate);
    if (!file)
        return 2;
    const size_t bytes = static_cast<size_t>(file.tellg());
    if ((bytes & 1u) != 0 || bytes / 2 > UINT32_MAX)
        return 2;
    std::vector<uint16_t> input(bytes / 2), decoded(bytes / 2);
    std::vector<uint32_t> sign((input.size() + 31u) / 32u, 0), decoded_sign(sign.size(), 0);
    file.seekg(0);
    file.read(reinterpret_cast<char*>(input.data()), bytes);
    // The encoder always stores signs; use deterministic synthetic signs.
    for (size_t i = 0; i < input.size(); ++i)
        if (input[i] != 0 && (i % 3u) == 1u)
            sign[i >> 5u] |= 1u << (i & 31u);
    uint16_t *device_input = nullptr, *device_decoded = nullptr;
    uint32_t *device_sign = nullptr, *device_decoded_sign = nullptr;
    check(cudaMalloc(&device_input, bytes), "allocate input");
    check(cudaMalloc(&device_decoded, bytes), "allocate output");
    check(cudaMalloc(&device_sign, sign.size() * sizeof(uint32_t)), "allocate sign");
    check(cudaMalloc(&device_decoded_sign, sign.size() * sizeof(uint32_t)),
          "allocate decoded sign");
    check(cudaMemcpy(
              device_sign, sign.data(), sign.size() * sizeof(uint32_t), cudaMemcpyHostToDevice),
          "copy sign");
    check(cudaMemcpy(device_input, input.data(), bytes, cudaMemcpyHostToDevice), "copy input");
    cudaStream_t stream{};
    check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "create stream");
    const uint32_t n = static_cast<uint32_t>(input.size());
    WALTZ::lossless::wzp_prealloc(n, stream);
    double encode_ms = 0, decode_ms = 0;
    uint32_t nnz = 0;
    const size_t compressed = WALTZ::lossless::wzp_encode_with_sign(
        device_input, device_sign, n, &nnz, &encode_ms, stream);
    if (const char* dump = std::getenv("WZP_DUMP")) {
        std::vector<unsigned char> archive(compressed);
        check(cudaMemcpy(archive.data(),
                         WALTZ::lossless::wzp_encoded_buf(),
                         archive.size(),
                         cudaMemcpyDeviceToHost),
              "archive copy");
        std::ofstream out(dump, std::ios::binary);
        out.write(reinterpret_cast<const char*>(archive.data()), archive.size());
    }
    WALTZ::lossless::wzp_decode_with_sign(WALTZ::lossless::wzp_encoded_buf(),
                                        n, device_decoded, device_decoded_sign, &decode_ms, stream);
    check(cudaMemcpy(decoded.data(), device_decoded, bytes, cudaMemcpyDeviceToHost),
          "copy decoded");
    check(cudaMemcpy(decoded_sign.data(), device_decoded_sign,
                     sign.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost),
          "copy decoded sign");
    size_t mismatch = 0;
    while (mismatch < input.size() && input[mismatch] == decoded[mismatch])
        ++mismatch;
    size_t sign_mismatch = 0;
    while (sign_mismatch < sign.size() && sign[sign_mismatch] == decoded_sign[sign_mismatch])
        ++sign_mismatch;
    const bool same = mismatch == input.size() && sign_mismatch == sign.size();
    std::printf("input=%zu compressed=%zu ratio=%.6f encode=%.6fms %.3fGB/s "
                "decode=%.6fms %.3fGB/s verify=%s mismatch=%zu sign_mismatch=%zu nnz=%u\n",
                bytes,
                compressed,
                static_cast<double>(bytes) / compressed,
                encode_ms,
                bytes / (encode_ms * 1.0e6),
                decode_ms,
                bytes / (decode_ms * 1.0e6),
                same ? "OK" : "FAIL",
                mismatch == input.size() ? 0 : mismatch,
                sign_mismatch == sign.size() ? 0 : sign_mismatch,
                nnz);
    cudaStreamDestroy(stream);
    cudaFree(device_decoded);
    cudaFree(device_decoded_sign);
    cudaFree(device_sign);
    cudaFree(device_input);
    return same ? 0 : 1;
}
