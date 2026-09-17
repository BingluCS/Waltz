#pragma once

#include "utils/Config.hpp"
#include "utils/def.hpp"
#include "utils/io.hpp"
#include "utils/Statistics.hpp"

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <stdexcept>

namespace WALTZ {

// Output capacity for the current fixed-cap side channels plus one input-volume
// worth of magnitude/header storage.  The extra MiB covers small-volume codec
// metadata and section alignment.
template <typename T> size_t compressed_buffer_capacity(uint64_t num) {
    constexpr uint64_t pwe_cap = uint64_t{1} << 22U;
    constexpr uint64_t qout_cap = uint64_t{1} << 20U;
    constexpr uint64_t safety_bytes = uint64_t{1} << 20U;
    const uint64_t pwe_count = num < pwe_cap ? num : pwe_cap;
    const uint64_t qout_count = num < qout_cap ? num : qout_cap;
    const uint64_t side_bytes = pwe_count * (sizeof(uint32_t) + sizeof(T)) +
                                qout_count * (sizeof(uint32_t) + sizeof(int32_t)) + safety_bytes;
    const uint64_t max_bytes = std::numeric_limits<size_t>::max();
    if (side_bytes > max_bytes || num > (max_bytes - side_bytes) / sizeof(T))
        throw std::overflow_error("Waltz compressed buffer capacity exceeds size_t");
    return static_cast<size_t>(num * sizeof(T) + side_bytes);
}

template <typename T, int NDIMS>
size_t d_WALTZ_compress(
    T* d_oridata, char* d_cmpData, const WALTZ::Config& config, const char* cmpPath, void* stream);

template <typename T>
void d_extrema(T* d_data, uint64_t n, double& out_min, double& out_max, void* stream);

template <typename T, int NDIMS>
size_t d_WALTZ_decompress(
    char* d_cmpData, T* d_decData, const WALTZ::Config& config, const char* decPath, void* stream);

// The caller resolves REL to conf.absErrorBound before entering this API.
// Keep relErrorBound as the autotuner's relative-bound hint.
template <typename T>
size_t d_compress(
    const WALTZ::Config& config, T* d_oridata, char* d_cmpData, const char* cmpPath, void* stream) {

    Config conf(config);
    if (conf.ndims != 3)
        throw std::invalid_argument("compression currently only supports 3D data.");
    const size_t cmpDataLen = d_WALTZ_compress<T, 3>(d_oridata, d_cmpData, conf, cmpPath, stream);

    return cmpDataLen;
}

// Decompress a blob produced by d_compress (device->device).  d_decData must hold
// config.dimx*config.dimy*config.dimz elements of T. Config must contain the exact
// original dimensions: the blob does not store dimensions or element count.
// T must match the compression type; no data-type tag is stored in the blob.
// Returns the decompressed byte count.
template <typename T>
size_t d_decompress(
    const WALTZ::Config& config, char* d_cmpData, T* d_decData, const char* decPath, void* stream) {
    Config conf(config);
    size_t decLen = 0;
    if (conf.ndims == 3) {
        decLen = d_WALTZ_decompress<T, 3>(d_cmpData, d_decData, conf, decPath, stream);
    } else {
        throw std::invalid_argument("decompression currently only supports 3D data.");
    }
    return decLen;
}

} // namespace WALTZ
