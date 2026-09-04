#pragma once

#include "utils/Config.hpp"
#include "utils/def.hpp"
#include "utils/io.hpp"

#include <chrono>
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

// Warm one-time modules and workspaces before the REL scan.
template <typename T>
void d_pipeline_prealloc(const WALTZ::Config& config, void* stream);

// Standalone fast-tuner entry (validation harness; same code path as the
// pipeline's tuner).  Returns 0 = dyadic, 1 = XY+Z.
template <typename T>
int d_autotune_fast(T* d_data, uint32_t dimx, uint32_t dimy, uint32_t dimz, double absBound, double relHint,
    void* stream, double* out_ms);

template <typename T, int NDIMS>
size_t d_WALTZ_decompress(
    char* d_cmpData, T* d_decData, const WALTZ::Config& config, const char* decPath, void* stream);

template <typename T>
size_t d_compress(
    const WALTZ::Config& config, T* d_oridata, char* d_cmpData, const char* cmpPath, void* stream) {

    Config conf(config);
    const char* const warmup = std::getenv("WALTZ_ASYNC_ALLOC");
    if (warmup != nullptr && std::atoi(warmup) != 0)
        d_pipeline_prealloc<T>(conf, stream);
    // CAL_range: resolve a REL bound into the absolute bound HERE, so the
    // pipeline below only ever sees EB_ABS (conf.absErrorBound is updated).
    if (conf.errorBoundMode == EB_REL) {
        double dmin = 0.0, dmax = 0.0;
        d_extrema<T>(d_oridata, static_cast<uint64_t>(conf.num), dmin, dmax, stream);
        const double range = dmax - dmin;
        conf.absErrorBound = conf.relErrorBound * range;
        conf.errorBoundMode = EB_ABS;
        // std::printf( "[WALTZ] CAL_range: REL %.3e x range(min=%.6g max=%.6g range=%.6g) -> abs error %.6g\n", conf.relErrorBound,
        //     dmin, dmax, range, conf.absErrorBound);
    }

    const bool report_ce2e = std::getenv("WALTZ_REPORT_CE2E") != nullptr;
    const auto ce2e_begin = std::chrono::steady_clock::now();
    size_t cmpDataLen = 0;
    if (conf.ndims == 1) {
        // cmpDataLen = WALTZ_compress<T, 1>(conf, data, cmpDataPos, cmpDataCap);
    } else if (conf.ndims == 2) {
        // cmpDataLen = WALTZ_compress<T, 2>(conf, data, cmpDataPos, cmpDataCap);
    } else if (conf.ndims == 3) {
        cmpDataLen = d_WALTZ_compress<T, 3>(d_oridata, d_cmpData, conf, cmpPath, stream);
    } else if (conf.ndims == 4) {
        // cmpDataLen = WALTZ_compress<T, 4>(conf, data, cmpDataPos, cmpDataCap);
    } else {
        throw std::invalid_argument("Data dimension higher than 4 is not supported.");
    }
    if (report_ce2e) {
        cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(stream));
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - ce2e_begin).count();
        const double gib = static_cast<double>(conf.num) * sizeof(T) / (1024.0 * 1024.0 * 1024.0);
        std::printf("[WALTZ-CE2E] compress_ms=%.6f throughput_GBs=%.6f\n", ms, gib * 1000.0 / ms);
    }

    // auto cmpConfPos = reinterpret_cast<uchar *>(d_cmpData);

    // auto confSize = conf.save(cmpConfPos);
    // if (confSize > confEstSize) {
    //     throw std::length_error("buffer allocated for config is not large enough.");
    // }

    // return confSize + cmpDataLen;
    return cmpDataLen;
}

// Decompress a blob produced by d_compress (device->device).  d_decData must hold
// dimx*dimy*dimz elements of T.  Returns the decompressed byte count.
template <typename T>
size_t d_decompress(
    const WALTZ::Config& config, char* d_cmpData, T* d_decData, const char* decPath, void* stream) {
    Config conf(config);
    const bool report_ce2e = std::getenv("WALTZ_REPORT_CE2E") != nullptr;
    const auto ce2e_begin = std::chrono::steady_clock::now();
    size_t decLen = 0;
    if (conf.ndims == 3) {
        decLen = d_WALTZ_decompress<T, 3>(d_cmpData, d_decData, conf, decPath, stream);
    } else {
        throw std::invalid_argument("decompression currently only supports 3D data.");
    }
    if (report_ce2e) {
        cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(stream));
        const double ms =
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - ce2e_begin)
                .count();
        const double gib = static_cast<double>(conf.num) * sizeof(T) / (1024.0 * 1024.0 * 1024.0);
        std::printf("[WALTZ-CE2E] decompress_ms=%.6f throughput_GBs=%.6f\n", ms, gib * 1000.0 / ms);
    }
    return decLen;
}

} // namespace WALTZ
