//
// Created by Bing LU on 25/5/26.
//

#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cuda_runtime.h>
#include <limits>
#include <stdexcept>

#ifdef __CUDACC__
namespace waltz {
namespace detail {

template <int BW> struct matchby;
template <> struct matchby<4> {
    using utype = unsigned int;
    using itype = int;
};
template <> struct matchby<8> {
    using utype = unsigned long long;
    using itype = long long;
};

template <typename T> __device__ __forceinline__ T atomicMinFp(T* addr, T value) {
    using itype = typename matchby<sizeof(T)>::itype;
    using utype = typename matchby<sizeof(T)>::utype;
    if (!signbit(value)) {
        itype old = atomicMin(reinterpret_cast<itype*>(addr), *reinterpret_cast<itype*>(&value));
        return *reinterpret_cast<T*>(&old);
    }
    utype old = atomicMax(reinterpret_cast<utype*>(addr), *reinterpret_cast<utype*>(&value));
    return *reinterpret_cast<T*>(&old);
}

template <typename T> __device__ __forceinline__ T atomicMaxFp(T* addr, T value) {
    using itype = typename matchby<sizeof(T)>::itype;
    using utype = typename matchby<sizeof(T)>::utype;
    if (!signbit(value)) {
        itype old = atomicMax(reinterpret_cast<itype*>(addr), *reinterpret_cast<itype*>(&value));
        return *reinterpret_cast<T*>(&old);
    }
    utype old = atomicMin(reinterpret_cast<utype*>(addr), *reinterpret_cast<utype*>(&value));
    return *reinterpret_cast<T*>(&old);
}

template <typename T>
__global__ void extrema_kernel(const T* in, uint64_t len, T* minel, T* maxel, int R) {
    __shared__ T shared_minv, shared_maxv;
    const T failsafe = in[0];
    const uint64_t entry =
        static_cast<uint64_t>(blockDim.x) * static_cast<uint64_t>(R) * blockIdx.x + threadIdx.x;
    T tp_minv = failsafe, tp_maxv = failsafe;
    if (threadIdx.x == 0) {
        shared_minv = failsafe;
        shared_maxv = failsafe;
    }
    __syncthreads();
    for (int r = 0; r < R; ++r) {
        const uint64_t idx = entry + static_cast<uint64_t>(r) * blockDim.x;
        if (idx < len) {
            const T v = in[idx];
            tp_minv = min(tp_minv, v);
            tp_maxv = max(tp_maxv, v);
        }
    }
    __syncthreads();
    atomicMinFp<T>(&shared_minv, tp_minv);
    atomicMaxFp<T>(&shared_maxv, tp_maxv);
    __syncthreads();
    if (threadIdx.x == 0) {
        atomicMinFp<T>(minel, shared_minv);
        atomicMaxFp<T>(maxel, shared_maxv);
    }
}

} // namespace detail

// Compute min/max of d_in[0..len) on the GPU.  result is a host array of 2: {min, max}.
template <typename T>
inline void extrema_scan(const T* d_in, uint64_t len, T* result, cudaStream_t stream = 0) {
    if (len == 0)
        throw std::invalid_argument("extrema_scan requires at least one element");
    constexpr int nworker = 512;
    constexpr uint64_t chunk = 32768;
    constexpr int R = static_cast<int>(chunk / nworker); // elements per thread
    const uint64_t grid64 = (len - 1) / chunk + 1;
    if (grid64 > std::numeric_limits<uint32_t>::max())
        throw std::length_error("extrema_scan CUDA grid exceeds uint32 capacity");
    const uint32_t grid = static_cast<uint32_t>(grid64);

    T* d_minel = nullptr;
    cudaMalloc(reinterpret_cast<void**>(&d_minel), 2U * sizeof(T));
    T* d_maxel = d_minel + 1;
    cudaMemcpyAsync(d_minel, d_in, sizeof(T), cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(d_maxel, d_in, sizeof(T), cudaMemcpyDeviceToDevice, stream);
    detail::extrema_kernel<T>
        <<<grid, nworker, 0, stream>>>(d_in, len, d_minel, d_maxel, R);
    cudaMemcpyAsync(result, d_minel, 2U * sizeof(T), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    cudaFree(d_minel);
}

} // namespace waltz
#endif // __CUDACC__

// Host-side reporting shared by the compression and decompression pipelines.
namespace WALTZ {

// The CLI hides the untimed warm-up call's stage report.
inline thread_local bool Waltz_report_enabled = true;

struct CompressionTimes {
    double autotune, block_rank, dwt, quant, wzp, idwt, outlier;
};

struct DecompressionTimes {
    double block_offset, wzp, unquant, idwt, outlier;
};

// Preserve the existing binary-byte throughput convention and zero-time output.
inline double Waltz_throughput(size_t bytes, double ms) {
    return ms > 0.0 ? static_cast<double>(bytes) / (1024.0 * 1024.0 * 1024.0) / ms * 1000.0 : 0.0;
}

class WaltzE2ETimer {
    const bool enabled = std::getenv("WALTZ_REPORT_CE2E") != nullptr;
    const std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

  public:
    void report(const char* operation, size_t bytes, void* stream) const {
        if (!enabled)
            return;
        cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(stream));
        const double ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - begin).count();
        std::printf("  [WALTZ-CE2E] %s_ms=%.6f throughput_GBs=%.6f\n", operation, ms, Waltz_throughput(bytes, ms));
    }
};

inline void Waltz_print_stage(const char* name, size_t bytes, double ms,
                             const char* note = "") {
    std::printf("  %-12s %-12.6f %-10.4f%s\n", name, ms, Waltz_throughput(bytes, ms), note);
}

inline void Waltz_print_header(bool decompress = false) {
    std::printf("\n  \033[1m%-12s %-12s %-20s\033[0m%s\n",
                "type", "time (ms)", "throughput (GB/s)", decompress ? "  [decompress]" : "");
}

inline void Waltz_print_total(size_t bytes, double ms) {
    std::printf("  \033[1m%-12s %-12.6f %-10.4f\033[0m\n\n",
                "total", ms, Waltz_throughput(bytes, ms));
}

template <typename T>
inline void Waltz_print_compression(size_t bytes, const CompressionTimes& t,
                                   size_t mag_bytes, uint32_t pwe_count,
                                   uint32_t qout_count, size_t blob_bytes) {
    if (!Waltz_report_enabled)
        return;
    Waltz_print_header();
    Waltz_print_stage("autotune", bytes, t.autotune);
    Waltz_print_stage("block-rank", bytes, t.block_rank);
    Waltz_print_stage("DWT", bytes, t.dwt);
    Waltz_print_stage("quant", bytes, t.quant);
    Waltz_print_stage("mag+sign-WZP", bytes, t.wzp, "   (sign fused in)");
    Waltz_print_stage("IDWT", bytes, t.idwt);
    Waltz_print_stage("outlier", bytes, t.outlier);
    // WZP and reconstruction overlap; this is the existing stage total, not API e2e.
    Waltz_print_total(bytes, t.autotune + t.block_rank + t.dwt + t.quant +
                                std::max(t.wzp, t.idwt) + t.outlier);

    const size_t index_bytes = static_cast<size_t>(pwe_count) * sizeof(uint32_t);
    const size_t error_bytes = static_cast<size_t>(pwe_count) * sizeof(T);
    const size_t out_bytes = index_bytes + error_bytes;
    const size_t qout_bytes = static_cast<size_t>(qout_count) * (sizeof(uint32_t) + sizeof(int32_t));
    const size_t stored_bytes = blob_bytes != 0U ? blob_bytes : mag_bytes + out_bytes + qout_bytes;
    std::printf("  %-17s = %.5f\n", "Compression Ratio",
                static_cast<double>(bytes) / static_cast<double>(stored_bytes));
}

inline void Waltz_print_decompression(size_t bytes, size_t blob_bytes, const DecompressionTimes& t) {
    if (!Waltz_report_enabled)
        return;
    Waltz_print_header(true);
    Waltz_print_stage("block-offset", bytes, t.block_offset);
    Waltz_print_stage("WZP+sign", bytes, t.wzp, "   (sign fused in)");
    Waltz_print_stage("unquant", bytes, t.unquant);
    Waltz_print_stage("IDWT", bytes, t.idwt);
    Waltz_print_stage("outlier-fix", bytes, t.outlier);
    Waltz_print_total(bytes, t.block_offset + t.wzp + t.unquant + t.idwt + t.outlier);
    std::printf("  %-17s = %.5f\n", "Compression Ratio",
                static_cast<double>(bytes) / static_cast<double>(blob_bytes));
}

} // namespace WALTZ

template <typename T> void statistic(const T* ori_data, const T* data, size_t num_elements) {
    if (num_elements == 0)
        throw std::invalid_argument("statistic requires at least one element");
    double max_value = ori_data[0], min_value = ori_data[0];
    double max_err = 0.0, l2_err = 0.0;
    size_t max_err_idx = 0;
    for (size_t i = 0; i < num_elements; ++i) {
        if (max_value < ori_data[i])
            max_value = ori_data[i];
        if (min_value > ori_data[i])
            min_value = ori_data[i];
        // Preserve the previous T-precision subtraction and accumulation order.
        const double err = std::fabs(data[i] - ori_data[i]);
        if (max_err < err) {
            max_err = err;
            max_err_idx = i;
        }
        l2_err += err * err;
    }
    const double mse = l2_err / num_elements;
    const double range = max_value - min_value;
    const double psnr = 20 * std::log10(range) - 10 * std::log10(mse);
    const double nrmse = std::sqrt(mse) / range;
    std::printf("  Max_E  = %.15lf, idx = %zu\n", max_err, max_err_idx);
    std::printf("  Max_RE = %G\n", max_err / range);
    std::printf("  PSNR = %.7lf, NRMSE = %.10G\n", psnr, nrmse);
}
