//
// Created by Bing LU on 25/5/26.
//

#pragma once

#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

#ifdef __CUDACC__
#include <cstddef>
#include <cuda_runtime.h>

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
__global__ void extrema_kernel(const T* in, uint64_t len, T* minel, T* maxel, T failsafe, int R) {
    __shared__ T shared_minv, shared_maxv;
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

    T *d_minel = nullptr, *d_maxel = nullptr;
    cudaMalloc(reinterpret_cast<void**>(&d_minel), sizeof(T));
    cudaMalloc(reinterpret_cast<void**>(&d_maxel), sizeof(T));
    T failsafe = T(0); // init min/max to in[0]
    cudaMemcpyAsync(&failsafe, d_in, sizeof(T), cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(d_minel, d_in, sizeof(T), cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(d_maxel, d_in, sizeof(T), cudaMemcpyDeviceToDevice, stream);
    cudaStreamSynchronize(stream); // need `failsafe` on host before launch
    detail::extrema_kernel<T>
        <<<grid, nworker, 0, stream>>>(d_in, len, d_minel, d_maxel, failsafe, R);
    cudaMemcpyAsync(&result[0], d_minel, sizeof(T), cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(&result[1], d_maxel, sizeof(T), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    cudaFree(d_minel);
    cudaFree(d_maxel);
}

} // namespace waltz
#endif // __CUDACC__

template <typename T> void statistic(T* ori_data, T* data, size_t num_elements) {
    size_t i = 0;
    double psnr, nrmse, max_err, range, l2_err = 0;
    psnr = nrmse = max_err = range = 0;
    double Max = ori_data[0];
    double Min = ori_data[0];
    // float Max = ori_data[0];
    // float Min = ori_data[0];
    // float psnr, nrmse, max_err, range, l2_err = 0;
    // psnr = nrmse = max_err = range = 0;
    // max_err = fabs(data[0] - ori_data[0]);
    double diff_sum = 0;
    double maxpw_relerr = 0;
    double sum1 = 0, sum2 = 0;
    for (i = 0; i < num_elements; i++) {
        sum1 += ori_data[i];
        sum2 += data[i];
    }
    double mean1 = sum1 / num_elements;
    double mean2 = sum2 / num_elements;
    size_t max_err_idx = 0;

    double sum3 = 0, sum4 = 0;
    double prodSum = 0, relerr = 0;
    double* diff = (double*)malloc(num_elements * sizeof(double));

    for (i = 0; i < num_elements; i++) {
        diff[i] = data[i] - ori_data[i];
        diff_sum += data[i] - ori_data[i];
        if (Max < ori_data[i])
            Max = ori_data[i];
        if (Min > ori_data[i])
            Min = ori_data[i];
        double err = fabs(data[i] - ori_data[i]);
        if (ori_data[i] != 0) {
            relerr = err / fabs(ori_data[i]);
            if (maxpw_relerr < relerr)
                maxpw_relerr = relerr;
        }

        if (max_err < err) {
            max_err = err;
            max_err_idx = i;
        }
        prodSum += (ori_data[i] - mean1) * (data[i] - mean2);
        sum3 += (ori_data[i] - mean1) * (ori_data[i] - mean1);
        sum4 += (data[i] - mean2) * (data[i] - mean2);
        l2_err += err * err;
    }
    double std1 = sqrt(sum3 / num_elements);
    double std2 = sqrt(sum4 / num_elements);
    double ee = prodSum / num_elements;
    double acEff = ee / std1 / std2;

    double mse = l2_err / num_elements;
    range = Max - Min;
    psnr = 20 * log10(range) - 10 * log10(mse);
    nrmse = sqrt(mse) / range;

    // printf("[Verify]L2 error = %.10G\n", l2_err);
    // printf("[Verify]Min=%.20G, Max=%.20G, range=%.20G\n", Min, Max, range);
    printf("  Max_E  = %.15lf, idx = %d\n", max_err, (int)max_err_idx);
    printf("  Max_RE = %G\n", max_err / (Max - Min));
    //        printf("Max pw relative error = %.2G\n", maxpw_relerr);
    printf("  PSNR = %.7lf, NRMSE = %.10G\n", psnr, nrmse);
    //        printf("PSNR = %f, NRMSE= %.10G L2Error= %.10G\n", psnr, nrmse, l2_err);
    //        printf("acEff=%f\n", acEff);
    //        printf("errAutoCorr=%.10f\n", autocorrelation1DLag1<double>(diff, num_elements,
    //        diff_sum / num_elements));
    free(diff);
}

template <typename T, int decmp = 1, int progressive = 1>
void print_result(size_t size,
                  size_t total_compressed_size,
                  double targeteb,
                  double targetreb,
                  double time_enum,
                  double time_pred,
                  double time_bitplane,
                  double time_lossless) {
    printf("\n  \e[1m%-12s %-12s %-20s\e[0m\n",    //
           const_cast<char*>("type"),              //
           const_cast<char*>("time (ms)"),         //
           const_cast<char*>("throughtput (GB/s)") //
    );
    size_t total_bytes = size * sizeof(T);
    auto th = [total_bytes](auto time) {
        return 1.0 * total_bytes / 1024 / 1024 / 1024 / time * 1000;
    };
    if constexpr (decmp == 1) {
        if constexpr (progressive == 1) {
            printf("  %-12s %'-12f %'-10.4f\n", "loadStrategy", time_enum, th(time_enum));
        }
        printf("  %-12s %'-12f %'-10.4f\n", "ipredict", time_pred, th(time_pred));
        printf("  %-12s %'-12f %'-10.4f\n", "ibitplane", time_bitplane, th(time_bitplane));
        printf("  %-12s %'-12f %'-10.4f\n", "ilossless", time_lossless, th(time_lossless));
        double total_time = time_pred + time_bitplane + time_lossless;
        if constexpr (progressive == 1) {
        }
        total_time += time_enum;
        printf("  \e[1m%-12s\e[0m %'-12f %'-10.4f\n\n", "itotal", total_time, th(total_time));
        // printf("  target Error: %G\n", targeteb);
        // printf("  target RError: %G\n",  targetreb);
        // if constexpr (progressive == 0)
        //     printf("--------------------------------------\n");
    }
    if constexpr (decmp == 0) {
        printf("  %-12s %'-12f %'-10.4f\n", "predict", time_pred, th(time_pred));
        printf("  %-12s %'-12f %'-10.4f\n", "bitplane", time_bitplane, th(time_bitplane));
        printf("  %-12s %'-12f %'-10.4f\n", "lossless", time_lossless, th(time_lossless));
        double total_time = time_lossless + time_bitplane + time_pred;
        printf("  \e[1m%-12s\e[0m %'-12f %'-10.4f\n\n", "total", total_time, th(total_time));
        double cr = 1.0 * total_bytes / total_compressed_size;
        printf("  compression ratio: %lf\n", cr);
        printf("--------------------------------------\n");
    }
    // printf("  target Error: %G\n", config->target_ebs[i] * input->range);
    // printf("  target RError: %G\n",  config->target_ebs[i]);
    //
}
