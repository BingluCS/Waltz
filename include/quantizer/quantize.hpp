#pragma once

// Device-resident midtread quantize / dequantize for Waltz.
//
// Quantize is FUSED with optional dequantize: a single read of each wavelet
// coefficient produces both the quantized integer (for z-order -> LC) and, when
// EmitDequant is true (PWE only), the dequantized coefficient written back for the
// inverse-DWT verify path.  PSNR sets EmitDequant=false and skips the dequant write.
//
// Format: SIGNED quantized integers (sign carried in the value, templated on `Q`,
// default int16).  Bit-exact with the CPU oracles quantize_signed_int16 /
// dequantize_int16 in transformers/h_CDF97.hpp.  The magnitude/sign-bitmask split
// that SPECK wants is deferred to the LC stage.
//
// All entry points take device pointers + a stream; no malloc / H2D / D2H.

#include <cstdint>
#include <cuda_runtime.h>
#include <limits>
#include <type_traits>

namespace WALTZ {
namespace quantizer {

// llrint with the per-precision overload that matches std::llrint(Real) on the CPU
// (round to nearest, ties to even under FE_TONEAREST).
template <typename T> __device__ __forceinline__ long long llrint_real(T x) {
    if constexpr (std::is_same_v<T, float>)
        return llrintf(x);
    else if constexpr (std::is_same_v<T, double>)
        return llrint(x);
}

// Clamp a 64-bit rounded value into Q's range (generalises saturate_int16).
template <typename Q> __device__ __forceinline__ Q saturate_q(long long v) {
    constexpr long long lo = static_cast<long long>(std::numeric_limits<Q>::min());
    constexpr long long hi = static_cast<long long>(std::numeric_limits<Q>::max());
    if (v < lo)
        return static_cast<Q>(lo);
    if (v > hi)
        return static_cast<Q>(hi);
    return static_cast<Q>(v);
}

// Fused quantize.  coeff -> qint (signed, saturated); if EmitDequant also writes
// dequant[i] = (T)qint[i] * q (the value the decoder will reconstruct, so the PWE
// verify path is exact).  `dequant` may alias `coeff` (read happens before write)
// or be a distinct buffer; pass nullptr when EmitDequant is false.  No __restrict__
// on coeff/dequant precisely because they are allowed to alias.
template <typename T, typename Q, bool EmitDequant, bool Abs>
__global__ void quantize_kernel(
    const T* coeff, Q* __restrict__ qint, T* dequant, unsigned char* sign, uint32_t n, double q) {
    const T inv_q = (q != 0.0) ? static_cast<T>(1.0 / q) : static_cast<T>(0);
    const T qT = static_cast<T>(q);
    const uint32_t stride = gridDim.x * blockDim.x;
    for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
        const T prod = coeff[i] * inv_q;
        long long r = llrint_real(prod);
        // Abs: store the magnitude (sign dropped), like SPERR's |round(coeff/q)|.
        const Q qi = saturate_q<Q>(Abs ? (r < 0 ? -r : r) : r);
        qint[i] = qi;
        // Optional separate sign: 1 for negative, 0 for zero/positive (sparse -> RZE-friendly).
        if (sign != nullptr)
            sign[i] = (r < 0) ? static_cast<unsigned char>(1) : static_cast<unsigned char>(0);
        if (EmitDequant)
            dequant[i] = static_cast<T>(qi) * qT; // note: with Abs the sign is lost
    }
}

// Standalone inverse quantize (decompress path): qint -> coeff.
template <typename T, typename Q>
__global__ void
dequantize_kernel(const Q* __restrict__ qint, T* __restrict__ coeff, uint32_t n, double q) {
    const T qT = static_cast<T>(q);
    const uint32_t stride = gridDim.x * blockDim.x;
    for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride)
        coeff[i] = static_cast<T>(qint[i]) * qT;
}

// ---- Thin host launchers (device pointers, optional stream) -----------------

inline int quantize_grid(uint32_t n, int block) {
    long long g = (static_cast<long long>(n) + block - 1) / block;
    if (g < 1)
        g = 1;
    if (g > 65535)
        g = 65535; // grid-stride loop covers the rest
    return static_cast<int>(g);
}

// d_dequant == nullptr  -> PSNR (quantize only).
// d_dequant != nullptr  -> PWE  (also emit dequantized coeffs; may equal d_coeff).
// abs_value == true      -> store magnitudes (sign dropped), SPERR-style.
// d_sign != nullptr      -> also write a per-element sign byte (1=negative).
template <typename T, typename Q>
inline void quantize(const T* d_coeff,
                     Q* d_qint,
                     T* d_dequant,
                     uint32_t n,
                     double q,
                     cudaStream_t stream = 0,
                     bool abs_value = false,
                     unsigned char* d_sign = nullptr) {
    if (n == 0)
        return;
    constexpr int block = 256;
    const int grid = quantize_grid(n, block);
    const bool dq = (d_dequant != nullptr);
    if (dq && abs_value)
        quantize_kernel<T, Q, true, true>
            <<<grid, block, 0, stream>>>(d_coeff, d_qint, d_dequant, d_sign, n, q);
    else if (dq)
        quantize_kernel<T, Q, true, false>
            <<<grid, block, 0, stream>>>(d_coeff, d_qint, d_dequant, d_sign, n, q);
    else if (abs_value)
        quantize_kernel<T, Q, false, true>
            <<<grid, block, 0, stream>>>(d_coeff, d_qint, nullptr, d_sign, n, q);
    else
        quantize_kernel<T, Q, false, false>
            <<<grid, block, 0, stream>>>(d_coeff, d_qint, nullptr, d_sign, n, q);
}

template <typename T, typename Q>
inline void dequantize(const Q* d_qint, T* d_coeff, uint32_t n, double q, cudaStream_t stream = 0) {
    if (n == 0)
        return;
    constexpr int block = 256;
    const int grid = quantize_grid(n, block);
    dequantize_kernel<T, Q><<<grid, block, 0, stream>>>(d_qint, d_coeff, n, q);
}

} // namespace quantizer
} // namespace WALTZ
