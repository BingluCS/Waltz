#pragma once

// Rounding and saturation helpers used by fused DWT quantization.
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

} // namespace quantizer
} // namespace WALTZ
