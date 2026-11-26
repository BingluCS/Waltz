#ifndef WALTZ_CPU_QUANTIZE_AVX2_HPP
#define WALTZ_CPU_QUANTIZE_AVX2_HPP

#include <cstdint>
#include <cstddef>
#include "transform.hpp"

namespace waltz_cpu_detail {

// Quantize native coefficients in place, retaining the signed, int32-saturated
// magnitude separately for subsequent row-major/Morton packing. For each i:
//   scaled = T(data[i] * native_inv);
//   rounded = nearest-even signed int64(scaled);
//   code = sign(rounded) * min(abs(rounded), INT32_MAX);
//   data[i] = code ? T(code) * native_q : T(+0);
// Signs of rounded zero (including negative-zero inputs) are discarded.
//
// Dispatches to AVX2/FMA-targeted 8-float / 4-double kernels only when CPU/OS
// support is present, otherwise uses the same scalar arithmetic. No global
// architecture flag, allocation, STL container, or padding is required.
//
// The caller must establish FE_TONEAREST and preserve subnormals (no FTZ/DAZ),
// as the outer CPU codec's FPGuard does. Incorrect rounding/FTZ/DAZ settings
// return false, rather than silently changing archive bytes. native_q must be
// positive finite; native_inv must be finite and nonnegative (zero is allowed
// for an underflowed reciprocal). For n>0, arrays contain n native-aligned,
// nonoverlapping elements. n==0 accepts null pointers and performs no access.
//
// Returns false for invalid arguments/environment, nonfinite scaled values, or
// abs(scaled)>=2^63. On a domain failure earlier elements may already have been
// written; callers must abort encoding and discard the partial result. As in
// the scalar codec, later synthesis/strict-bound checks validate dequantized
// representability; these helpers do not promise that inconsistent q/inv
// arguments cannot produce a nonfinite dequantized coefficient.
bool quantize_f32(float* data, std::uint32_t n, float native_q, float native_inv,
                   std::int32_t* signed_magnitudes) noexcept;
bool quantize_f64(double* data, std::uint32_t n, double native_q, double native_inv,
                   std::int32_t* signed_magnitudes) noexcept;
bool quantize_has_avx2() noexcept;

// Internal codec path. The caller has validated q/inv and the FP environment.
// data/mag point to this brick, sign and exceptions to the full volume.
bool quantize_morton_f32(float* data, waltz_cpu::Shape field, waltz_cpu::Shape brick,
                    const std::uint16_t* rank, std::uint16_t* mag, std::uint32_t* sign,
                    std::size_t origin, std::size_t base, float q, float inv,
                    std::uint32_t* indices, std::uint32_t* values, std::uint32_t& count) noexcept;

} // namespace waltz_cpu_detail
#endif
