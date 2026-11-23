#pragma once

#include <cstddef>
#include <cstdint>

namespace waltz_cpu_detail {

// Decode one <=16 x <=16 x <=32 brick directly into its row-major output
// view. rank[(z*ny+y)*nx+x] is a trusted permutation of [0,nx*ny*nz), and
// mag already points to the brick's first packed magnitude. sign is the
// complete natural packed-index bitmap; sign_base is the brick's first
// packed ELEMENT index, not a word offset. No padded input allocation is
// required. Magnitudes and rank need only uint16_t alignment.
//
// Computes v=T(mag[rank])*q, followed by the sign-selected unary minus, just
// like codec.cpp. In particular a set sign on a zero magnitude produces -0.
// The caller supplies the codec's native rounding/subnormal environment.
// Buffers must be disjoint; row_stride/plane_stride are in T elements.
//
// Returns true after writing the brick with runtime-checked AVX2. Returns
// false WITHOUT writing anything if avx2 is false, unavailable, or the brick
// is too narrow for the vector path; the caller then uses its scalar loop.
// There are no allocations and no full-volume temporary.
bool morton_decode_f32(const std::uint16_t* rank,
                       const std::uint16_t* mag,
                       const std::uint32_t* sign,
                       std::size_t sign_base,
                       float* output,
                       std::size_t row_stride,
                       std::size_t plane_stride,
                       std::uint32_t nx,
                       std::uint32_t ny,
                       std::uint32_t nz,
                       float q,
                       bool avx2 = true);
bool morton_decode_f64(const std::uint16_t* rank,
                       const std::uint16_t* mag,
                       const std::uint32_t* sign,
                       std::size_t sign_base,
                       double* output,
                       std::size_t row_stride,
                       std::size_t plane_stride,
                       std::uint32_t nx,
                       std::uint32_t ny,
                       std::uint32_t nz,
                       double q,
                       bool avx2 = true);

} // namespace waltz_cpu_detail
