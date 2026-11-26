#ifndef WALTZ_CPU_REPAIR_AVX2_HPP
#define WALTZ_CPU_REPAIR_AVX2_HPP

#include <cstdint>

namespace waltz_cpu_detail {

// Native PWE collection and strict post-repair verification, without modifying
// original/reconstructed. Per sample, exactly as the scalar codec:
//   e = T(reconstructed[i] - original[i]);
//   patch = abs(e) > T(bound);
//   corrected = patch ? T(reconstructed[i] - e) : reconstructed[i];
// Record (i,e) iff patch; require native intermediates finite and the *exact*
// represented-value difference |corrected-original[i]| <= double bound.
// TwoDiff resolves strict-bound ties; there is no tolerance/epsilon allowance.
// Successful indices are ascending original indices and error bits are native.
//
// Arrays are native aligned and disjoint from outputs. Each output array must
// hold n entries; only the first *count are defined (SIMD compaction can store
// spare entries, but never outside that n-entry capacity). count is required,
// starts at zero, and may report a partial result on failure. For n==0 the
// arrays may be null. Nothing allocates memory or uses a container.
//
// The caller must establish FE_TONEAREST and preserve subnormals (FTZ/DAZ off),
// as codec FPGuard does. Invalid environment/bound/arguments return false.
// AVX2/FMA is runtime gated; unsupported CPUs use the same scalar rule.
template<class T>
bool collect_repairs(const T* original,const T* reconstructed,std::uint32_t n,double bound,
                     std::uint32_t* indices,T* errors,std::uint32_t* count,bool avx2) noexcept;

// Bitwise exponent classification: finite scans do not require a particular
// floating-point environment and do not raise FP exceptions for signaling
// NaNs. Empty scans accept null; nonempty null scans return false.
bool finite_f32(const float* values, std::uint32_t n) noexcept;
bool finite_f64(const double* values, std::uint32_t n) noexcept;
// Encoder-only destination is private scratch and may be partially filled on failure.
template<class T>
bool copy_finite(const T* input,T* dest,std::uint32_t n,bool avx2) noexcept;
bool repair_has_avx2() noexcept;

} // namespace waltz_cpu_detail
#endif
