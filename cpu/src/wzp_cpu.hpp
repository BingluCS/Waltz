#pragma once

#include <cstddef>
#include <cstdint>

namespace waltz_cpu_detail {

// Standalone, little-endian implementation of the current Waltz GPU WZP
// bitstream (8192 magnitudes/chunk, 16 chunks/group, 24-bit metadata).
// Runtime functions use fixed-size stack scratch and no heap allocations.
// Invalid arguments, insufficient capacity, and malformed archives throw
// std::runtime_error. Input/output buffers must not overlap. Output buffers
// may have been modified when an exception is thrown. The current GPU
// format requires n <= UINT32_MAX/2 and total payload <= UINT32_MAX bytes.
//
// Signs are natural-index bitmaps: ceil(n / 32) uint32_t words are required,
// not a chunk-rounded allocation. Signs on zero magnitudes are discarded;
// decode clears unused bits in the final bitmap word. A full packed sign slot
// is 1024 bytes and is represented by sign code zero, not by an absent slot.
// n == 0 has an empty archive and permits null pointers.
// avx2 requests a runtime-checked AVX2 implementation; false always uses the
// scalar implementation. Both paths produce identical archives and outputs.
std::size_t wzp_max_bytes(std::uint32_t n);
std::size_t wzp_encode(const std::uint16_t* mag,
                       const std::uint32_t* sign,
                       std::uint32_t n,
                       std::uint8_t* out,
                       std::size_t capacity,
                       bool avx2 = true);
void wzp_decode(const std::uint8_t* in,
                std::size_t bytes,
                std::uint32_t n,
                std::uint16_t* mag,
                std::uint32_t* sign,
                bool avx2 = true);

} // namespace waltz_cpu_detail
