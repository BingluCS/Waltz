#ifndef WALTZ_CPU_TRANSFORM_HPP
#define WALTZ_CPU_TRANSFORM_HPP

#include <cstddef>
#include <cstdint>

namespace waltz_cpu {

struct Shape {
    std::uint32_t nx, ny, nz;
};

// Number of T elements required by either transform. The scratch range must
// not overlap data. No transform allocates memory.
// Includes line-local compatibility state for very thin geometries; this is
// not necessarily equal to the number of input elements.
std::size_t workspace_elements(Shape shape);
bool transform_has_avx2() noexcept;

// In-place split-band/octant layout, x fastest. Both directions use synthesis
// 2 (HaloAllLevels).
// avx2=true requests runtime-checked AVX2/FMA; unavailable CPUs use scalar.
// Compile transform.cpp with -ffp-contract=off -fno-fast-math. The caller must
// use round-to-nearest and preserve subnormals for GPU bitwise compatibility.
// WALTZ_CPU_NOHALO enables full logical lines without halo tiles. Selected
// inverse endpoint expressions preserve the existing GPU arithmetic profile.
void forward_f32(float* data, float* scratch, Shape shape,
                 std::uint8_t levels_xy, std::uint8_t levels_z,
                 bool dyadic, std::uint8_t synthesis, bool avx2);
void forward_f64(double* data, double* scratch, Shape shape,
                 std::uint8_t levels_xy, std::uint8_t levels_z,
                 bool dyadic, std::uint8_t synthesis, bool avx2);
void inverse_f32(float* data, float* scratch, Shape shape,
                 std::uint8_t levels_xy, std::uint8_t levels_z,
                 bool dyadic, std::uint8_t synthesis, bool avx2);
void inverse_f64(double* data, double* scratch, Shape shape,
                 std::uint8_t levels_xy, std::uint8_t levels_z,
                 bool dyadic, std::uint8_t synthesis, bool avx2);

} // namespace waltz_cpu

#endif
