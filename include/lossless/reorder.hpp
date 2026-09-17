#pragma once

// Block-local Z-order indexing for fused quantization and inverse reorder.
// Odd dimensions use a ceil-half low region, matching SPERR's recursive order.

#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>

namespace WALTZ {
namespace lossless {

// Forward lookup used by producers that already own a row-major coefficient:
// table[local row-major index] = within-block z-order rank.
// Slot 0 is regular; slots 1..7 select short X/Y/Z extents with bits 0/1/2.
// Edge slots share one buffer, each with the regular block's stride.
__global__ inline void d_build_block_rank(uint16_t* table, uint32_t Bx, uint32_t By, uint32_t Bz,
                                         dim3 dims = dim3(0, 0, 0)) {
    const uint32_t shape = blockIdx.y;
    table += shape * Bx * By * Bz;
    if (shape & 1U) Bx = dims.x % Bx;
    if (shape & 2U) By = dims.y % By;
    if (shape & 4U) Bz = dims.z % Bz;
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t n = Bx * By * Bz;
    if (i < n) {
        // Block-local recursive Z-order rank (ceil-half low region).
        const uint64_t plane = static_cast<uint64_t>(Bx) * static_cast<uint64_t>(By);
        uint64_t x = static_cast<uint64_t>(i) % static_cast<uint64_t>(Bx);
        uint64_t y = (static_cast<uint64_t>(i) / static_cast<uint64_t>(Bx)) % By;
        uint64_t z = static_cast<uint64_t>(i) / plane;
        uint64_t z_rank = 0;
        uint64_t lx = Bx, ly = By, lz = Bz;

        while (lx > 1 || ly > 1 || lz > 1) {
            const uint64_t sx[2] = {lx - lx / 2, lx / 2};
            const uint64_t sy[2] = {ly - ly / 2, ly / 2};
            const uint64_t sz[2] = {lz - lz / 2, lz / 2};
            const unsigned int cx = (x >= sx[0]) ? 1U : 0U;
            const unsigned int cy = (y >= sy[0]) ? 1U : 0U;
            const unsigned int cz = (z >= sz[0]) ? 1U : 0U;

            // Closed form for the total volume of the octants ordered before this one
            // (child = cx | cy<<1 | cz<<2), replacing the per-octant inner loop:
            //   cz -> all 4 lower-z octants = sz0*lx*ly
            //   cy -> lower-y octants at this z = sz[cz]*sy0*lx
            //   cx -> lower-x octant at this y,z = sz[cz]*sy[cy]*sx0
            if (cz)
                z_rank += sz[0] * lx * ly;
            if (cy)
                z_rank += sz[cz] * sy[0] * lx;
            if (cx)
                z_rank += sz[cz] * sy[cy] * sx[0];

            if (cx)
                x -= sx[0];
            if (cy)
                y -= sy[0];
            if (cz)
                z -= sz[0];
            lx = sx[cx];
            ly = sy[cy];
            lz = sz[cz];
        }
        table[i] = static_cast<uint16_t>(z_rank);
    }
}

// Inverse lookup for fused WZP dequantization: Z-order rank -> row-major offset.
// Slot 0 is regular; slots 1..7 cover the short XYZ edge shapes.
__global__ inline void d_build_block_ioffset(
    uint32_t* table, uint32_t Bx, uint32_t By, uint32_t Bz, uint32_t nx, size_t nxy,
    dim3 dims = dim3(0, 0, 0)) {
    const uint32_t shape = blockIdx.y;
    table += shape * Bx * By * Bz;
    if (shape & 1U) Bx = dims.x % Bx;
    if (shape & 2U) By = dims.y % By;
    if (shape & 4U) Bz = dims.z % Bz;
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t n = Bx * By * Bz;
    if (i >= n)
        return;
    const uint32_t lx = i % Bx, ly = (i / Bx) % By, lz = i / (Bx * By);
    uint32_t rank;
    {
        // Block-local recursive Z-order rank (ceil-half low region).
        const uint64_t plane = static_cast<uint64_t>(Bx) * static_cast<uint64_t>(By);
        uint64_t x = static_cast<uint64_t>(i) % static_cast<uint64_t>(Bx);
        uint64_t y = (static_cast<uint64_t>(i) / static_cast<uint64_t>(Bx)) % By;
        uint64_t z = static_cast<uint64_t>(i) / plane;
        uint64_t z_rank = 0;
        uint64_t lx = Bx, ly = By, lz = Bz;

        while (lx > 1 || ly > 1 || lz > 1) {
            const uint64_t sx[2] = {lx - lx / 2, lx / 2};
            const uint64_t sy[2] = {ly - ly / 2, ly / 2};
            const uint64_t sz[2] = {lz - lz / 2, lz / 2};
            const unsigned int cx = (x >= sx[0]) ? 1U : 0U;
            const unsigned int cy = (y >= sy[0]) ? 1U : 0U;
            const unsigned int cz = (z >= sz[0]) ? 1U : 0U;

            // Closed form for the total volume of the octants ordered before this one
            // (child = cx | cy<<1 | cz<<2), replacing the per-octant inner loop:
            //   cz -> all 4 lower-z octants = sz0*lx*ly
            //   cy -> lower-y octants at this z = sz[cz]*sy0*lx
            //   cx -> lower-x octant at this y,z = sz[cz]*sy[cy]*sx0
            if (cz)
                z_rank += sz[0] * lx * ly;
            if (cy)
                z_rank += sz[cz] * sy[0] * lx;
            if (cx)
                z_rank += sz[cz] * sy[cy] * sx[0];

            if (cx)
                x -= sx[0];
            if (cy)
                y -= sy[0];
            if (cz)
                z -= sz[0];
            lx = sx[cx];
            ly = sy[cy];
            lz = sz[cz];
        }
        rank = static_cast<uint32_t>(z_rank);
    }
    const size_t offset = lx + static_cast<size_t>(ly) * nx + static_cast<size_t>(lz) * nxy;
    table[rank] = static_cast<uint32_t>(offset);
}

inline uint32_t select_block_zband(dim3 dims, int levels_z, bool use_dyadic) {
    if (use_dyadic) {
        const uint32_t d = 1U << levels_z;
        const uint32_t zb = (dims.z + d - 1U) / d;
        return zb < 1U ? 1U : (zb > 16U ? 16U : zb);
    }
    if (dims.z < 4U)
        return dims.z;

    const uint64_t nbx = (static_cast<uint64_t>(dims.x) + 15U) / 16U;
    const uint64_t nby = (static_cast<uint64_t>(dims.y) + 15U) / 16U;
    const uint64_t blocks = nbx * nby;
    const uint64_t full_blocks = static_cast<uint64_t>(dims.x / 16U) * (dims.y / 16U);
    return (blocks - full_blocks) * 32U >= blocks ? 4U : 2U;
}

} // namespace lossless
} // namespace WALTZ
