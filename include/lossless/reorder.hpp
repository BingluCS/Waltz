#pragma once

// Device-resident z-order (Morton) reorder for Waltz.
//
// Reorders a row-major chunk of quantized integers into the same z-order curve as
// SPERR's z_order_sort (recursive split rule `length - length/2` for the low half,
// see SPERR src/SPECK_FLT.cpp visit_z_order_region).  Element-typed (works on the
// signed quantized ints produced by lossless/quantize.hpp, sign carried along).
//
// Unlike transformers/h_CDF97.hpp::reorder_zorder_cuda (which mallocs + H2D + D2H
// every call), everything here is device-to-device: the caller owns d_src / d_dst.
//
// Power-of-two cubes use 8^3 shared-memory tiles. Arbitrary dimensions use the
// general octree traversal, or a reusable block permutation table in production.

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cuda_runtime.h>

namespace WALTZ {
namespace lossless {

// Expand/extract one of three interleaved 10-bit lanes from a Morton code.
__device__ __forceinline__ uint32_t compact_morton3(uint32_t v) {
    v &= 0x09249249U;
    v = (v ^ (v >> 2U)) & 0x030c30c3U;
    v = (v ^ (v >> 4U)) & 0x0300f00fU;
    v = (v ^ (v >> 8U)) & 0xff0000ffU;
    v = (v ^ (v >> 16U)) & 0x000003ffU;
    return v;
}

// Row-major src index -> z-order rank, matching SPERR visit_z_order_region exactly.
__device__ __forceinline__ size_t zorder_rank(size_t src_idx,
                                              uint32_t dimx,
                                              uint32_t dimy,
                                              uint32_t dimz) {
    const uint64_t plane = static_cast<uint64_t>(dimx) * static_cast<uint64_t>(dimy);
    uint64_t x = static_cast<uint64_t>(src_idx) % static_cast<uint64_t>(dimx);
    uint64_t y = (static_cast<uint64_t>(src_idx) / static_cast<uint64_t>(dimx)) % dimy;
    uint64_t z = static_cast<uint64_t>(src_idx) / plane;
    uint64_t rank = 0;
    uint64_t lx = dimx, ly = dimy, lz = dimz;

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
            rank += sz[0] * lx * ly;
        if (cy)
            rank += sz[cz] * sy[0] * lx;
        if (cx)
            rank += sz[cz] * sy[cy] * sx[0];

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
    return static_cast<size_t>(rank);
}

// Fast path for a power-of-two CUBE of edge D: each block stages one 8x8x8 tile in
// shared memory and writes it out in Morton order, so global accesses are coalesced
// and the reorder happens entirely in shared (no random global scatter).  Reads a
// sub-cube at origin (x0,y0,z0) of a volume with row/plane strides (nx, nxy) and
// writes to dst[dst_base + Morton].  For pow2 D the SPERR split equals an exact
// halving, so this reproduces the z-order.  (Contiguous whole-cube: x0=y0=z0=0,
// nx=D, nxy=D*D, dst_base=0.)
template <typename U>
__global__ void d_zorder_tiled_pow2(const U* src,
                                    U* dst,
                                    uint32_t D,
                                    uint32_t x0,
                                    uint32_t y0,
                                    uint32_t z0,
                                    uint32_t nx,
                                    size_t nxy,
                                    size_t dst_base) {
    constexpr uint32_t TE = 8;
    constexpr uint32_t TV = TE * TE * TE; // 512
    __shared__ U tile[TV];

    const uint32_t row = threadIdx.x;  // 0..511 (raster within tile)
    const uint32_t trank = blockIdx.x; // 0..(D/8)^3-1 (Morton tile index)
    const uint32_t tx = compact_morton3(trank);
    const uint32_t ty = compact_morton3(trank >> 1U);
    const uint32_t tz = compact_morton3(trank >> 2U);

    const uint32_t gx = x0 + tx * TE + (row & 7U);
    const uint32_t gy = y0 + ty * TE + ((row >> 3U) & 7U);
    const uint32_t gz = z0 + tz * TE + (row >> 6U);
    tile[row] =
        src[static_cast<size_t>(gx) + static_cast<size_t>(gy) * nx + static_cast<size_t>(gz) * nxy];
    __syncthreads();

    const uint32_t mx = compact_morton3(row);
    const uint32_t my = compact_morton3(row >> 1U);
    const uint32_t mz = compact_morton3(row >> 2U);
    dst[dst_base + static_cast<size_t>(trank) * TV + row] = tile[mx + my * TE + mz * TE * TE];
}

// General path (any dims): per-element octree descent, coalesced reads / scattered
// writes.  dst[rank(i)] = src[i].
template <typename U>
__global__ void
d_zorder_general(const U* src, U* dst, size_t total, uint32_t dimx, uint32_t dimy, uint32_t dimz) {
    const size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;
    for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < total;
         i += stride) {
        dst[zorder_rank(i, dimx, dimy, dimz)] = src[i];
    }
}

// ---- Host launchers (device pointers, optional stream) ----------------------

inline int reorder_grid(size_t total, int block, int max_grid = 4096) {
    long long g = (static_cast<long long>(total) + block - 1) / block;
    if (g < 1)
        g = 1;
    if (g > max_grid)
        g = max_grid;
    return static_cast<int>(g);
}

// Returns true if `v` is a power of two and >= 8.
inline bool host_pow2(uint32_t v) {
    return v >= 8u && (v & (v - 1u)) == 0u;
}

// One-shot WHOLE-VOLUME reorder.  pow2 cube -> tiled (shared-memory, coalesced);
// otherwise the general per-element octree (random scatter).
template <typename U>
inline void reorder_zorder(const U* d_src, U* d_dst, dim3 dims, cudaStream_t stream = 0) {
    const size_t total = static_cast<size_t>(dims.x) * dims.y * dims.z;
    if (total == 0)
        return;
    if (dims.x == dims.y && dims.y == dims.z && host_pow2(dims.x)) {
        const uint32_t D = dims.x;
        const uint32_t t = D / 8;
        d_zorder_tiled_pow2<U><<<t * t * t, 512, 0, stream>>>(
            d_src, d_dst, D, 0, 0, 0, D, static_cast<size_t>(D) * D, 0);
    } else {
        constexpr int block = 256;
        const int grid = reorder_grid(total, block);
        d_zorder_general<U>
            <<<grid, block, 0, stream>>>(d_src, d_dst, total, dims.x, dims.y, dims.z);
    }
}

// ---- Fast Bx*By*Bz block reorder via a PRE-BUILT iperm table ------------------
// For a FIXED block size the within-block octree (rank<->coords) is one fixed
// mapping of Bx*By*Bz entries.  Build it ONCE (octree evaluated Bx*By*Bz times
// total, on-device, no H2D): table[rank] = packed local coords (10b each).  The
// reorder is then a coalesced-write GATHER with a table lookup -- no per-element
// octree.  Edge blocks fall back to the octree scatter.
__global__ inline void d_build_block_iperm(uint16_t* table, uint32_t Bx, uint32_t By, uint32_t Bz) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t n = Bx * By * Bz;
    if (i >= n)
        return;
    const uint32_t lx = i % Bx, ly = (i / Bx) % By, lz = i / (Bx * By);
    const uint32_t rank = static_cast<uint32_t>(zorder_rank(i, Bx, By, Bz));
    table[rank] =
        static_cast<uint16_t>(lx | (ly << 5) | (lz << 10)); // 5-bit fields (Bx,By,Bz <= 32)
}

// Forward lookup used by producers that already own a row-major coefficient:
// table[local row-major index] = within-block z-order rank.
__global__ inline void d_build_block_rank(uint16_t* table, uint32_t Bx, uint32_t By, uint32_t Bz) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t n = Bx * By * Bz;
    if (i < n)
        table[i] = static_cast<uint16_t>(zorder_rank(i, Bx, By, Bz));
}

// Build the iperm table on-device (one-time; table size = Bx*By*Bz uint16).
inline void build_block_iperm(
    uint16_t* d_table, uint32_t Bx, uint32_t By, uint32_t Bz, cudaStream_t stream = 0) {
    const uint32_t n = Bx * By * Bz;
    d_build_block_iperm<<<(n + 255) / 256, 256, 0, stream>>>(d_table, Bx, By, Bz);
}

template <typename U>
__global__ void d_block_table_kernel(const U* __restrict__ src,
                                     U* __restrict__ dst,
                                     uint32_t nx,
                                     uint32_t ny,
                                     uint32_t nz,
                                     uint32_t Bx,
                                     uint32_t By,
                                     uint32_t Bz,
                                     uint32_t nbx,
                                     uint32_t nby,
                                     const uint16_t* __restrict__ table) {
    const uint32_t bi = blockIdx.x;
    const uint32_t bx = bi % nbx;
    const uint32_t by = (bi / nbx) % nby;
    const uint32_t bz = bi / (static_cast<size_t>(nbx) * nby);
    const uint32_t x0 = bx * Bx, y0 = by * By, z0 = bz * Bz;
    const uint32_t bdx = min(Bx, nx - x0), bdy = min(By, ny - y0), bdz = min(Bz, nz - z0);
    const size_t nxy = static_cast<size_t>(nx) * ny;
    const size_t b0 = static_cast<size_t>(nx) * ny * z0 + static_cast<size_t>(nx) * bdz * y0 +
                      static_cast<size_t>(bdy) * bdz * x0;
    if (bdx == Bx && bdy == By && bdz == Bz) { // full block -> table gather (coalesced write)
        const uint32_t bn = Bx * By * Bz;
        for (uint32_t r = threadIdx.x; r < bn; r += blockDim.x) {
            const uint32_t p = table[r];
            dst[b0 + r] = src[(x0 + (p & 31u)) + static_cast<size_t>(y0 + ((p >> 5) & 31u)) * nx +
                              static_cast<size_t>(z0 + ((p >> 10) & 31u)) * nxy];
        }
    } else { // edge block -> within-block octree scatter
        const uint32_t bn = bdx * bdy * bdz;
        for (uint32_t i = threadIdx.x; i < bn; i += blockDim.x) {
            const uint32_t lx = i % bdx, ly = (i / bdx) % bdy, lz = i / (bdx * bdy);
            dst[b0 + zorder_rank(i, bdx, bdy, bdz)] =
                src[(x0 + lx) + static_cast<size_t>(y0 + ly) * nx +
                    static_cast<size_t>(z0 + lz) * nxy];
        }
    }
}

template <typename U>
inline void reorder_block_table(const U* d_src,
                                U* d_dst,
                                dim3 dims,
                                uint32_t Bx,
                                uint32_t By,
                                uint32_t Bz,
                                const uint16_t* d_table,
                                cudaStream_t stream = 0) {
    const uint32_t nbx = (dims.x + Bx - 1) / Bx, nby = (dims.y + By - 1) / By,
                   nbz = (dims.z + Bz - 1) / Bz;
    const size_t nblocks = static_cast<size_t>(nbx) * nby * nbz;
    if (nblocks == 0)
        return;
    d_block_table_kernel<U><<<static_cast<int>(nblocks), 256, 0, stream>>>(
        d_src, d_dst, dims.x, dims.y, dims.z, Bx, By, Bz, nbx, nby, d_table);
}

// ---- Production reorder: 16 x 16 x z_band block z-order --------------------
// Dyadic transforms use the coarsest Z extent, capped at 16. A plane transform
// normally uses two slices, which best preserves long XY zero runs. Four slices
// are used only when at least 1/32 of the XY bricks are partial: in that case the
// smaller block count repays the edge-address overhead without applying the CR
// tradeoff to a volume with only a negligible boundary. This O(1) model reads no
// samples and never recognizes field names or exact dimensions.
inline uint32_t block_zband(uint32_t nz, int levels_z) {
    const uint32_t d = 1u << levels_z;
    const uint32_t zb = (nz + d - 1) / d;
    return zb < 1u ? 1u : (zb > 16u ? 16u : zb);
}

inline uint32_t select_block_zband(dim3 dims, int levels_z, bool use_dyadic) {
    if (use_dyadic)
        return block_zband(dims.z, levels_z);
    if (dims.z <= 1U)
        return 1U;

    const uint64_t nbx = (static_cast<uint64_t>(dims.x) + 15U) / 16U;
    const uint64_t nby = (static_cast<uint64_t>(dims.y) + 15U) / 16U;
    const uint64_t full_x = dims.x / 16U;
    const uint64_t full_y = dims.y / 16U;
    const uint64_t blocks = nbx * nby;
    const uint64_t edge_blocks = blocks - full_x * full_y;
    const bool edge_heavy = edge_blocks * 32U >= blocks;
    return edge_heavy && dims.z >= 4U ? 4U : 2U;
}

} // namespace lossless
} // namespace WALTZ
