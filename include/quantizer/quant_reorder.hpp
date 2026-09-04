#pragma once

// Fused quantize + 16x16x(z_band) block z-order reorder.
//
// A single output-driven kernel replaces the separate quantize() + reorder pass:
// iterating the REORDERED output positions, each thread computes its source
// row-major index s, reads ONE coefficient, quantizes it, and writes
//   * the MAGNITUDE   to the reordered position  d_qint_z[b0 + r]   (coalesced)
//   * the NEGATIVE bit to the reordered bitmap position (warp-packed for aligned
//     full blocks; sparse atomicOr on edge/misaligned blocks)
//   * (quant only) the signed DEQUANT value to row-major d_iquant[s] (block-local
//     scatter; feeds the inverse-DWT / PWE verify path).
// The reorder is a bijection, so every coefficient is read exactly once.  Full
// 16x16xBz blocks gather their source coords from the prebuilt iperm table; edge
// blocks use the inline octree rank.  Base offset is the separable closed form.
//
//   quant       -> EmitDequant=true  (PWE: also writes d_iquant)
//   quant_psnr  -> EmitDequant=false (PSNR: no dequant, no d_iquant buffer)
//
// Sign bits are in REORDERED (z-order) positions, so the sign nz-compaction must rank
// nonzeros over the reordered magnitudes.

#include "quantize.hpp" // llrint_real, saturate_q
#include "lossless/reorder.hpp" // zorder_rank, block_zband, build_block_iperm

#include <cstdint>
#include <cuda_runtime.h>

namespace WALTZ {
namespace quantizer {

// Full-tile table used by the default non-shared path. Unlike the generic iperm
// table, each rank directly stores its row-major source offset relative to the
// tile origin.  nx/nxy are fixed for a volume, so the hot quant/unquant kernels
// avoid unpacking Morton coordinates and two per-element stride multiplies.
// The current Waltz API is uint32-indexed, hence every valid source offset fits
// in uint32_t.  Edge tiles retain the generic octree path and do not read this
// table.
__global__ inline void d_build_block_ioffset(
    uint32_t* table, uint32_t Bx, uint32_t By, uint32_t Bz, uint32_t nx, size_t nxy) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t n = Bx * By * Bz;
    if (i >= n)
        return;
    const uint32_t lx = i % Bx, ly = (i / Bx) % By, lz = i / (Bx * By);
    const uint32_t rank = static_cast<uint32_t>(lossless::zorder_rank(i, Bx, By, Bz));
    const size_t offset = lx + static_cast<size_t>(ly) * nx + static_cast<size_t>(lz) * nxy;
    table[rank] = static_cast<uint32_t>(offset);
}

inline void build_block_ioffset(uint32_t* d_table,
                                uint32_t Bx,
                                uint32_t By,
                                uint32_t Bz,
                                uint32_t nx,
                                uint32_t ny,
                                cudaStream_t stream = 0) {
    const uint32_t n = Bx * By * Bz;
    d_build_block_ioffset<<<(n + 255) / 256, 256, 0, stream>>>(
        d_table, Bx, By, Bz, nx, static_cast<size_t>(nx) * ny);
}

// Magnitudes are stored as uint16 (the LC payload).  The rare |mag| > 65535 cases are
// CLAMPED in qint_z and recorded exactly as "q_outliers" {z-order position, true int32
// magnitude} so the decoder can patch them back -- the dequant output (and therefore the
// reconstruction / PWE outliers) uses the TRUE magnitude.
// Sign is a 1-bit-per-element NEGATIVE bitmap (bit set <=> coefficient < 0), z-ordered.
// The zero state is implicit (mag == 0), so 1 bit suffices.  In a full block whose
// output base is 32-bit aligned, each warp owns one complete consecutive bitmap word:
// ballot packs its 32 signs and lane 0 writes that word directly.  Edge/misaligned
// blocks retain sparse atomicOr into the pre-zeroed bitmap (the launcher memsets it).
template <typename T, bool EmitDequant, bool OffsetTable = false>
__global__ void d_quant_reorder_kernel(const T* coeff,
                                       uint16_t* __restrict__ qint_z,
                                       uint32_t* __restrict__ sign_bm,
                                       T* iquant,
                                       uint32_t nx,
                                       uint32_t ny,
                                       uint32_t nz,
                                       uint32_t Bx,
                                       uint32_t By,
                                       uint32_t Bz,
                                       uint32_t nbx,
                                       uint32_t nby,
                                       const uint16_t* __restrict__ table,
                                       double q,
                                       uint32_t* __restrict__ qout_idx,
                                       int32_t* __restrict__ qout_val,
                                       unsigned int* __restrict__ qout_cnt,
                                       uint32_t qout_cap) {
    const uint32_t bi = blockIdx.x;
    const uint32_t bx = bi % nbx;
    const uint32_t by = (bi / nbx) % nby;
    const uint32_t bz = bi / (static_cast<size_t>(nbx) * nby);
    const uint32_t x0 = bx * Bx, y0 = by * By, z0 = bz * Bz;
    const uint32_t bdx = min(Bx, nx - x0), bdy = min(By, ny - y0), bdz = min(Bz, nz - z0);
    const size_t nxy = static_cast<size_t>(nx) * ny;
    const size_t b0 = static_cast<size_t>(nx) * ny * z0 + static_cast<size_t>(nx) * bdz * y0 +
                      static_cast<size_t>(bdy) * bdz * x0;
    const T inv_q = (q != 0.0) ? static_cast<T>(1.0 / q) : static_cast<T>(0);
    const T qT = static_cast<T>(q);
    const bool full = (bdx == Bx && bdy == By && bdz == Bz);
    const uint32_t bn = full ? (Bx * By * Bz) : (bdx * bdy * bdz);
    size_t src0 = 0;
    if constexpr (OffsetTable)
        src0 = x0 + static_cast<size_t>(y0) * nx + static_cast<size_t>(z0) * nxy;
    // With a warp-multiple launch, a full, word-aligned output segment consists of
    // disjoint 32-element ranges.  Thus every active warp can replace up to 32
    // contending atomicOr operations with one exact bitmap-word store.  Keep the
    // explicit guards so future block shapes/launch sizes safely fall back.
    // Ballot/store only amortises for shallow, medium-width volumes. Large planes
    // and deep cubes are gather/scatter dominated, while very sparse signs make
    // negative-only atomics cheaper. Keep this predicate uniform per launch.
    const bool warp_sign_shape = nx <= 512u && ny <= 512u && nz <= 128u;
    const bool warp_sign_words = warp_sign_shape && full && ((b0 & 31u) == 0u) &&
                                 ((bn & 31u) == 0u) && ((blockDim.x & 31u) == 0u);
    for (uint32_t r = threadIdx.x; r < bn; r += blockDim.x) {
        uint32_t lx, ly, lz;
        size_t outpos;
        size_t s;
        if (full) {
            if constexpr (OffsetTable) {
                outpos = b0 + r;
                s = src0 + reinterpret_cast<const uint32_t*>(table)[r];
            } else {
                outpos = b0 + r;
                const uint32_t p = table[r];
                lx = p & 31u;
                ly = (p >> 5) & 31u;
                lz = (p >> 10) & 31u;
                s = (x0 + lx) + static_cast<size_t>(y0 + ly) * nx +
                    static_cast<size_t>(z0 + lz) * nxy;
            }
        } else { // r is the linear within-block index; octree gives the rank
            lx = r % bdx;
            ly = (r / bdx) % bdy;
            lz = r / (bdx * bdy);
            outpos = b0 + lossless::zorder_rank(r, bdx, bdy, bdz);
            s = (x0 + lx) + static_cast<size_t>(y0 + ly) * nx + static_cast<size_t>(z0 + lz) * nxy;
        }
        const long long rll = llrint_real(coeff[s] * inv_q);
        const bool neg = (rll < 0);
        const int32_t mag = saturate_q<int32_t>(neg ? -rll : rll);
        qint_z[outpos] = static_cast<uint16_t>(mag > 65535 ? 65535 : mag);
        if (mag > 65535 && qout_cnt != nullptr) { // record the exact value (decoder patch)
            const unsigned int slot = atomicAdd(qout_cnt, 1u);
            if (slot < qout_cap) {
                qout_idx[slot] =
                    static_cast<uint32_t>(s); // ROW-MAJOR index (decoder patches post-unquant)
                qout_val[slot] = mag;
            }
        }
        const bool sign = neg && mag != 0;
        if (warp_sign_words) {
            // outpos is consecutive across the warp and lane 0 is word-aligned.
            const uint32_t bits = __ballot_sync(0xffffffffu, sign);
            if ((threadIdx.x & 31u) == 0u)
                sign_bm[outpos >> 5] = bits;
        } else if (sign) {
            // Edge blocks use a rank scatter and misaligned full blocks can share a
            // word with a neighbour, so retain the race-safe sparse update there.
            atomicOr(sign_bm + (outpos >> 5), 1u << (outpos & 31u));
        }
        if (EmitDequant)
            iquant[s] = static_cast<T>(neg ? -mag : mag) * qT; // signed reconstruction (TRUE mag)
    }
}

// INVERSE of d_quant_reorder_kernel (decompression): read the z-ordered u16 magnitude +
// negative bit at outpos, write the row-major dequantized coefficient
// iquant[s] = (+/-) mag * q.  Same block decomposition / table gather as the forward.
template <typename T, bool OffsetTable = false>
__global__ void d_unquant_reorder_kernel(const uint16_t* __restrict__ qint_z,
                                         const uint32_t* __restrict__ sign_bm,
                                         T* __restrict__ iquant,
                                         uint32_t nx,
                                         uint32_t ny,
                                         uint32_t nz,
                                         uint32_t Bx,
                                         uint32_t By,
                                         uint32_t Bz,
                                         uint32_t nbx,
                                         uint32_t nby,
                                         const uint16_t* __restrict__ table,
                                         double q) {
    const uint32_t bi = blockIdx.x;
    const uint32_t bx = bi % nbx;
    const uint32_t by = (bi / nbx) % nby;
    const uint32_t bz = bi / (static_cast<size_t>(nbx) * nby);
    const uint32_t x0 = bx * Bx, y0 = by * By, z0 = bz * Bz;
    const uint32_t bdx = min(Bx, nx - x0), bdy = min(By, ny - y0), bdz = min(Bz, nz - z0);
    const size_t nxy = static_cast<size_t>(nx) * ny;
    const size_t b0 = static_cast<size_t>(nx) * ny * z0 + static_cast<size_t>(nx) * bdz * y0 +
                      static_cast<size_t>(bdy) * bdz * x0;
    const T qT = static_cast<T>(q);
    const bool full = (bdx == Bx && bdy == By && bdz == Bz);
    const uint32_t bn = full ? (Bx * By * Bz) : (bdx * bdy * bdz);
    size_t src0 = 0;
    if constexpr (OffsetTable)
        src0 = x0 + static_cast<size_t>(y0) * nx + static_cast<size_t>(z0) * nxy;
    for (uint32_t r = threadIdx.x; r < bn; r += blockDim.x) {
        uint32_t lx, ly, lz;
        size_t outpos;
        size_t s;
        if (full) {
            outpos = b0 + r;
            if constexpr (OffsetTable) {
                s = src0 + reinterpret_cast<const uint32_t*>(table)[r];
            } else {
                const uint32_t p = table[r];
                lx = p & 31u;
                ly = (p >> 5) & 31u;
                lz = (p >> 10) & 31u;
                s = (x0 + lx) + static_cast<size_t>(y0 + ly) * nx +
                    static_cast<size_t>(z0 + lz) * nxy;
            }
        } else {
            lx = r % bdx;
            ly = (r / bdx) % bdy;
            lz = r / (bdx * bdy);
            outpos = b0 + lossless::zorder_rank(r, bdx, bdy, bdz);
            s = (x0 + lx) + static_cast<size_t>(y0 + ly) * nx + static_cast<size_t>(z0 + lz) * nxy;
        }
        const T mag = static_cast<T>(qint_z[outpos]);
        const uint32_t bit = (sign_bm[outpos >> 5] >> (outpos & 31u)) & 1u;
        iquant[s] = bit ? -mag * qT : mag * qT; // mag==0 -> writes (+/-)0
    }
}

// ---- Host launchers ---------------------------------------------------------
// d_table: caller-owned scratch of at least 16*16*16 uint32. The table is
// rebuilt on-device.
// d_qint_z / d_sign_bm are reordered outputs.
// d_qout_*: q-outlier record buffers (idx/val/count, capacity qout_cap); pass count =
// nullptr to clamp silently (the recorded values are required for exact decoding).

template <typename T>
inline void quant(const T* d_coeff,
                  uint16_t* d_qint_z,
                  uint32_t* d_sign_bm,
                  T* d_iquant,
                  dim3 dims,
                  int levels_z,
                  uint16_t* d_table,
                  double q,
                  cudaStream_t stream = 0,
                  uint32_t* d_qout_idx = nullptr,
                  int32_t* d_qout_val = nullptr,
                  unsigned int* d_qout_cnt = nullptr,
                  uint32_t qout_cap = 0) {
    const uint32_t zb = lossless::block_zband(dims.z, levels_z);
    const uint32_t nbx = (dims.x + 15u) / 16u, nby = (dims.y + 15u) / 16u,
                   nbz = (dims.z + zb - 1u) / zb;
    const size_t nblocks = static_cast<size_t>(nbx) * nby * nbz;
    if (nblocks == 0)
        return;
    const size_t n = static_cast<size_t>(dims.x) * dims.y * dims.z;
    cudaMemsetAsync(
        d_sign_bm, 0, ((n + 31) / 32) * 4, stream); // sparse atomicOr needs zeroed bitmap
    build_block_ioffset(
        reinterpret_cast<uint32_t*>(d_table), 16, 16, zb, dims.x, dims.y, stream);
    d_quant_reorder_kernel<T, true, true>
        <<<static_cast<int>(nblocks), 256, 0, stream>>>(d_coeff,
                                                        d_qint_z,
                                                        d_sign_bm,
                                                        d_iquant,
                                                        dims.x,
                                                        dims.y,
                                                        dims.z,
                                                        16,
                                                        16,
                                                        zb,
                                                        nbx,
                                                        nby,
                                                        d_table,
                                                        q,
                                                        d_qout_idx,
                                                        d_qout_val,
                                                        d_qout_cnt,
                                                        qout_cap);
}

template <typename T>
inline void quant_psnr(const T* d_coeff,
                       uint16_t* d_qint_z,
                       uint32_t* d_sign_bm,
                       dim3 dims,
                       int levels_z,
                       uint16_t* d_table,
                       double q,
                       cudaStream_t stream = 0,
                       uint32_t* d_qout_idx = nullptr,
                       int32_t* d_qout_val = nullptr,
                       unsigned int* d_qout_cnt = nullptr,
                       uint32_t qout_cap = 0) {
    const uint32_t zb = lossless::block_zband(dims.z, levels_z);
    const uint32_t nbx = (dims.x + 15u) / 16u, nby = (dims.y + 15u) / 16u,
                   nbz = (dims.z + zb - 1u) / zb;
    const size_t nblocks = static_cast<size_t>(nbx) * nby * nbz;
    if (nblocks == 0)
        return;
    const size_t n = static_cast<size_t>(dims.x) * dims.y * dims.z;
    cudaMemsetAsync(
        d_sign_bm, 0, ((n + 31) / 32) * 4, stream); // sparse atomicOr needs zeroed bitmap
    build_block_ioffset(
        reinterpret_cast<uint32_t*>(d_table), 16, 16, zb, dims.x, dims.y, stream);
    d_quant_reorder_kernel<T, false, true>
        <<<static_cast<int>(nblocks), 256, 0, stream>>>(d_coeff,
                                                        d_qint_z,
                                                        d_sign_bm,
                                                        nullptr,
                                                        dims.x,
                                                        dims.y,
                                                        dims.z,
                                                        16,
                                                        16,
                                                        zb,
                                                        nbx,
                                                        nby,
                                                        d_table,
                                                        q,
                                                        d_qout_idx,
                                                        d_qout_val,
                                                        d_qout_cnt,
                                                        qout_cap);
}

// Decompression: z-ordered u16 magnitudes + 3-state sign bytes -> row-major dequantized
// coefficients (the IDWT input).  Builds the iperm table on-device like the forward.
template <typename T>
inline void unquant(const uint16_t* d_qint_z,
                    const uint32_t* d_sign_bm,
                    T* d_iquant,
                    dim3 dims,
                    int levels_z,
                    uint16_t* d_table,
                    double q,
                    cudaStream_t stream = 0) {
    const uint32_t zb = lossless::block_zband(dims.z, levels_z);
    const uint32_t nbx = (dims.x + 15u) / 16u, nby = (dims.y + 15u) / 16u,
                   nbz = (dims.z + zb - 1u) / zb;
    const size_t nblocks = static_cast<size_t>(nbx) * nby * nbz;
    if (nblocks == 0)
        return;
    build_block_ioffset(
        reinterpret_cast<uint32_t*>(d_table), 16, 16, zb, dims.x, dims.y, stream);
    d_unquant_reorder_kernel<T, true>
        <<<static_cast<int>(nblocks), 256, 0, stream>>>(d_qint_z,
                                                        d_sign_bm,
                                                        d_iquant,
                                                        dims.x,
                                                        dims.y,
                                                        dims.z,
                                                        16,
                                                        16,
                                                        zb,
                                                        nbx,
                                                        nby,
                                                        d_table,
                                                        q);
}

} // namespace quantizer
} // namespace WALTZ
