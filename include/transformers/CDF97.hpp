#pragma once

#include "lossless/reorder.hpp"
#include "quantizer/quantize.hpp"
#include "utils/Config.hpp"

#include <algorithm>
#include <cstdlib>
#include <cooperative_groups.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stddef.h>
#include <stdint.h>
#include <type_traits>

#ifdef WALTZ_P4_TRANSFORM
#define CDF97_L0 1.1335628809201543
#define CDF97_L1 0.24748737341529164
#define CDF97_L2 -0.12374368670764582
#define CDF97_L3 0.0
#define CDF97_L4 0.016581654018824540
#define CDF97_H0 0.70710678118654746
#define CDF97_H1 -0.40092954493277239
#define CDF97_H2 0.0
#define CDF97_H3 0.047376154339498683
#else
#define CDF97_L0 0.852698679008896
#define CDF97_L1 0.377402855612830
#define CDF97_L2 -0.110624404418436
#define CDF97_L3 -0.023849465019557
#define CDF97_L4 0.037828455507264
#define CDF97_H0 -0.788485616405213
#define CDF97_H1 0.418092273222947
#define CDF97_H2 0.040689417609764
#define CDF97_H3 -0.064538882629244
#endif

#define TPB 384

#define CS 8192
#define CS_LARGE_X 12000
#define CS_PLANE 10000
#define CDF97_MAX_LEVELS 5

// Forward and inverse transforms share the 384-thread block size.  Min resident
// blocks/SM caps registers so enough warps remain to hide memory latency.  Split
// by precision: inverse float kernels hit 4 blocks at <=42 regs, but the double
// kernels need more live registers (double address math) and spill to local
// memory if forced that low, so they stay at 3.  Forward targets 3/2 blocks.
// Only sm_80
// (A100, 164KB smem) and sm_90 (H100, 228KB) can afford 4x the ~34KB tile; the
// lower-shared parts (sm_70/86/89) are shared-limited anyway, so we don't over-
// constrain their register budget.  Selected per-T in launch_bounds.
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 800 || __CUDA_ARCH__ >= 900)
#define CDF97_MIN_BLOCKS_F 4
#define CDF97_MIN_BLOCKS_D 3
#define CDF97_DWT_MIN_BLOCKS_F 3
#define CDF97_DWT_MIN_BLOCKS_D 2
#else
#define CDF97_MIN_BLOCKS_F 2
#define CDF97_MIN_BLOCKS_D 2
#define CDF97_DWT_MIN_BLOCKS_F 2
#define CDF97_DWT_MIN_BLOCKS_D 2
#endif
#define CDF97_MIN_BLOCKS(T) (std::is_same_v<T, float> ? CDF97_MIN_BLOCKS_F : CDF97_MIN_BLOCKS_D)
#define CDF97_DWT_MIN_BLOCKS(T) (std::is_same_v<T, float> ? CDF97_DWT_MIN_BLOCKS_F : CDF97_DWT_MIN_BLOCKS_D)

namespace CDF97 {

using WALTZ::DWTConfig;

static __device__ unsigned int x_count[CDF97_MAX_LEVELS];
static __device__ unsigned int y_count[CDF97_MAX_LEVELS];
static __device__ unsigned int z_count[CDF97_MAX_LEVELS];

// Device-side state for fused double DWT + quantization. Keeping it in one
// record avoids expanding the cooperative-kernel ABI and carrying the stream
// pointers through transform phases that do not emit finalized coefficients.
struct DwtDoubleQuantParams {
    uint16_t* qint_z;
    uint32_t* sign_bm;
    double* iquant;
    const uint16_t* rank_table;
    uint32_t q_bz;
    uint32_t block_nx;
    uint32_t block_ny;
    uint32_t bz_shift;
    uint32_t block_elems;
    uint32_t regular_blocks;
    double inv_q;
    double q;
    uint32_t* qout_idx;
    int32_t* qout_val;
    unsigned int* qout_cnt;
    uint32_t qout_cap;
};

static __device__ DwtDoubleQuantParams double_quant_params;

__device__ __forceinline__ void dwt_quant_emit_double(
    double value, size_t s, uint32_t x, uint32_t y, uint32_t z, dim3 data_dims,
    const DwtDoubleQuantParams* __restrict__ p) {
    constexpr uint32_t BX = 16U, BY = 16U;
    const uint32_t bz_size = p->q_bz;
    const uint32_t x0 = x & ~(BX - 1U);
    const uint32_t y0 = y & ~(BY - 1U);
    uint32_t z0, lz;
    if ((bz_size & (bz_size - 1U)) == 0U) {
        z0 = z & ~(bz_size - 1U);
        lz = z & (bz_size - 1U);
    } else {
        z0 = (z / bz_size) * bz_size;
        lz = z - z0;
    }
    const uint32_t bdx = min(BX, data_dims.x - x0);
    const uint32_t bdy = min(BY, data_dims.y - y0);
    const uint32_t bdz = min(bz_size, data_dims.z - z0);
    const uint32_t lx = x & (BX - 1U), ly = y & (BY - 1U);
    const uint32_t local = lx + ly * bdx + lz * bdx * bdy;
    const size_t base = static_cast<size_t>(data_dims.x) * data_dims.y * z0 +
                        static_cast<size_t>(data_dims.x) * bdz * y0 +
                        static_cast<size_t>(bdy) * bdz * x0;
    const bool full = bdx == BX && bdy == BY && bdz == bz_size;
    const uint32_t rank =
        full ? p->rank_table[local]
             : static_cast<uint32_t>(WALTZ::lossless::zorder_rank(local, bdx, bdy, bdz));
    const size_t outpos = base + rank;
    const long long rll = WALTZ::quantizer::llrint_real(value * p->inv_q);
    const bool neg = rll < 0;
    const int32_t mag = WALTZ::quantizer::saturate_q<int32_t>(neg ? -rll : rll);
    p->qint_z[outpos] = static_cast<uint16_t>(mag > 65535 ? 65535 : mag);
    if (neg && mag != 0)
        atomicOr(p->sign_bm + (outpos >> 5U), 1U << (outpos & 31U));
    p->iquant[s] = static_cast<double>(neg ? -mag : mag) * p->q;
    if (mag > 65535 && p->qout_cnt != nullptr) {
        const unsigned int slot = atomicAdd(p->qout_cnt, 1U);
        if (slot < p->qout_cap) {
            p->qout_idx[slot] = static_cast<uint32_t>(s);
            p->qout_val[slot] = mag;
        }
    }
}

__device__ __forceinline__ void dwt_quant_emit_regular_double(
    double value, size_t s, uint32_t x, uint32_t y, uint32_t z,
    const DwtDoubleQuantParams* __restrict__ p) {
    constexpr uint32_t BX = 16U, BY = 16U;
    const uint32_t bz = p->q_bz;
    const uint32_t block = (x >> 4U) + p->block_nx * (y >> 4U) +
                           p->block_nx * p->block_ny * (z >> p->bz_shift);
    const uint32_t local = (x & (BX - 1U)) + (y & (BY - 1U)) * BX +
                           (z & (bz - 1U)) * BX * BY;
    const size_t outpos =
        static_cast<size_t>(block) * p->block_elems + p->rank_table[local];
    const long long rll = WALTZ::quantizer::llrint_real(value * p->inv_q);
    const bool neg = rll < 0;
    const int32_t mag = WALTZ::quantizer::saturate_q<int32_t>(neg ? -rll : rll);
    p->qint_z[outpos] = static_cast<uint16_t>(mag > 65535 ? 65535 : mag);
    if (neg && mag != 0)
        atomicOr(p->sign_bm + (outpos >> 5U), 1U << (outpos & 31U));
    p->iquant[s] = static_cast<double>(neg ? -mag : mag) * p->q;
    if (mag > 65535 && p->qout_cnt != nullptr) {
        const unsigned int slot = atomicAdd(p->qout_cnt, 1U);
        if (slot < p->qout_cap) {
            p->qout_idx[slot] = static_cast<uint32_t>(s);
            p->qout_val[slot] = mag;
        }
    }
}

// Reset the cooperative-kernel work-distribution counters (atomicAdd chunk dispatch);
// must run before each forward dwt3d / dwt3d_plane launch.
static __global__ void d_reset() {
    for (int i = 0; i < CDF97_MAX_LEVELS; ++i) {
        x_count[i] = 0U;
        y_count[i] = 0U;
        z_count[i] = 0U;
    }
}

template <typename T> __device__ __forceinline__ T add_rn(T a, T b);

template <> __device__ __forceinline__ float add_rn<float>(float a, float b) {
    return __fadd_rn(a, b);
}

template <> __device__ __forceinline__ double add_rn<double>(double a, double b) {
    return __dadd_rn(a, b);
}

template <typename T>
__device__ __forceinline__ T low_filter(T center, T sum1, T sum2, T sum3, T sum4) {
    if constexpr (std::is_same_v<T, float>) {
        float acc = static_cast<float>(CDF97_L0) * center;
        acc = __fmaf_rn(static_cast<float>(CDF97_L1), sum1, acc);
        acc = __fmaf_rn(static_cast<float>(CDF97_L2), sum2, acc);
        acc = __fmaf_rn(static_cast<float>(CDF97_L3), sum3, acc);
        return __fmaf_rn(static_cast<float>(CDF97_L4), sum4, acc);
    } else if constexpr (std::is_same_v<T, double>) {
        double acc = CDF97_L0 * center;
        acc = __fma_rn(CDF97_L1, sum1, acc);
        acc = __fma_rn(CDF97_L2, sum2, acc);
        acc = __fma_rn(CDF97_L3, sum3, acc);
        return __fma_rn(CDF97_L4, sum4, acc);
    }
}

template <typename T> __device__ __forceinline__ T high_filter(T center, T sum1, T sum2, T sum3) {
    if constexpr (std::is_same_v<T, float>) {
        float acc = static_cast<float>(CDF97_H0) * center;
        acc = __fmaf_rn(static_cast<float>(CDF97_H1), sum1, acc);
        acc = __fmaf_rn(static_cast<float>(CDF97_H2), sum2, acc);
        return __fmaf_rn(static_cast<float>(CDF97_H3), sum3, acc);
    } else if constexpr (std::is_same_v<T, double>) {
        double acc = CDF97_H0 * center;
        acc = __fma_rn(CDF97_H1, sum1, acc);
        acc = __fma_rn(CDF97_H2, sum2, acc);
        return __fma_rn(CDF97_H3, sum3, acc);
    }
}

template <typename T, bool Chain2 = false>
__device__ __forceinline__ void dwt_y_rows(T* tmp,
                                           const T* input,
                                           uint32_t row0,
                                           uint32_t rows,
                                           uint32_t active_x,
                                           uint32_t active_y,
                                           dim3 data_leaps) {
    const uint32_t even_y = active_y - (active_y >> 1);
    const int stride_y = data_leaps.y;

    if constexpr (Chain2 && sizeof(T) == sizeof(float)) {
        // Convert the half-open flattened row chunk to its unique range of
        // even-Y owners.  This mapping is valid for both even and odd heights
        // and never assigns the same owner to adjacent row chunks.
        const uint32_t row1 = row0 + rows;
        const uint32_t z0 = row0 / active_y;
        const uint32_t y0 = row0 - z0 * active_y;
        const uint32_t z1 = row1 / active_y;
        const uint32_t y1 = row1 - z1 * active_y;
        const uint32_t pair0 = z0 * even_y + ((y0 + 1U) >> 1U);
        const uint32_t pair1 = z1 * even_y + ((y1 + 1U) >> 1U);
        const uint32_t pair_count = pair1 - pair0;

        for (uint32_t e = threadIdx.x; e < pair_count * active_x; e += blockDim.x) {
            const uint32_t q = pair0 + e / active_x;
            const int idx = static_cast<int>(e % active_x);
            const int gidz = static_cast<int>(q / even_y);
            const uint32_t pair = q - static_cast<uint32_t>(gidz) * even_y;
            const int gidy = static_cast<int>(pair << 1U);

            // Even local pair indices lead a possible two-pair chain.  An odd
            // owner is skipped only when its preceding even owner is in this
            // same row chunk and had a fully interior 11-sample union window.
            if ((pair & 1U) != 0U) {
                const int prev_y = gidy - 2;
                const bool consumed = q > pair0 && pair > 0U && prev_y >= 4 &&
                                      prev_y + 6 < static_cast<int>(active_y);
                if (consumed)
                    continue;
            }

            const bool chain = (pair & 1U) == 0U && q + 1U < pair1 && pair + 1U < even_y &&
                               gidy >= 4 && gidy + 6 < static_cast<int>(active_y);
            const int global_id = idx + gidy * data_leaps.y + gidz * data_leaps.z;

            if (chain) {
                // Two adjacent even owners have an 11-value union.  Keep only
                // nine values live for the first pair, store it, then reuse c0
                // and c1 for the final two loads.  Every add/filter expression
                // retains the scalar single-pair operand and evaluation order.
                T c0 = input[global_id - 4 * stride_y];
                T c1 = input[global_id - 3 * stride_y];
                const T c2 = input[global_id - 2 * stride_y];
                const T c3 = input[global_id - stride_y];
                const T c4 = input[global_id];
                const T c5 = input[global_id + stride_y];
                const T c6 = input[global_id + 2 * stride_y];
                const T c7 = input[global_id + 3 * stride_y];
                const T c8 = input[global_id + 4 * stride_y];

                const T sum1 = add_rn<T>(c3, c5);
                const T sum2 = add_rn<T>(c2, c6);
                const T sum3 = add_rn<T>(c1, c7);
                const T sum4 = add_rn<T>(c0, c8);
                const int low0 = idx + static_cast<int>(pair) * data_leaps.y + gidz * data_leaps.z;
                tmp[low0] = low_filter<T>(c4, sum1, sum2, sum3, sum4);

                const T h1 = add_rn<T>(c4, c6);
                const T h2 = add_rn<T>(c3, c7);
                const T h3 = add_rn<T>(c2, c8);
                const int high0 =
                    idx + static_cast<int>(even_y + pair) * data_leaps.y + gidz * data_leaps.z;
                tmp[high0] = high_filter<T>(c5, h1, h2, h3);

                c0 = input[global_id + 5 * stride_y];
                c1 = input[global_id + 6 * stride_y];
                const T sum5 = add_rn<T>(c5, c7);
                const T sum6 = add_rn<T>(c4, c8);
                const T sum7 = add_rn<T>(c3, c0);
                const T sum8 = add_rn<T>(c2, c1);
                const int low1 =
                    idx + static_cast<int>(pair + 1U) * data_leaps.y + gidz * data_leaps.z;
                tmp[low1] = low_filter<T>(c6, sum5, sum6, sum7, sum8);

                const T h4 = add_rn<T>(c6, c8);
                const T h5 = add_rn<T>(c5, c0);
                const T h6 = add_rn<T>(c4, c1);
                const int high1 =
                    idx + static_cast<int>(even_y + pair + 1U) * data_leaps.y + gidz * data_leaps.z;
                tmp[high1] = high_filter<T>(c7, h4, h5, h6);
                continue;
            }

            // Boundary, row-chunk, z-slice, and tail owners retain the exact
            // original one-pair path, including whole-sample reflection.
            int ym1 = gidy - 1, yp1 = gidy + 1;
            int ym2 = gidy - 2, yp2 = gidy + 2;
            int ym3 = gidy - 3, yp3 = gidy + 3;
            int ym4 = gidy - 4, yp4 = gidy + 4;
            if (ym1 < 0)
                ym1 = -ym1;
            if (ym2 < 0)
                ym2 = -ym2;
            if (ym3 < 0)
                ym3 = -ym3;
            if (ym4 < 0)
                ym4 = -ym4;
            if (yp1 >= static_cast<int>(active_y))
                yp1 = 2 * active_y - 2 - yp1;
            if (yp2 >= static_cast<int>(active_y))
                yp2 = 2 * active_y - 2 - yp2;
            if (yp3 >= static_cast<int>(active_y))
                yp3 = 2 * active_y - 2 - yp3;
            if (yp4 >= static_cast<int>(active_y))
                yp4 = 2 * active_y - 2 - yp4;

            const T c0 = input[global_id];
            const T vm1 = input[global_id + (ym1 - gidy) * stride_y];
            const T vp1 = input[global_id + (yp1 - gidy) * stride_y];
            const T vm2 = input[global_id + (ym2 - gidy) * stride_y];
            const T vp2 = input[global_id + (yp2 - gidy) * stride_y];
            const T vm3 = input[global_id + (ym3 - gidy) * stride_y];
            const T vp3 = input[global_id + (yp3 - gidy) * stride_y];
            const T vm4 = input[global_id + (ym4 - gidy) * stride_y];
            const T vp4 = input[global_id + (yp4 - gidy) * stride_y];
            const T sum1 = add_rn<T>(vm1, vp1);
            const T sum2 = add_rn<T>(vm2, vp2);
            const T sum3 = add_rn<T>(vm3, vp3);
            const T sum4 = add_rn<T>(vm4, vp4);
            const int low_out = idx + static_cast<int>(pair) * data_leaps.y + gidz * data_leaps.z;
            tmp[low_out] = low_filter<T>(c0, sum1, sum2, sum3, sum4);
            if (gidy + 1 < static_cast<int>(active_y)) {
                const T h1 = add_rn<T>(c0, vp2);
                const T h2 = add_rn<T>(vm1, vp3);
                const T h3 = add_rn<T>(vm2, vp4);
                const int high_out =
                    idx + static_cast<int>(even_y + pair) * data_leaps.y + gidz * data_leaps.z;
                tmp[high_out] = high_filter<T>(vp1, h1, h2, h3);
            }
        }
        return;
    }

    if ((active_y & 1U) != 0U) {
        // Pair-fuse odd-height slices too.  A naive flattened row+=2 traversal is
        // incorrect here because row parity flips at every odd-height z boundary.
        // Map each half-open input-row chunk [row0,row0+rows) to its unique range
        // of even-y owners instead: F(r)=slice*ceil(N/2)+ceil(local_y/2).
        // An owner emits the even low and its adjacent odd high from one 9-point
        // window.  Therefore a chunk beginning on an odd row does no duplicate
        // work for that row: the preceding chunk's even owner already emitted it.
        const uint32_t row1 = row0 + rows;
        const uint32_t z0 = row0 / active_y;
        const uint32_t y0 = row0 - z0 * active_y;
        const uint32_t z1 = row1 / active_y;
        const uint32_t y1 = row1 - z1 * active_y;
        const uint32_t pair0 = z0 * even_y + ((y0 + 1U) >> 1U);
        const uint32_t pair1 = z1 * even_y + ((y1 + 1U) >> 1U);
        const uint32_t pair_count = pair1 - pair0;

        for (uint32_t e = threadIdx.x; e < pair_count * active_x; e += blockDim.x) {
            const uint32_t q = pair0 + e / active_x;
            const int idx = static_cast<int>(e % active_x);
            const int gidz = static_cast<int>(q / even_y);
            const int pair = static_cast<int>(q - static_cast<uint32_t>(gidz) * even_y);
            const int gidy = pair << 1;
            const int global_id = idx + gidy * data_leaps.y + gidz * data_leaps.z;

            int ym1 = gidy - 1, yp1 = gidy + 1;
            int ym2 = gidy - 2, yp2 = gidy + 2;
            int ym3 = gidy - 3, yp3 = gidy + 3;
            int ym4 = gidy - 4, yp4 = gidy + 4;
            if (ym1 < 0)
                ym1 = -ym1;
            if (ym2 < 0)
                ym2 = -ym2;
            if (ym3 < 0)
                ym3 = -ym3;
            if (ym4 < 0)
                ym4 = -ym4;
            if (yp1 >= static_cast<int>(active_y))
                yp1 = 2 * active_y - 2 - yp1;
            if (yp2 >= static_cast<int>(active_y))
                yp2 = 2 * active_y - 2 - yp2;
            if (yp3 >= static_cast<int>(active_y))
                yp3 = 2 * active_y - 2 - yp3;
            if (yp4 >= static_cast<int>(active_y))
                yp4 = 2 * active_y - 2 - yp4;

            const T c0 = input[global_id];
            const T vm1 = input[global_id + (ym1 - gidy) * stride_y];
            const T vp1 = input[global_id + (yp1 - gidy) * stride_y];
            const T vm2 = input[global_id + (ym2 - gidy) * stride_y];
            const T vp2 = input[global_id + (yp2 - gidy) * stride_y];
            const T vm3 = input[global_id + (ym3 - gidy) * stride_y];
            const T vp3 = input[global_id + (yp3 - gidy) * stride_y];
            const T vm4 = input[global_id + (ym4 - gidy) * stride_y];
            const T vp4 = input[global_id + (yp4 - gidy) * stride_y];

            const T sum1 = add_rn<T>(vm1, vp1);
            const T sum2 = add_rn<T>(vm2, vp2);
            const T sum3 = add_rn<T>(vm3, vp3);
            const T sum4 = add_rn<T>(vm4, vp4);
            const int low_out = idx + pair * data_leaps.y + gidz * data_leaps.z;
            tmp[low_out] = low_filter<T>(c0, sum1, sum2, sum3, sum4);

            if (gidy + 1 < static_cast<int>(active_y)) {
                const T h1 = add_rn<T>(c0, vp2);
                const T h2 = add_rn<T>(vm1, vp3);
                const T h3 = add_rn<T>(vm2, vp4);
                const int high_out = idx + (even_y + pair) * data_leaps.y + gidz * data_leaps.z;
                tmp[high_out] = high_filter<T>(vp1, h1, h2, h3);
            }
        }
        return;
    }

    const uint32_t first_low = (row0 & 1U) ? 1U : 0U;
    const uint32_t low_rows = (first_low < rows) ? ((rows - first_low + 1U) >> 1U) : 0U;

    // Pair-fused (always): each even row loads its 9-tap y-window once and emits
    // both the low coefficient (row gidy) and the high coefficient (row gidy+1),
    // reusing the same shifted window -> ~1/2 the neighbour loads of split low/high
    // passes. Bit-identical to the removed split path (same operands and order).
    {
        for (uint32_t i = threadIdx.x; i < low_rows * active_x; i += blockDim.x) {
            const uint32_t row = row0 + first_low + ((i / active_x) << 1U);
            const int idx = i % active_x;
            const int gidy = row % active_y;
            const int gidz = row / active_y;
            const int global_id = idx + gidy * data_leaps.y + gidz * data_leaps.z;

            // Interior rows are the overwhelmingly common case on the wide
            // plane levels.  Avoid four reflection-index branches and the
            // associated integer arithmetic; the boundary/tail rows below
            // retain the exact whole-sample mirror semantics.
            if (gidy >= 4 && gidy + 4 < static_cast<int>(active_y)) {
                const T c0 = input[global_id];
                const T vm1 = input[global_id - stride_y];
                const T vp1 = input[global_id + stride_y];
                const T vm2 = input[global_id - 2 * stride_y];
                const T vp2 = input[global_id + 2 * stride_y];
                const T vm3 = input[global_id - 3 * stride_y];
                const T vp3 = input[global_id + 3 * stride_y];
                const T vm4 = input[global_id - 4 * stride_y];
                const T vp4 = input[global_id + 4 * stride_y];
                const T sum1 = add_rn<T>(vm1, vp1);
                const T sum2 = add_rn<T>(vm2, vp2);
                const T sum3 = add_rn<T>(vm3, vp3);
                const T sum4 = add_rn<T>(vm4, vp4);
                const int low_out = idx + (gidy >> 1) * data_leaps.y + gidz * data_leaps.z;
                tmp[low_out] = low_filter<T>(c0, sum1, sum2, sum3, sum4);
                if (gidy + 1 < static_cast<int>(active_y)) {
                    const T h1 = add_rn<T>(c0, vp2);
                    const T h2 = add_rn<T>(vm1, vp3);
                    const T h3 = add_rn<T>(vm2, vp4);
                    const int high_out =
                        idx + ((gidy >> 1) + even_y) * data_leaps.y + gidz * data_leaps.z;
                    tmp[high_out] = high_filter<T>(vp1, h1, h2, h3);
                }
                continue;
            }

            int ym1 = gidy - 1;
            int yp1 = gidy + 1;
            int ym2 = gidy - 2;
            int yp2 = gidy + 2;
            int ym3 = gidy - 3;
            int yp3 = gidy + 3;
            int ym4 = gidy - 4;
            int yp4 = gidy + 4;

            if (ym1 < 0)
                ym1 = -ym1;
            if (ym2 < 0)
                ym2 = -ym2;
            if (ym3 < 0)
                ym3 = -ym3;
            if (ym4 < 0)
                ym4 = -ym4;

            if (yp1 >= active_y)
                yp1 = 2 * active_y - 2 - yp1;
            if (yp2 >= active_y)
                yp2 = 2 * active_y - 2 - yp2;
            if (yp3 >= active_y)
                yp3 = 2 * active_y - 2 - yp3;
            if (yp4 >= active_y)
                yp4 = 2 * active_y - 2 - yp4;

            const T c0 = input[global_id];
            const T vm1 = input[global_id + (ym1 - gidy) * stride_y];
            const T vp1 = input[global_id + (yp1 - gidy) * stride_y];
            const T vm2 = input[global_id + (ym2 - gidy) * stride_y];
            const T vp2 = input[global_id + (yp2 - gidy) * stride_y];
            const T vm3 = input[global_id + (ym3 - gidy) * stride_y];
            const T vp3 = input[global_id + (yp3 - gidy) * stride_y];
            const T vm4 = input[global_id + (ym4 - gidy) * stride_y];
            const T vp4 = input[global_id + (yp4 - gidy) * stride_y];

            {
                const T sum1 = add_rn<T>(vm1, vp1);
                const T sum2 = add_rn<T>(vm2, vp2);
                const T sum3 = add_rn<T>(vm3, vp3);
                const T sum4 = add_rn<T>(vm4, vp4);
                const int out_id = idx + (gidy >> 1) * data_leaps.y + gidz * data_leaps.z;
                tmp[out_id] = low_filter<T>(c0, sum1, sum2, sum3, sum4);
            }

            if (gidy + 1 < static_cast<int>(active_y)) {
                const T sum1 = add_rn<T>(c0, vp2);
                const T sum2 = add_rn<T>(vm1, vp3);
                const T sum3 = add_rn<T>(vm2, vp4);
                const int out_id =
                    idx + ((gidy >> 1) + even_y) * data_leaps.y + gidz * data_leaps.z;
                tmp[out_id] = high_filter<T>(vp1, sum1, sum2, sum3);
            }
        }
    }
}

// X-direction DWT on a batch of `rows` rows (a row = a fixed (y,z) line of length
// active_x), the X counterpart of dwt_y_rows.  Unlike Y, the x-axis is contiguous,
// so each row is staged into shared `s_data` once (coalesced) and transformed there;
// `s_rowbase` precomputes the per-row global base to kill per-element /active_y,
// %active_y.  Pair-fused: each even owner emits BOTH low (col lx) and high
// (col even_tile+lx, for gidx+1) from one 9-tap window -> ~1/2 the index/clamp work.
// Reads `src`, writes `dst` deinterleaved low|high; safe in place (src==dst) since a
// row is fully staged before any write.  Caller owns the chunk work-queue (like the
// Y passes) and provides the shared staging tile `s_data` (>= rows*active_x elems).
// Requires rows <= 256 (s_rowbase capacity); callers cap tile_y accordingly.
template <typename T>
__device__ __forceinline__ void dwt_x_rows(T* dst,
                                           const T* src,
                                           uint32_t row0,
                                           uint32_t rows,
                                           uint32_t active_x,
                                           uint32_t active_y,
                                           dim3 data_leaps,
                                           T* s_data) {
    __shared__ int s_rowbase[256];
    const int odd_tile = active_x >> 1;
    const int even_tile = active_x - odd_tile;

    for (uint32_t r = threadIdx.x; r < rows; r += blockDim.x) {
        const uint32_t row = row0 + r;
        s_rowbase[r] = (row % active_y) * data_leaps.y + (row / active_y) * data_leaps.z;
    }
    __syncthreads();

    const int len = rows * active_x;
    const int low_len = rows * even_tile;

    // When the active x-line spans the full physical row and the volume has no
    // gaps (first level: active_x == row stride, active_y == full y), the chunk's
    // rows are contiguous in memory, so stage them with a flat coalesced copy
    // (skips the per-element row/col split + s_rowbase lookup).  The condition is
    // loop-invariant, so the compiler hoists it into two specialised load loops.
    if (active_x == (uint32_t)data_leaps.y &&
        (uint32_t)data_leaps.z == (uint32_t)data_leaps.y * active_y) {
        const int base0 = (int)row0 * data_leaps.y;
        if constexpr (sizeof(T) == sizeof(double)) {
            // A flat, 16-byte-aligned double X pass can move two coefficients
            // per instruction. Keep a scalar fallback for arbitrary pointers
            // and a possible one-element tail.
            const T* const in = src + base0;
            if ((reinterpret_cast<uintptr_t>(in) & 15U) == 0U) {
                const int nv = len >> 1;
                const double2* const in2 = reinterpret_cast<const double2*>(in);
                double2* const sh2 = reinterpret_cast<double2*>(s_data);
                for (int v = threadIdx.x; v < nv; v += blockDim.x)
                    sh2[v] = in2[v];
                for (int i = (nv << 1) + threadIdx.x; i < len; i += blockDim.x)
                    s_data[i] = in[i];
            } else {
                for (int i = threadIdx.x; i < len; i += blockDim.x)
                    s_data[i] = in[i];
            }
        } else {
            const T* const in = src + base0;
            if ((len & 3) == 0 && (reinterpret_cast<uintptr_t>(in) & 15U) == 0U) {
                const int nv = len >> 2;
                const float4* const in4 = reinterpret_cast<const float4*>(in);
                float4* const sh4 = reinterpret_cast<float4*>(s_data);
                for (int v = threadIdx.x; v < nv; v += blockDim.x)
                    sh4[v] = in4[v];
            } else {
                for (int i = threadIdx.x; i < len; i += blockDim.x)
                    s_data[i] = in[i];
            }
        }
    } else {
        if constexpr (sizeof(T) == sizeof(float)) {
            const bool vec_ok = (active_x & 3U) == 0U &&
                                (static_cast<uint32_t>(data_leaps.y) & 3U) == 0U &&
                                (reinterpret_cast<uintptr_t>(src) & 15U) == 0U;
            if (vec_ok) {
                const int nvrow = static_cast<int>(active_x >> 2U);
                const int totalv = static_cast<int>(rows) * nvrow;
                for (int v = threadIdx.x; v < totalv; v += blockDim.x) {
                    const int local_row = v / nvrow;
                    const int vx = v - local_row * nvrow;
                    *reinterpret_cast<float4*>(s_data + local_row * static_cast<int>(active_x) +
                                               (vx << 2)) =
                        *reinterpret_cast<const float4*>(src + s_rowbase[local_row] + (vx << 2));
                }
            } else {
                for (int i = threadIdx.x; i < len; i += blockDim.x) {
                    const int local_row = i / static_cast<int>(active_x);
                    const int idx = i - local_row * static_cast<int>(active_x);
                    s_data[i] = src[s_rowbase[local_row] + idx];
                }
            }
        } else {
            for (int i = threadIdx.x; i < len; i += blockDim.x) {
                const int local_row = i / static_cast<int>(active_x);
                const int idx = i - local_row * static_cast<int>(active_x);
                s_data[i] = src[s_rowbase[local_row] + idx];
            }
        }
    }
    __syncthreads();

    for (int i = threadIdx.x; i < low_len; i += blockDim.x) {
        const int local_row = i / even_tile;
        const int lx = i - local_row * even_tile;
        const int gidx = lx << 1;
        const int id = local_row * (int)active_x + gidx;
        int xm1 = gidx - 1, xp1 = gidx + 1, xm2 = gidx - 2, xp2 = gidx + 2;
        int xm3 = gidx - 3, xp3 = gidx + 3, xm4 = gidx - 4, xp4 = gidx + 4;
        if (xm1 < 0)
            xm1 = -xm1;
        if (xm2 < 0)
            xm2 = -xm2;
        if (xm3 < 0)
            xm3 = -xm3;
        if (xm4 < 0)
            xm4 = -xm4;
        if (xp1 >= (int)active_x)
            xp1 = 2 * (int)active_x - 2 - xp1;
        if (xp2 >= (int)active_x)
            xp2 = 2 * (int)active_x - 2 - xp2;
        if (xp3 >= (int)active_x)
            xp3 = 2 * (int)active_x - 2 - xp3;
        if (xp4 >= (int)active_x)
            xp4 = 2 * (int)active_x - 2 - xp4;

        const T c0 = s_data[id];
        const T vm1 = s_data[id + (xm1 - gidx)];
        const T vp1 = s_data[id + (xp1 - gidx)];
        const T vm2 = s_data[id + (xm2 - gidx)];
        const T vp2 = s_data[id + (xp2 - gidx)];
        const T vm3 = s_data[id + (xm3 - gidx)];
        const T vp3 = s_data[id + (xp3 - gidx)];
        const T vm4 = s_data[id + (xm4 - gidx)];
        const T vp4 = s_data[id + (xp4 - gidx)];

        dst[s_rowbase[local_row] + lx] = low_filter<T>(
            c0, add_rn<T>(vm1, vp1), add_rn<T>(vm2, vp2), add_rn<T>(vm3, vp3), add_rn<T>(vm4, vp4));

        if (gidx + 1 < (int)active_x) {
            dst[s_rowbase[local_row] + even_tile + lx] =
                high_filter<T>(vp1, add_rn<T>(c0, vp2), add_rn<T>(vm1, vp3), add_rn<T>(vm2, vp4));
        }
    }
    __syncthreads();
}


template <typename T, uint32_t CS_BYTES = CS, uint32_t BLOCK_THREADS = TPB>
__global__ __launch_bounds__(BLOCK_THREADS, CDF97_DWT_MIN_BLOCKS(T)) void dwt3d(
    T* data, T* tmp, dim3 data_dims, dim3 data_leaps, uint32_t datasize, uint8_t max_level) {
    cooperative_groups::grid_group grid = cooperative_groups::this_grid();
    dim3 chunk_dims = data_dims;
    dim3 chunk_leaps = data_leaps;
    constexpr uint32_t num = CS_BYTES / sizeof(T) * 4;
    __shared__ alignas(128) T s_data[num];

    uint32_t tile_y = num / data_dims.x; //(num / (data_dims.x << 1)) << 1;
    uint32_t tile_x = data_dims.x;

    uint32_t N = tile_x * tile_y;
    uint32_t max_count = (datasize - 1) / N + 1;

    __shared__ int s_chunkID;
    // for (int l = 0; l < 1; ++l) {

    int odd_tile = data_dims.x >> 1;
    int even_tile = data_dims.x - odd_tile;
    // X transform: deinterleave low|high along x, in place (data -> data).
    const uint32_t total_rows_x0 = data_dims.y * data_dims.z;
    const uint32_t tile_yx0 = min(tile_y, 256u); // dwt_x_rows: rows <= 256
    const uint32_t max_count_x0 = (total_rows_x0 - 1U) / tile_yx0 + 1U;
    do {
        if (threadIdx.x == 0)
            s_chunkID = atomicAdd(&x_count[0], 1U);
        __syncthreads();
        int chunkID = s_chunkID;
        __syncthreads();
        if (chunkID >= (int)max_count_x0) {
            break;
        }
        uint32_t row0 = chunkID * tile_yx0;
        uint32_t rows = min(tile_yx0, total_rows_x0 - row0);
        dwt_x_rows<T>(data, data, row0, rows, data_dims.x, data_dims.y, data_leaps, s_data);
    } while (true);
    grid.sync();

    odd_tile = data_dims.y >> 1;
    even_tile = data_dims.y - odd_tile;
    // Y transform
    do {
        if (threadIdx.x == 0)
            s_chunkID = atomicAdd(&y_count[0], 1U);
        __syncthreads();
        int chunkID = s_chunkID;
        __syncthreads();
        if (chunkID >= max_count) {
            break;
        }

        uint32_t row0 = chunkID * tile_y;
        uint32_t full_rows = data_dims.y * data_dims.z;
        uint32_t rows = min(tile_y, full_rows - row0);
        T* input = data;
        dwt_y_rows<T>(tmp, input, row0, rows, tile_x, data_dims.y, data_leaps);

    } while (true);
    grid.sync();

    odd_tile = data_dims.z >> 1;
    even_tile = data_dims.z - odd_tile;
    // Z transform.  float is bandwidth-bound, so pair-fusion (even-z owner
    // loads its 9-plane window once and emits BOTH low and high) cuts neighbour
    // reads ~1.8x and wins.  double is latency-bound on the large z-stride, so
    // halving the active owner-threads hurts MLP; baseline per-element is faster.
    // (Re-measured on H100: forcing double through pair-fusion cost CMD1 ~+6%.)
    if constexpr (std::is_same_v<T, float>) {
        do {
            if (threadIdx.x == 0)
                s_chunkID = atomicAdd(&z_count[0], 1U);
            __syncthreads();
            int chunkID = s_chunkID;
            __syncthreads();
            if (chunkID >= max_count)
                break;

            int start = chunkID * N;
            int len = min(N, datasize - start);
            int stride_z = data_leaps.z;
            T* input = data + start;
            for (int i = threadIdx.x; i < len; i += BLOCK_THREADS) {
                int global_id = start + i;
                if (global_id >= datasize)
                    break;
                int gidz = global_id / data_leaps.z;
                if ((gidz & 1) != 0)
                    continue;

                int zm1 = gidz - 1;
                int zp1 = gidz + 1;
                int zm2 = gidz - 2;
                int zp2 = gidz + 2;
                int zm3 = gidz - 3;
                int zp3 = gidz + 3;
                int zm4 = gidz - 4;
                int zp4 = gidz + 4;

                if (zm1 < 0)
                    zm1 = -zm1;
                if (zm2 < 0)
                    zm2 = -zm2;
                if (zm3 < 0)
                    zm3 = -zm3;
                if (zm4 < 0)
                    zm4 = -zm4;

                if (zp1 >= data_dims.z)
                    zp1 = 2 * data_dims.z - 2 - zp1;
                if (zp2 >= data_dims.z)
                    zp2 = 2 * data_dims.z - 2 - zp2;
                if (zp3 >= data_dims.z)
                    zp3 = 2 * data_dims.z - 2 - zp3;
                if (zp4 >= data_dims.z)
                    zp4 = 2 * data_dims.z - 2 - zp4;

                const T c0 = tmp[global_id];
                const T vm1 = tmp[global_id + (zm1 - gidz) * stride_z];
                const T vp1 = tmp[global_id + (zp1 - gidz) * stride_z];
                const T vm2 = tmp[global_id + (zm2 - gidz) * stride_z];
                const T vp2 = tmp[global_id + (zp2 - gidz) * stride_z];
                const T vm3 = tmp[global_id + (zm3 - gidz) * stride_z];
                const T vp3 = tmp[global_id + (zp3 - gidz) * stride_z];
                const T vm4 = tmp[global_id + (zm4 - gidz) * stride_z];
                const T vp4 = tmp[global_id + (zp4 - gidz) * stride_z];

                {
                    T sum1 = add_rn<T>(vm1, vp1);
                    T sum2 = add_rn<T>(vm2, vp2);
                    T sum3 = add_rn<T>(vm3, vp3);
                    T sum4 = add_rn<T>(vm4, vp4);
                    int corrected_gidz = (gidz / 2) * data_leaps.z;
                    input[i - corrected_gidz] = low_filter<T>(c0, sum1, sum2, sum3, sum4);
                }

                if (gidz + 1 < static_cast<int>(data_dims.z)) {
                    T sum1 = add_rn<T>(c0, vp2);
                    T sum2 = add_rn<T>(vm1, vp3);
                    T sum3 = add_rn<T>(vm2, vp4);
                    int corrected_gidz = (gidz / 2 + even_tile - gidz) * data_leaps.z;
                    input[i + corrected_gidz] = high_filter<T>(vp1, sum1, sum2, sum3);
                }
            }
        } while (true);
    } else {
        do {
            if (threadIdx.x == 0)
                s_chunkID = atomicAdd(&z_count[0], 1U);
            __syncthreads();
            int chunkID = s_chunkID;
            __syncthreads();
            if (chunkID >= max_count) {
                break;
            }

            int start = chunkID * N;
            int len = min(N, datasize - start);

            int stride_z = data_leaps.z;
            T* input = data + start;
            for (int i = threadIdx.x; i < len; i += BLOCK_THREADS) {
                int global_id = start + i;
                if (global_id >= datasize)
                    break;

                int gidz = global_id / data_leaps.z;

                if (gidz % 2 == 1) {
                    int corrected_gidz = (gidz / 2 + even_tile - gidz) * data_leaps.z;
                    int zm1 = gidz - 1;
                    int zp1 = gidz + 1;
                    int zm2 = gidz - 2;
                    int zp2 = gidz + 2;
                    int zm3 = gidz - 3;
                    int zp3 = gidz + 3;

                    if (zm1 < 0)
                        zm1 = -zm1;
                    if (zm2 < 0)
                        zm2 = -zm2;
                    if (zm3 < 0)
                        zm3 = -zm3;

                    if (zp1 >= data_dims.z)
                        zp1 = 2 * data_dims.z - 2 - zp1;
                    if (zp2 >= data_dims.z)
                        zp2 = 2 * data_dims.z - 2 - zp2;
                    if (zp3 >= data_dims.z)
                        zp3 = 2 * data_dims.z - 2 - zp3;

                    int id_m1 = global_id + (zm1 - gidz) * stride_z;
                    int id_p1 = global_id + (zp1 - gidz) * stride_z;
                    int id_m2 = global_id + (zm2 - gidz) * stride_z;
                    int id_p2 = global_id + (zp2 - gidz) * stride_z;
                    int id_m3 = global_id + (zm3 - gidz) * stride_z;
                    int id_p3 = global_id + (zp3 - gidz) * stride_z;
                    {
                        T sum1 = add_rn<T>(tmp[id_m1], tmp[id_p1]);
                        T sum2 = add_rn<T>(tmp[id_m2], tmp[id_p2]);
                        T sum3 = add_rn<T>(tmp[id_m3], tmp[id_p3]);

                        T acc = high_filter<T>(tmp[global_id], sum1, sum2, sum3);

                        input[i + corrected_gidz] = acc;
                    }

                } else {
                    int corrected_gidz = (gidz / 2) * data_leaps.z;
                    int zm1 = gidz - 1;
                    int zp1 = gidz + 1;
                    int zm2 = gidz - 2;
                    int zp2 = gidz + 2;
                    int zm3 = gidz - 3;
                    int zp3 = gidz + 3;
                    int zm4 = gidz - 4;
                    int zp4 = gidz + 4;

                    if (zm1 < 0)
                        zm1 = -zm1;
                    if (zm2 < 0)
                        zm2 = -zm2;
                    if (zm3 < 0)
                        zm3 = -zm3;
                    if (zm4 < 0)
                        zm4 = -zm4;

                    if (zp1 >= data_dims.z)
                        zp1 = 2 * data_dims.z - 2 - zp1;
                    if (zp2 >= data_dims.z)
                        zp2 = 2 * data_dims.z - 2 - zp2;
                    if (zp3 >= data_dims.z)
                        zp3 = 2 * data_dims.z - 2 - zp3;
                    if (zp4 >= data_dims.z)
                        zp4 = 2 * data_dims.z - 2 - zp4;

                    int id_m1 = global_id + (zm1 - gidz) * stride_z;
                    int id_p1 = global_id + (zp1 - gidz) * stride_z;
                    int id_m2 = global_id + (zm2 - gidz) * stride_z;
                    int id_p2 = global_id + (zp2 - gidz) * stride_z;
                    int id_m3 = global_id + (zm3 - gidz) * stride_z;
                    int id_p3 = global_id + (zp3 - gidz) * stride_z;
                    int id_m4 = global_id + (zm4 - gidz) * stride_z;
                    int id_p4 = global_id + (zp4 - gidz) * stride_z;
                    {
                        T sum1 = add_rn<T>(tmp[id_m1], tmp[id_p1]);
                        T sum2 = add_rn<T>(tmp[id_m2], tmp[id_p2]);
                        T sum3 = add_rn<T>(tmp[id_m3], tmp[id_p3]);
                        T sum4 = add_rn<T>(tmp[id_m4], tmp[id_p4]);

                        T acc = low_filter<T>(tmp[global_id], sum1, sum2, sum3, sum4);

                        input[i - corrected_gidz] = acc;
                    }
                }
            }
        } while (true);
    }
    grid.sync();

    // }

    chunk_dims.x = chunk_dims.x - (chunk_dims.x >> 1);
    chunk_dims.y = chunk_dims.y - (chunk_dims.y >> 1);
    chunk_dims.z = chunk_dims.z - (chunk_dims.z >> 1);
    chunk_leaps.y = chunk_dims.x;
    chunk_leaps.z = chunk_dims.y * chunk_dims.x;

    for (int l = 1; l < max_level; ++l) {

        uint32_t tile_y = num / chunk_dims.x;
        uint32_t tile_x = chunk_dims.x;

        uint32_t N = tile_x * tile_y;
        uint32_t total_elems = chunk_dims.x * chunk_dims.y * chunk_dims.z;
        uint32_t max_count = (total_elems - 1) / N + 1;

        odd_tile = chunk_dims.x >> 1;
        even_tile = chunk_dims.x - odd_tile;
        uint32_t total_rows = chunk_dims.y * chunk_dims.z;
        // X transform: deinterleave low|high along x, in place (data -> data).
        const uint32_t tile_yx = min(tile_y, 256u); // dwt_x_rows: rows <= 256
        const uint32_t max_count_x = (total_rows - 1U) / tile_yx + 1U;
        do {
            if (threadIdx.x == 0)
                s_chunkID = atomicAdd(&x_count[l], 1U);
            __syncthreads();
            int chunkID = s_chunkID;
            __syncthreads();
            if (chunkID >= (int)max_count_x) {
                break;
            }
            uint32_t row0 = chunkID * tile_yx;
            uint32_t rows = min(tile_yx, total_rows - row0);
            dwt_x_rows<T>(data, data, row0, rows, chunk_dims.x, chunk_dims.y, data_leaps, s_data);
        } while (true);
        grid.sync();

        odd_tile = chunk_dims.y >> 1;
        even_tile = chunk_dims.y - odd_tile;
        // Y transform
        do {
            if (threadIdx.x == 0)
                s_chunkID = atomicAdd(&y_count[l], 1U);
            __syncthreads();
            int chunkID = s_chunkID;
            __syncthreads();
            if (chunkID >= max_count) {
                break;
            }

            uint32_t row0 = chunkID * tile_y;
            uint32_t rows = min(tile_y, total_rows - row0);
            T* input = data;
            dwt_y_rows<T>(tmp, input, row0, rows, chunk_dims.x, chunk_dims.y, data_leaps);

        } while (true);
        grid.sync();

        odd_tile = chunk_dims.z >> 1;
        even_tile = chunk_dims.z - odd_tile;
        // Z transform (see level-0 note: float -> pair-fusion, double -> baseline).
        if constexpr (std::is_same_v<T, float>) {
            do {
                if (threadIdx.x == 0)
                    s_chunkID = atomicAdd(&z_count[l], 1U);
                __syncthreads();
                int chunkID = s_chunkID;
                __syncthreads();
                if (chunkID >= max_count) {
                    break;
                }

                uint32_t local_start = chunkID * N;
                uint32_t len = min(N, total_elems - local_start);

                int stride_z = data_leaps.z;
                for (int i = threadIdx.x; i < len; i += BLOCK_THREADS) {
                    uint32_t local_id = local_start + i;
                    int idx = local_id % chunk_dims.x;
                    int gidy = (local_id / chunk_dims.x) % chunk_dims.y;
                    int gidz = local_id / chunk_leaps.z;
                    int global_id = idx + gidy * data_leaps.y + gidz * data_leaps.z;
                    if ((gidz & 1) != 0)
                        continue;

                    int zm1 = gidz - 1;
                    int zp1 = gidz + 1;
                    int zm2 = gidz - 2;
                    int zp2 = gidz + 2;
                    int zm3 = gidz - 3;
                    int zp3 = gidz + 3;
                    int zm4 = gidz - 4;
                    int zp4 = gidz + 4;

                    if (zm1 < 0)
                        zm1 = -zm1;
                    if (zm2 < 0)
                        zm2 = -zm2;
                    if (zm3 < 0)
                        zm3 = -zm3;
                    if (zm4 < 0)
                        zm4 = -zm4;

                    if (zp1 >= chunk_dims.z)
                        zp1 = 2 * chunk_dims.z - 2 - zp1;
                    if (zp2 >= chunk_dims.z)
                        zp2 = 2 * chunk_dims.z - 2 - zp2;
                    if (zp3 >= chunk_dims.z)
                        zp3 = 2 * chunk_dims.z - 2 - zp3;
                    if (zp4 >= chunk_dims.z)
                        zp4 = 2 * chunk_dims.z - 2 - zp4;

                    const T c0 = tmp[global_id];
                    const T vm1 = tmp[global_id + (zm1 - gidz) * stride_z];
                    const T vp1 = tmp[global_id + (zp1 - gidz) * stride_z];
                    const T vm2 = tmp[global_id + (zm2 - gidz) * stride_z];
                    const T vp2 = tmp[global_id + (zp2 - gidz) * stride_z];
                    const T vm3 = tmp[global_id + (zm3 - gidz) * stride_z];
                    const T vp3 = tmp[global_id + (zp3 - gidz) * stride_z];
                    const T vm4 = tmp[global_id + (zm4 - gidz) * stride_z];
                    const T vp4 = tmp[global_id + (zp4 - gidz) * stride_z];

                    {
                        T sum1 = add_rn<T>(vm1, vp1);
                        T sum2 = add_rn<T>(vm2, vp2);
                        T sum3 = add_rn<T>(vm3, vp3);
                        T sum4 = add_rn<T>(vm4, vp4);
                        int out_id = idx + gidy * data_leaps.y + (gidz >> 1) * data_leaps.z;
                        data[out_id] = low_filter<T>(c0, sum1, sum2, sum3, sum4);
                    }

                    if (gidz + 1 < static_cast<int>(chunk_dims.z)) {
                        T sum1 = add_rn<T>(c0, vp2);
                        T sum2 = add_rn<T>(vm1, vp3);
                        T sum3 = add_rn<T>(vm2, vp4);
                        int out_id =
                            idx + gidy * data_leaps.y + ((gidz >> 1) + even_tile) * data_leaps.z;
                        data[out_id] = high_filter<T>(vp1, sum1, sum2, sum3);
                    }
                }
            } while (true);
        } else {
            do {
                if (threadIdx.x == 0)
                    s_chunkID = atomicAdd(&z_count[l], 1U);
                __syncthreads();
                int chunkID = s_chunkID;
                __syncthreads();
                if (chunkID >= max_count) {
                    break;
                }

                uint32_t local_start = chunkID * N;
                uint32_t len = min(N, total_elems - local_start);

                int stride_z = data_leaps.z;
                for (int i = threadIdx.x; i < len; i += BLOCK_THREADS) {
                    uint32_t local_id = local_start + i;
                    int idx = local_id % chunk_dims.x;
                    int gidy = (local_id / chunk_dims.x) % chunk_dims.y;
                    int gidz = local_id / chunk_leaps.z;
                    int global_id = idx + gidy * data_leaps.y + gidz * data_leaps.z;

                    if ((gidz & 1) == 1) {
                        int out_id =
                            idx + gidy * data_leaps.y + ((gidz >> 1) + even_tile) * data_leaps.z;
                        int zm1 = gidz - 1;
                        int zp1 = gidz + 1;
                        int zm2 = gidz - 2;
                        int zp2 = gidz + 2;
                        int zm3 = gidz - 3;
                        int zp3 = gidz + 3;

                        // whole-sample symmetric: -1 -> 1, -2 -> 2, n -> n-2
                        if (zm1 < 0)
                            zm1 = -zm1;
                        if (zm2 < 0)
                            zm2 = -zm2;
                        if (zm3 < 0)
                            zm3 = -zm3;

                        if (zp1 >= chunk_dims.z)
                            zp1 = 2 * chunk_dims.z - 2 - zp1;
                        if (zp2 >= chunk_dims.z)
                            zp2 = 2 * chunk_dims.z - 2 - zp2;
                        if (zp3 >= chunk_dims.z)
                            zp3 = 2 * chunk_dims.z - 2 - zp3;

                        int id_m1 = global_id + (zm1 - gidz) * stride_z;
                        int id_p1 = global_id + (zp1 - gidz) * stride_z;
                        int id_m2 = global_id + (zm2 - gidz) * stride_z;
                        int id_p2 = global_id + (zp2 - gidz) * stride_z;
                        int id_m3 = global_id + (zm3 - gidz) * stride_z;
                        int id_p3 = global_id + (zp3 - gidz) * stride_z;
                        {
                            T sum1 = add_rn<T>(tmp[id_m1], tmp[id_p1]);
                            T sum2 = add_rn<T>(tmp[id_m2], tmp[id_p2]);
                            T sum3 = add_rn<T>(tmp[id_m3], tmp[id_p3]);

                            T acc = high_filter<T>(tmp[global_id], sum1, sum2, sum3);

                            data[out_id] = acc;
                        }

                    } else {
                        int out_id = idx + gidy * data_leaps.y + (gidz >> 1) * data_leaps.z;
                        int zm1 = gidz - 1;
                        int zp1 = gidz + 1;
                        int zm2 = gidz - 2;
                        int zp2 = gidz + 2;
                        int zm3 = gidz - 3;
                        int zp3 = gidz + 3;
                        int zm4 = gidz - 4;
                        int zp4 = gidz + 4;

                        // whole-sample symmetric: -1 -> 1, -2 -> 2, n -> n-2
                        if (zm1 < 0)
                            zm1 = -zm1;
                        if (zm2 < 0)
                            zm2 = -zm2;
                        if (zm3 < 0)
                            zm3 = -zm3;
                        if (zm4 < 0)
                            zm4 = -zm4;

                        if (zp1 >= chunk_dims.z)
                            zp1 = 2 * chunk_dims.z - 2 - zp1;
                        if (zp2 >= chunk_dims.z)
                            zp2 = 2 * chunk_dims.z - 2 - zp2;
                        if (zp3 >= chunk_dims.z)
                            zp3 = 2 * chunk_dims.z - 2 - zp3;
                        if (zp4 >= chunk_dims.z)
                            zp4 = 2 * chunk_dims.z - 2 - zp4;

                        int id_m1 = global_id + (zm1 - gidz) * stride_z;
                        int id_p1 = global_id + (zp1 - gidz) * stride_z;
                        int id_m2 = global_id + (zm2 - gidz) * stride_z;
                        int id_p2 = global_id + (zp2 - gidz) * stride_z;
                        int id_m3 = global_id + (zm3 - gidz) * stride_z;
                        int id_p3 = global_id + (zp3 - gidz) * stride_z;
                        int id_m4 = global_id + (zm4 - gidz) * stride_z;
                        int id_p4 = global_id + (zp4 - gidz) * stride_z;
                        {
                            T sum1 = add_rn<T>(tmp[id_m1], tmp[id_p1]);
                            T sum2 = add_rn<T>(tmp[id_m2], tmp[id_p2]);
                            T sum3 = add_rn<T>(tmp[id_m3], tmp[id_p3]);
                            T sum4 = add_rn<T>(tmp[id_m4], tmp[id_p4]);

                            T acc = low_filter<T>(tmp[global_id], sum1, sum2, sum3, sum4);

                            data[out_id] = acc;
                        }
                    }
                }
            } while (true);
        }
        grid.sync();

        chunk_dims.x = chunk_dims.x - (chunk_dims.x >> 1);
        chunk_dims.y = chunk_dims.y - (chunk_dims.y >> 1);
        chunk_dims.z = chunk_dims.z - (chunk_dims.z >> 1);
        chunk_leaps.y = chunk_dims.x;
        chunk_leaps.z = chunk_dims.y * chunk_dims.x;
    }
}

// One X-axis kernel serves float/double and both in-place/out-of-place paths.
// dwt_x_rows stages each complete row before writing, so src == dst is safe.
template <typename T>
__global__ __launch_bounds__(TPB, sizeof(T) == sizeof(float) ? 3 : 2) void dwt_x_axis(
    const T* src, T* dst, uint32_t nx, uint32_t ny, uint32_t lx, uint32_t ly, uint32_t lz,
    uint32_t counter_index) {
    constexpr uint32_t cap = 32768U / sizeof(T);
    __shared__ alignas(128) T s_data[cap];
    __shared__ uint32_t chunk_id;
    const dim3 leaps(1, nx, nx * ny);
    const uint32_t rpt = min(max(cap / lx, 1U), 256U);
    const uint32_t rows_total = ly * lz;
    const uint32_t chunks = (rows_total + rpt - 1U) / rpt;
    while (true) {
        if (threadIdx.x == 0)
            chunk_id = atomicAdd(&x_count[counter_index], 1U);
        __syncthreads();
        const uint32_t chunk = chunk_id;
        __syncthreads();
        if (chunk >= chunks)
            break;
        const uint32_t row0 = chunk * rpt;
        dwt_x_rows<T>(dst, src, row0, min(rpt, rows_total - row0), lx, ly, leaps, s_data);
    }
}

template <typename T>
__global__ __launch_bounds__(TPB, sizeof(T) == sizeof(float) ? 3 : 2) void dwt_y_axis(
    const T* src, T* dst, uint32_t nx, uint32_t ny, uint32_t lx, uint32_t ly, uint32_t lz,
    uint32_t counter_index) {
    constexpr uint32_t cap = 32768U / sizeof(T);
    __shared__ uint32_t chunk_id;
    const dim3 leaps(1, nx, nx * ny);
    const uint32_t rpt = max(cap / lx, 1U);
    const uint32_t rows_total = ly * lz;
    const uint32_t chunks = (rows_total + rpt - 1U) / rpt;
    while (true) {
        if (threadIdx.x == 0)
            chunk_id = atomicAdd(&y_count[counter_index], 1U);
        __syncthreads();
        const uint32_t chunk = chunk_id;
        __syncthreads();
        if (chunk >= chunks)
            break;
        const uint32_t row0 = chunk * rpt;
        dwt_y_rows<T>(dst, src, row0, min(rpt, rows_total - row0), lx, ly, leaps);
    }
}

constexpr int DWT_HALO_R = 4;

__device__ __forceinline__ int dwt_halo_reflect(int i, int n) {
    if (n <= 1)
        return 0;
    while (i < 0 || i >= n) {
        if (i < 0)
            i = -i;
        if (i >= n)
            i = 2 * n - 2 - i;
    }
    return i;
}

template <typename T, uint8_t Quant = 0, bool FullColumns = false>
__global__ void dwt_3d_z_s(const T* src, T* dst, dim3 data_dims, dim3 chunk_dims,
                           uint32_t counter_index, uint32_t levels_to_run,
                           uint32_t shared_capacity);
// Fused XY analysis for the non-dyadic (plane) path.  The plane transform first
// runs Z to the requested depth, then recursively analyzes every XY plane.  The
// regular implementation materialises an X volume and a Y volume at every
// level.  At the finest XY level this kernel keeps a 4-sample halo in shared
// memory, performs X and Y without the intermediate global traffic, and emits
// the final de-interleaved XY coefficients directly to `dst`.
//
// `src` and `dst` must be different allocations.  The host launcher below
// swaps the two pointers after this first out-of-place level; this is important
// because an in-place launch would race with a neighbouring tile still loading
// its halo.  The convolution itself is deliberately the same low_filter /
// high_filter sequence used by dwt_x_rows and dwt_y_rows (including explicit
// round-to-nearest additions), so the resulting coefficient stream is
// bit-identical to the unfused plane path.
// The 32x16 tile is the conservative baseline (about 7 KiB shared memory).
// Wide planes can use the optional 64x32 specialization (about 22 KiB), which
// lowers the halo/owner ratio; the launcher below keeps both variants available
// for shape A/B tests without changing the default path.
constexpr int DWT_TILE_BX32 = 32;
constexpr int DWT_TILE_BY16 = 16;
constexpr int DWT_TILE_BX64 = 64;
constexpr int DWT_TILE_BY32 = 32;

// Kept in one device-side record so the optional DWT->quant path does not keep
// ten independent pointers/scalars live in every halo thread.  The default
// (Quant=0) kernel never dereferences this record.
struct DwtFuseQuantParams {
    uint16_t* qint_z;
    uint32_t* sign_bm;
    float* iquant;
    const uint16_t* rank_table;
    uint32_t q_bz;
    uint32_t block_nx;
    uint32_t block_ny;
    uint32_t bz_shift;
    uint32_t block_elems;
    uint32_t regular_blocks;
    float inv_qf;
    float qf;
    uint32_t* qout_idx;
    int32_t* qout_val;
    unsigned int* qout_cnt;
    uint32_t qout_cap;
};

static __constant__ DwtFuseQuantParams float_quant_params;

template <typename T>
inline cudaError_t set_quant_params(uint16_t* qint_z, uint32_t* sign_bm, T* iquant,
                                    const uint16_t* rank_table, dim3 dims, uint32_t q_bz,
                                    double q, uint32_t* qout_idx, int32_t* qout_val,
                                    unsigned int* qout_cnt, uint32_t qout_cap,
                                    cudaStream_t stream = 0) {
    if constexpr (std::is_same_v<T, double>) {
        DwtDoubleQuantParams p{qint_z,
                               sign_bm,
                               iquant,
                               rank_table,
                               q_bz,
                               (dims.x + 15U) >> 4U,
                               (dims.y + 15U) >> 4U,
                               q_bz != 0U
                                   ? static_cast<uint32_t>(
                                         __builtin_ctz(static_cast<unsigned int>(q_bz)))
                                   : 0U,
                               16U * 16U * q_bz,
                               q_bz != 0U && (q_bz & (q_bz - 1U)) == 0U &&
                                   (dims.x & 15U) == 0U && (dims.y & 15U) == 0U &&
                                   dims.z % q_bz == 0U,
                               q != 0.0 ? 1.0 / q : 0.0,
                               q,
                               qout_idx,
                               qout_val,
                               qout_cnt,
                               qout_cap};
        return cudaMemcpyToSymbolAsync(
            double_quant_params, &p, sizeof(p), 0, cudaMemcpyHostToDevice, stream);
    } else {
        static_assert(std::is_same_v<T, float>, "fused quantization supports float and double");
        DwtFuseQuantParams p{qint_z,
                             sign_bm,
                             iquant,
                             rank_table,
                             q_bz,
                             (dims.x + 15U) >> 4U,
                             (dims.y + 15U) >> 4U,
                             q_bz != 0U
                                 ? static_cast<uint32_t>(
                                       __builtin_ctz(static_cast<unsigned int>(q_bz)))
                                 : 0U,
                             16U * 16U * q_bz,
                             q_bz != 0U && (q_bz & (q_bz - 1U)) == 0U &&
                                 (dims.x & 15U) == 0U && (dims.y & 15U) == 0U &&
                                 dims.z % q_bz == 0U,
                             q != 0.0 ? static_cast<float>(1.0 / q) : 0.0f,
                             static_cast<float>(q),
                             qout_idx,
                             qout_val,
                             qout_cnt,
                             qout_cap};
        return cudaMemcpyToSymbolAsync(
            float_quant_params, &p, sizeof(p), 0, cudaMemcpyHostToDevice, stream);
    }
}


__device__ __forceinline__ void dwt_quant_emit_float(float value, size_t s, uint32_t x,
                                                     uint32_t y,
                                                     uint32_t z,
                                                     dim3 data_dims,
                                                     const DwtFuseQuantParams* __restrict__ p) {
    const long long rll = WALTZ::quantizer::llrint_real(value * p->inv_qf);
    const bool neg = rll < 0;
    const int32_t mag = WALTZ::quantizer::saturate_q<int32_t>(neg ? -rll : rll);
    constexpr uint32_t Bx = 16U, By = 16U;
    const uint32_t x0 = x & ~(Bx - 1U), y0 = y & ~(By - 1U);
    const uint32_t Bz = p->q_bz;
    uint32_t z0, lz;
    if ((Bz & (Bz - 1U)) == 0U) {
        z0 = z & ~(Bz - 1U);
        lz = z & (Bz - 1U);
    } else {
        z0 = (z / Bz) * Bz;
        lz = z - z0;
    }
    size_t b0, rank;
    const bool full =
        x0 + Bx <= data_dims.x && y0 + By <= data_dims.y && z0 + Bz <= data_dims.z &&
        p->rank_table != nullptr;
    if (full) {
        const uint32_t local = (x & (Bx - 1U)) + ((y & (By - 1U)) << 4U) + (lz << 8U);
        b0 = static_cast<size_t>(data_dims.x) * data_dims.y * z0 +
             static_cast<size_t>(data_dims.x) * Bz * y0 + static_cast<size_t>(Bx) * Bz * x0;
        rank = p->rank_table[local];
    } else {
        const uint32_t bdx = min(Bx, data_dims.x - x0);
        const uint32_t bdy = min(By, data_dims.y - y0);
        const uint32_t bdz = min(Bz, data_dims.z - z0);
        b0 = static_cast<size_t>(data_dims.x) * data_dims.y * z0 +
             static_cast<size_t>(data_dims.x) * bdz * y0 + static_cast<size_t>(bdy) * bdz * x0;
        const size_t local = (x & (Bx - 1U)) + static_cast<size_t>(y & (By - 1U)) * bdx +
                             static_cast<size_t>(lz) * bdx * bdy;
        rank = WALTZ::lossless::zorder_rank(local, bdx, bdy, bdz);
    }
    const size_t outpos = b0 + rank;
    p->qint_z[outpos] = static_cast<uint16_t>(mag > 65535 ? 65535 : mag);
    if (neg && mag != 0)
        atomicOr(p->sign_bm + (outpos >> 5U), 1U << (outpos & 31U));
    p->iquant[s] = static_cast<float>(neg ? -mag : mag) * p->qf;
    if (mag > 65535 && p->qout_cnt != nullptr) {
        const unsigned int slot = atomicAdd(p->qout_cnt, 1U);
        if (slot < p->qout_cap) {
            p->qout_idx[slot] = static_cast<uint32_t>(s);
            p->qout_val[slot] = mag;
        }
    }
}

// Fast emitter for dynamic volumes made entirely of full 16x16xBZ blocks.
// The dimensions remain runtime values; only the absence of edge blocks is a
// compile-time property of this kernel variant.  Irregular volumes continue to
// use the generic emitter above.
__device__ __forceinline__ void dwt_quant_emit_regular_float(float value, size_t s, 
    uint32_t x, uint32_t y, uint32_t z, const DwtFuseQuantParams* __restrict__ p) {
    constexpr uint32_t BX = 16U, BY = 16U;
    const uint32_t bz = p->q_bz;
    const uint32_t block = (x >> 4U) + p->block_nx * (y >> 4U) + p->block_nx * p->block_ny * (z >> p->bz_shift);
    const uint32_t local = (x & (BX - 1U)) + (y & (BY - 1U)) * BX + (z & (bz - 1U)) * BX * BY;
    const size_t outpos = static_cast<size_t>(block) * p->block_elems + p->rank_table[local];
    const long long rll = WALTZ::quantizer::llrint_real(value * p->inv_qf);
    const bool neg = rll < 0;
    const int32_t mag = WALTZ::quantizer::saturate_q<int32_t>(neg ? -rll : rll);
    p->qint_z[outpos] = static_cast<uint16_t>(mag > 65535 ? 65535 : mag);
    
    if (neg && mag != 0)
        atomicOr(p->sign_bm + (outpos >> 5U), 1U << (outpos & 31U));
    
    p->iquant[s] = static_cast<float>(neg ? -mag : mag) * p->qf;
    if (mag > 65535 && p->qout_cnt != nullptr) {
        const unsigned int slot = atomicAdd(p->qout_cnt, 1U);
        if (slot < p->qout_cap) {
            p->qout_idx[slot] = static_cast<uint32_t>(s);
            p->qout_val[slot] = mag;
        }
    }
}

template <typename T>
__device__ __forceinline__ void quantization(
    T value, size_t s, uint32_t x, uint32_t y, uint32_t z, dim3 data_dims) {
    if constexpr (std::is_same_v<T, float>) {
        if (float_quant_params.regular_blocks != 0U)
            dwt_quant_emit_regular_float(value, s, x, y, z, &float_quant_params);
        else
            dwt_quant_emit_float(value, s, x, y, z, data_dims, &float_quant_params);
    } else if constexpr (std::is_same_v<T, double>) {
        if (double_quant_params.regular_blocks != 0U)
            dwt_quant_emit_regular_double(value, s, x, y, z, &double_quant_params);
        else
            dwt_quant_emit_double(value, s, x, y, z, data_dims, &double_quant_params);
    }
}

template <int BX, int BY, uint8_t Quant = 0>
__global__ void dwt3d_xy_float(
    const float* __restrict__ src, float* __restrict__ dst, dim3 data_dims, dim3 chunk_dims) {
    static_assert(Quant <= 2, "Quant must be 0, 1, or 2");
    constexpr int R = DWT_HALO_R;
    constexpr int EX = BX + 2 * R;
    constexpr int EY = BY + 2 * R;

    __shared__ alignas(128) float raw[EX * EY];
    __shared__ alignas(128) float xbuf[BX * EY];

    const int nx = chunk_dims.x;
    const int ny = chunk_dims.y;
    const int sx = data_dims.x;
    const int x0 = blockIdx.x * BX;
    const int y0 = blockIdx.y * BY;
    const int even_x = nx - (nx >> 1);
    const int even_y = ny - (ny >> 1);
    const size_t leap_z = static_cast<size_t>(data_dims.x) * data_dims.y;

    const bool tile_interior = x0 >= R && x0 + BX + R <= nx && y0 >= R && y0 + BY + R <= ny;
    const int z = static_cast<int>(blockIdx.z);

    constexpr int VEX = EX / 4;
    const bool base_aligned = ((sx & 3) == 0) && ((reinterpret_cast<uintptr_t>(src) & 15U) == 0);
    for (int i = threadIdx.x; i < VEX * EY; i += blockDim.x) {
        const int vx = i % VEX;
        const int ey = i / VEX;
        const int ex = vx << 2;
        const int gx0 = x0 + ex - R;
        float* const out = raw + ey * EX + ex;
        if (tile_interior && base_aligned) {
            const int gy = y0 + ey - R;
            const size_t g = static_cast<size_t>(gx0) + static_cast<size_t>(gy) * sx +
                             static_cast<size_t>(z) * leap_z;
            *reinterpret_cast<float4*>(out) = *reinterpret_cast<const float4*>(src + g);
        } else if (base_aligned && gx0 >= 0 && gx0 + 3 < nx) {
            const int gy = dwt_halo_reflect(y0 + ey - R, ny);
            const size_t g = static_cast<size_t>(gx0) + static_cast<size_t>(gy) * sx +
                             static_cast<size_t>(z) * leap_z;
            *reinterpret_cast<float4*>(out) = *reinterpret_cast<const float4*>(src + g);
        } else {
            const int gy = dwt_halo_reflect(y0 + ey - R, ny);
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int gx = dwt_halo_reflect(gx0 + j, nx);
                out[j] = src[gx + static_cast<size_t>(gy) * sx +
                             static_cast<size_t>(z) * leap_z];
            }
        }
    }
    __syncthreads();

    // BX and x0 are even.  One owner evaluates an adjacent even/odd pair:
    // their 9/7 windows have a nine-value union, versus 9+7 independent
    // shared loads in the scalar-owner form.
    constexpr int XPAIRS = (BX + 1) / 2;
    for (int i = threadIdx.x; i < XPAIRS * EY; i += blockDim.x) {
        const int px = i % XPAIRS;
        const int ey = i / XPAIRS;
        const int lx = px << 1;
        const int gx = x0 + lx;
        const int xo = ey * BX + lx;
        if (gx >= nx)
            continue;
        const int c = ey * EX + lx + R;
        xbuf[xo] = low_filter<float>(raw[c],
                                     raw[c - 1] + raw[c + 1],
                                     raw[c - 2] + raw[c + 2],
                                     raw[c - 3] + raw[c + 3],
                                     raw[c - 4] + raw[c + 4]);
        if (gx + 1 < nx) {
            xbuf[xo + 1] = high_filter<float>(raw[c + 1],
                                              raw[c] + raw[c + 2],
                                              raw[c - 1] + raw[c + 3],
                                              raw[c - 2] + raw[c + 4]);
        }
    }
    __syncthreads();

    constexpr int YPAIRS = (BY + 1) / 2;
    for (int i = threadIdx.x; i < BX * YPAIRS; i += blockDim.x) {
        const int lx = i % BX;
        const int py = i / BX;
        const int ly = py << 1;
        const int gx = x0 + lx;
        const int gy = y0 + ly;
        if (gx >= nx || gy >= ny)
            continue;

        const int c = (ly + DWT_HALO_R) * BX + lx;
        const int output_x = (gx & 1) ? even_x + (gx >> 1) : (gx >> 1);
        const int low_y = gy >> 1;
        const size_t zoff = static_cast<size_t>(z) * leap_z;
        const float low_value = low_filter<float>(xbuf[c],
                                                  xbuf[c - BX] + xbuf[c + BX],
                                                  xbuf[c - 2 * BX] + xbuf[c + 2 * BX],
                                                  xbuf[c - 3 * BX] + xbuf[c + 3 * BX],
                                                  xbuf[c - 4 * BX] + xbuf[c + 4 * BX]);
        const size_t low_index = output_x + static_cast<size_t>(low_y) * sx + zoff;
        if constexpr (Quant == 0)
            dst[low_index] = low_value;
        else if constexpr (Quant == 2)
            quantization(low_value, low_index, output_x, low_y, z, data_dims);
        else if constexpr (Quant == 1) {
            if (gx & 1)
                quantization(low_value, low_index, output_x, low_y, z, data_dims);
            else
                dst[low_index] = low_value;
        }
        if (gy + 1 < ny) {
            const float high_value =
                high_filter<float>(xbuf[c + BX],
                                   xbuf[c] + xbuf[c + 2 * BX],
                                   xbuf[c - BX] + xbuf[c + 3 * BX],
                                   xbuf[c - 2 * BX] + xbuf[c + 4 * BX]);
            const uint32_t high_y = even_y + low_y;
            const size_t high_index = output_x + static_cast<size_t>(high_y) * sx + zoff;
            if constexpr (Quant != 0)
                quantization(high_value, high_index, output_x, high_y, z, data_dims);
            else
                dst[high_index] = high_value;
        }
    }
}

// Double-precision XY kernel.  Load two adjacent doubles at once when the
// source row is 16-byte aligned; physical X edges and odd row strides retain
// the scalar reflected path.
template <int BX, int BY, uint8_t Quant = 0>
__global__ void dwt3d_xy_double(
    const double* __restrict__ src, double* __restrict__ dst, dim3 data_dims, dim3 chunk_dims) {
    static_assert(Quant <= 2, "Quant must be 0, 1, or 2");
    constexpr int R = DWT_HALO_R;
    constexpr int EX = BX + 2 * R;
    constexpr int EY = BY + 2 * R;
    static_assert((EX & 1) == 0, "double2 loads require an even extended tile width");
    __shared__ alignas(128) double raw[EX * EY];
    __shared__ alignas(128) double xbuf[BX * EY];

    const int nx = static_cast<int>(chunk_dims.x);
    const int ny = static_cast<int>(chunk_dims.y);
    const int sx = static_cast<int>(data_dims.x);
    const int x0 = static_cast<int>(blockIdx.x) * BX;
    const int y0 = static_cast<int>(blockIdx.y) * BY;
    const int even_x = nx - (nx >> 1);
    const int even_y = ny - (ny >> 1);
    const size_t leap_z = static_cast<size_t>(data_dims.x) * data_dims.y;
    const int z = static_cast<int>(blockIdx.z);
    const bool tile_interior =
        x0 >= R && x0 + BX + R <= nx && y0 >= R && y0 + BY + R <= ny;
    constexpr int VEX = EX / 2;
    const bool base_aligned =
        (sx & 1) == 0 && (reinterpret_cast<uintptr_t>(src) & 15U) == 0U;
    for (int i = threadIdx.x; i < VEX * EY; i += blockDim.x) {
        const int vx = i % VEX;
        const int ey = i / VEX;
        const int ex = vx << 1;
        const int gx0 = x0 + ex - R;
        double* const out = raw + ey * EX + ex;
        if (tile_interior && base_aligned) {
            const int gy = y0 + ey - R;
            const size_t g = static_cast<size_t>(gx0) + static_cast<size_t>(gy) * sx +
                             static_cast<size_t>(z) * leap_z;
            *reinterpret_cast<double2*>(out) =
                *reinterpret_cast<const double2*>(src + g);
        } else if (base_aligned && gx0 >= 0 && gx0 + 1 < nx) {
            const int gy = dwt_halo_reflect(y0 + ey - R, ny);
            const size_t g = static_cast<size_t>(gx0) + static_cast<size_t>(gy) * sx +
                             static_cast<size_t>(z) * leap_z;
            *reinterpret_cast<double2*>(out) =
                *reinterpret_cast<const double2*>(src + g);
        } else {
            const int gy = dwt_halo_reflect(y0 + ey - R, ny);
#pragma unroll
            for (int j = 0; j < 2; ++j) {
                const int gx = dwt_halo_reflect(gx0 + j, nx);
                out[j] = src[gx + static_cast<size_t>(gy) * sx +
                             static_cast<size_t>(z) * leap_z];
            }
        }
    }
    __syncthreads();

    constexpr int XPAIRS = (BX + 1) / 2;
    for (int i = threadIdx.x; i < XPAIRS * EY; i += blockDim.x) {
        const int px = i % XPAIRS;
        const int ey = i / XPAIRS;
        const int lx = px << 1;
        const int gx = x0 + lx;
        const int xo = ey * BX + lx;
        if (gx >= nx)
            continue;
        const int c = ey * EX + lx + R;
        xbuf[xo] = low_filter<double>(raw[c],
                                      raw[c - 1] + raw[c + 1],
                                      raw[c - 2] + raw[c + 2],
                                      raw[c - 3] + raw[c + 3],
                                      raw[c - 4] + raw[c + 4]);
        if (gx + 1 < nx)
            xbuf[xo + 1] = high_filter<double>(raw[c + 1],
                                               raw[c] + raw[c + 2],
                                               raw[c - 1] + raw[c + 3],
                                               raw[c - 2] + raw[c + 4]);
    }
    __syncthreads();

    constexpr int YPAIRS = (BY + 1) / 2;
    for (int i = threadIdx.x; i < BX * YPAIRS; i += blockDim.x) {
        const int lx = i % BX;
        const int ly = (i / BX) << 1;
        const int gx = x0 + lx;
        const int gy = y0 + ly;
        if (gx >= nx || gy >= ny)
            continue;
        const int c = (ly + R) * BX + lx;
        const int output_x = (gx & 1) ? even_x + (gx >> 1) : (gx >> 1);
        const int low_y = gy >> 1;
        const size_t zoff = static_cast<size_t>(z) * leap_z;
        const double low_value = low_filter<double>(xbuf[c],
                                                    xbuf[c - BX] + xbuf[c + BX],
                                                    xbuf[c - 2 * BX] + xbuf[c + 2 * BX],
                                                    xbuf[c - 3 * BX] + xbuf[c + 3 * BX],
                                                    xbuf[c - 4 * BX] + xbuf[c + 4 * BX]);
        const size_t low_index = output_x + static_cast<size_t>(low_y) * sx + zoff;
        if constexpr (Quant == 0)
            dst[low_index] = low_value;
        else if constexpr (Quant == 2)
            quantization(low_value, low_index, output_x, low_y, z, data_dims);
        else if constexpr (Quant == 1) {
            if (gx & 1)
                quantization(low_value, low_index, output_x, low_y, z, data_dims);
            else
                dst[low_index] = low_value;
        }
        if (gy + 1 < ny) {
            const double high_value = high_filter<double>(xbuf[c + BX],
                                                          xbuf[c] + xbuf[c + 2 * BX],
                                                          xbuf[c - BX] + xbuf[c + 3 * BX],
                                                          xbuf[c - 2 * BX] + xbuf[c + 4 * BX]);
            const uint32_t high_y = even_y + low_y;
            const size_t high_index = output_x + static_cast<size_t>(high_y) * sx + zoff;
            if constexpr (Quant != 0)
                quantization(high_value, high_index, output_x, high_y, z, data_dims);
            else
                dst[high_index] = high_value;
        }
    }
}

template <typename T, uint8_t Quant = 0>
inline cudaError_t dwt3d_z_launch(const T* src, T* dst, dim3 data_dims, dim3 chunk_dims,
                                  uint32_t counter_index, cudaStream_t stream,
                                  uint32_t levels_to_run = 1U);

template <typename T>
inline cudaError_t dwt_z_active_staged_levels_launch(
    const T* src, T* dst, dim3 dims, int levels_z, cudaStream_t stream);

// At coarse plane levels the number of row tiles quickly drops below the
// occupancy-sized persistent grid.  Launching hundreds of CTAs that immediately
// fail the grid-stride loop is a measurable fixed cost on shallow volumes.
// Cap each launch at its exact tile count; the kernels and arithmetic are
// unchanged, and fine levels still use the full occupancy grid.
template <typename T>
inline int dwt_plane_x_launch_grid(uint32_t lx, uint32_t ly, uint32_t nz) {
    constexpr uint32_t cap = 32768U / sizeof(T);
    const uint32_t rpt = min(max(cap / lx, 1U), 256U);
    const uint64_t rows = static_cast<uint64_t>(ly) * nz;
    const uint64_t chunks = (rows + rpt - 1U) / rpt;
    const int occupancy_blocks = DWTConfig<T>::x_blocks;
    return chunks < static_cast<uint64_t>(occupancy_blocks)
               ? max(1, static_cast<int>(chunks))
               : occupancy_blocks;
}

template <typename T>
inline int dwt_plane_y_launch_grid(uint32_t lx, uint32_t ly, uint32_t nz) {
    constexpr uint32_t cap = 32768U / sizeof(T);
    const uint32_t rpt = max(cap / lx, 1U);
    const uint64_t rows = static_cast<uint64_t>(ly) * nz;
    const uint64_t chunks = (rows + rpt - 1U) / rpt;
    const int occupancy_blocks = DWTConfig<T>::y_blocks;
    return chunks < static_cast<uint64_t>(occupancy_blocks)
               ? max(1, static_cast<int>(chunks))
               : occupancy_blocks;
}

template <typename T>
inline void dwt_plane_y_coarse_launch(const T* src, T* dst, uint32_t nx, uint32_t ny, uint32_t lx, uint32_t ly,
                                      uint32_t lz, uint32_t counter_index, cudaStream_t stream) {
    const int gy = dwt_plane_y_launch_grid<T>(lx, ly, lz);
    dwt_y_axis<T><<<gy, TPB, 0, stream>>>(src, dst, nx, ny, lx, ly, lz, counter_index);
}

// Launch one fused XY level over the active low-low rectangle while retaining
// the original volume strides. Finalized subbands are quantized immediately;
// only the next low-low rectangle is materialized for the following level.
template <typename T, uint8_t Quant>
inline cudaError_t dwt3d_plane_xy_quant_launch(
    const T* src, T* dst, dim3 data_dims, dim3 chunk_dims, cudaStream_t stream) {
    static_assert(Quant == 1 || Quant == 2, "quantized XY launch requires Quant=1 or Quant=2");
    if constexpr (std::is_same_v<T, float>) {
        const dim3 grid((chunk_dims.x + DWT_TILE_BX64 - 1U) / DWT_TILE_BX64,
                        (chunk_dims.y + DWT_TILE_BY32 - 1U) / DWT_TILE_BY32,
                        chunk_dims.z);
        dwt3d_xy_float<DWT_TILE_BX64, DWT_TILE_BY32, Quant>
            <<<grid, 256, 0, stream>>>(src, dst, data_dims, chunk_dims);
    } else {
        const dim3 grid((chunk_dims.x + DWT_TILE_BX64 - 1U) / DWT_TILE_BX64,
                        (chunk_dims.y + DWT_TILE_BY16 - 1U) / DWT_TILE_BY16,
                        chunk_dims.z);
        dwt3d_xy_double<DWT_TILE_BX64, DWT_TILE_BY16, Quant>
            <<<grid, 256, 0, stream>>>(src, dst, data_dims, chunk_dims);
    }
    return cudaGetLastError();
}

// Non-dyadic plane transform: run all Z levels in the existing column kernel,
// then launch one ordinary fused XY+quant kernel for every XY level. Kernel
// launch boundaries provide the required global synchronization between XY
// levels without a cooperative persistent grid.
template <uint8_t Quant>
inline cudaError_t dwt3d_plane_xy_halo_float(
    float*& data, float*& tmp, dim3 dims, int levels_xy, int levels_z, const float* z_src,
    cudaStream_t stream) {
    if (levels_z > 0) {
        const cudaError_t z_error = dwt_z_active_staged_levels_launch<float>(
            z_src != nullptr ? z_src : data, data, dims, levels_z, stream);
        if (z_error != cudaSuccess)
            return z_error;
    } else if (z_src != nullptr && z_src != data) {
        const size_t bytes = static_cast<size_t>(dims.x) * dims.y * dims.z * sizeof(float);
        const cudaError_t copy_error =
            cudaMemcpyAsync(data, z_src, bytes, cudaMemcpyDeviceToDevice, stream);
        if (copy_error != cudaSuccess)
            return copy_error;
    }
    if (levels_xy <= 0)
        return cudaGetLastError();

    static_assert(Quant <= 2, "Quant must be 0, 1, or 2");
    if constexpr (Quant == 0) {
        const bool wide_tile = dims.x >= 384U && dims.y >= 384U;
        if (wide_tile) {
            const dim3 grid((dims.x + DWT_TILE_BX64 - 1U) / DWT_TILE_BX64,
                            (dims.y + DWT_TILE_BY32 - 1U) / DWT_TILE_BY32,
                            dims.z);
            dwt3d_xy_float<DWT_TILE_BX64, DWT_TILE_BY32, 0>
                <<<grid, 256, 0, stream>>>(data, tmp, dims, dims);
        } else {
            const dim3 grid((dims.x + DWT_TILE_BX32 - 1U) / DWT_TILE_BX32,
                            (dims.y + DWT_TILE_BY16 - 1U) / DWT_TILE_BY16,
                            dims.z);
            dwt3d_xy_float<DWT_TILE_BX32, DWT_TILE_BY16, 0>
                <<<grid, 256, 0, stream>>>(data, tmp, dims, dims);
        }
        float* current = tmp;
        float* scratch = data;
        uint32_t lx = dims.x - (dims.x >> 1U);
        uint32_t ly = dims.y - (dims.y >> 1U);
        for (int lev = 1; lev < levels_xy; ++lev) {
            const int gx = dwt_plane_x_launch_grid<float>(lx, ly, dims.z);
            dwt_x_axis<float><<<gx, TPB, 0, stream>>>(
                current, scratch, dims.x, dims.y, lx, ly, dims.z,
                static_cast<uint32_t>(lev));
            dwt_plane_y_coarse_launch<float>(scratch, current, dims.x, dims.y, lx, ly,
                                             dims.z, static_cast<uint32_t>(lev), stream);
            lx -= lx >> 1U;
            ly -= ly >> 1U;
        }
        data = current;
        tmp = scratch;
        return cudaGetLastError();
    }

    float* current = data;
    float* scratch = tmp;
    dim3 chunk_dims = dims;
    for (int lev = 0; lev < levels_xy; ++lev) {
        cudaError_t error;
        if (lev + 1 == levels_xy) {
            if constexpr (Quant == 2)
                error = dwt3d_plane_xy_quant_launch<float, 2>(
                    current, scratch, dims, chunk_dims, stream);
            else
                error = dwt3d_plane_xy_quant_launch<float, 1>(
                    current, scratch, dims, chunk_dims, stream);
        } else {
            error = dwt3d_plane_xy_quant_launch<float, 1>(
                current, scratch, dims, chunk_dims, stream);
        }
        if (error != cudaSuccess)
            return error;
        float* const swap = current;
        current = scratch;
        scratch = swap;
        chunk_dims.x -= chunk_dims.x >> 1U;
        chunk_dims.y -= chunk_dims.y >> 1U;
    }
    data = current;
    tmp = scratch;
    return cudaGetLastError();
}

template <typename T, uint8_t Quant = 0>
inline cudaError_t dwt_global_level_generic(
    T* cur, T* scratch, dim3 dims, uint32_t lx, uint32_t ly, uint32_t lz, uint32_t counter_index,
    cudaStream_t stream) {
    constexpr uint32_t xcap = 32768U / sizeof(T);
    if (lx == 0U || lx > xcap)
        return cudaErrorInvalidValue;

    dwt_x_axis<T><<<DWTConfig<T>::x_blocks, TPB, 0, stream>>>(
        cur, cur, dims.x, dims.y, lx, ly, lz, counter_index);
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess)
        return e;
    dwt_y_axis<T><<<DWTConfig<T>::y_blocks, TPB, 0, stream>>>(
        cur, scratch, dims.x, dims.y, lx, ly, lz, counter_index);
    e = cudaGetLastError();
    if (e != cudaSuccess)
        return e;
    return dwt3d_z_launch<T, Quant>(
        scratch, cur, dims, dim3(lx, ly, lz), counter_index, stream);
}

// Double-precision entry point for the same Z-all + per-level fused XY path.
template <typename T, uint8_t Quant>
inline cudaError_t dwt3d_plane_xy_halo_generic(
    T*& data, T*& tmp, dim3 dims, int levels_xy, int levels_z, const T* z_src,
    cudaStream_t stream) {
    if (data == nullptr || tmp == nullptr || dims.x == 0U || dims.y == 0U || dims.z == 0U)
        return cudaErrorInvalidValue;
    if (levels_z > 0) {
        const cudaError_t z_error = dwt_z_active_staged_levels_launch<T>(
            z_src != nullptr ? z_src : data, data, dims, levels_z, stream);
        if (z_error != cudaSuccess)
            return z_error;
    } else if (z_src != nullptr && z_src != data) {
        const size_t bytes = static_cast<size_t>(dims.x) * dims.y * dims.z * sizeof(T);
        const cudaError_t copy_error =
            cudaMemcpyAsync(data, z_src, bytes, cudaMemcpyDeviceToDevice, stream);
        if (copy_error != cudaSuccess)
            return copy_error;
    }
    const cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess || levels_xy <= 0)
        return error;

    static_assert(Quant <= 2, "Quant must be 0, 1, or 2");
    if constexpr (Quant == 0) {
        const dim3 grid((dims.x + DWT_TILE_BX64 - 1U) / DWT_TILE_BX64,
                        (dims.y + DWT_TILE_BY16 - 1U) / DWT_TILE_BY16,
                        dims.z);
        dwt3d_xy_double<DWT_TILE_BX64, DWT_TILE_BY16, 0>
            <<<grid, 256, 0, stream>>>(data, tmp, dims, dims);
        cudaError_t level_error = cudaGetLastError();
        if (level_error != cudaSuccess)
            return level_error;

        T* current = tmp;
        T* scratch = data;
        uint32_t lx = dims.x - (dims.x >> 1U);
        uint32_t ly = dims.y - (dims.y >> 1U);
        for (int lev = 1; lev < levels_xy; ++lev) {
            const int gx = dwt_plane_x_launch_grid<T>(lx, ly, dims.z);
            dwt_x_axis<T><<<gx, TPB, 0, stream>>>(
                current, scratch, dims.x, dims.y, lx, ly, dims.z,
                static_cast<uint32_t>(lev));
            level_error = cudaGetLastError();
            if (level_error != cudaSuccess)
                return level_error;
            dwt_plane_y_coarse_launch<T>(scratch, current, dims.x, dims.y, lx, ly, dims.z,
                                         static_cast<uint32_t>(lev), stream);
            level_error = cudaGetLastError();
            if (level_error != cudaSuccess)
                return level_error;
            lx -= lx >> 1U;
            ly -= ly >> 1U;
        }
        data = current;
        tmp = scratch;
        return cudaGetLastError();
    }

    T* current = data;
    T* scratch = tmp;
    dim3 chunk_dims = dims;
    for (int lev = 0; lev < levels_xy; ++lev) {
        cudaError_t level_error;
        if (lev + 1 == levels_xy) {
            if constexpr (Quant == 2)
                level_error = dwt3d_plane_xy_quant_launch<T, 2>(
                    current, scratch, dims, chunk_dims, stream);
            else
                level_error = dwt3d_plane_xy_quant_launch<T, 1>(
                    current, scratch, dims, chunk_dims, stream);
        } else {
            level_error = dwt3d_plane_xy_quant_launch<T, 1>(
                current, scratch, dims, chunk_dims, stream);
        }
        if (level_error != cudaSuccess)
            return level_error;
        T* const swap = current;
        current = scratch;
        scratch = swap;
        chunk_dims.x -= chunk_dims.x >> 1U;
        chunk_dims.y -= chunk_dims.y >> 1U;
    }
    data = current;
    tmp = scratch;
    return cudaGetLastError();
}

template <bool RegularBlocks>
__global__ __launch_bounds__(TPB, 3) void dwt_plane_z_l0_quant_float(
    const float* src, float* dst, dim3 dims);


// Non-cooperative dyadic path. Every level fuses X+Y in one shared-halo kernel,
// then runs the dependent Z stage and quantizes the subbands finalized at that
// level. Ordinary kernel-launch ordering supplies the required global barrier;
// no cooperative launch or grid-wide synchronization is used.
template <typename T, uint8_t Quant>
inline void dwt3d_xy_halo_zstage(T* data, T* tmp, dim3 dims, int levels, uint32_t q_bz, cudaStream_t stream) {
    dim3 chunk_dims = dims;
    for (int lev = 0; lev < levels; ++lev) {
        if constexpr (std::is_same_v<T, float>) {
            const dim3 grid((chunk_dims.x + DWT_TILE_BX64 - 1U) / DWT_TILE_BX64,
                            (chunk_dims.y + DWT_TILE_BY32 - 1U) / DWT_TILE_BY32,
                            chunk_dims.z);
            dwt3d_xy_float<DWT_TILE_BX64, DWT_TILE_BY32, 0><<<grid, 256, 0, stream>>>(data, tmp, dims, chunk_dims);
        } else if constexpr (std::is_same_v<T, double>) {
            const dim3 grid((chunk_dims.x + DWT_TILE_BX64 - 1U) / DWT_TILE_BX64,
                            (chunk_dims.y + DWT_TILE_BY16 - 1U) / DWT_TILE_BY16,
                            chunk_dims.z);
            dwt3d_xy_double<DWT_TILE_BX64, DWT_TILE_BY16, 0><<<grid, 256, 0, stream>>>(data, tmp, dims, chunk_dims);
        }

        if constexpr (Quant != 0) {
            const bool final_level = lev + 1 == levels;
            bool used_regular_l0 = false;
            if constexpr (std::is_same_v<T, float>) {
                if (lev == 0 && !final_level) {
                    uint32_t tile_cols = min(CS / (dims.z * 2U), dims.x * dims.y);
                    if ((tile_cols & ~3U) >= 4U)
                        tile_cols &= ~3U;
                    const bool regular_blocks = tile_cols != 0U && dims.x % tile_cols == 0U &&
                        (dims.x & 15U) == 0U && (dims.y & 15U) == 0U &&
                        dims.z % q_bz == 0U && (q_bz & (q_bz - 1U)) == 0U;

                    if (regular_blocks) {
                        dwt_plane_z_l0_quant_float<true><<<DWTConfig<float>::z_blocks, TPB, 0, stream>>>
                            (reinterpret_cast<const float*>(tmp), reinterpret_cast<float*>(data), dims);
                        used_regular_l0 = true;
                    }
                }
            }
            if (!used_regular_l0) {
                if (final_level) {
                    if constexpr (Quant == 2)
                        dwt3d_z_launch<T, 2>(tmp, data, dims, chunk_dims, static_cast<uint32_t>(lev), stream);
                    else
                        dwt3d_z_launch<T, 1>(tmp, data, dims, chunk_dims, static_cast<uint32_t>(lev), stream);
                } else {
                    dwt3d_z_launch<T, 1>(tmp, data, dims, chunk_dims, static_cast<uint32_t>(lev), stream);
                }
            }
        } else {
            dwt3d_z_launch<T>(tmp, data, dims, chunk_dims, static_cast<uint32_t>(lev), stream);
        }

        chunk_dims.x -= chunk_dims.x >> 1U;
        chunk_dims.y -= chunk_dims.y >> 1U;
        chunk_dims.z -= chunk_dims.z >> 1U;
    }
}

// ===========================================================================
// Non-dyadic (wavelet-packet) 3D DWT: transform Z to its full depth first,
// then transform every XY plane to its full depth.  Used when the XY and Z
// axes support a different number of levels (see can_use_dyadic).  Convolution
// form, same coefficients/rounding as dwt3d, so it matches the CPU reference
// TESTDWT::dwt3d_plane_convolution_levels_cpu bit-for-bit.
//
// Correctness-first implementation: `data` always holds the complete evolving
// volume; `tmp` is scratch for the single out-of-place axis pass.  Each axis
// pass below is a sub-cube transform parameterised by `sub` (active extent) and
// the global `data_leaps` strides -- identical math to the dyadic kernel's
// level>0 X/Y/Z blocks.
// ===========================================================================

// All `levels_z` levels of Z transform done in shared memory, in place on
// `data`.  A tile of columns (each = fixed (x,y), all z) is loaded once
// (coalesced), transformed through every level via shared ping-pong, and
// written back once.  This avoids the huge-z-stride redundant global reads
// (L2 thrash) AND the per-level copy-back of the baseline path.
// Requires full_z * 2 <= num (two ping-pong buffers); caller checks.
// Pair-fused convolution -> bit-identical to the baseline / CPU reference.
__device__ __forceinline__ bool
plane_z_final_owner_b(uint32_t z, uint32_t full_z, uint8_t levels_z) {
    // Four Z levels are common for shallow volumes. Keep that writeback path
    // branch-light; the generic loop handles every other legal depth.
    if (levels_z == 4U) {
        const uint32_t a1 = full_z - (full_z >> 1U);
        const uint32_t a2 = a1 - (a1 >> 1U);
        const uint32_t a3 = a2 - (a2 >> 1U);
        return z >= a1 || (z >= a3 && z < a2);
    }
    uint32_t active = full_z;
    for (uint8_t lev = 0; lev < levels_z; ++lev) {
        const uint32_t next = active - (active >> 1U);
        if (z >= next)
            return (lev & 1U) == 0U;
        active = next;
    }
    return (levels_z & 1U) != 0U;
}

template <typename T>
__device__ __forceinline__ T plane_z_low_filter(T center, T sum1, T sum2, T sum3, T sum4) {
    return low_filter<T>(center, sum1, sum2, sum3, sum4);
}

template <typename T>
__device__ __forceinline__ T plane_z_high_filter(T center, T sum1, T sum2, T sum3) {
    return high_filter<T>(center, sum1, sum2, sum3);
}

__device__ __forceinline__ void plane_z_forward_pair_float(const float* s_src,
                                                           float* s_dst,
                                                           uint32_t ncol,
                                                           uint32_t active_z,
                                                           uint32_t even_z,
                                                           uint32_t pair,
                                                           uint32_t lc) {
    const int z = static_cast<int>(pair << 1U);
    int zm1 = z - 1, zp1 = z + 1, zm2 = z - 2, zp2 = z + 2;
    int zm3 = z - 3, zp3 = z + 3, zm4 = z - 4, zp4 = z + 4;
    if (zm1 < 0)
        zm1 = -zm1;
    if (zm2 < 0)
        zm2 = -zm2;
    if (zm3 < 0)
        zm3 = -zm3;
    if (zm4 < 0)
        zm4 = -zm4;
    if (zp1 >= static_cast<int>(active_z))
        zp1 = 2 * active_z - 2 - zp1;
    if (zp2 >= static_cast<int>(active_z))
        zp2 = 2 * active_z - 2 - zp2;
    if (zp3 >= static_cast<int>(active_z))
        zp3 = 2 * active_z - 2 - zp3;
    if (zp4 >= static_cast<int>(active_z))
        zp4 = 2 * active_z - 2 - zp4;

    const float v0 = s_src[static_cast<uint32_t>(z) * ncol + lc];
    const float vm1 = s_src[static_cast<uint32_t>(zm1) * ncol + lc];
    const float vp1 = s_src[static_cast<uint32_t>(zp1) * ncol + lc];
    const float vm2 = s_src[static_cast<uint32_t>(zm2) * ncol + lc];
    const float vp2 = s_src[static_cast<uint32_t>(zp2) * ncol + lc];
    const float vm3 = s_src[static_cast<uint32_t>(zm3) * ncol + lc];
    const float vp3 = s_src[static_cast<uint32_t>(zp3) * ncol + lc];
    const float vm4 = s_src[static_cast<uint32_t>(zm4) * ncol + lc];
    const float vp4 = s_src[static_cast<uint32_t>(zp4) * ncol + lc];
    s_dst[pair * ncol + lc] = plane_z_low_filter<float>(v0,
                                                        add_rn<float>(vm1, vp1),
                                                        add_rn<float>(vm2, vp2),
                                                        add_rn<float>(vm3, vp3),
                                                        add_rn<float>(vm4, vp4));
    if (z + 1 < static_cast<int>(active_z)) {
        s_dst[(even_z + pair) * ncol + lc] = plane_z_high_filter<float>(
            vp1, add_rn<float>(v0, vp2), add_rn<float>(vm1, vp3), add_rn<float>(vm2, vp4));
    }
}

template <typename T, bool FuseDoubleQuant = false, bool RegularFloatQuant = false>
__device__ __forceinline__ void plane_z_columns_all(const T* input,
                                                    T* data,
                                                    dim3 data_dims,
                                                    dim3 data_leaps,
                                                    uint8_t levels_z,
                                                    T* s_data,
                                                    uint32_t num,
                                                    unsigned int* counter,
                                                    int* s_chunkID,
                                                    const DwtFuseQuantParams* fuse_l0 = nullptr) {
    const uint32_t full_z = data_dims.z;
    const uint32_t total_columns = data_dims.x * data_dims.y; // == data_leaps.z
    uint32_t cols = num / (full_z * 2U);
    cols = min(cols, total_columns);
    // Align float column tiles to float4.  This makes every full tile and both
    // ping-pong buffers 16-byte aligned; at most three columns of capacity are
    // traded for four times fewer global/shared staging instructions.
    if constexpr (sizeof(T) == sizeof(float)) {
        const uint32_t aligned_cols = cols & ~3U;
        if (aligned_cols >= 4U)
            cols = aligned_cols;
    }
    const uint32_t max_count = (total_columns - 1U) / cols + 1U;
    const uint32_t buf = cols * full_z;
    const int stride_z = data_leaps.z;

    // A dynamic queue keeps persistent CTAs balanced across many column tiles;
    // static grid-stride assignment has less balanced per-CTA tails.
    do {
        if (threadIdx.x == 0)
            *s_chunkID = static_cast<int>(atomicAdd(counter, 1U));
        __syncthreads();
        const int queued = *s_chunkID;
        __syncthreads();
        if (queued >= static_cast<int>(max_count))
            break;
        const uint32_t chunkID = static_cast<uint32_t>(queued);
        const uint32_t col0 = chunkID * cols;
        const uint32_t ncol = min(cols, total_columns - col0);
        const uint32_t tile_elems = ncol * full_z;
        uint32_t tile_x0 = 0U, tile_y = 0U;
        if constexpr (RegularFloatQuant) {
            tile_y = col0 / data_dims.x;
            tile_x0 = col0 - tile_y * data_dims.x;
        }
        T* const s_a = s_data;
        T* const s_b = s_data + buf;
        T* s_src = s_a;
        T* s_dst = s_b;

        // Load tile: s_src[z*ncol + lc] = data[(col0+lc) + z*stride_z].
        // Only the possible tail tile uses the scalar fallback.
        const bool vector_tile = sizeof(T) == sizeof(float) && cols >= 4U && ncol == cols &&
                                 (ncol & 3U) == 0U &&
                                 ((col0 | static_cast<uint32_t>(stride_z)) & 3U) == 0U &&
                                 (reinterpret_cast<uintptr_t>(input) & 15U) == 0U &&
                                 (reinterpret_cast<uintptr_t>(data) & 15U) == 0U;
        if constexpr (sizeof(T) == sizeof(float)) {
            if (vector_tile) {
                const uint32_t vec_cols = ncol >> 2U;
                const uint32_t vec_elems = full_z * vec_cols;
                float4* const sh4 = reinterpret_cast<float4*>(s_src);
                for (uint32_t i = threadIdx.x; i < vec_elems; i += TPB) {
                    const uint32_t z = i / vec_cols;
                    const uint32_t vc = i - z * vec_cols;
                    sh4[i] =
                        *reinterpret_cast<const float4*>(input + col0 + (vc << 2U) + z * stride_z);
                }
            } else {
                for (uint32_t i = threadIdx.x; i < tile_elems; i += TPB) {
                    const uint32_t z = i / ncol;
                    const uint32_t lc = i - z * ncol;
                    s_src[i] = input[(col0 + lc) + z * stride_z];
                }
            }
        } else {
            for (uint32_t i = threadIdx.x; i < tile_elems; i += TPB) {
                const uint32_t z = i / ncol;
                const uint32_t lc = i - z * ncol;
                s_src[i] = input[(col0 + lc) + z * stride_z];
            }
        }
        __syncthreads();

        uint32_t active_z = full_z;
        for (int lev = 0; lev < levels_z; ++lev) {
            const uint32_t even_z = active_z - (active_z >> 1U);
            if constexpr (sizeof(T) == sizeof(float)) {
                const uint32_t groups = (even_z + 1U) >> 1U;
                const uint32_t work = groups * ncol;
                for (uint32_t w = threadIdx.x; w < work; w += TPB) {
                    const uint32_t group = w / ncol;
                    const uint32_t lc = w - group * ncol;
                    const uint32_t pair = group << 1U;
                    const int z = static_cast<int>(pair << 1U);
                    if (pair + 1U < even_z && z >= 4 && z + 6 < static_cast<int>(active_z)) {
                        const float c0 = s_src[static_cast<uint32_t>(z - 4) * ncol + lc];
                        const float c1 = s_src[static_cast<uint32_t>(z - 3) * ncol + lc];
                        const float c2 = s_src[static_cast<uint32_t>(z - 2) * ncol + lc];
                        const float c3 = s_src[static_cast<uint32_t>(z - 1) * ncol + lc];
                        const float c4 = s_src[static_cast<uint32_t>(z) * ncol + lc];
                        const float c5 = s_src[static_cast<uint32_t>(z + 1) * ncol + lc];
                        const float c6 = s_src[static_cast<uint32_t>(z + 2) * ncol + lc];
                        const float c7 = s_src[static_cast<uint32_t>(z + 3) * ncol + lc];
                        const float c8 = s_src[static_cast<uint32_t>(z + 4) * ncol + lc];
                        const float c9 = s_src[static_cast<uint32_t>(z + 5) * ncol + lc];
                        const float c10 = s_src[static_cast<uint32_t>(z + 6) * ncol + lc];
                        s_dst[pair * ncol + lc] = plane_z_low_filter<float>(c4,
                                                                            add_rn<float>(c3, c5),
                                                                            add_rn<float>(c2, c6),
                                                                            add_rn<float>(c1, c7),
                                                                            add_rn<float>(c0, c8));
                        s_dst[(even_z + pair) * ncol + lc] =
                            plane_z_high_filter<float>(c5,
                                                       add_rn<float>(c4, c6),
                                                       add_rn<float>(c3, c7),
                                                       add_rn<float>(c2, c8));
                        s_dst[(pair + 1U) * ncol + lc] =
                            plane_z_low_filter<float>(c6,
                                                      add_rn<float>(c5, c7),
                                                      add_rn<float>(c4, c8),
                                                      add_rn<float>(c3, c9),
                                                      add_rn<float>(c2, c10));
                        if (z + 3 < static_cast<int>(active_z)) {
                            s_dst[(even_z + pair + 1U) * ncol + lc] =
                                plane_z_high_filter<float>(c7,
                                                           add_rn<float>(c6, c8),
                                                           add_rn<float>(c5, c9),
                                                           add_rn<float>(c4, c10));
                        }
                    } else {
                        plane_z_forward_pair_float(reinterpret_cast<const float*>(s_src),
                                                   reinterpret_cast<float*>(s_dst),
                                                   ncol,
                                                   active_z,
                                                   even_z,
                                                   pair,
                                                   lc);
                        if (pair + 1U < even_z)
                            plane_z_forward_pair_float(reinterpret_cast<const float*>(s_src),
                                                       reinterpret_cast<float*>(s_dst),
                                                       ncol,
                                                       active_z,
                                                       even_z,
                                                       pair + 1U,
                                                       lc);
                    }
                }
            } else {
                const uint32_t pairs = even_z * ncol;
                for (uint32_t i = threadIdx.x; i < pairs; i += TPB) {
                    const uint32_t pair = i / ncol;
                    const uint32_t lc = i - pair * ncol;
                    const int z = static_cast<int>(pair << 1U);
                    int zm1 = z - 1, zp1 = z + 1, zm2 = z - 2, zp2 = z + 2;
                    int zm3 = z - 3, zp3 = z + 3, zm4 = z - 4, zp4 = z + 4;
                    if (zm1 < 0)
                        zm1 = -zm1;
                    if (zm2 < 0)
                        zm2 = -zm2;
                    if (zm3 < 0)
                        zm3 = -zm3;
                    if (zm4 < 0)
                        zm4 = -zm4;
                    if (zp1 >= (int)active_z)
                        zp1 = 2 * active_z - 2 - zp1;
                    if (zp2 >= (int)active_z)
                        zp2 = 2 * active_z - 2 - zp2;
                    if (zp3 >= (int)active_z)
                        zp3 = 2 * active_z - 2 - zp3;
                    if (zp4 >= (int)active_z)
                        zp4 = 2 * active_z - 2 - zp4;
                    const T v0 = s_src[(uint32_t)z * ncol + lc];
                    const T vm1 = s_src[(uint32_t)zm1 * ncol + lc];
                    const T vp1 = s_src[(uint32_t)zp1 * ncol + lc];
                    const T vm2 = s_src[(uint32_t)zm2 * ncol + lc];
                    const T vp2 = s_src[(uint32_t)zp2 * ncol + lc];
                    const T vm3 = s_src[(uint32_t)zm3 * ncol + lc];
                    const T vp3 = s_src[(uint32_t)zp3 * ncol + lc];
                    const T vm4 = s_src[(uint32_t)zm4 * ncol + lc];
                    const T vp4 = s_src[(uint32_t)zp4 * ncol + lc];
                    s_dst[pair * ncol + lc] = plane_z_low_filter<T>(v0,
                                                                    add_rn<T>(vm1, vp1),
                                                                    add_rn<T>(vm2, vp2),
                                                                    add_rn<T>(vm3, vp3),
                                                                    add_rn<T>(vm4, vp4));
                    if (z + 1 < (int)active_z)
                        s_dst[(even_z + pair) * ncol + lc] = plane_z_high_filter<T>(
                            vp1, add_rn<T>(v0, vp2), add_rn<T>(vm1, vp3), add_rn<T>(vm2, vp4));
                }
            }
            __syncthreads();

            T* swap = s_src;
            s_src = s_dst;
            s_dst = swap;
            active_z = even_z;
        }

        // Write the fully z-transformed columns back to `data` once. An optional
        // fused producer can consume finalized L0 detail while it is still in
        // shared memory, materializing only LLL for later coarse levels.
        if constexpr (sizeof(T) == sizeof(float)) {
            if (fuse_l0 != nullptr) {
                const uint32_t low_x = data_dims.x - (data_dims.x >> 1U);
                const uint32_t low_y = data_dims.y - (data_dims.y >> 1U);
                const uint32_t low_z = data_dims.z - (data_dims.z >> 1U);
                for (uint32_t i = threadIdx.x; i < tile_elems; i += TPB) {
                    const uint32_t z = i / ncol;
                    const uint32_t lc = i - z * ncol;
                    const uint32_t col = col0 + lc;
                    uint32_t x, y;
                    if constexpr (RegularFloatQuant) {
                        x = tile_x0 + lc;
                        y = tile_y;
                    } else {
                        x = col % data_dims.x;
                        y = col / data_dims.x;
                    }
                    const T* const owner = plane_z_final_owner_b(z, full_z, levels_z) ? s_b : s_a;
                    const float value = static_cast<float>(owner[i]);
                    const size_t s = static_cast<size_t>(col) + static_cast<size_t>(z) * stride_z;
                    if (x < low_x && y < low_y && z < low_z) {
                        data[s] = static_cast<T>(value);
                    } else if constexpr (RegularFloatQuant) {
                        dwt_quant_emit_regular_float(value, s, x, y, z, fuse_l0);
                    } else {
                        dwt_quant_emit_float(value, s, x, y, z, data_dims, fuse_l0);
                    }
                }
            } else if (vector_tile) {
                const uint32_t vec_cols = ncol >> 2U;
                const uint32_t vec_elems = full_z * vec_cols;
                const float4* const a4 = reinterpret_cast<const float4*>(s_a);
                const float4* const b4 = reinterpret_cast<const float4*>(s_b);
                for (uint32_t i = threadIdx.x; i < vec_elems; i += TPB) {
                    const uint32_t z = i / vec_cols;
                    const uint32_t vc = i - z * vec_cols;
                    const float4 value = plane_z_final_owner_b(z, full_z, levels_z) ? b4[i] : a4[i];
                    *reinterpret_cast<float4*>(data + col0 + (vc << 2U) + z * stride_z) = value;
                }
            } else {
                for (uint32_t i = threadIdx.x; i < tile_elems; i += TPB) {
                    const uint32_t z = i / ncol;
                    const uint32_t lc = i - z * ncol;
                    const T* const owner = plane_z_final_owner_b(z, full_z, levels_z) ? s_b : s_a;
                    data[(col0 + lc) + z * stride_z] = owner[i];
                }
            }
        } else {
            for (uint32_t i = threadIdx.x; i < tile_elems; i += TPB) {
                const uint32_t z = i / ncol;
                const uint32_t lc = i - z * ncol;
                const uint32_t col = col0 + lc;
                const T* const owner = plane_z_final_owner_b(z, full_z, levels_z) ? s_b : s_a;
                const size_t s = static_cast<size_t>(col) + static_cast<size_t>(z) * stride_z;
                if constexpr (FuseDoubleQuant) {
                    static_assert(sizeof(T) == sizeof(double),
                                  "fused double quantization requires double");
                    const uint32_t x = col % data_dims.x;
                    const uint32_t y = col / data_dims.x;
                    const uint32_t low_x = data_dims.x - (data_dims.x >> 1U);
                    const uint32_t low_y = data_dims.y - (data_dims.y >> 1U);
                    const uint32_t low_z = data_dims.z - (data_dims.z >> 1U);
                    if (x < low_x && y < low_y && z < low_z)
                        data[s] = owner[i];
                    else
                        quantization(
                            static_cast<double>(owner[i]), s, x, y, z, data_dims);
                } else {
                    data[s] = owner[i];
                }
            }
        }
        __syncthreads();
    } while (true);
}

template <typename T>
__device__ __forceinline__ void plane_z_columns_all(T* data,
                                                    dim3 data_dims,
                                                    dim3 data_leaps,
                                                    uint8_t levels_z,
                                                    T* s_data,
                                                    uint32_t num,
                                                    unsigned int* counter,
                                                    int* s_chunkID) {
    plane_z_columns_all<T>(
        data, data, data_dims, data_leaps, levels_z, s_data, num, counter, s_chunkID);
}

// One level of Z transform on sub-cube `sub`, reading `src`, writing `dst`
// (deinterleaved low|high along Z).  Per-element baseline (no shared).
template <typename T>
__device__ __forceinline__ void plane_z_level(const T* src,
                                              T* dst,
                                              dim3 sub,
                                              dim3 data_leaps,
                                              uint32_t num,
                                              unsigned int* counter,
                                              int* s_chunkID) {
    const uint32_t chunk_leaps_z = sub.x * sub.y; // compact z-stride
    const uint32_t total_elems = chunk_leaps_z * sub.z;
    const uint32_t max_count = (total_elems - 1U) / num + 1U;
    const int even_tile = sub.z - (sub.z >> 1);
    const int stride_z = data_leaps.z;

    do {
        if (threadIdx.x == 0)
            *s_chunkID = static_cast<int>(atomicAdd(counter, 1U));
        __syncthreads();
        int chunkID = *s_chunkID;
        __syncthreads();
        if (chunkID >= static_cast<int>(max_count))
            break;
        const uint32_t local_start = chunkID * num;
        const uint32_t len = min(num, total_elems - local_start);
        for (int i = threadIdx.x; i < (int)len; i += TPB) {
            uint32_t local_id = local_start + i;
            int idx = local_id % sub.x;
            int gidy = (local_id / sub.x) % sub.y;
            int gidz = local_id / chunk_leaps_z;
            int global_id = idx + gidy * data_leaps.y + gidz * data_leaps.z;

            if ((gidz & 1) == 1) {
                int out_id = idx + gidy * data_leaps.y + ((gidz >> 1) + even_tile) * data_leaps.z;
                int zm1 = gidz - 1, zp1 = gidz + 1, zm2 = gidz - 2, zp2 = gidz + 2;
                int zm3 = gidz - 3, zp3 = gidz + 3;
                if (zm1 < 0)
                    zm1 = -zm1;
                if (zm2 < 0)
                    zm2 = -zm2;
                if (zm3 < 0)
                    zm3 = -zm3;
                if (zp1 >= (int)sub.z)
                    zp1 = 2 * sub.z - 2 - zp1;
                if (zp2 >= (int)sub.z)
                    zp2 = 2 * sub.z - 2 - zp2;
                if (zp3 >= (int)sub.z)
                    zp3 = 2 * sub.z - 2 - zp3;

                T sum1 = add_rn<T>(src[global_id + (zm1 - gidz) * stride_z],
                                   src[global_id + (zp1 - gidz) * stride_z]);
                T sum2 = add_rn<T>(src[global_id + (zm2 - gidz) * stride_z],
                                   src[global_id + (zp2 - gidz) * stride_z]);
                T sum3 = add_rn<T>(src[global_id + (zm3 - gidz) * stride_z],
                                   src[global_id + (zp3 - gidz) * stride_z]);
                dst[out_id] = high_filter<T>(src[global_id], sum1, sum2, sum3);
            } else {
                int out_id = idx + gidy * data_leaps.y + (gidz >> 1) * data_leaps.z;
                int zm1 = gidz - 1, zp1 = gidz + 1, zm2 = gidz - 2, zp2 = gidz + 2;
                int zm3 = gidz - 3, zp3 = gidz + 3, zm4 = gidz - 4, zp4 = gidz + 4;
                if (zm1 < 0)
                    zm1 = -zm1;
                if (zm2 < 0)
                    zm2 = -zm2;
                if (zm3 < 0)
                    zm3 = -zm3;
                if (zm4 < 0)
                    zm4 = -zm4;
                if (zp1 >= (int)sub.z)
                    zp1 = 2 * sub.z - 2 - zp1;
                if (zp2 >= (int)sub.z)
                    zp2 = 2 * sub.z - 2 - zp2;
                if (zp3 >= (int)sub.z)
                    zp3 = 2 * sub.z - 2 - zp3;
                if (zp4 >= (int)sub.z)
                    zp4 = 2 * sub.z - 2 - zp4;

                T sum1 = add_rn<T>(src[global_id + (zm1 - gidz) * stride_z],
                                   src[global_id + (zp1 - gidz) * stride_z]);
                T sum2 = add_rn<T>(src[global_id + (zm2 - gidz) * stride_z],
                                   src[global_id + (zp2 - gidz) * stride_z]);
                T sum3 = add_rn<T>(src[global_id + (zm3 - gidz) * stride_z],
                                   src[global_id + (zp3 - gidz) * stride_z]);
                T sum4 = add_rn<T>(src[global_id + (zm4 - gidz) * stride_z],
                                   src[global_id + (zp4 - gidz) * stride_z]);
                dst[out_id] = low_filter<T>(src[global_id], sum1, sum2, sum3, sum4);
            }
        }
    } while (true);
}

// Contiguous copy of `count` elements src -> dst (grid-stride, no work queue).
template <typename T>
__device__ __forceinline__ void plane_copy_contiguous(T* dst, const T* src, uint32_t count) {
    const uint32_t stride = gridDim.x * blockDim.x;
    for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += stride)
        dst[i] = src[i];
}

// Copy the leading `per_slice` elements of every z-slice src -> dst (the region
// (all x, y < active_y) is contiguous per slice when the x extent is full).
template <typename T>
__device__ __forceinline__ void
plane_copy_rows(T* dst, const T* src, size_t per_slice, uint32_t nz, size_t leap_z) {
    const size_t count = per_slice * nz;
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < count;
         i += stride) {
        const size_t z = i / per_slice, j = i - z * per_slice;
        dst[z * leap_z + j] = src[z * leap_z + j];
    }
}

template <typename T, uint32_t CS_BYTES = CS>
__global__ __launch_bounds__(TPB, CDF97_DWT_MIN_BLOCKS(T)) void dwt3d_plane(
    T* data, T* tmp, dim3 data_dims, dim3 data_leaps, uint8_t max_level_xy,
    uint8_t max_level_z) {
    cooperative_groups::grid_group grid = cooperative_groups::this_grid();

    constexpr uint32_t num = CS_BYTES / sizeof(T) * 4;
    __shared__ alignas(128) T s_data[num];
    __shared__ int s_chunkID;

    // Transform every complete Z column to full depth before the XY stages.
    if (max_level_z > 0) {
        if (data_dims.z * 2U <= num) {
            plane_z_columns_all<T>(
                data, data_dims, data_leaps, max_level_z, s_data, num, &z_count[0], &s_chunkID);
            grid.sync();
        } else {
            uint32_t active_z = data_dims.z;
            for (int lz = 0; lz < max_level_z; ++lz) {
                dim3 sub(data_dims.x, data_dims.y, active_z);
                plane_z_level<T>(data, tmp, sub, data_leaps, num, &z_count[lz], &s_chunkID);
                grid.sync();
                plane_copy_contiguous<T>(data, tmp, active_z * data_leaps.z);
                grid.sync();
                active_z -= active_z >> 1;
            }
        }
    }

    // Transform every XY plane to full depth with locked X/Y levels.
    uint32_t active_x = data_dims.x;
    uint32_t active_y = data_dims.y;
    for (int lxy = 0; lxy < max_level_xy; ++lxy) {
        // X: data -> tmp (reuse dwt_x_rows over chunks of rows, like the Y pass)
        {
            const uint32_t tile_yx = min(num / active_x, 256u);
            const uint32_t total_rows = active_y * data_dims.z;
            const uint32_t max_count = (total_rows - 1U) / tile_yx + 1U;
            do {
                if (threadIdx.x == 0)
                    s_chunkID = static_cast<int>(atomicAdd(&x_count[lxy], 1U));
                __syncthreads();
                int chunkID = s_chunkID;
                __syncthreads();
                if (chunkID >= static_cast<int>(max_count))
                    break;
                uint32_t row0 = chunkID * tile_yx;
                uint32_t rows = min(tile_yx, total_rows - row0);
                dwt_x_rows<T>(tmp, data, row0, rows, active_x, active_y, data_leaps, s_data);
            } while (true);
        }
        grid.sync();
        // Y: tmp -> data (reuse dwt_y_rows over chunks of rows)
        {
            const uint32_t tile_y = num / active_x;
            const uint32_t total_rows = active_y * data_dims.z;
            const uint32_t max_count = (total_rows - 1U) / tile_y + 1U;
            do {
                if (threadIdx.x == 0)
                    s_chunkID = static_cast<int>(atomicAdd(&y_count[lxy], 1U));
                __syncthreads();
                int chunkID = s_chunkID;
                __syncthreads();
                if (chunkID >= static_cast<int>(max_count))
                    break;
                uint32_t row0 = chunkID * tile_y;
                uint32_t rows = min(tile_y, total_rows - row0);
                dwt_y_rows<T>(data, tmp, row0, rows, active_x, active_y, data_leaps);
            } while (true);
        }
        grid.sync();
        active_x -= active_x >> 1;
        active_y -= active_y >> 1;
    }
}

__device__ __forceinline__ int dwt_reflect(int i, int n) {
    if (n <= 1)
        return 0;
    int reflected = i < 0 ? -i : i;
    if (reflected >= n)
        reflected = 2 * (n - 1) - reflected;
    if (reflected >= 0 && reflected < n)
        return reflected;
    const int period = 2 * (n - 1);
    i %= period;
    if (i < 0)
        i += period;
    return i < n ? i : period - i;
}

template <typename T, uint8_t Quant>
__device__ __forceinline__ void dwt_z_active_store(T value, T* dst, size_t s, uint32_t x,
                                                   uint32_t y, uint32_t z, uint32_t low_x,
                                                   uint32_t low_y, bool low_z, dim3 data_dims) {
    if constexpr (Quant != 0) {
        if (!low_z || Quant == 2 || x >= low_x || y >= low_y) {
            if constexpr (sizeof(T) == sizeof(float))
                quantization(static_cast<float>(value), s, x, y, z, data_dims);
            else
                quantization(static_cast<double>(value), s, x, y, z, data_dims);
        } else {
            dst[s] = value;
        }
    } else {
        dst[s] = value;
    }
}

template <typename T, uint8_t Quant>
__device__ __forceinline__ void dwt_z_active_transform_pair(
    const T* s_data, T* dst, uint32_t ncol, dim3 data_dims, uint32_t lx, uint32_t lz,
    uint32_t column0, uint32_t pair, uint32_t local_column, size_t leap_z, uint32_t even_z,
    uint32_t low_x, uint32_t low_y) {
    const uint32_t active_column = column0 + local_column;
    const uint32_t x = active_column % lx;
    const uint32_t y = active_column / lx;
    const int z = static_cast<int>(pair << 1U);
    const int zm1 = dwt_reflect(z - 1, lz);
    const int zp1 = dwt_reflect(z + 1, lz);
    const int zm2 = dwt_reflect(z - 2, lz);
    const int zp2 = dwt_reflect(z + 2, lz);
    const int zm3 = dwt_reflect(z - 3, lz);
    const int zp3 = dwt_reflect(z + 3, lz);
    const int zm4 = dwt_reflect(z - 4, lz);
    const int zp4 = dwt_reflect(z + 4, lz);
    const T c0 = s_data[static_cast<uint32_t>(z) * ncol + local_column];
    const T vm1 = s_data[static_cast<uint32_t>(zm1) * ncol + local_column];
    const T vp1 = s_data[static_cast<uint32_t>(zp1) * ncol + local_column];
    const T vm2 = s_data[static_cast<uint32_t>(zm2) * ncol + local_column];
    const T vp2 = s_data[static_cast<uint32_t>(zp2) * ncol + local_column];
    const T vm3 = s_data[static_cast<uint32_t>(zm3) * ncol + local_column];
    const T vp3 = s_data[static_cast<uint32_t>(zp3) * ncol + local_column];
    const T vm4 = s_data[static_cast<uint32_t>(zm4) * ncol + local_column];
    const T vp4 = s_data[static_cast<uint32_t>(zp4) * ncol + local_column];
    const size_t xy = x + static_cast<size_t>(y) * data_dims.x;
    const size_t low_s = xy + static_cast<size_t>(pair) * leap_z;
    const T low = plane_z_low_filter<T>(c0,
                                        add_rn<T>(vm1, vp1),
                                        add_rn<T>(vm2, vp2),
                                        add_rn<T>(vm3, vp3),
                                        add_rn<T>(vm4, vp4));
    dwt_z_active_store<T, Quant>(
        low, dst, low_s, x, y, pair, low_x, low_y, true, data_dims);
    if (z + 1 < static_cast<int>(lz)) {
        const T high = plane_z_high_filter<T>(
            vp1, add_rn<T>(c0, vp2), add_rn<T>(vm1, vp3), add_rn<T>(vm2, vp4));
        const uint32_t high_z = even_z + pair;
        const size_t high_s = xy + static_cast<size_t>(high_z) * leap_z;
        dwt_z_active_store<T, Quant>(
            high, dst, high_s, x, y, high_z, low_x, low_y, false, data_dims);
    }
}

// Direct global-memory kernel for a single active Z level. It is used when the
// active cube has too few column chunks to occupy the shared-memory scheduler.
template <typename T, uint8_t Quant>
__global__ __launch_bounds__(TPB, std::is_same_v<T, float> && Quant == 0 ? 3 : 2) void
dwt_3d_z(const T* src, T* dst, dim3 data_dims, dim3 chunk_dims) {
    const uint32_t nx = data_dims.x;
    const uint32_t ny = data_dims.y;
    const uint32_t lx = chunk_dims.x;
    const uint32_t ly = chunk_dims.y;
    const uint32_t lz = chunk_dims.z;
    const size_t leap_z = static_cast<size_t>(nx) * ny;
    const uint32_t low_x = lx - (lx >> 1U);
    const uint32_t low_y = ly - (ly >> 1U);
    const uint32_t low_z = lz - (lz >> 1U);
    const size_t active_plane = static_cast<size_t>(lx) * ly;
    const size_t work = active_plane * low_z;
    const size_t grid_stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    for (size_t g = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; g < work;
         g += grid_stride) {
        const uint32_t pair = static_cast<uint32_t>(g / active_plane);
        const size_t local_xy = g - static_cast<size_t>(pair) * active_plane;
        const uint32_t x = static_cast<uint32_t>(local_xy % lx);
        const uint32_t y = static_cast<uint32_t>(local_xy / lx);
        const int z = static_cast<int>(pair << 1U);
        const size_t xy = x + static_cast<size_t>(y) * nx;
        const auto value = [&](int zi) -> T {
            return src[xy + static_cast<size_t>(dwt_reflect(zi, lz)) * leap_z];
        };
        const T low = plane_z_low_filter<T>(
            value(z), add_rn<T>(value(z - 1), value(z + 1)),
            add_rn<T>(value(z - 2), value(z + 2)),
            add_rn<T>(value(z - 3), value(z + 3)),
            add_rn<T>(value(z - 4), value(z + 4)));
        const size_t low_s = xy + static_cast<size_t>(pair) * leap_z;
        dwt_z_active_store<T, Quant>(
            low, dst, low_s, x, y, pair, low_x, low_y, true, data_dims);
        if (z + 1 < static_cast<int>(lz)) {
            const int odd_z = z + 1;
            const T high = plane_z_high_filter<T>(
                value(odd_z), add_rn<T>(value(odd_z - 1), value(odd_z + 1)),
                add_rn<T>(value(odd_z - 2), value(odd_z + 2)),
                add_rn<T>(value(odd_z - 3), value(odd_z + 3)));
            const uint32_t high_z = low_z + pair;
            const size_t high_s = xy + static_cast<size_t>(high_z) * leap_z;
            dwt_z_active_store<T, Quant>(
                high, dst, high_s, x, y, high_z, low_x, low_y, false, data_dims);
        }
    }
}

// Shared-memory Z kernel. `data_dims` describes physical storage strides;
// `chunk_dims` describes only the current active region.
template <typename T, uint8_t Quant, bool FullColumns>
__global__ __launch_bounds__(TPB, std::is_same_v<T, float> && (Quant == 0 || FullColumns) ? 3 : 2) void
dwt_3d_z_s(const T* src, T* dst, dim3 data_dims, dim3 chunk_dims, uint32_t counter_index,
           uint32_t levels_to_run, uint32_t shared_capacity) {
    const uint32_t nx = data_dims.x;
    const uint32_t ny = data_dims.y;
    const uint32_t lx = chunk_dims.x;
    const uint32_t ly = chunk_dims.y;
    const uint32_t lz = chunk_dims.z;
    extern __shared__ __align__(128) unsigned char dwt_z_active_smem[];
    T* const s_data = reinterpret_cast<T*>(dwt_z_active_smem);
    __shared__ int chunk_id;

    if (lx == 0U || ly == 0U || lz == 0U || levels_to_run == 0U || shared_capacity < lz)
        return;

    if constexpr (FullColumns) {
        static_assert(Quant != 2, "full-column Z never finalizes the last dyadic low cube");
        const dim3 active_dims(nx, ny, lz);
        const dim3 storage_leaps(1U, nx, nx * ny);
        if constexpr (std::is_same_v<T, float> && Quant != 0) {
            plane_z_columns_all<T, false, false>(
                src, dst, active_dims, storage_leaps, static_cast<uint8_t>(levels_to_run), s_data,
                shared_capacity, &z_count[0], &chunk_id, &float_quant_params);
        } else if constexpr (std::is_same_v<T, double> && Quant != 0) {
            constexpr uint32_t fixed_capacity = 32768U / sizeof(T);
            __shared__ alignas(128) T fixed_s_data[fixed_capacity];
            plane_z_columns_all<T, true>(src, dst, active_dims, storage_leaps,
                                         static_cast<uint8_t>(levels_to_run), fixed_s_data,
                                         fixed_capacity, &z_count[0], &chunk_id);
        } else {
            plane_z_columns_all<T>(src, dst, active_dims, storage_leaps,
                                   static_cast<uint8_t>(levels_to_run), s_data, shared_capacity,
                                   &z_count[0], &chunk_id);
        }
        return;
    } else {
    const size_t leap_z = static_cast<size_t>(nx) * ny;
    const uint32_t total_columns = lx * ly;
    uint32_t columns_per_chunk = min(shared_capacity / lz, total_columns);
    if constexpr (sizeof(T) == sizeof(float)) {
        if (columns_per_chunk >= 4U)
            columns_per_chunk &= ~3U;
    } else {
        if (columns_per_chunk >= 2U)
            columns_per_chunk &= ~1U;
    }
    columns_per_chunk = max(columns_per_chunk, 1U);
    const uint32_t chunks = (total_columns + columns_per_chunk - 1U) / columns_per_chunk;
    const uint32_t even_z = lz - (lz >> 1U);
    const uint32_t low_x = lx - (lx >> 1U);
    const uint32_t low_y = ly - (ly >> 1U);

    while (true) {
        if (threadIdx.x == 0)
            chunk_id = static_cast<int>(atomicAdd(&z_count[counter_index], 1U));
        __syncthreads();
        const uint32_t chunk = static_cast<uint32_t>(chunk_id);
        if (chunk >= chunks)
            break;

        const uint32_t column0 = chunk * columns_per_chunk;
        const uint32_t ncol = min(columns_per_chunk, total_columns - column0);
        const uint32_t tile_elements = ncol * lz;
        const bool contiguous_columns = lx == nx;

        if constexpr (sizeof(T) == sizeof(float)) {
            const bool vector_load = contiguous_columns && ncol == columns_per_chunk &&
                                     (ncol & 3U) == 0U && (column0 & 3U) == 0U &&
                                     (leap_z & 3U) == 0U &&
                                     (reinterpret_cast<uintptr_t>(src) & 15U) == 0U;
            if (vector_load) {
                const uint32_t vector_columns = ncol >> 2U;
                const uint32_t vector_elements = lz * vector_columns;
                float4* const shared4 = reinterpret_cast<float4*>(s_data);
                for (uint32_t i = threadIdx.x; i < vector_elements; i += TPB) {
                    const uint32_t z = i / vector_columns;
                    const uint32_t vc = i - z * vector_columns;
                    shared4[i] = *reinterpret_cast<const float4*>(
                        src + column0 + (vc << 2U) + static_cast<size_t>(z) * leap_z);
                }
            } else {
                for (uint32_t i = threadIdx.x; i < tile_elements; i += TPB) {
                    const uint32_t z = i / ncol;
                    const uint32_t local_column = i - z * ncol;
                    const uint32_t active_column = column0 + local_column;
                    const uint32_t x = active_column % lx;
                    const uint32_t y = active_column / lx;
                    s_data[i] = src[x + static_cast<size_t>(y) * nx +
                                    static_cast<size_t>(z) * leap_z];
                }
            }
        } else {
            const bool vector_load = contiguous_columns && ncol == columns_per_chunk &&
                                     (ncol & 1U) == 0U && (column0 & 1U) == 0U &&
                                     (leap_z & 1U) == 0U &&
                                     (reinterpret_cast<uintptr_t>(src) & 15U) == 0U;
            if (vector_load) {
                const uint32_t vector_columns = ncol >> 1U;
                const uint32_t vector_elements = lz * vector_columns;
                double2* const shared2 = reinterpret_cast<double2*>(s_data);
                for (uint32_t i = threadIdx.x; i < vector_elements; i += TPB) {
                    const uint32_t z = i / vector_columns;
                    const uint32_t vc = i - z * vector_columns;
                    shared2[i] = *reinterpret_cast<const double2*>(
                        src + column0 + (vc << 1U) + static_cast<size_t>(z) * leap_z);
                }
            } else {
                for (uint32_t i = threadIdx.x; i < tile_elements; i += TPB) {
                    const uint32_t z = i / ncol;
                    const uint32_t local_column = i - z * ncol;
                    const uint32_t active_column = column0 + local_column;
                    const uint32_t x = active_column % lx;
                    const uint32_t y = active_column / lx;
                    s_data[i] = src[x + static_cast<size_t>(y) * nx +
                                    static_cast<size_t>(z) * leap_z];
                }
            }
        }
        __syncthreads();

        if constexpr (sizeof(T) == sizeof(float)) {
            const uint32_t groups = (even_z + 1U) >> 1U;
            const uint32_t work = groups * ncol;
            for (uint32_t i = threadIdx.x; i < work; i += TPB) {
                const uint32_t group = i / ncol;
                const uint32_t local_column = i - group * ncol;
                const uint32_t pair = group << 1U;
                const int z = static_cast<int>(pair << 1U);
                if (pair + 1U < even_z && z >= 4 && z + 6 < static_cast<int>(lz)) {
                    const float c0 = s_data[static_cast<uint32_t>(z - 4) * ncol + local_column];
                    const float c1 = s_data[static_cast<uint32_t>(z - 3) * ncol + local_column];
                    const float c2 = s_data[static_cast<uint32_t>(z - 2) * ncol + local_column];
                    const float c3 = s_data[static_cast<uint32_t>(z - 1) * ncol + local_column];
                    const float c4 = s_data[static_cast<uint32_t>(z) * ncol + local_column];
                    const float c5 = s_data[static_cast<uint32_t>(z + 1) * ncol + local_column];
                    const float c6 = s_data[static_cast<uint32_t>(z + 2) * ncol + local_column];
                    const float c7 = s_data[static_cast<uint32_t>(z + 3) * ncol + local_column];
                    const float c8 = s_data[static_cast<uint32_t>(z + 4) * ncol + local_column];
                    const float c9 = s_data[static_cast<uint32_t>(z + 5) * ncol + local_column];
                    const float c10 = s_data[static_cast<uint32_t>(z + 6) * ncol + local_column];
                    const uint32_t active_column = column0 + local_column;
                    const uint32_t x = active_column % lx;
                    const uint32_t y = active_column / lx;
                    const size_t xy = x + static_cast<size_t>(y) * nx;
                    const float low0 = plane_z_low_filter<float>(c4,
                                                                 add_rn<float>(c3, c5),
                                                                 add_rn<float>(c2, c6),
                                                                 add_rn<float>(c1, c7),
                                                                 add_rn<float>(c0, c8));
                    const float high0 = plane_z_high_filter<float>(
                        c5, add_rn<float>(c4, c6), add_rn<float>(c3, c7),
                        add_rn<float>(c2, c8));
                    const float low1 = plane_z_low_filter<float>(c6,
                                                                 add_rn<float>(c5, c7),
                                                                 add_rn<float>(c4, c8),
                                                                 add_rn<float>(c3, c9),
                                                                 add_rn<float>(c2, c10));
                    dwt_z_active_store<T, Quant>(
                        low0, dst, xy + static_cast<size_t>(pair) * leap_z, x, y, pair, low_x,
                        low_y, true, data_dims);
                    dwt_z_active_store<T, Quant>(
                        high0, dst, xy + static_cast<size_t>(even_z + pair) * leap_z, x, y,
                        even_z + pair, low_x, low_y, false, data_dims);
                    dwt_z_active_store<T, Quant>(
                        low1, dst, xy + static_cast<size_t>(pair + 1U) * leap_z, x, y, pair + 1U,
                        low_x, low_y, true, data_dims);
                    if (z + 3 < static_cast<int>(lz)) {
                        const float high1 = plane_z_high_filter<float>(
                            c7, add_rn<float>(c6, c8), add_rn<float>(c5, c9),
                            add_rn<float>(c4, c10));
                        dwt_z_active_store<T, Quant>(
                            high1, dst, xy + static_cast<size_t>(even_z + pair + 1U) * leap_z, x,
                            y, even_z + pair + 1U, low_x, low_y, false, data_dims);
                    }
                } else {
                    dwt_z_active_transform_pair<T, Quant>(
                        s_data, dst, ncol, data_dims, lx, lz, column0, pair, local_column, leap_z,
                        even_z, low_x, low_y);
                    if (pair + 1U < even_z)
                        dwt_z_active_transform_pair<T, Quant>(
                            s_data, dst, ncol, data_dims, lx, lz, column0, pair + 1U, local_column,
                            leap_z, even_z, low_x, low_y);
                }
            }
        } else {
            const uint32_t work = even_z * ncol;
            for (uint32_t i = threadIdx.x; i < work; i += TPB) {
                const uint32_t pair = i / ncol;
                const uint32_t local_column = i - pair * ncol;
                dwt_z_active_transform_pair<T, Quant>(
                    s_data, dst, ncol, data_dims, lx, lz, column0, pair, local_column, leap_z,
                    even_z, low_x, low_y);
            }
        }
        __syncthreads();
    }
    }
}

template <typename T, uint8_t Quant>
inline cudaError_t dwt3d_z_launch(const T* src, T* dst, dim3 data_dims, dim3 chunk_dims,
                                  uint32_t counter_index, cudaStream_t stream,
                                  uint32_t levels_to_run) {
    const size_t shared_bytes = DWTConfig<T>::z_smem_bytes;
    const uint32_t shared_capacity = static_cast<uint32_t>(shared_bytes / sizeof(T));
    if (shared_capacity < chunk_dims.z || DWTConfig<T>::z_blocks == 0)
        return cudaErrorInvalidConfiguration;

    uint32_t columns_per_chunk = min(shared_capacity / chunk_dims.z, chunk_dims.x * chunk_dims.y);
    if constexpr (std::is_same_v<T, float>) {
        if (columns_per_chunk >= 4U)
            columns_per_chunk &= ~3U;
    } else if constexpr (std::is_same_v<T, double>) {
        if (columns_per_chunk >= 2U)
            columns_per_chunk &= ~1U;
    }
    columns_per_chunk = max(columns_per_chunk, 1U);
    const uint32_t chunks = (chunk_dims.x * chunk_dims.y + columns_per_chunk - 1U) / columns_per_chunk;
    const int blocks = min(DWTConfig<T>::z_blocks, max(1, static_cast<int>(chunks)));
    const bool use_direct_chunk = levels_to_run == 1U &&
                                  chunks < static_cast<uint32_t>(WALTZ::gpu_config().sm_count());
    const bool full_columns = chunk_dims.x == data_dims.x && chunk_dims.y == data_dims.y;
    if constexpr (Quant != 2) {
        if (full_columns) {
            if constexpr (std::is_same_v<T, float> && Quant != 0) {
                dwt_3d_z_s<T, Quant, true><<<blocks, TPB, shared_bytes, stream>>>(
                        src, dst, data_dims, chunk_dims, counter_index, levels_to_run,
                        shared_capacity);
            } else if constexpr (std::is_same_v<T, double> && Quant != 0) {
                constexpr uint32_t fixed_capacity = 32768U / sizeof(T);
                if (chunk_dims.z > fixed_capacity)
                    return cudaErrorInvalidConfiguration;
                dwt_3d_z_s<T, Quant, true><<<512, TPB, 0, stream>>>(src, dst,
                    data_dims, chunk_dims, counter_index, levels_to_run, fixed_capacity);
            } else {
                dwt_3d_z_s<T, Quant, true><<<blocks, TPB, shared_bytes, stream>>>(src, dst,
                    data_dims, chunk_dims, counter_index, levels_to_run, shared_capacity);
            }
            return cudaGetLastError();
        }
    }
    if (use_direct_chunk)
        dwt_3d_z<T, Quant><<<512, TPB, 0, stream>>>(src, dst, data_dims, chunk_dims);
    else
        dwt_3d_z_s<T, Quant, false>
            <<<blocks, TPB, shared_bytes, stream>>>(
                src, dst, data_dims, chunk_dims, counter_index, levels_to_run, shared_capacity);
    return cudaGetLastError();
}

template <typename T>
inline cudaError_t dwt_z_active_staged_levels_launch(
    const T* src, T* dst, dim3 dims, int levels_z, cudaStream_t stream) {
    if (levels_z < 0 || levels_z > CDF97_MAX_LEVELS)
        return cudaErrorInvalidValue;
    if (levels_z == 0)
        return cudaSuccess;
    return dwt3d_z_launch<T>(
        src, dst, dims, dims, 0U, stream, static_cast<uint32_t>(levels_z));
}

// Exact Z-L0 + finalized-detail quantization producer.  It uses the
// same column-staged convolution routine and therefore produces the same
// floating-point coefficients as one full-column dwt3d_z level.
// The only difference is ownership of the output: the LLL cube remains
// in `dst`, while the other seven octants go straight to the canonical
// quantized streams and the PWE dequantized coefficient volume.
template <bool RegularBlocks>
__global__ __launch_bounds__(TPB,
                             3) void dwt_plane_z_l0_quant_float(const float* src,
                                                               float* dst,
                                                               dim3 dims) {
    __shared__ alignas(128) float s_data[CS];
    __shared__ int s_chunkID;
    __shared__ DwtFuseQuantParams params;
    if (threadIdx.x == 0)
        params = float_quant_params;
    __syncthreads();
    const dim3 leaps(1U, dims.x, dims.x * dims.y);
    plane_z_columns_all<float, false, RegularBlocks>(
        src, dst, dims, leaps, 1U, s_data, CS, &z_count[0], &s_chunkID, &params);
}

template <typename T> inline cudaError_t idwt3d_plane_prealloc(dim3 dims);

template <typename T>
inline void dwt3d_split_prealloc(dim3 dims, bool use_dyadic) {
    if (!use_dyadic)
        idwt3d_plane_prealloc<T>(dims);

    const cudaDeviceProp& prop = WALTZ::gpu_config().properties;
    const int multiprocessor_count = prop.multiProcessorCount;
    const auto initialize_grid_blocks = [multiprocessor_count](auto kernel, int& grid_blocks) {
        if (grid_blocks != 0)
            return;
        int blocks_per_sm = 0;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_per_sm, kernel, TPB, 0);
        grid_blocks = max(blocks_per_sm, 1) * multiprocessor_count;
    };

    initialize_grid_blocks(dwt_x_axis<T>, DWTConfig<T>::x_blocks);
    initialize_grid_blocks(dwt_y_axis<T>, DWTConfig<T>::y_blocks);

    if (DWTConfig<T>::z_configured_nz == dims.z && DWTConfig<T>::z_blocks != 0)
        return;

    DWTConfig<T>::z_blocks = 0;
    DWTConfig<T>::z_smem_bytes = 0;
    DWTConfig<T>::z_configured_nz = 0;
    constexpr uint32_t column_step = 16 / sizeof(T);
    const size_t minimum_bytes = static_cast<size_t>(dims.z) * column_step * sizeof(T);
    const size_t baseline_bytes = max(static_cast<size_t>(32768U), minimum_bytes);
    if (baseline_bytes > static_cast<size_t>(prop.sharedMemPerBlockOptin))
        return;
    // Shared-memory Z variants: Quant, FullColumns.
    if (baseline_bytes >= static_cast<size_t>(prop.sharedMemPerBlock))
        cudaFuncSetAttribute(dwt_3d_z_s<T, 0, true>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                             static_cast<int>(baseline_bytes));
    int target_blocks_per_sm = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&target_blocks_per_sm, dwt_3d_z_s<T, 0, true>,
        TPB, baseline_bytes);
    target_blocks_per_sm = max(1, target_blocks_per_sm);

    const size_t shared_budget = min(static_cast<size_t>(prop.sharedMemPerBlockOptin), 
        static_cast<size_t>(prop.sharedMemPerMultiprocessor) / static_cast<size_t>(target_blocks_per_sm));
    uint32_t try_columns = static_cast<uint32_t>(shared_budget / (dims.z * sizeof(T)));
    if constexpr (std::is_same_v<T, float>) 
        try_columns &= ~3U;
    else if constexpr (std::is_same_v<T, double>) 
        try_columns &= ~1U;

    // Keep the old downward search available for comparison.  The active path below
    // uses the first (largest) aligned try_columns value directly.
#if 0
    for (; try_columns >= column_step; try_columns -= column_step) {
        const uint32_t try_capacity = dims.z * try_columns;
        const size_t try_bytes = static_cast<size_t>(try_capacity) * sizeof(T);

        const cudaError_t base_attribute = cudaFuncSetAttribute(dwt_3d_z_s<T, 0, false>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(try_bytes));

        const cudaError_t quant_attribute = cudaFuncSetAttribute(dwt_3d_z_s<T, 1, false>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(try_bytes));

        const cudaError_t final_attribute = cudaFuncSetAttribute(dwt_3d_z_s<T, 2, false>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(try_bytes));

        const cudaError_t full_attribute = cudaFuncSetAttribute(dwt_3d_z_s<T, 0, true>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(try_bytes));

        cudaError_t full_irregular_quant_attribute = cudaSuccess;
        if constexpr (std::is_same_v<T, float>)
            full_irregular_quant_attribute = cudaFuncSetAttribute(dwt_3d_z_s<T, 1, true>,
                cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(try_bytes));

        if (base_attribute != cudaSuccess || quant_attribute != cudaSuccess ||
            final_attribute != cudaSuccess || full_attribute != cudaSuccess ||
            full_irregular_quant_attribute != cudaSuccess)
            continue;

        int blocks_per_sm = 0;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_per_sm, dwt_3d_z_s<T, 0, true>, TPB, try_bytes);

        if (blocks_per_sm >= target_blocks_per_sm) {
            DWTConfig<T>::z_blocks = max(1, blocks_per_sm) * prop.multiProcessorCount;
            DWTConfig<T>::z_smem_bytes = try_bytes;
            DWTConfig<T>::z_configured_nz = dims.z;
            return;
        }
    }
#endif

    const uint32_t try_capacity = dims.z * try_columns;
    const size_t try_bytes = static_cast<size_t>(try_capacity) * sizeof(T);

    if (try_bytes >= static_cast<size_t>(prop.sharedMemPerBlock)) {
        cudaFuncSetAttribute(dwt_3d_z_s<T, 0, false>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(try_bytes));
        cudaFuncSetAttribute(dwt_3d_z_s<T, 1, false>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(try_bytes));
        cudaFuncSetAttribute(dwt_3d_z_s<T, 2, false>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(try_bytes));
        cudaFuncSetAttribute(dwt_3d_z_s<T, 0, true>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(try_bytes));
        if constexpr (std::is_same_v<T, float>)
            cudaFuncSetAttribute(dwt_3d_z_s<T, 1, true>,
                cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(try_bytes));
    }

    int blocks_per_sm = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_per_sm,
        dwt_3d_z_s<T, 0, true>, TPB, try_bytes);
    DWTConfig<T>::z_blocks = max(1, blocks_per_sm) * prop.multiProcessorCount;
    DWTConfig<T>::z_smem_bytes = try_bytes;
    DWTConfig<T>::z_configured_nz = dims.z;
}

// ---------------------------------------------------------------------------
} // namespace CDF97

// //Predict
// for (int i = threadIdx.x; i < half_len; i += TPB) {
//     int id = i * stidex2 + 1;
//     int global_id = start + id;
//     if (global_id >= datasize)
//         break;
//     int gidx = global_id % data_dims.x;
//     int chunkx = gidx % chunk_dims.x;
//     // int global_id = start + id;
//     // dim3 gid(global_id % data_dims.x, (global_id / data_dims.x) % data_dims.y, global_id /
//     (data_dims.x * data_dims.y)); if (chunkx != chunk_dims.x - 1) {
//         // s_data[id] += alpha * ( s_data[id-1] +  s_data[id+1]);
//         s_data[id] = __fadd_rn(alpha * (s_data[id - 1] + s_data[id + 1]),
//         s_data[id]);//__fadd_rn(s_data[id], __fmul_rn(alpha, sum));
//     }
//     else {
//         // s_data[id] += 2 * alpha * s_data[id - 1];
//         s_data[id] = __fadd_rn(2 * alpha * s_data[id - 1], s_data[id]); //2 * alpha *
//         s_data[id-1];
//     }
// }
// __syncthreads();
// for (int i = threadIdx.x; i < half_len; i += TPB) {
//     int id = i * stidex2 ;
//     int global_id = start + id;
//     if (global_id >= datasize)
//         break;
//     int gidx = global_id % data_dims.x;
//     int chunkx = gidx % chunk_dims.x;
//     // int global_id = start + id;
//     // dim3 gid(global_id % data_dims.x, (global_id / data_dims.x) % data_dims.y, global_id /
//     (data_dims.x * data_dims.y)); if (chunkx == 0) {
//         // s_data[id] += 2 * beta * s_data[id + 1];
//         s_data[id] = __fadd_rn(2 * beta * (s_data[id + 1]), s_data[id]);
//     }
//     else if (chunkx == chunk_dims.x - 1) {
//         // s_data[id] += 2 * beta * s_data[id - 1];
//         s_data[id] = __fadd_rn(2 * beta * (s_data[id - 1]), s_data[id]); //2 * alpha *
//         s_data[id-1];
//     }
//     else {
//         // s_data[id] += beta * (s_data[id-1] +  s_data[id+1]);
//         s_data[id] = __fadd_rn(beta * (s_data[id - 1] + s_data[id + 1]),
//         s_data[id]);//__fadd_rn(s_data[id], __fmul_rn(alpha, sum));
//     }
// }
// __syncthreads();
// for (int i = threadIdx.x; i < half_len; i += TPB) {
//     int id = i * stidex2 + 1;
//     int global_id = start + id;
//     if (global_id >= datasize)
//         break;
//     int gidx = global_id % data_dims.x;
//     int chunkx = gidx % chunk_dims.x;
//     // int global_id = start + id;
//     // dim3 gid(global_id % data_dims.x, (global_id / data_dims.x) % data_dims.y, global_id /
//     (data_dims.x * data_dims.y));

//     if (chunkx != chunk_dims.x - 1) {
//         // s_data[id] += gamma * (s_data[id-1] +  s_data[id+1]);
//         s_data[id] = __fadd_rn(gamma * (s_data[id - 1] + s_data[id + 1]), s_data[id]);
//     }
//     else {
//         //  s_data[id] += 2 * gamma * s_data[id - 1];
//         s_data[id] = __fadd_rn(2 * gamma * s_data[id - 1], s_data[id]);
//     }
// }
// __syncthreads();
// for (int i = threadIdx.x; i < half_len; i += TPB) {
//     int id = i * stidex2 ;
//     int global_id = start + id;
//     if (global_id >= datasize)
//         break;
//     int gidx = global_id % data_dims.x;
//     int chunkx = gidx % chunk_dims.x;
//     // int global_id = start + id;
//     // dim3 gid(global_id % data_dims.x, (global_id / data_dims.x) % data_dims.y, global_id /
//     (data_dims.x * data_dims.y));

//     if (chunkx == 0) {
//         // s_data[id] += 2 * delta * s_data[id + 1];
//         input[id] = s_data[id] = epsilon * __fadd_rn(2 * delta * (s_data[id + 1]), s_data[id]);
//     }
//     else if (chunkx == chunk_dims.x - 1) {
//         // s_data[id] += 2 * delta * s_data[id - 1];
//         input[id] = s_data[id] = epsilon * __fadd_rn(2 * delta * (s_data[id - 1]), s_data[id]);
//         //2 * alpha * s_data[id-1];
//     }
//     else {
//         // s_data[id] += delta * (s_data[id-1] +  s_data[id+1]);
//         input[id] = s_data[id] = epsilon * __fadd_rn(delta * (s_data[id - 1] + s_data[id + 1]),
//         s_data[id]);//__fadd_rn(s_data[id], __fmul_rn(alpha, sum));
//     }
// }
// __syncthreads();
// for (int i = threadIdx.x; i < half_len; i += TPB) {
//     int id = i * stidex2 + 1;
//     int global_id = start + id;
//     if (global_id >= datasize)
//         break;
//     int gidx = global_id % data_dims.x;
//     int chunkx = gidx % chunk_dims.x;
//     // int global_id = start + id;
//     // dim3 gid(global_id % data_dims.x, (global_id / data_dims.x) % data_dims.y, global_id /
//     (data_dims.x * data_dims.y)); input[id] = s_data[id] *= -inv_epsilon;
// }
