#pragma once

#include "lossless/reorder.hpp"
#include "quantizer/quantize.hpp"
#include "utils/Config.hpp"

#include <algorithm>
#include <cstdlib>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stddef.h>
#include <stdint.h>
#include <type_traits>
#include <utility>

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

#define DWT_TPB 512
#define TPB DWT_TPB

#define CS 8192
#define CDF97_MAX_LEVELS 5

// Min resident blocks/SM cap inverse-kernel register use so enough warps remain
// to hide memory latency. The targets remain split by precision because double
// kernels need more live registers.
// Only sm_80
// (A100, 164KB smem) and sm_90 (H100, 228KB) can afford 4x the ~34KB tile; the
// lower-shared parts (sm_70/86/89) are shared-limited anyway, so we don't over-
// constrain their register budget.  Selected per-T in launch_bounds.
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 800 || __CUDA_ARCH__ >= 900)
#define CDF97_MIN_BLOCKS_F 4
#define CDF97_MIN_BLOCKS_D 3
#else
#define CDF97_MIN_BLOCKS_F 2
#define CDF97_MIN_BLOCKS_D 2
#endif
#define CDF97_MIN_BLOCKS(T) (std::is_same_v<T, float> ? CDF97_MIN_BLOCKS_F : CDF97_MIN_BLOCKS_D)

namespace CDF97 {

using WALTZ::DWTConfig;
using WALTZ::QuantMode;

static __device__ unsigned int x_count[CDF97_MAX_LEVELS];
static __device__ unsigned int y_count[CDF97_MAX_LEVELS];
static __device__ unsigned int z_count[CDF97_MAX_LEVELS];

// Device-side state for fused double DWT + quantization.
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

// Reset the multi-kernel work-distribution counters used by atomic chunk dispatch.
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

template <typename T>
__device__ __forceinline__ void dwt_y_rows(T* tmp,
                                           const T* input,
                                           uint32_t row0,
                                           uint32_t rows,
                                           uint32_t active_x,
                                           uint32_t active_y,
                                           dim3 data_leaps) {
    const uint32_t even_y = active_y - (active_y >> 1);
    const int stride_y = data_leaps.y;


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
constexpr int BX_32 = 32;
constexpr int BY_16 = 16;
constexpr int BX_64 = 64;
constexpr int BY_32 = 32;
constexpr int BX_128 = 128;

// Kept in one device-side record so the optional DWT->quant path does not keep
// ten independent pointers/scalars live in every halo thread.  The default
// QuantMode::None kernels never dereference this record.
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
    const bool regular_blocks =
        q_bz != 0U && (q_bz & (q_bz - 1U)) == 0U && (dims.x & 15U) == 0U &&
        (dims.y & 15U) == 0U && dims.z % q_bz == 0U;
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
                               regular_blocks,
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
                             regular_blocks,
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

template <bool RegularBlocks, typename T>
__device__ __forceinline__ void quantization(
    T value, size_t s, uint32_t x, uint32_t y, uint32_t z, dim3 data_dims) {
    if constexpr (std::is_same_v<T, float>) {
        if constexpr (RegularBlocks)
            dwt_quant_emit_regular_float(value, s, x, y, z, &float_quant_params);
        else
            dwt_quant_emit_float(value, s, x, y, z, data_dims, &float_quant_params);
    } else if constexpr (std::is_same_v<T, double>) {
        if constexpr (RegularBlocks)
            dwt_quant_emit_regular_double(value, s, x, y, z, &double_quant_params);
        else
            dwt_quant_emit_double(value, s, x, y, z, data_dims, &double_quant_params);
    }
}

template <int BX, int BY, QuantMode Quant = QuantMode::None, bool QuantBlocksRegular = false>
__global__ void dwt3d_xy_float(
    const float* __restrict__ src, float* __restrict__ dst, dim3 data_dims, dim3 chunk_dims) {
    constexpr int R = DWT_HALO_R;
#if defined(__CUDA_ARCH__)
    // Keep the measured Hopper float improvement; other targets retain one pair.
    constexpr bool pair2_x = (__CUDA_ARCH__ == 900) && ((BX & 3) == 0);
    constexpr bool pair2_y =
        (__CUDA_ARCH__ == 900 && Quant == QuantMode::None) && ((BY & 3) == 0);
#else
    constexpr bool pair2_x = false;
    constexpr bool pair2_y = false;
#endif
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

    if constexpr (pair2_x) {
        // BX and x0 are even.  One owner evaluates two adjacent even/odd
        // pairs.  Their windows have an eleven-value union (c-4..c+6), so
        // the seven values shared by both pairs are loaded only once.
        constexpr int XPAIR_GROUPS = (BX + 3) / 4;
        for (int i = threadIdx.x; i < XPAIR_GROUPS * EY; i += blockDim.x) {
            const int px = i % XPAIR_GROUPS;
            const int ey = i / XPAIR_GROUPS;
            const int lx = px << 2;
            const int gx = x0 + lx;
            const int xo = ey * BX + lx;
            if (gx >= nx)
                continue;
            const int c = ey * EX + lx + R;

            // c-4..c+6 are exactly the union of the two pair windows.  Keep
            // the filter argument order identical to the one-pair path.
            const float r0 = raw[c - 4];
            const float r1 = raw[c - 3];
            const float r2 = raw[c - 2];
            const float r3 = raw[c - 1];
            const float r4 = raw[c];
            const float r5 = raw[c + 1];
            const float r6 = raw[c + 2];
            const float r7 = raw[c + 3];
            const float r8 = raw[c + 4];
            const float r9 = raw[c + 5];
            const float r10 = raw[c + 6];

            xbuf[xo] = low_filter<float>(r4, r3 + r5, r2 + r6, r1 + r7, r0 + r8);
            if (gx + 1 < nx) {
                xbuf[xo + 1] = high_filter<float>(r5, r4 + r6, r3 + r7, r2 + r8);
            }
            if (gx + 2 < nx) {
                xbuf[xo + 2] = low_filter<float>(r6, r5 + r7, r4 + r8, r3 + r9, r2 + r10);
                if (gx + 3 < nx) {
                    xbuf[xo + 3] = high_filter<float>(r7, r6 + r8, r5 + r9, r4 + r10);
                }
            }
        }
    } else {
        // Generic one-pair owner for architectures/quantization modes not
        // selected by the controlled pair-grouping experiment.
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
    }
    __syncthreads();

    if constexpr (pair2_y) {
        // One owner evaluates two adjacent even/odd row pairs.  Their 9/7
        // windows have an eleven-row union (c-4*BX..c+6*BX), so the seven
        // rows shared by the pair windows are loaded into locals once.
        constexpr int YPAIR_GROUPS = (BY + 3) / 4;
        for (int i = threadIdx.x; i < BX * YPAIR_GROUPS; i += blockDim.x) {
            const int lx = i % BX;
            const int py = i / BX;
            const int ly = py << 2;
            const int gx = x0 + lx;
            const int gy = y0 + ly;
            if (gx >= nx || gy >= ny)
                continue;

            const int c = (ly + DWT_HALO_R) * BX + lx;
            const float r0 = xbuf[c - 4 * BX];
            const float r1 = xbuf[c - 3 * BX];
            const float r2 = xbuf[c - 2 * BX];
            const float r3 = xbuf[c - BX];
            const float r4 = xbuf[c];
            const float r5 = xbuf[c + BX];
            const float r6 = xbuf[c + 2 * BX];
            const float r7 = xbuf[c + 3 * BX];
            const float r8 = xbuf[c + 4 * BX];
            const float r9 = xbuf[c + 5 * BX];
            const float r10 = xbuf[c + 6 * BX];
            const int output_x = (gx & 1) ? even_x + (gx >> 1) : (gx >> 1);
            const int low_y = gy >> 1;
            const size_t zoff = static_cast<size_t>(z) * leap_z;
            const float low_value =
                low_filter<float>(r4, r3 + r5, r2 + r6, r1 + r7, r0 + r8);
            const size_t low_index = output_x + static_cast<size_t>(low_y) * sx + zoff;
            if constexpr (Quant == QuantMode::None)
                dst[low_index] = low_value;
            else if constexpr (Quant == QuantMode::All)
                quantization<QuantBlocksRegular>(
                    low_value, low_index, output_x, low_y, z, data_dims);
            else if constexpr (Quant == QuantMode::High) {
                if (gx & 1)
                    quantization<QuantBlocksRegular>(
                        low_value, low_index, output_x, low_y, z, data_dims);
                else
                    dst[low_index] = low_value;
            }
            if (gy + 1 < ny) {
                const float high_value =
                    high_filter<float>(r5, r4 + r6, r3 + r7, r2 + r8);
                const uint32_t high_y = even_y + low_y;
                const size_t high_index = output_x + static_cast<size_t>(high_y) * sx + zoff;
                if constexpr (Quant != QuantMode::None)
                    quantization<QuantBlocksRegular>(
                        high_value, high_index, output_x, high_y, z, data_dims);
                else
                    dst[high_index] = high_value;
            }
            if (gy + 2 < ny) {
                const int gy2 = gy + 2;
                const int low_y2 = gy2 >> 1;
                const float low_value2 =
                    low_filter<float>(r6, r5 + r7, r4 + r8, r3 + r9, r2 + r10);
                const size_t low_index2 =
                    output_x + static_cast<size_t>(low_y2) * sx + zoff;
                if constexpr (Quant == QuantMode::None)
                    dst[low_index2] = low_value2;
                else if constexpr (Quant == QuantMode::All)
                    quantization<QuantBlocksRegular>(
                        low_value2, low_index2, output_x, low_y2, z, data_dims);
                else if constexpr (Quant == QuantMode::High) {
                    if (gx & 1)
                        quantization<QuantBlocksRegular>(
                            low_value2, low_index2, output_x, low_y2, z, data_dims);
                    else
                        dst[low_index2] = low_value2;
                }
                if (gy + 3 < ny) {
                    const float high_value2 =
                        high_filter<float>(r7, r6 + r8, r5 + r9, r4 + r10);
                    const uint32_t high_y2 = even_y + low_y2;
                    const size_t high_index2 =
                        output_x + static_cast<size_t>(high_y2) * sx + zoff;
                    if constexpr (Quant != QuantMode::None)
                        quantization<QuantBlocksRegular>(
                            high_value2, high_index2, output_x, high_y2, z, data_dims);
                    else
                        dst[high_index2] = high_value2;
                }
            }
        }
    } else {
        // Generic one-pair owner for architectures/quantization modes not
        // selected by the controlled pair-grouping experiment.
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
            if constexpr (Quant == QuantMode::None)
                dst[low_index] = low_value;
            else if constexpr (Quant == QuantMode::All)
                quantization<QuantBlocksRegular>(
                    low_value, low_index, output_x, low_y, z, data_dims);
            else if constexpr (Quant == QuantMode::High) {
                if (gx & 1)
                    quantization<QuantBlocksRegular>(
                        low_value, low_index, output_x, low_y, z, data_dims);
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
                if constexpr (Quant != QuantMode::None)
                    quantization<QuantBlocksRegular>(
                        high_value, high_index, output_x, high_y, z, data_dims);
                else
                    dst[high_index] = high_value;
            }
        }
    }
}

// Double-precision XY kernel.  Load two adjacent doubles at once when the
// source row is 16-byte aligned; physical X edges and odd row strides retain
// the scalar reflected path.
template <int BX, int BY, QuantMode Quant = QuantMode::None, bool QuantBlocksRegular = false>
__global__ void dwt3d_xy_double(
    const double* __restrict__ src, double* __restrict__ dst, dim3 data_dims, dim3 chunk_dims) {
    constexpr int R = DWT_HALO_R;
#if defined(__CUDA_ARCH__)
    constexpr bool pair2_x =
        (__CUDA_ARCH__ == 1200 && Quant == QuantMode::None) && ((BX & 3) == 0);
    constexpr bool pair2_y =
        (__CUDA_ARCH__ == 1200 && Quant == QuantMode::None) && ((BY & 3) == 0);
#else
    constexpr bool pair2_x = false;
    constexpr bool pair2_y = false;
#endif
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

    if constexpr (pair2_x) {
        // BX and x0 are even.  One owner evaluates two adjacent even/odd
        // pairs.  Their windows have an eleven-value union (c-4..c+6), so
        // the seven values shared by both pairs are loaded only once.
        constexpr int XPAIR_GROUPS = (BX + 3) / 4;
        for (int i = threadIdx.x; i < XPAIR_GROUPS * EY; i += blockDim.x) {
            const int px = i % XPAIR_GROUPS;
            const int ey = i / XPAIR_GROUPS;
            const int lx = px << 2;
            const int gx = x0 + lx;
            const int xo = ey * BX + lx;
            if (gx >= nx)
                continue;
            const int c = ey * EX + lx + R;

            // c-4..c+6 are exactly the union of the two pair windows.  Keep
            // the filter argument order identical to the one-pair path.
            const double r0 = raw[c - 4];
            const double r1 = raw[c - 3];
            const double r2 = raw[c - 2];
            const double r3 = raw[c - 1];
            const double r4 = raw[c];
            const double r5 = raw[c + 1];
            const double r6 = raw[c + 2];
            const double r7 = raw[c + 3];
            const double r8 = raw[c + 4];
            const double r9 = raw[c + 5];
            const double r10 = raw[c + 6];

            xbuf[xo] = low_filter<double>(r4, r3 + r5, r2 + r6, r1 + r7, r0 + r8);
            if (gx + 1 < nx) {
                xbuf[xo + 1] = high_filter<double>(r5, r4 + r6, r3 + r7, r2 + r8);
            }
            if (gx + 2 < nx) {
                xbuf[xo + 2] = low_filter<double>(r6, r5 + r7, r4 + r8, r3 + r9, r2 + r10);
                if (gx + 3 < nx) {
                    xbuf[xo + 3] = high_filter<double>(r7, r6 + r8, r5 + r9, r4 + r10);
                }
            }
        }
    } else {
        // Generic one-pair owner for architectures/quantization modes not
        // selected by the controlled pair-grouping experiment.
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
    }
    __syncthreads();

    if constexpr (pair2_y) {
        // One owner evaluates two adjacent even/odd row pairs.  Their 9/7
        // windows have an eleven-row union (c-4*BX..c+6*BX), so the seven
        // rows shared by the pair windows are loaded into locals once.
        constexpr int YPAIR_GROUPS = (BY + 3) / 4;
        for (int i = threadIdx.x; i < BX * YPAIR_GROUPS; i += blockDim.x) {
            const int lx = i % BX;
            const int py = i / BX;
            const int ly = py << 2;
            const int gx = x0 + lx;
            const int gy = y0 + ly;
            if (gx >= nx || gy >= ny)
                continue;

            const int c = (ly + R) * BX + lx;
            const double r0 = xbuf[c - 4 * BX];
            const double r1 = xbuf[c - 3 * BX];
            const double r2 = xbuf[c - 2 * BX];
            const double r3 = xbuf[c - BX];
            const double r4 = xbuf[c];
            const double r5 = xbuf[c + BX];
            const double r6 = xbuf[c + 2 * BX];
            const double r7 = xbuf[c + 3 * BX];
            const double r8 = xbuf[c + 4 * BX];
            const double r9 = xbuf[c + 5 * BX];
            const double r10 = xbuf[c + 6 * BX];
            const int output_x = (gx & 1) ? even_x + (gx >> 1) : (gx >> 1);
            const int low_y = gy >> 1;
            const size_t zoff = static_cast<size_t>(z) * leap_z;
            const double low_value =
                low_filter<double>(r4, r3 + r5, r2 + r6, r1 + r7, r0 + r8);
            const size_t low_index = output_x + static_cast<size_t>(low_y) * sx + zoff;
            if constexpr (Quant == QuantMode::None)
                dst[low_index] = low_value;
            else if constexpr (Quant == QuantMode::All)
                quantization<QuantBlocksRegular>(
                    low_value, low_index, output_x, low_y, z, data_dims);
            else if constexpr (Quant == QuantMode::High) {
                if (gx & 1)
                    quantization<QuantBlocksRegular>(
                        low_value, low_index, output_x, low_y, z, data_dims);
                else
                    dst[low_index] = low_value;
            }
            if (gy + 1 < ny) {
                const double high_value =
                    high_filter<double>(r5, r4 + r6, r3 + r7, r2 + r8);
                const uint32_t high_y = even_y + low_y;
                const size_t high_index = output_x + static_cast<size_t>(high_y) * sx + zoff;
                if constexpr (Quant != QuantMode::None)
                    quantization<QuantBlocksRegular>(
                        high_value, high_index, output_x, high_y, z, data_dims);
                else
                    dst[high_index] = high_value;
            }
            if (gy + 2 < ny) {
                const int gy2 = gy + 2;
                const int low_y2 = gy2 >> 1;
                const double low_value2 =
                    low_filter<double>(r6, r5 + r7, r4 + r8, r3 + r9, r2 + r10);
                const size_t low_index2 =
                    output_x + static_cast<size_t>(low_y2) * sx + zoff;
                if constexpr (Quant == QuantMode::None)
                    dst[low_index2] = low_value2;
                else if constexpr (Quant == QuantMode::All)
                    quantization<QuantBlocksRegular>(
                        low_value2, low_index2, output_x, low_y2, z, data_dims);
                else if constexpr (Quant == QuantMode::High) {
                    if (gx & 1)
                        quantization<QuantBlocksRegular>(
                            low_value2, low_index2, output_x, low_y2, z, data_dims);
                    else
                        dst[low_index2] = low_value2;
                }
                if (gy + 3 < ny) {
                    const double high_value2 =
                        high_filter<double>(r7, r6 + r8, r5 + r9, r4 + r10);
                    const uint32_t high_y2 = even_y + low_y2;
                    const size_t high_index2 =
                        output_x + static_cast<size_t>(high_y2) * sx + zoff;
                    if constexpr (Quant != QuantMode::None)
                        quantization<QuantBlocksRegular>(
                            high_value2, high_index2, output_x, high_y2, z, data_dims);
                    else
                        dst[high_index2] = high_value2;
                }
            }
        }
    } else {
        // Generic one-pair owner for architectures/quantization modes not
        // selected by the controlled pair-grouping experiment.
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
            if constexpr (Quant == QuantMode::None)
                dst[low_index] = low_value;
            else if constexpr (Quant == QuantMode::All)
                quantization<QuantBlocksRegular>(
                    low_value, low_index, output_x, low_y, z, data_dims);
            else if constexpr (Quant == QuantMode::High) {
                if (gx & 1)
                    quantization<QuantBlocksRegular>(
                        low_value, low_index, output_x, low_y, z, data_dims);
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
                if constexpr (Quant != QuantMode::None)
                    quantization<QuantBlocksRegular>(
                        high_value, high_index, output_x, high_y, z, data_dims);
                else
                    dst[high_index] = high_value;
            }
        }
    }
}

template <typename T>
__global__ void dwt_3d_z_single_global(
    const T* src, T* dst, dim3 data_dims, dim3 chunk_dims, QuantMode quant_mode);

// The plane path's full-Z pass always uses this fixed shared-memory footprint.
// The capacity is expressed in elements inside the kernel so float and double
// use the same 45 KiB allocation and the same host-side fit check.
constexpr uint32_t DWT_Z_STATIC_BYTES = 45U * 1024U;

template <typename T>
__global__ __launch_bounds__(TPB, std::is_same_v<T, float> ? 3 : 2)
void dwt_3d_z_all_static(const T* src, T* dst, dim3 data_dims,
                         uint8_t levels_to_run);

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
template <typename T, QuantMode Quant>
inline cudaError_t dwt3d_plane_xy_quant_launch(
    const T* src, T* dst, dim3 data_dims, dim3 chunk_dims, bool quant_blocks_regular,
    cudaStream_t stream) {
    if constexpr (std::is_same_v<T, float>) {
        const dim3 grid((chunk_dims.x + BX_64 - 1U) / BX_64,
                        (chunk_dims.y + BY_32 - 1U) / BY_32,
                        chunk_dims.z);
        if (quant_blocks_regular)
            dwt3d_xy_float<BX_64, BY_32, Quant, true>
                <<<grid, 256, 0, stream>>>(src, dst, data_dims, chunk_dims);
        else
            dwt3d_xy_float<BX_64, BY_32, Quant, false>
                <<<grid, 256, 0, stream>>>(src, dst, data_dims, chunk_dims);
    } else {
        const dim3 grid((chunk_dims.x + BX_64 - 1U) / BX_64,
                        (chunk_dims.y + BY_16 - 1U) / BY_16,
                        chunk_dims.z);
        if (quant_blocks_regular)
            dwt3d_xy_double<BX_64, BY_16, Quant, true>
                <<<grid, 256, 0, stream>>>(src, dst, data_dims, chunk_dims);
        else
            dwt3d_xy_double<BX_64, BY_16, Quant, false>
                <<<grid, 256, 0, stream>>>(src, dst, data_dims, chunk_dims);
    }
    return cudaGetLastError();
}


template <typename T>
inline cudaError_t dwt3d_plane_xy_halo_z(
    T*& data, T*& tmp, dim3 dims, int levels_xy, int levels_z, const T* z_src,
    uint32_t q_bz, cudaStream_t stream) {
    const T* input = z_src != nullptr ? z_src : data;
    if (levels_z > 0) {
        constexpr uint32_t shared_capacity = DWT_Z_STATIC_BYTES / sizeof(T);
        // dwt_z_float/double use two ping-pong buffers for every full-Z column.
        // Reject a shape that cannot fit before launching: the device kernel
        // cannot report an error, and allowing zero columns would divide by 0.
        if (dims.z > shared_capacity / 2U)
            return cudaErrorInvalidConfiguration;
        dwt_3d_z_all_static<T><<<DWTConfig<T>::z_blocks, TPB, 0, stream>>>(
            input, data, dims, static_cast<uint8_t>(levels_z));
        input = data;
    }
    const bool quant_blocks_regular = q_bz != 0U && (q_bz & (q_bz - 1U)) == 0U &&
                                      (dims.x & 15U) == 0U &&
                                      (dims.y & 15U) == 0U && dims.z % q_bz == 0U;
    T* current = data;
    T* scratch = tmp;
    if (input == scratch) {
        T* const temp = current;
        current = scratch;
        scratch = temp;
    }
    dim3 chunk_dims = dims;
    for (int lev = 0; lev < levels_xy; ++lev) {
        cudaError_t error;
        if (lev + 1 == levels_xy) {
            error = dwt3d_plane_xy_quant_launch<T, QuantMode::All>(
                input, scratch, dims, chunk_dims, quant_blocks_regular, stream);
        } else {
            error = dwt3d_plane_xy_quant_launch<T, QuantMode::High>(
                input, scratch, dims, chunk_dims, quant_blocks_regular, stream);
        }
        if (error != cudaSuccess)
            return error;
        T* const temp = current;
        current = scratch;
        scratch = temp;
        input = current;
        chunk_dims.x -= chunk_dims.x >> 1U;
        chunk_dims.y -= chunk_dims.y >> 1U;
    }
    data = current;
    tmp = scratch;
    return cudaGetLastError();
}

template <typename T, bool ZTilesStayInRow, bool QuantBlocksRegular>
__global__ void dwt_3d_z_single_static(
    const T* src, T* dst, dim3 data_dims, dim3 chunk_dims,
    uint32_t counter_index, QuantMode quant_mode);

// Dyadic multi-kernel path. Every level fuses X+Y in one shared-halo kernel,
// then runs the dependent Z stage and quantizes the subbands finalized at that
// level. Ordinary kernel-launch ordering supplies the required global barrier;
// no grid-wide synchronization is needed.
template <typename T>
inline void dwt3d_xy_halo_z(T* data, T* tmp, dim3 dims, int levels, uint32_t q_bz, cudaStream_t stream) {
    constexpr uint32_t static_capacity = 32768U / sizeof(T);
    const bool quant_blocks_regular = (q_bz & (q_bz - 1U)) == 0U &&
        (dims.x & 15U) == 0U && (dims.y & 15U) == 0U && dims.z % q_bz == 0U;
    const uint32_t sm_count =
        static_cast<uint32_t>(WALTZ::gpu_config().properties.multiProcessorCount);
    dim3 chunk_dims = dims;
    for (int lev = 0; lev < levels; ++lev) {
        if constexpr (std::is_same_v<T, float>) {
            const dim3 grid((chunk_dims.x + BX_64 - 1U) / BX_64,
                            (chunk_dims.y + BY_32 - 1U) / BY_32,
                            chunk_dims.z);
            dwt3d_xy_float<BX_64, BY_32, QuantMode::None>
                <<<grid, 256, 0, stream>>>(data, tmp, dims, chunk_dims);
        } else if constexpr (std::is_same_v<T, double>) {
            const dim3 grid((chunk_dims.x + BX_64 - 1U) / BX_64,
                            (chunk_dims.y + BY_16 - 1U) / BY_16,
                            chunk_dims.z);
            dwt3d_xy_double<BX_64, BY_16, QuantMode::None>
                <<<grid, 256, 0, stream>>>(data, tmp, dims, chunk_dims);
        }

        const QuantMode quant_mode = lev + 1 == levels ? QuantMode::All : QuantMode::High;
        const uint32_t total_columns = chunk_dims.x * chunk_dims.y;
        uint32_t columns_per_chunk = min(static_capacity / (chunk_dims.z * (std::is_same_v<T, float> ? 2U : 1U)),
                total_columns);
        const bool has_column_tile = columns_per_chunk != 0U;
        if constexpr (std::is_same_v<T, float>) {
            if (columns_per_chunk >= 4U)
                columns_per_chunk &= ~3U;
        } else if constexpr (std::is_same_v<T, double>) {
            if (columns_per_chunk >= 2U)
                columns_per_chunk &= ~1U;
        }
        columns_per_chunk = max(columns_per_chunk, 1U);
        const uint32_t chunks =
            (total_columns + columns_per_chunk - 1U) / columns_per_chunk;
        // Once fewer than two waves of shared-memory chunks remain, the direct
        // global kernel is faster than staging a mostly empty persistent grid.
        if (chunks <= 2U * sm_count) {
            dwt_3d_z_single_global<T><<<512, TPB, 0, stream>>>(tmp, data, dims, chunk_dims, quant_mode);
            chunk_dims.x -= chunk_dims.x >> 1U;
            chunk_dims.y -= chunk_dims.y >> 1U;
            chunk_dims.z -= chunk_dims.z >> 1U;
            continue;
        }
        const int z_grid = min(384, max(1, static_cast<int>(chunks)));
        const bool z_tiles_stay_in_row =
            has_column_tile && chunk_dims.x % columns_per_chunk == 0U;
        const unsigned int path =
            (static_cast<unsigned int>(z_tiles_stay_in_row) << 1U) |
            static_cast<unsigned int>(quant_blocks_regular);
        switch (path) {
            case 3U:
                dwt_3d_z_single_static<T, true, true><<<z_grid, TPB, 0, stream>>>(
                    tmp, data, dims, chunk_dims, static_cast<uint32_t>(lev), quant_mode);
                break;
            case 2U:
                dwt_3d_z_single_static<T, true, false><<<z_grid, TPB, 0, stream>>>(
                    tmp, data, dims, chunk_dims, static_cast<uint32_t>(lev), quant_mode);
                break;
            case 1U:
                dwt_3d_z_single_static<T, false, true><<<z_grid, TPB, 0, stream>>>(
                    tmp, data, dims, chunk_dims, static_cast<uint32_t>(lev), quant_mode);
                break;
            default:
                dwt_3d_z_single_static<T, false, false><<<z_grid, TPB, 0, stream>>>(
                    tmp, data, dims, chunk_dims, static_cast<uint32_t>(lev), quant_mode);
                break;
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

template <bool Quantize = false, bool ZTilesStayInRow = false,
          bool QuantBlocksKnown = false, bool QuantBlocksRegular = false>
__device__ __forceinline__ void dwt_z_float(const float* input, float* data, dim3 data_dims, dim3 data_leaps,
    uint8_t levels_z, float* s_data, uint32_t num, unsigned int* counter, int* s_chunkID,
    QuantMode quant_mode) {
    const uint32_t full_z = data_dims.z;
    const uint32_t total_columns = data_dims.x * data_dims.y; // == data_leaps.z
    uint32_t cols = num / (full_z * 2U);
    cols = min(cols, total_columns);
    // Align float column tiles to float4.  This makes every full tile and both
    // ping-pong buffers 16-byte aligned; at most three columns of capacity are
    // traded for four times fewer global/shared staging instructions.
    const uint32_t aligned_cols = cols & ~3U;
    if (aligned_cols >= 4U)
        cols = aligned_cols;
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
        if constexpr (ZTilesStayInRow) {
            tile_y = col0 / data_dims.x;
            tile_x0 = col0 - tile_y * data_dims.x;
        }
        float* const s_a = s_data;
        float* const s_b = s_data + buf;
        float* s_src = s_a;
        float* s_dst = s_b;

        // Load tile: s_src[z*ncol + lc] = data[(col0+lc) + z*stride_z].
        // Only the possible tail tile uses the scalar fallback.
        const bool vector_tile = cols >= 4U && ncol == cols &&
                                 (ncol & 3U) == 0U &&
                                 ((col0 | static_cast<uint32_t>(stride_z)) & 3U) == 0U &&
                                 (reinterpret_cast<uintptr_t>(input) & 15U) == 0U &&
                                 (reinterpret_cast<uintptr_t>(data) & 15U) == 0U;
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
        __syncthreads();

        uint32_t active_z = full_z;
        for (int lev = 0; lev < levels_z; ++lev) {
            const uint32_t even_z = active_z - (active_z >> 1U);
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
                    plane_z_forward_pair_float(
                        s_src, s_dst, ncol, active_z, even_z, pair, lc);
                    if (pair + 1U < even_z)
                        plane_z_forward_pair_float(
                            s_src, s_dst, ncol, active_z, even_z, pair + 1U, lc);
                }
            }
            __syncthreads();

            float* swap = s_src;
            s_src = s_dst;
            s_dst = swap;
            active_z = even_z;
        }

        // Write each fully transformed column once. QuantMode::High preserves
        // the low-frequency cube; QuantMode::All also quantizes that final cube.
        if constexpr (Quantize) {
            const uint32_t low_x = data_dims.x - (data_dims.x >> 1U);
            const uint32_t low_y = data_dims.y - (data_dims.y >> 1U);
            const uint32_t low_z = data_dims.z - (data_dims.z >> 1U);
            for (uint32_t i = threadIdx.x; i < tile_elems; i += TPB) {
                const uint32_t z = i / ncol;
                const uint32_t lc = i - z * ncol;
                const uint32_t col = col0 + lc;
                uint32_t x, y;
                if constexpr (ZTilesStayInRow) {
                    x = tile_x0 + lc;
                    y = tile_y;
                } else {
                    x = col % data_dims.x;
                    y = col / data_dims.x;
                }
                const float* const owner =
                    plane_z_final_owner_b(z, full_z, levels_z) ? s_b : s_a;
                const float value = owner[i];
                const size_t s = static_cast<size_t>(col) + static_cast<size_t>(z) * stride_z;
                if (quant_mode == QuantMode::High &&
                    x < low_x && y < low_y && z < low_z) {
                    data[s] = value;
                    continue;
                }
                if constexpr (QuantBlocksKnown)
                    quantization<QuantBlocksRegular>(value, s, x, y, z, data_dims);
                else
                    quantization(value, s, x, y, z, data_dims);
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
                const float* const owner =
                    plane_z_final_owner_b(z, full_z, levels_z) ? s_b : s_a;
                data[(col0 + lc) + z * stride_z] = owner[i];
            }
        }
        __syncthreads();
    } while (true);
}

template <bool Quantize = false, bool ZTilesStayInRow = false,
          bool QuantBlocksKnown = false, bool QuantBlocksRegular = false>
__device__ __forceinline__ void dwt_z_double(const double* input,
                                             double* data,
                                             dim3 data_dims,
                                             dim3 data_leaps,
                                             uint8_t levels_z,
                                             double* s_data,
                                             uint32_t num,
                                             unsigned int* counter,
                                             int* s_chunkID,
                                             QuantMode quant_mode) {
    const uint32_t full_z = data_dims.z;
    const uint32_t total_columns = data_dims.x * data_dims.y;
    uint32_t cols = min(num / (full_z * 2U), total_columns);
    const uint32_t aligned_cols = cols & ~1U;
    if (aligned_cols >= 2U)
        cols = aligned_cols;
    const uint32_t max_count = (total_columns - 1U) / cols + 1U;
    const uint32_t buf = cols * full_z;
    const int stride_z = data_leaps.z;

    do {
        if (threadIdx.x == 0)
            *s_chunkID = static_cast<int>(atomicAdd(counter, 1U));
        __syncthreads();
        const int queued = *s_chunkID;
        __syncthreads();
        if (queued >= static_cast<int>(max_count))
            break;

        const uint32_t col0 = static_cast<uint32_t>(queued) * cols;
        const uint32_t ncol = min(cols, total_columns - col0);
        const uint32_t tile_elems = ncol * full_z;
        uint32_t tile_x0 = 0U, tile_y = 0U;
        if constexpr (ZTilesStayInRow) {
            tile_y = col0 / data_dims.x;
            tile_x0 = col0 - tile_y * data_dims.x;
        }

        double* const s_a = s_data;
        double* const s_b = s_data + buf;
        double* s_src = s_a;
        double* s_dst = s_b;
        const bool vector_tile = cols >= 2U && ncol == cols &&
                                 (ncol & 1U) == 0U &&
                                 ((col0 | static_cast<uint32_t>(stride_z)) & 1U) == 0U &&
                                 (reinterpret_cast<uintptr_t>(input) & 15U) == 0U &&
                                 (reinterpret_cast<uintptr_t>(data) & 15U) == 0U;
        if (vector_tile) {
            const uint32_t vector_columns = ncol >> 1U;
            const uint32_t vector_elements = full_z * vector_columns;
            double2* const shared2 = reinterpret_cast<double2*>(s_src);
            for (uint32_t i = threadIdx.x; i < vector_elements; i += TPB) {
                const uint32_t z = i / vector_columns;
                const uint32_t vc = i - z * vector_columns;
                shared2[i] = *reinterpret_cast<const double2*>(
                    input + col0 + (vc << 1U) + z * stride_z);
            }
        } else {
            for (uint32_t i = threadIdx.x; i < tile_elems; i += TPB) {
                const uint32_t z = i / ncol;
                const uint32_t local_column = i - z * ncol;
                s_src[i] = input[col0 + local_column + z * stride_z];
            }
        }
        __syncthreads();

        uint32_t active_z = full_z;
        for (int level = 0; level < levels_z; ++level) {
            const uint32_t even_z = active_z - (active_z >> 1U);
            const uint32_t work = even_z * ncol;
            for (uint32_t i = threadIdx.x; i < work; i += TPB) {
                const uint32_t pair = i / ncol;
                const uint32_t local_column = i - pair * ncol;
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

                const double value = s_src[static_cast<uint32_t>(z) * ncol + local_column];
                const double vm1 = s_src[static_cast<uint32_t>(zm1) * ncol + local_column];
                const double vp1 = s_src[static_cast<uint32_t>(zp1) * ncol + local_column];
                const double vm2 = s_src[static_cast<uint32_t>(zm2) * ncol + local_column];
                const double vp2 = s_src[static_cast<uint32_t>(zp2) * ncol + local_column];
                const double vm3 = s_src[static_cast<uint32_t>(zm3) * ncol + local_column];
                const double vp3 = s_src[static_cast<uint32_t>(zp3) * ncol + local_column];
                const double vm4 = s_src[static_cast<uint32_t>(zm4) * ncol + local_column];
                const double vp4 = s_src[static_cast<uint32_t>(zp4) * ncol + local_column];
                s_dst[pair * ncol + local_column] =
                    plane_z_low_filter<double>(value,
                                               add_rn<double>(vm1, vp1),
                                               add_rn<double>(vm2, vp2),
                                               add_rn<double>(vm3, vp3),
                                               add_rn<double>(vm4, vp4));
                if (z + 1 < static_cast<int>(active_z))
                    s_dst[(even_z + pair) * ncol + local_column] =
                        plane_z_high_filter<double>(vp1,
                                                    add_rn<double>(value, vp2),
                                                    add_rn<double>(vm1, vp3),
                                                    add_rn<double>(vm2, vp4));
            }
            __syncthreads();

            double* const swap = s_src;
            s_src = s_dst;
            s_dst = swap;
            active_z = even_z;
        }

        if constexpr (Quantize) {
            for (uint32_t i = threadIdx.x; i < tile_elems; i += TPB) {
                const uint32_t z = i / ncol;
                const uint32_t local_column = i - z * ncol;
                const uint32_t column = col0 + local_column;
                const double* const owner =
                    plane_z_final_owner_b(z, full_z, levels_z) ? s_b : s_a;
                const size_t index =
                    static_cast<size_t>(column) + static_cast<size_t>(z) * stride_z;
                uint32_t x, y;
                if constexpr (ZTilesStayInRow) {
                    x = tile_x0 + local_column;
                    y = tile_y;
                } else {
                    x = column % data_dims.x;
                    y = column / data_dims.x;
                }
                if (quant_mode == QuantMode::High) {
                    const uint32_t low_x = data_dims.x - (data_dims.x >> 1U);
                    const uint32_t low_y = data_dims.y - (data_dims.y >> 1U);
                    const uint32_t low_z = data_dims.z - (data_dims.z >> 1U);
                    if (x < low_x && y < low_y && z < low_z) {
                        data[index] = owner[i];
                        continue;
                    }
                }
                if constexpr (QuantBlocksKnown)
                    quantization<QuantBlocksRegular>(
                        owner[i], index, x, y, z, data_dims);
                else
                    quantization(owner[i], index, x, y, z, data_dims);
            }
        } else if (vector_tile) {
            const uint32_t vector_columns = ncol >> 1U;
            const uint32_t vector_elements = full_z * vector_columns;
            const double2* const a2 = reinterpret_cast<const double2*>(s_a);
            const double2* const b2 = reinterpret_cast<const double2*>(s_b);
            for (uint32_t i = threadIdx.x; i < vector_elements; i += TPB) {
                const uint32_t z = i / vector_columns;
                const uint32_t vc = i - z * vector_columns;
                const double2 value =
                    plane_z_final_owner_b(z, full_z, levels_z) ? b2[i] : a2[i];
                *reinterpret_cast<double2*>(
                    data + col0 + (vc << 1U) + z * stride_z) = value;
            }
        } else {
            for (uint32_t i = threadIdx.x; i < tile_elems; i += TPB) {
                const uint32_t z = i / ncol;
                const uint32_t local_column = i - z * ncol;
                const double* const owner =
                    plane_z_final_owner_b(z, full_z, levels_z) ? s_b : s_a;
                data[col0 + local_column + static_cast<size_t>(z) * stride_z] = owner[i];
            }
        }
        __syncthreads();
    } while (true);
}

// Full-column, all-level Z pass for the non-dyadic plane path.  The kernel
// deliberately has no runtime QuantMode/quantization argument: plane Z is
// always an unquantized staging pass, while the XY launcher emits quantized
// subbands.  The existing typed dwt_z_*<false> bodies retain the baseline math
// and ping-pong data flow.
template <typename T>
__global__ __launch_bounds__(TPB, std::is_same_v<T, float> ? 3 : 2)
void dwt_3d_z_all_static(const T* src, T* dst, dim3 data_dims,
                         uint8_t levels_to_run) {
    constexpr uint32_t capacity = DWT_Z_STATIC_BYTES / sizeof(T);
    __shared__ alignas(128) T s_data[capacity];
    __shared__ int s_chunkID;
    const dim3 storage_leaps(1U, data_dims.x, data_dims.x * data_dims.y);
    if constexpr (std::is_same_v<T, float>) {
        dwt_z_float<false>(src, dst, data_dims, storage_leaps, levels_to_run,
                           s_data, capacity, &z_count[0], &s_chunkID,
                           QuantMode::None);
    } else if constexpr (std::is_same_v<T, double>) {
        dwt_z_double<false>(src, dst, data_dims, storage_leaps, levels_to_run,
                            s_data, capacity, &z_count[0], &s_chunkID,
                            QuantMode::None);
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

template <typename T, bool Quantize, bool QuantBlocksKnown = false,
          bool QuantBlocksRegular = false>
__device__ __forceinline__ void dwt_z_active_store(T value, T* dst, size_t s, uint32_t x,
                                                   uint32_t y, uint32_t z, uint32_t low_x,
                                                   uint32_t low_y, bool low_z, dim3 data_dims,
                                                   QuantMode quant_mode) {
    if constexpr (Quantize) {
        if (!low_z || quant_mode == QuantMode::All || x >= low_x || y >= low_y) {
            if constexpr (QuantBlocksKnown)
                quantization<QuantBlocksRegular>(value, s, x, y, z, data_dims);
            else
                quantization(value, s, x, y, z, data_dims);
        } else {
            dst[s] = value;
        }
    } else {
        dst[s] = value;
    }
}

template <typename T, bool Quantize, bool ZTilesStayInRow = false,
          bool QuantBlocksKnown = false, bool QuantBlocksRegular = false>
__device__ __forceinline__ void dwt_z_active_transform_pair(
    const T* s_data, T* dst, uint32_t ncol, dim3 data_dims, uint32_t lx, uint32_t lz,
    uint32_t column0, uint32_t pair, uint32_t local_column, size_t leap_z, uint32_t even_z,
    uint32_t low_x, uint32_t low_y, QuantMode quant_mode, uint32_t tile_x0 = 0U,
    uint32_t tile_y = 0U) {
    const uint32_t active_column = column0 + local_column;
    const uint32_t x = ZTilesStayInRow ? tile_x0 + local_column : active_column % lx;
    const uint32_t y = ZTilesStayInRow ? tile_y : active_column / lx;
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
    dwt_z_active_store<T, Quantize, QuantBlocksKnown, QuantBlocksRegular>(
        low, dst, low_s, x, y, pair, low_x, low_y, true, data_dims, quant_mode);
    if (z + 1 < static_cast<int>(lz)) {
        const T high = plane_z_high_filter<T>(
            vp1, add_rn<T>(c0, vp2), add_rn<T>(vm1, vp3), add_rn<T>(vm2, vp4));
        const uint32_t high_z = even_z + pair;
        const size_t high_s = xy + static_cast<size_t>(high_z) * leap_z;
        dwt_z_active_store<T, Quantize, QuantBlocksKnown, QuantBlocksRegular>(
            high, dst, high_s, x, y, high_z, low_x, low_y, false, data_dims,
            quant_mode);
    }
}

// Double-precision one-level Z body inlined into dwt_3d_z_single_static<double>.
// It stages one or more active columns in one buffer and writes the
// low/high coefficients directly to global memory; no all-level ping-pong
// state is carried into this specialized path.
template <bool ZTilesStayInRow, bool QuantBlocksRegular>
__device__ __forceinline__ void dwt_3d_single_shared_double(
    const double* src, double* dst, dim3 data_dims, dim3 chunk_dims, double* s_data,
    uint32_t capacity, unsigned int* counter, int* chunk_id,
    QuantMode quant_mode) {
    const uint32_t nx = data_dims.x;
    const uint32_t ny = data_dims.y;
    const uint32_t lx = chunk_dims.x;
    const uint32_t ly = chunk_dims.y;
    const uint32_t lz = chunk_dims.z;
    const size_t leap_z = static_cast<size_t>(nx) * ny;
    const uint32_t total_columns = lx * ly;
    uint32_t columns_per_chunk = min(capacity / lz, total_columns);
    if (columns_per_chunk >= 2U)
        columns_per_chunk &= ~1U;
    columns_per_chunk = max(columns_per_chunk, 1U);

    const uint32_t chunks =
        (total_columns + columns_per_chunk - 1U) / columns_per_chunk;
    const uint32_t even_z = lz - (lz >> 1U);
    const uint32_t low_x = lx - (lx >> 1U);
    const uint32_t low_y = ly - (ly >> 1U);

    while (true) {
        if (threadIdx.x == 0)
            *chunk_id = static_cast<int>(atomicAdd(counter, 1U));
        __syncthreads();
        const uint32_t chunk = static_cast<uint32_t>(*chunk_id);
        if (chunk >= chunks)
            break;

        const uint32_t column0 = chunk * columns_per_chunk;
        const uint32_t ncol = min(columns_per_chunk, total_columns - column0);
        const uint32_t tile_elements = ncol * lz;
        uint32_t tile_x0 = 0U;
        uint32_t tile_y = 0U;
        if constexpr (ZTilesStayInRow) {
            tile_y = column0 / lx;
            tile_x0 = column0 - tile_y * lx;
        }

        const bool contiguous_columns = lx == nx || ZTilesStayInRow;
        const size_t global_column0 = ZTilesStayInRow
                                          ? tile_x0 + static_cast<size_t>(tile_y) * nx
                                          : column0;
        const bool vector_load = contiguous_columns && ncol == columns_per_chunk &&
                                 (ncol & 1U) == 0U &&
                                 (global_column0 & 1U) == 0U &&
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
                    src + global_column0 + (vc << 1U) +
                    static_cast<size_t>(z) * leap_z);
            }
        } else {
            for (uint32_t i = threadIdx.x; i < tile_elements; i += TPB) {
                const uint32_t z = i / ncol;
                const uint32_t local_column = i - z * ncol;
                const uint32_t active_column = column0 + local_column;
                const uint32_t x = ZTilesStayInRow
                                       ? tile_x0 + local_column
                                       : active_column % lx;
                const uint32_t y = ZTilesStayInRow ? tile_y : active_column / lx;
                s_data[i] = src[x + static_cast<size_t>(y) * nx +
                                static_cast<size_t>(z) * leap_z];
            }
        }
        __syncthreads();

        const uint32_t work = even_z * ncol;
        for (uint32_t i = threadIdx.x; i < work; i += TPB) {
            const uint32_t pair = i / ncol;
            const uint32_t local_column = i - pair * ncol;
            dwt_z_active_transform_pair<double, true, ZTilesStayInRow, true,
                                        QuantBlocksRegular>(
                s_data, dst, ncol, data_dims, lx, lz, column0, pair,
                local_column, leap_z, even_z, low_x, low_y, quant_mode,
                tile_x0, tile_y);
        }
        __syncthreads();
    }
}

template <typename T>
__global__ __launch_bounds__(TPB, 2) void dwt_3d_z_single_global(const T* src, T* dst, dim3 data_dims, dim3 chunk_dims,
    QuantMode quant_mode) {
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
            value(z), value(z - 1) + value(z + 1),
            value(z - 2) + value(z + 2),
            value(z - 3) + value(z + 3),
            value(z - 4) + value(z + 4));
        const size_t low_s = xy + static_cast<size_t>(pair) * leap_z;
        dwt_z_active_store<T, true>(
            low, dst, low_s, x, y, pair, low_x, low_y, true, data_dims,
            quant_mode);
        if (z + 1 < static_cast<int>(lz)) {
            const int odd_z = z + 1;
            const T high = plane_z_high_filter<T>(
                value(odd_z), value(odd_z - 1) + value(odd_z + 1),
                value(odd_z - 2) + value(odd_z + 2),
                value(odd_z - 3) + value(odd_z + 3));
            const uint32_t high_z = low_z + pair;
            const size_t high_s = xy + static_cast<size_t>(high_z) * leap_z;
            dwt_z_active_store<T, true>(
                high, dst, high_s, x, y, high_z, low_x, low_y, false, data_dims,
                quant_mode);
        }
    }
}

// One-level float body inlined into dwt_3d_z_single_static<float>.
// It deliberately keeps the original two-buffer data flow: stage the
// input, form a complete transformed tile, then quantize that tile in linear
// order. __forceinline__ makes this body part of each kernel's generated SASS;
// it does not call the multi-level Z-all implementation.
template <bool ZTilesStayInRow, bool QuantBlocksRegular>
__device__ __forceinline__ void dwt_3d_single_shared_float(
    const float* src, float* dst, dim3 data_dims, dim3 chunk_dims,
    float* s_data, uint32_t capacity, unsigned int* counter, int* chunk_id,
    QuantMode quant_mode) {
    const uint32_t nx = data_dims.x;
    const uint32_t ny = data_dims.y;
    const uint32_t lx = chunk_dims.x;
    const uint32_t ly = chunk_dims.y;
    const uint32_t lz = chunk_dims.z;
    const size_t leap_z = static_cast<size_t>(nx) * ny;
    const uint32_t total_columns = lx * ly;
    uint32_t columns_per_chunk = min(capacity / (2U * lz), total_columns);
    if (columns_per_chunk >= 4U)
        columns_per_chunk &= ~3U;
    columns_per_chunk = max(columns_per_chunk, 1U);
    const uint32_t chunks =
        (total_columns + columns_per_chunk - 1U) / columns_per_chunk;
    const uint32_t even_z = lz - (lz >> 1U);
    const uint32_t low_x = lx - (lx >> 1U);
    const uint32_t low_y = ly - (ly >> 1U);

    while (true) {
        if (threadIdx.x == 0)
            *chunk_id = static_cast<int>(atomicAdd(counter, 1U));
        __syncthreads();
        const uint32_t chunk = static_cast<uint32_t>(*chunk_id);
        if (chunk >= chunks)
            break;

        const uint32_t column0 = chunk * columns_per_chunk;
        const uint32_t ncol = min(columns_per_chunk, total_columns - column0);
        const uint32_t tile_elements = ncol * lz;
        const uint32_t buffer_elements = columns_per_chunk * lz;
        float* const s_src = s_data;
        float* const s_dst = s_data + buffer_elements;
        uint32_t tile_x0 = 0U;
        uint32_t tile_y = 0U;
        if constexpr (ZTilesStayInRow) {
            tile_y = column0 / lx;
            tile_x0 = column0 - tile_y * lx;
        }

        const bool contiguous_columns = lx == nx || ZTilesStayInRow;
        const size_t global_column0 = ZTilesStayInRow
                                          ? tile_x0 + static_cast<size_t>(tile_y) * nx
                                          : column0;
        const bool vector_load = contiguous_columns && ncol == columns_per_chunk &&
                                 (ncol & 3U) == 0U &&
                                 (global_column0 & 3U) == 0U &&
                                 (leap_z & 3U) == 0U &&
                                 (reinterpret_cast<uintptr_t>(src) & 15U) == 0U;
        if (vector_load) {
            const uint32_t vector_columns = ncol >> 2U;
            const uint32_t vector_elements = lz * vector_columns;
            float4* const shared4 = reinterpret_cast<float4*>(s_src);
            for (uint32_t i = threadIdx.x; i < vector_elements; i += TPB) {
                const uint32_t z = i / vector_columns;
                const uint32_t vc = i - z * vector_columns;
                shared4[i] = *reinterpret_cast<const float4*>(
                    src + global_column0 + (vc << 2U) +
                    static_cast<size_t>(z) * leap_z);
            }
        } else {
            for (uint32_t i = threadIdx.x; i < tile_elements; i += TPB) {
                const uint32_t z = i / ncol;
                const uint32_t local_column = i - z * ncol;
                const uint32_t active_column = column0 + local_column;
                const uint32_t x = ZTilesStayInRow
                                       ? tile_x0 + local_column
                                       : active_column % lx;
                const uint32_t y = ZTilesStayInRow ? tile_y : active_column / lx;
                s_src[i] = src[x + static_cast<size_t>(y) * nx +
                               static_cast<size_t>(z) * leap_z];
            }
        }
        __syncthreads();

        const uint32_t groups = (even_z + 1U) >> 1U;
        const uint32_t work = groups * ncol;
        for (uint32_t i = threadIdx.x; i < work; i += TPB) {
            const uint32_t group = i / ncol;
            const uint32_t local_column = i - group * ncol;
            const uint32_t pair = group << 1U;
            const int z = static_cast<int>(pair << 1U);
            if (pair + 1U < even_z && z >= 4 && z + 6 < static_cast<int>(lz)) {
                const float c0 = s_src[static_cast<uint32_t>(z - 4) * ncol + local_column];
                const float c1 = s_src[static_cast<uint32_t>(z - 3) * ncol + local_column];
                const float c2 = s_src[static_cast<uint32_t>(z - 2) * ncol + local_column];
                const float c3 = s_src[static_cast<uint32_t>(z - 1) * ncol + local_column];
                const float c4 = s_src[static_cast<uint32_t>(z) * ncol + local_column];
                const float c5 = s_src[static_cast<uint32_t>(z + 1) * ncol + local_column];
                const float c6 = s_src[static_cast<uint32_t>(z + 2) * ncol + local_column];
                const float c7 = s_src[static_cast<uint32_t>(z + 3) * ncol + local_column];
                const float c8 = s_src[static_cast<uint32_t>(z + 4) * ncol + local_column];
                const float c9 = s_src[static_cast<uint32_t>(z + 5) * ncol + local_column];
                const float c10 = s_src[static_cast<uint32_t>(z + 6) * ncol + local_column];
                s_dst[pair * ncol + local_column] = plane_z_low_filter<float>(
                    c4, add_rn<float>(c3, c5), add_rn<float>(c2, c6),
                    add_rn<float>(c1, c7), add_rn<float>(c0, c8));
                s_dst[(even_z + pair) * ncol + local_column] =
                    plane_z_high_filter<float>(
                        c5, add_rn<float>(c4, c6), add_rn<float>(c3, c7),
                        add_rn<float>(c2, c8));
                s_dst[(pair + 1U) * ncol + local_column] =
                    plane_z_low_filter<float>(
                        c6, add_rn<float>(c5, c7), add_rn<float>(c4, c8),
                        add_rn<float>(c3, c9), add_rn<float>(c2, c10));
                if (z + 3 < static_cast<int>(lz))
                    s_dst[(even_z + pair + 1U) * ncol + local_column] =
                        plane_z_high_filter<float>(
                            c7, add_rn<float>(c6, c8), add_rn<float>(c5, c9),
                            add_rn<float>(c4, c10));
            } else {
                plane_z_forward_pair_float(
                    s_src, s_dst, ncol, lz, even_z, pair, local_column);
                if (pair + 1U < even_z)
                    plane_z_forward_pair_float(
                        s_src, s_dst, ncol, lz, even_z, pair + 1U, local_column);
            }
        }
        __syncthreads();

        for (uint32_t i = threadIdx.x; i < tile_elements; i += TPB) {
            const uint32_t z = i / ncol;
            const uint32_t local_column = i - z * ncol;
            const uint32_t active_column = column0 + local_column;
            const uint32_t x = ZTilesStayInRow
                                   ? tile_x0 + local_column
                                   : active_column % lx;
            const uint32_t y = ZTilesStayInRow ? tile_y : active_column / lx;
            const size_t index = x + static_cast<size_t>(y) * nx +
                                 static_cast<size_t>(z) * leap_z;
            const float value = s_dst[i];
            if (quant_mode == QuantMode::High && x < low_x && y < low_y && z < even_z)
                dst[index] = value;
            else
                quantization<QuantBlocksRegular>(
                    value, index, x, y, z, data_dims);
        }
        __syncthreads();
    }
}

// Exact single-level Z transform and quantization producer. `quant_mode` is a
// CUDA runtime value: High preserves the low-frequency cube, while All also
// quantizes that final cube.
template <typename T, bool ZTilesStayInRow, bool QuantBlocksRegular>
__global__ __launch_bounds__(TPB, std::is_same_v<T, float> ? 3 : 2) void
dwt_3d_z_single_static(
    const T* src, T* dst, dim3 data_dims, dim3 chunk_dims,
    uint32_t counter_index, QuantMode quant_mode) {
    constexpr uint32_t capacity = 32768U / sizeof(T);
    __shared__ alignas(128) T s_data[capacity];
    __shared__ int s_chunkID;
    if constexpr (std::is_same_v<T, float>) {
        dwt_3d_single_shared_float<ZTilesStayInRow, QuantBlocksRegular>(
            src, dst, data_dims, chunk_dims, s_data, capacity,
            &z_count[counter_index], &s_chunkID, quant_mode);
    } else if constexpr (std::is_same_v<T, double>) {
        dwt_3d_single_shared_double<ZTilesStayInRow, QuantBlocksRegular>(
            src, dst, data_dims, chunk_dims, s_data, capacity,
            &z_count[counter_index], &s_chunkID, quant_mode);
    }
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
    initialize_grid_blocks(dwt_3d_z_all_static<T>, DWTConfig<T>::z_blocks);
}

// ---------------------------------------------------------------------------
} // namespace CDF97
