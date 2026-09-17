#pragma once

#include "CDF97.hpp"

#include <cstdint>

#undef TPB
#define IDWT_TPB 384
#define TPB IDWT_TPB

// The IDWT wide-X halo fallback has its own 128x16 tile shape.  Four owner
// positions on each side keep every X tap window inside the IDWT shared tile;
// this allocation is independent of the DWT raw/x-buffer layout.
inline constexpr size_t IDWT_YX_HALO_SHARED_BYTES =
    static_cast<size_t>(CDF97::BY_16) *
    (static_cast<size_t>(CDF97::BX_128) + 8U) * sizeof(float);

// even-output (low-phase) polyphase filter, symmetric: {IL0,IL1,IL2,IL3,IL2,IL1,IL0}
#define IL0 -0.0238494650195568
#define IL1 -0.0406894176097634
#define IL2 0.3774028556128310
#define IL3 0.7884856164052130

// odd-output (high-phase) polyphase filter, symmetric: {IH0,IH1,IH2,IH3,IH4,IH3,IH2,IH1,IH0}
#define IH0 -0.0378284555072640
#define IH1 -0.0645388826292432
#define IH2 0.1106244044184370
#define IH3 0.4180922732229480
#define IH4 -0.8526986790088940


namespace CDF97 {

using WALTZ::IDWTConfig;
using WALTZ::PWE;

__device__ __forceinline__ uint32_t approx_len_dev(uint32_t n, int lev) {
    uint32_t low = n;
    for (int i = 0; i < lev; ++i)
        low -= low >> 1;
    return low;
}

// whole-sample symmetric (mirror, no edge repeat) folding -- SPERR's symmetric_idx
__device__ __forceinline__ uint32_t sym_idx(int64_t idx, int64_t last) {
    while (idx < 0 || idx > last) {
        if (idx < 0)
            idx = -idx;
        if (idx > last) {
            // Form the full 64-bit reflection with explicit carry/borrow.
            // Avoid the sm120 codegen issue in the C++ expression 2*last-idx.
            int64_t reflected;
            asm volatile(
                "{\n\t"
                ".reg .b32 a0, a1, b0, b1, t0, t1, r0, r1;\n\t"
                "mov.b64 {a0, a1}, %1;\n\t"
                "mov.b64 {b0, b1}, %2;\n\t"
                "add.cc.u32 t0, a0, a0;\n\t"
                "addc.u32 t1, a1, a1;\n\t"
                "sub.cc.u32 r0, t0, b0;\n\t"
                "subc.u32 r1, t1, b1;\n\t"
                "mov.b64 %0, {r0, r1};\n\t"
                "}"
                : "=l"(reflected)
                : "l"(last), "l"(idx));
            idx = reflected;
        }
    }
    return static_cast<uint32_t>(idx);
}

// Z shared float4 boundary reflection: idx/last are bounded by the shared
// tile capacity, so use 32-bit signed coordinates.
// The while/mirror recurrence is intentionally identical to sym_idx above.
__device__ __forceinline__ uint32_t sym_idx32(int idx, int last) {
    while (idx < 0 || idx > last) {
        if (idx < 0)
            idx = -idx;
        if (idx > last)
            idx = 2 * last - idx;
    }
    return static_cast<uint32_t>(idx);
}

// PAIR-FUSED synthesis: emit the two adjacent outputs at positions 2i (even) and 2i+1
// (odd) from ONE loaded window c[0..8] = C(2i-3 .. 2i+5).  Mirrors the forward's
// pair-fusion (load the tap window once, produce two outputs) so each input is read
// ~4.5x not ~8x.  C(idx) returns the deinterleaved coeff at logical index idx.
template <typename T, typename F>
__device__ __forceinline__ void synth_pair(uint32_t i, F C, T& out_even, T& out_odd) {
    const long long b = 2 * static_cast<long long>(i) - 3; // c[0] is logical 2i-3
    const T c0 = C(b), c1 = C(b + 1), c2 = C(b + 2), c3 = C(b + 3), c4 = C(b + 4), c5 = C(b + 5),
            c6 = C(b + 6), c7 = C(b + 7), c8 = C(b + 8);
    out_even = T(IL0) * (c0 + c6) + T(IL1) * (c1 + c5) + T(IL2) * (c2 + c4) + T(IL3) * c3;
    out_odd = T(IH0) * (c0 + c8) + T(IH1) * (c1 + c7) + T(IH2) * (c2 + c6) + T(IH3) * (c3 + c5) +
              T(IH4) * c4;
}

// REGISTER-BLOCKED: two adjacent pairs (i, i+1) -> four outputs at 2i,2i+1,2i+2,2i+3
// from ONE window c[0..10] = C(2i-3 .. 2i+7).  The two pairs overlap by 7 taps, so 11
// loads feed 4 outputs (vs 18 for two separate pairs) -- fewer L2 round-trips and more
// in-flight loads to hide the deinterleaved-read latency.
template <typename T, typename F>
__device__ __forceinline__ void synth_quad(uint32_t i, F C, T& e0, T& o0, T& e1, T& o1) {
    const long long b = 2 * static_cast<long long>(i) - 3;
    const T c0 = C(b), c1 = C(b + 1), c2 = C(b + 2), c3 = C(b + 3), c4 = C(b + 4), c5 = C(b + 5),
            c6 = C(b + 6), c7 = C(b + 7), c8 = C(b + 8), c9 = C(b + 9), c10 = C(b + 10);
    e0 = T(IL0) * (c0 + c6) + T(IL1) * (c1 + c5) + T(IL2) * (c2 + c4) + T(IL3) * c3;
    o0 = T(IH0) * (c0 + c8) + T(IH1) * (c1 + c7) + T(IH2) * (c2 + c6) + T(IH3) * (c3 + c5) +
         T(IH4) * c4;
    e1 = T(IL0) * (c2 + c8) + T(IL1) * (c3 + c7) + T(IL2) * (c4 + c6) + T(IL3) * c5;
    o1 = T(IH0) * (c2 + c10) + T(IH1) * (c3 + c9) + T(IH2) * (c4 + c8) + T(IH3) * (c5 + c7) +
         T(IH4) * c6;
}


__device__ __forceinline__ float4
idwt_even4(float4 c0, float4 c1, float4 c2, float4 c3, float4 c4, float4 c5, float4 c6) {
    return make_float4(float(IL0) * (c0.x + c6.x) + float(IL1) * (c1.x + c5.x) +
                           float(IL2) * (c2.x + c4.x) + float(IL3) * c3.x,
                       float(IL0) * (c0.y + c6.y) + float(IL1) * (c1.y + c5.y) +
                           float(IL2) * (c2.y + c4.y) + float(IL3) * c3.y,
                       float(IL0) * (c0.z + c6.z) + float(IL1) * (c1.z + c5.z) +
                           float(IL2) * (c2.z + c4.z) + float(IL3) * c3.z,
                       float(IL0) * (c0.w + c6.w) + float(IL1) * (c1.w + c5.w) +
                           float(IL2) * (c2.w + c4.w) + float(IL3) * c3.w);
}

__device__ __forceinline__ float4 idwt_odd4(float4 c0, float4 c1, float4 c2, float4 c3,
    float4 c4, float4 c5, float4 c6, float4 c7, float4 c8) {
    return make_float4(
        float(IH0) * (c0.x + c8.x) + float(IH1) * (c1.x + c7.x) + float(IH2) * (c2.x + c6.x) +
            float(IH3) * (c3.x + c5.x) + float(IH4) * c4.x,
        float(IH0) * (c0.y + c8.y) + float(IH1) * (c1.y + c7.y) + float(IH2) * (c2.y + c6.y) +
            float(IH3) * (c3.y + c5.y) + float(IH4) * c4.y,
        float(IH0) * (c0.z + c8.z) + float(IH1) * (c1.z + c7.z) + float(IH2) * (c2.z + c6.z) +
            float(IH3) * (c3.z + c5.z) + float(IH4) * c4.z,
        float(IH0) * (c0.w + c8.w) + float(IH1) * (c1.w + c7.w) + float(IH2) * (c2.w + c6.w) +
            float(IH3) * (c3.w + c5.w) + float(IH4) * c4.w);
}

// Two-column companion of the float4 Z synthesis helpers.  The vector only
// changes the shared/global transaction width; each lane keeps the scalar
// double-precision tap expression and ordering.
__device__ __forceinline__ double2 idwt_even2(double2 c0, double2 c1, double2 c2,
    double2 c3, double2 c4, double2 c5, double2 c6) {
    return make_double2(
        double(IL0) * (c0.x + c6.x) + double(IL1) * (c1.x + c5.x) +
            double(IL2) * (c2.x + c4.x) + double(IL3) * c3.x,
        double(IL0) * (c0.y + c6.y) + double(IL1) * (c1.y + c5.y) +
            double(IL2) * (c2.y + c4.y) + double(IL3) * c3.y);
}

__device__ __forceinline__ double2 idwt_odd2(double2 c0, double2 c1, double2 c2, double2 c3,
    double2 c4, double2 c5, double2 c6, double2 c7, double2 c8) {
    return make_double2(
        double(IH0) * (c0.x + c8.x) + double(IH1) * (c1.x + c7.x) +
            double(IH2) * (c2.x + c6.x) + double(IH3) * (c3.x + c5.x) + double(IH4) * c4.x,
        double(IH0) * (c0.y + c8.y) + double(IH1) * (c1.y + c7.y) +
            double(IH2) * (c2.y + c6.y) + double(IH3) * (c3.y + c5.y) + double(IH4) * c4.y);
}

// Four-column version of the four-pair register block used by the plane XY
// inverse.  Each lane follows the scalar synth_quad expression independently;
// the vector form only changes the memory transaction granularity.
__device__ __forceinline__ void synth_quad4(float4 c0, float4 c1, float4 c2, float4 c3, float4 c4,
    float4 c5, float4 c6, float4 c7, float4 c8, float4 c9, float4 c10,
    float4& e0, float4& o0, float4& e1, float4& o1) {
    e0 = idwt_even4(c0, c1, c2, c3, c4, c5, c6);
    o0 = idwt_odd4(c0, c1, c2, c3, c4, c5, c6, c7, c8);
    e1 = idwt_even4(c2, c3, c4, c5, c6, c7, c8);
    o1 = idwt_odd4(c2, c3, c4, c5, c6, c7, c8, c9, c10);
}

// Z synthesis: coalesced across x (innermost); float register-blocks 2 pairs/thread
// (synth_quad) for more in-flight loads, double uses 1 pair (lower register pressure).
template <typename T>
__device__ void idwt_z_pass(const T* __restrict__ src, T* __restrict__ dst,
    uint32_t nx, uint32_t ny, uint32_t lx, uint32_t ly, uint32_t lz) {
    if (lz < 2)
        return;
    const size_t leap_z = static_cast<size_t>(nx) * ny;
    const uint32_t low_lz = lz - (lz >> 1);
    const long long last = static_cast<long long>(lz) - 1;
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    const size_t g0 = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    // Float uses two pairs per thread for more memory-level parallelism; double
    // uses one pair to limit register pressure.
    if constexpr (std::is_same_v<T, float>) {
        const uint32_t dp = (low_lz + 1u) >> 1;
        const size_t total = static_cast<size_t>(lx) * ly * dp;
        // Power-of-two active extents use masks/shifts instead of 64-bit div/mod;
        // arbitrary extents retain the exact generic mapping below.
        const bool p2x = (lx & (lx - 1U)) == 0U;
        const bool p2y = (ly & (ly - 1U)) == 0U;
        const int sx = p2x ? (__ffs(static_cast<int>(lx)) - 1) : 0;
        const int sy = p2y ? (__ffs(static_cast<int>(ly)) - 1) : 0;
        for (size_t g = g0; g < total; g += stride) {
            const uint32_t x =
                p2x ? static_cast<uint32_t>(g) & (lx - 1U) : static_cast<uint32_t>(g % lx);
            const size_t r = p2x ? (g >> sx) : (g / lx);
            const uint32_t y =
                p2y ? static_cast<uint32_t>(r) & (ly - 1U) : static_cast<uint32_t>(r % ly);
            const uint32_t i = 2u * static_cast<uint32_t>(p2y ? (r >> sy) : (r / ly));
            const size_t xy = x + static_cast<size_t>(y) * nx;
            if (i >= 2U && 2U * i + 7U < lz) {
                auto Ci = [&](int idx) -> T {
                    const uint32_t u = static_cast<uint32_t>(idx);
                    const uint32_t row = (u & 1U) ? (low_lz + (u >> 1U)) : (u >> 1U);
                    return src[xy + static_cast<size_t>(row) * leap_z];
                };
                T e0, o0, e1, o1;
                synth_quad<T>(static_cast<int>(i), Ci, e0, o0, e1, o1);
                dst[xy + static_cast<size_t>(2U * i) * leap_z] = e0;
                dst[xy + static_cast<size_t>(2U * i + 1U) * leap_z] = o0;
                if (2U * i + 2U < lz)
                    dst[xy + static_cast<size_t>(2U * i + 2U) * leap_z] = e1;
                if (2U * i + 3U < lz)
                    dst[xy + static_cast<size_t>(2U * i + 3U) * leap_z] = o1;
                continue;
            }
            auto C = [&](long long idx) -> T {
                const uint32_t pos = sym_idx(idx, last);
                return src[xy +
                           static_cast<size_t>((pos & 1u) ? (low_lz + (pos >> 1)) : (pos >> 1)) *
                               leap_z];
            };
            if (i + 1u < low_lz) {
                T e0, o0, e1, o1;
                synth_quad<T>(i, C, e0, o0, e1, o1);
                dst[xy + static_cast<size_t>(2 * i) * leap_z] = e0;
                dst[xy + static_cast<size_t>(2 * i + 1) * leap_z] = o0;
                dst[xy + static_cast<size_t>(2 * i + 2) * leap_z] = e1;
                if (2 * i + 3 < lz)
                    dst[xy + static_cast<size_t>(2 * i + 3) * leap_z] = o1;
            } else {
                T e0, o0;
                synth_pair<T>(i, C, e0, o0);
                dst[xy + static_cast<size_t>(2 * i) * leap_z] = e0;
                if (2 * i + 1 < lz)
                    dst[xy + static_cast<size_t>(2 * i + 1) * leap_z] = o0;
            }
        }
    } else {
        const size_t total = static_cast<size_t>(lx) * ly * low_lz;
        for (size_t g = g0; g < total; g += stride) {
            const uint32_t x = static_cast<uint32_t>(g % lx);
            const size_t r = g / lx;
            const uint32_t y = static_cast<uint32_t>(r % ly);
            const uint32_t i = static_cast<uint32_t>(r / ly);
            const size_t xy = x + static_cast<size_t>(y) * nx;
            auto C = [&](long long idx) -> T {
                const uint32_t pos = sym_idx(idx, last);
                return src[xy +
                           static_cast<size_t>((pos & 1u) ? (low_lz + (pos >> 1)) : (pos >> 1)) *
                               leap_z];
            };
            T e0, o0;
            synth_pair<T>(i, C, e0, o0);
            dst[xy + static_cast<size_t>(2 * i) * leap_z] = e0;
            if (2 * i + 1 < lz)
                dst[xy + static_cast<size_t>(2 * i + 1) * leap_z] = o0;
        }
    }
}

template <typename T>
__device__ void idwt_z_level_shared(const T* src, T* dst, T* s_data, uint32_t cap, uint32_t nx, uint32_t ny,
    uint32_t lx, uint32_t ly, uint32_t lz, bool vector_z) {
    if (lz < 2)
        return;
    const size_t leap_z = static_cast<size_t>(nx) * ny;
    const uint32_t low_lz = lz - (lz >> 1);
    const uint32_t TX = cap / lz;
    if (TX == 0)
        return;
    const uint32_t nxt = (lx + TX - 1) / TX;
    const uint32_t ntiles = ly * nxt;
    const uint32_t half = lz >> 1;
    const uint32_t hi_i = (half >= 3u) ? (half - 3u) : 0u;
    for (uint32_t t = blockIdx.x; t < ntiles; t += gridDim.x) {
        const uint32_t y = t / nxt;
        const uint32_t x0 = (t % nxt) * TX;
        const uint32_t txn = min(TX, lx - x0);
        const size_t base = x0 + static_cast<size_t>(y) * nx;
        for (uint32_t e = threadIdx.x; e < lz * txn; e += TPB) { // stage deinterleaved column
            const uint32_t z = e / txn, tx = e - z * txn;
            s_data[z * txn + tx] = src[base + tx + static_cast<size_t>(z) * leap_z];
        }
        __syncthreads();
        // Optional vector synthesis: adjacent x-columns share the same z-window,
        // so float4/double2 shared loads and global stores remove scalar transaction
        // overhead.  Keep the original float4 gate; double2 additionally requires
        // even shared rows and alignment of every global output row.
        bool vector_tile = false;
        if constexpr (std::is_same_v<T, float>) {
            vector_tile = vector_z && txn >= 4U && (txn & 3U) == 0U;
        } else if constexpr (std::is_same_v<T, double>) {
            vector_tile = vector_z && txn >= 2U && (txn & 1U) == 0U && (nx & 1U) == 0U &&
                          (x0 & 1U) == 0U &&
                          (reinterpret_cast<uintptr_t>(dst + base) & 15U) == 0U;
        }
        if (vector_tile) {
            if constexpr (std::is_same_v<T, float>) {
                const int last32 = static_cast<int>(lz) - 1;
                const uint32_t nv = txn >> 2U;
                for (uint32_t e = threadIdx.x; e < low_lz * nv; e += TPB) {
                    const uint32_t i = e / nv, vc = e - i * nv;
                    const uint32_t tx = vc << 2U;
                    // The staged tile is bounded by IDWT_SHARED_BYTES (32 KiB for
                    // this path), so every shared element offset fits in uint32_t.
                    // Keep the output address arithmetic below in size_t: global
                    // rows use the full-volume 64-bit stride.
                    float4 oe;
                    auto C4 = [&](int idx) -> float4 {
                        const uint32_t pos = sym_idx32(idx, last32);
                        const uint32_t k = (pos & 1u) ? (low_lz + (pos >> 1)) : (pos >> 1);
                        return *reinterpret_cast<const float4*>(
                            s_data + k * txn + tx);
                    };
                    const bool interior = i >= 2u && i <= hi_i;
                    if (interior) {
                        // Stream the even tap pairs.  At most two float4 inputs are
                        // live at a time; the expression order is the same as
                        // idwt_even4 (pair sum, coefficient product, left-folded
                        // accumulation), including the compiler's FMA choices.
                        {
                            const float4 c0 = *reinterpret_cast<const float4*>(
                                s_data + (low_lz + i - 2U) * txn + tx);
                            const float4 c6 = *reinterpret_cast<const float4*>(
                                s_data + (low_lz + i + 1U) * txn + tx);
                            oe = make_float4(
                                float(IL0) * (c0.x + c6.x), float(IL0) * (c0.y + c6.y),
                                float(IL0) * (c0.z + c6.z), float(IL0) * (c0.w + c6.w));
                        }
                        {
                            const float4 c1 = *reinterpret_cast<const float4*>(
                                s_data + (i - 1U) * txn + tx);
                            const float4 c5 = *reinterpret_cast<const float4*>(
                                s_data + (i + 1U) * txn + tx);
                            oe.x = oe.x + float(IL1) * (c1.x + c5.x);
                            oe.y = oe.y + float(IL1) * (c1.y + c5.y);
                            oe.z = oe.z + float(IL1) * (c1.z + c5.z);
                            oe.w = oe.w + float(IL1) * (c1.w + c5.w);
                        }
                        {
                            const float4 c2 = *reinterpret_cast<const float4*>(
                                s_data + (low_lz + i - 1U) * txn + tx);
                            const float4 c4 = *reinterpret_cast<const float4*>(
                                s_data + (low_lz + i) * txn + tx);
                            oe.x = oe.x + float(IL2) * (c2.x + c4.x);
                            oe.y = oe.y + float(IL2) * (c2.y + c4.y);
                            oe.z = oe.z + float(IL2) * (c2.z + c4.z);
                            oe.w = oe.w + float(IL2) * (c2.w + c4.w);
                        }
                        {
                            const float4 c3 = *reinterpret_cast<const float4*>(
                                s_data + i * txn + tx);
                            oe.x = oe.x + float(IL3) * c3.x;
                            oe.y = oe.y + float(IL3) * c3.y;
                            oe.z = oe.z + float(IL3) * c3.z;
                            oe.w = oe.w + float(IL3) * c3.w;
                        }
                    } else {
                        const int b = 2 * static_cast<int>(i) - 3;
                        // Boundary taps use the same streamed schedule through
                        // C4; reflection remains 32-bit because it is confined to
                        // this shared tile.
                        {
                            const float4 c0 = C4(b + 0), c6 = C4(b + 6);
                            oe = make_float4(
                                float(IL0) * (c0.x + c6.x), float(IL0) * (c0.y + c6.y),
                                float(IL0) * (c0.z + c6.z), float(IL0) * (c0.w + c6.w));
                        }
                        {
                            const float4 c1 = C4(b + 1), c5 = C4(b + 5);
                            oe.x = oe.x + float(IL1) * (c1.x + c5.x);
                            oe.y = oe.y + float(IL1) * (c1.y + c5.y);
                            oe.z = oe.z + float(IL1) * (c1.z + c5.z);
                            oe.w = oe.w + float(IL1) * (c1.w + c5.w);
                        }
                        {
                            const float4 c2 = C4(b + 2), c4 = C4(b + 4);
                            oe.x = oe.x + float(IL2) * (c2.x + c4.x);
                            oe.y = oe.y + float(IL2) * (c2.y + c4.y);
                            oe.z = oe.z + float(IL2) * (c2.z + c4.z);
                            oe.w = oe.w + float(IL2) * (c2.w + c4.w);
                        }
                        {
                            const float4 c3 = C4(b + 3);
                            oe.x = oe.x + float(IL3) * c3.x;
                            oe.y = oe.y + float(IL3) * c3.y;
                            oe.z = oe.z + float(IL3) * c3.z;
                            oe.w = oe.w + float(IL3) * c3.w;
                        }
                    }
                    *reinterpret_cast<float4*>(dst + base + tx +
                                               static_cast<size_t>(2U * i) * leap_z) = oe;
                    if (2U * i + 1U < lz) {
                        // Store the even result before constructing the odd result.
                        // The odd tap pairs are reloaded, which keeps oe dead and
                        // bounds the vector live set while retaining the original
                        // odd-output association.
                        float4 oo;
                        if (interior) {
                            {
                                const float4 c0 = *reinterpret_cast<const float4*>(
                                    s_data + (low_lz + i - 2U) * txn + tx);
                                const float4 c8 = *reinterpret_cast<const float4*>(
                                    s_data + (low_lz + i + 2U) * txn + tx);
                                oo = make_float4(
                                    float(IH0) * (c0.x + c8.x), float(IH0) * (c0.y + c8.y),
                                    float(IH0) * (c0.z + c8.z), float(IH0) * (c0.w + c8.w));
                            }
                            {
                                const float4 c1 = *reinterpret_cast<const float4*>(
                                    s_data + (i - 1U) * txn + tx);
                                const float4 c7 = *reinterpret_cast<const float4*>(
                                    s_data + (i + 2U) * txn + tx);
                                oo.x = oo.x + float(IH1) * (c1.x + c7.x);
                                oo.y = oo.y + float(IH1) * (c1.y + c7.y);
                                oo.z = oo.z + float(IH1) * (c1.z + c7.z);
                                oo.w = oo.w + float(IH1) * (c1.w + c7.w);
                            }
                            {
                                const float4 c2 = *reinterpret_cast<const float4*>(
                                    s_data + (low_lz + i - 1U) * txn + tx);
                                const float4 c6 = *reinterpret_cast<const float4*>(
                                    s_data + (low_lz + i + 1U) * txn + tx);
                                oo.x = oo.x + float(IH2) * (c2.x + c6.x);
                                oo.y = oo.y + float(IH2) * (c2.y + c6.y);
                                oo.z = oo.z + float(IH2) * (c2.z + c6.z);
                                oo.w = oo.w + float(IH2) * (c2.w + c6.w);
                            }
                            {
                                const float4 c3 = *reinterpret_cast<const float4*>(
                                    s_data + i * txn + tx);
                                const float4 c5 = *reinterpret_cast<const float4*>(
                                    s_data + (i + 1U) * txn + tx);
                                oo.x = oo.x + float(IH3) * (c3.x + c5.x);
                                oo.y = oo.y + float(IH3) * (c3.y + c5.y);
                                oo.z = oo.z + float(IH3) * (c3.z + c5.z);
                                oo.w = oo.w + float(IH3) * (c3.w + c5.w);
                            }
                            {
                                const float4 c4 = *reinterpret_cast<const float4*>(
                                    s_data + (low_lz + i) * txn + tx);
                                oo.x = oo.x + float(IH4) * c4.x;
                                oo.y = oo.y + float(IH4) * c4.y;
                                oo.z = oo.z + float(IH4) * c4.z;
                                oo.w = oo.w + float(IH4) * c4.w;
                            }
                        } else {
                            const int b = 2 * static_cast<int>(i) - 3;
                            {
                                const float4 c0 = C4(b + 0), c8 = C4(b + 8);
                                oo = make_float4(
                                    float(IH0) * (c0.x + c8.x), float(IH0) * (c0.y + c8.y),
                                    float(IH0) * (c0.z + c8.z), float(IH0) * (c0.w + c8.w));
                            }
                            {
                                const float4 c1 = C4(b + 1), c7 = C4(b + 7);
                                oo.x = oo.x + float(IH1) * (c1.x + c7.x);
                                oo.y = oo.y + float(IH1) * (c1.y + c7.y);
                                oo.z = oo.z + float(IH1) * (c1.z + c7.z);
                                oo.w = oo.w + float(IH1) * (c1.w + c7.w);
                            }
                            {
                                const float4 c2 = C4(b + 2), c6 = C4(b + 6);
                                oo.x = oo.x + float(IH2) * (c2.x + c6.x);
                                oo.y = oo.y + float(IH2) * (c2.y + c6.y);
                                oo.z = oo.z + float(IH2) * (c2.z + c6.z);
                                oo.w = oo.w + float(IH2) * (c2.w + c6.w);
                            }
                            {
                                const float4 c3 = C4(b + 3), c5 = C4(b + 5);
                                oo.x = oo.x + float(IH3) * (c3.x + c5.x);
                                oo.y = oo.y + float(IH3) * (c3.y + c5.y);
                                oo.z = oo.z + float(IH3) * (c3.z + c5.z);
                                oo.w = oo.w + float(IH3) * (c3.w + c5.w);
                            }
                            {
                                const float4 c4 = C4(b + 4);
                                oo.x = oo.x + float(IH4) * c4.x;
                                oo.y = oo.y + float(IH4) * c4.y;
                                oo.z = oo.z + float(IH4) * c4.z;
                                oo.w = oo.w + float(IH4) * c4.w;
                            }
                        }
                        *reinterpret_cast<float4*>(
                            dst + base + tx + static_cast<size_t>(2U * i + 1U) * leap_z) = oo;
                    }
                }
            } else if constexpr (std::is_same_v<T, double>) {
                const uint32_t nv = txn >> 1U;
                for (uint32_t e = threadIdx.x; e < low_lz * nv; e += TPB) {
                    const uint32_t i = e / nv, vc = e - i * nv;
                    const uint32_t tx = vc << 1U;
                    double2 oe, oo;
                    // Shared-capacity-limited Z indices fit in int; avoid the sm120 64-bit reflection anomaly.
                    auto C2 = [&](int idx) -> double2 {
                        const uint32_t pos = sym_idx32(idx, static_cast<int>(lz) - 1);
                        const uint32_t k = (pos & 1u) ? (low_lz + (pos >> 1)) : (pos >> 1);
                        return *reinterpret_cast<const double2*>(
                            s_data + static_cast<size_t>(k) * txn + tx);
                    };
                    if (i >= 2u && i <= hi_i) {
                        const double2 c0 = *reinterpret_cast<const double2*>(
                            s_data + static_cast<size_t>(low_lz + i - 2U) * txn + tx);
                        const double2 c1 = *reinterpret_cast<const double2*>(
                            s_data + static_cast<size_t>(i - 1U) * txn + tx);
                        const double2 c2 = *reinterpret_cast<const double2*>(
                            s_data + static_cast<size_t>(low_lz + i - 1U) * txn + tx);
                        const double2 c3 = *reinterpret_cast<const double2*>(
                            s_data + static_cast<size_t>(i) * txn + tx);
                        const double2 c4 = *reinterpret_cast<const double2*>(
                            s_data + static_cast<size_t>(low_lz + i) * txn + tx);
                        const double2 c5 = *reinterpret_cast<const double2*>(
                            s_data + static_cast<size_t>(i + 1U) * txn + tx);
                        const double2 c6 = *reinterpret_cast<const double2*>(
                            s_data + static_cast<size_t>(low_lz + i + 1U) * txn + tx);
                        const double2 c7 = *reinterpret_cast<const double2*>(
                            s_data + static_cast<size_t>(i + 2U) * txn + tx);
                        const double2 c8 = *reinterpret_cast<const double2*>(
                            s_data + static_cast<size_t>(low_lz + i + 2U) * txn + tx);
                        oe = idwt_even2(c0, c1, c2, c3, c4, c5, c6);
                        oo = idwt_odd2(c0, c1, c2, c3, c4, c5, c6, c7, c8);
                    } else {
                        const int b = 2 * static_cast<int>(i) - 3;
                        oe = idwt_even2(C2(b + 0),
                                        C2(b + 1),
                                        C2(b + 2),
                                        C2(b + 3),
                                        C2(b + 4),
                                        C2(b + 5),
                                        C2(b + 6));
                        oo = idwt_odd2(C2(b + 0),
                                       C2(b + 1),
                                       C2(b + 2),
                                       C2(b + 3),
                                       C2(b + 4),
                                       C2(b + 5),
                                       C2(b + 6),
                                       C2(b + 7),
                                       C2(b + 8));
                    }
                    *reinterpret_cast<double2*>(dst + base + tx +
                                                static_cast<size_t>(2U * i) * leap_z) = oe;
                    if (2U * i + 1U < lz)
                        *reinterpret_cast<double2*>(
                            dst + base + tx + static_cast<size_t>(2U * i + 1U) * leap_z) = oo;
                }
            }
        }
        if (!vector_tile)
            for (uint32_t e = threadIdx.x; e < low_lz * txn; e += TPB) {
                const uint32_t i = e / txn, tx = e - i * txn;
                const T* col = s_data + tx;
                T oe, oo;
                if (i >= 2u && i <= hi_i) {
                    const size_t lo = static_cast<size_t>(i) * txn;
                    const size_t hi = static_cast<size_t>(low_lz + i) * txn;
                    const size_t s = txn;
                    const T l_m1 = col[lo - s], l_0 = col[lo], l_p1 = col[lo + s],
                            l_p2 = col[lo + 2 * s];
                    const T h_m2 = col[hi - 2 * s], h_m1 = col[hi - s], h_0 = col[hi],
                            h_p1 = col[hi + s], h_p2 = col[hi + 2 * s];
                    oe = T(IL0) * (h_m2 + h_p1) + T(IL1) * (l_m1 + l_p1) + T(IL2) * (h_m1 + h_0) +
                         T(IL3) * l_0;
                    oo = T(IH0) * (h_m2 + h_p2) + T(IH1) * (l_m1 + l_p2) + T(IH2) * (h_m1 + h_p1) +
                         T(IH3) * (l_0 + l_p1) + T(IH4) * h_0;
                } else {
                    // Shared-capacity-bounded indices use the same 32-bit
                    // reflection as the vector branch (including odd tiles).
                    auto C = [&](int idx) -> T {
                        const uint32_t pos = sym_idx32(idx, static_cast<int>(lz) - 1);
                        return col[static_cast<size_t>((pos & 1u) ? (low_lz + (pos >> 1))
                                                                  : (pos >> 1)) *
                                   txn];
                    };
                    synth_pair<T>(i, C, oe, oo);
                }
                dst[base + tx + static_cast<size_t>(2 * i) * leap_z] = oe;
                if (2 * i + 1 < lz)
                    dst[base + tx + static_cast<size_t>(2 * i + 1) * leap_z] = oo;
            }
        __syncthreads();
    }
}

// Double Y+X synthesis shared with the plane launcher. Aligned columns use
// double2 loads; adjacent outputs reuse coefficient windows. PWE selects only
// error recording at compile time; the vector-Y memory path is selected at runtime.
template <PWE Mode = PWE::F, int KTPB = TPB>
__device__ __forceinline__ void idwt_yx_tiled_double(const double* __restrict__ src, double* __restrict__ dst, double* rows,
    uint32_t cap, uint32_t nx, uint32_t ny, uint32_t lx, uint32_t ly, uint32_t lz, const double* orig = nullptr,
    double bound = 0.0, unsigned int* ocnt = nullptr, uint32_t* oidx = nullptr, double* oerr = nullptr,
    uint32_t ocap = 0, bool vector_y = false) {
    if (lx < 2U || ly < 2U)
        return;
    const uint32_t pairs_per_tile = cap / (2U * lx);
    if (pairs_per_tile == 0U)
        return;

    const size_t leap_z = static_cast<size_t>(nx) * ny;
    const uint32_t low_ly = ly - (ly >> 1U);
    const uint32_t low_lx = lx - (lx >> 1U);
    const long long last_y = static_cast<long long>(ly) - 1;
    const long long last_x = static_cast<long long>(lx) - 1;
    const uint32_t nty = (low_ly + pairs_per_tile - 1U) / pairs_per_tile;
    const uint32_t ntiles = nty * lz;

    for (uint32_t tile = blockIdx.x; tile < ntiles; tile += gridDim.x) {
        const uint32_t z = tile / nty;
        const uint32_t pt = tile - z * nty;
        const uint32_t pair0 = pt * pairs_per_tile;
        const uint32_t pairn = min(pairs_per_tile, low_ly - pair0);
        const uint32_t out_rows = min(2U * pairn, ly - 2U * pair0);

        const uint32_t groups = (pairn + 1U) >> 1U;
        const uint32_t chains = (groups + 1U) >> 1U;
        if (vector_y) {
            const uint32_t vec_lx = lx >> 1U;
            const uint32_t ywork = groups * vec_lx;
            for (uint32_t e = threadIdx.x; e < ywork; e += KTPB) {
                const uint32_t vx = e % vec_lx;
                const uint32_t x = vx << 1U;
                const uint32_t group = e / vec_lx;
                const uint32_t i = pair0 + (group << 1U);
                const size_t xz = static_cast<size_t>(x) + static_cast<size_t>(z) * leap_z;
                auto C2 = [&](long long idx) -> double2 {
                    const uint32_t pos = sym_idx(idx, last_y);
                    const uint32_t sy =
                        (pos & 1U) ? (low_ly + (pos >> 1U)) : (pos >> 1U);
                    return *reinterpret_cast<const double2*>(
                        src + xz + static_cast<size_t>(sy) * nx);
                };
                const uint32_t local = (i - pair0) << 1U;
                const long long b = 2LL * static_cast<long long>(i) - 3LL;
                const double2 c2 = C2(b + 2), c3 = C2(b + 3), c4 = C2(b + 4),
                              c5 = C2(b + 5), c6 = C2(b + 6), c7 = C2(b + 7),
                              c8 = C2(b + 8);
                {
                    const double2 c0 = C2(b + 0), c1 = C2(b + 1);
                    const double2 e0 = idwt_even2(c0, c1, c2, c3, c4, c5, c6);
                    *reinterpret_cast<double2*>(rows + static_cast<size_t>(local) * lx + x) = e0;
                    if (local + 1U < out_rows) {
                        const double2 o0 = idwt_odd2(c0, c1, c2, c3, c4, c5, c6, c7, c8);
                        *reinterpret_cast<double2*>(
                            rows + static_cast<size_t>(local + 1U) * lx + x) = o0;
                    }
                }
                if (local + 2U < out_rows) {
                    const double2 e1 = idwt_even2(c2, c3, c4, c5, c6, c7, c8);
                    *reinterpret_cast<double2*>(rows + static_cast<size_t>(local + 2U) * lx + x) =
                        e1;
                    if (local + 3U < out_rows) {
                        const double2 c9 = C2(b + 9), c10 = C2(b + 10);
                        const double2 o1 = idwt_odd2(c2, c3, c4, c5, c6, c7, c8, c9, c10);
                        *reinterpret_cast<double2*>(
                            rows + static_cast<size_t>(local + 3U) * lx + x) = o1;
                    }
                }
            }
        } else {
            const uint32_t ywork = chains * lx;
            for (uint32_t e = threadIdx.x; e < ywork; e += KTPB) {
                const uint32_t x = e % lx;
                const uint32_t chain = e / lx;
                const uint32_t i = pair0 + (chain << 2U);
                const size_t xz = static_cast<size_t>(x) + static_cast<size_t>(z) * leap_z;
                auto C = [&](long long idx) -> double {
                    const uint32_t pos = sym_idx(idx, last_y);
                    const uint32_t sy =
                        (pos & 1U) ? (low_ly + (pos >> 1U)) : (pos >> 1U);
                    return src[xz + static_cast<size_t>(sy) * nx];
                };
                const uint32_t local = (i - pair0) << 1U;
                const long long b = 2LL * static_cast<long long>(i) - 3LL;
                const double c0 = C(b + 0), c1 = C(b + 1), c2 = C(b + 2), c3 = C(b + 3),
                             c4 = C(b + 4), c5 = C(b + 5), c6 = C(b + 6), c7 = C(b + 7),
                             c8 = C(b + 8), c9 = C(b + 9), c10 = C(b + 10);
                const double e0 = double(IL0) * (c0 + c6) + double(IL1) * (c1 + c5) +
                                  double(IL2) * (c2 + c4) + double(IL3) * c3;
                const double o0 = double(IH0) * (c0 + c8) + double(IH1) * (c1 + c7) +
                                  double(IH2) * (c2 + c6) + double(IH3) * (c3 + c5) +
                                  double(IH4) * c4;
                rows[(local + 0U) * lx + x] = e0;
                if (local + 1U < out_rows)
                    rows[(local + 1U) * lx + x] = o0;
                if (i + 1U < pair0 + pairn) {
                    const double e1 = double(IL0) * (c2 + c8) + double(IL1) * (c3 + c7) +
                                      double(IL2) * (c4 + c6) + double(IL3) * c5;
                    const double o1 = double(IH0) * (c2 + c10) + double(IH1) * (c3 + c9) +
                                      double(IH2) * (c4 + c8) + double(IH3) * (c5 + c7) +
                                      double(IH4) * c6;
                    rows[(local + 2U) * lx + x] = e1;
                    if (local + 3U < out_rows)
                        rows[(local + 3U) * lx + x] = o1;
                }
                if (i + 2U < pair0 + pairn) {
                    const double c11 = C(b + 11), c12 = C(b + 12);
                    const double e2 = double(IL0) * (c4 + c10) + double(IL1) * (c5 + c9) +
                                      double(IL2) * (c6 + c8) + double(IL3) * c7;
                    const double o2 = double(IH0) * (c4 + c12) + double(IH1) * (c5 + c11) +
                                      double(IH2) * (c6 + c10) + double(IH3) * (c7 + c9) +
                                      double(IH4) * c8;
                    rows[(local + 4U) * lx + x] = e2;
                    if (local + 5U < out_rows)
                        rows[(local + 5U) * lx + x] = o2;
                    if (i + 3U < pair0 + pairn) {
                        const double c13 = C(b + 13), c14 = C(b + 14);
                        const double e3 = double(IL0) * (c6 + c12) +
                                          double(IL1) * (c7 + c11) +
                                          double(IL2) * (c8 + c10) + double(IL3) * c9;
                        const double o3 = double(IH0) * (c6 + c14) +
                                          double(IH1) * (c7 + c13) +
                                          double(IH2) * (c8 + c12) +
                                          double(IH3) * (c9 + c11) + double(IH4) * c10;
                        rows[(local + 6U) * lx + x] = e3;
                        if (local + 7U < out_rows)
                            rows[(local + 7U) * lx + x] = o3;
                    }
                }
            }
        }
        __syncthreads();

        const uint32_t xgroups = (low_lx + 1U) >> 1U;
        const uint32_t xwork = out_rows * xgroups;
        const uint32_t half = lx >> 1U;
        const uint32_t quad_hi = (half >= 4U) ? (half - 4U) : 0U;
        for (uint32_t e = threadIdx.x; e < xwork; e += KTPB) {
            const uint32_t r = e / xgroups;
            const uint32_t group = e - r * xgroups;
            const uint32_t i = group << 1U;
            const double* sh = rows + r * lx;
            double e0, o0, e1 = 0.0, o1 = 0.0;
            const bool have_second = i + 1U < low_lx;
            if (have_second && i >= 2U && i <= quad_hi) {
                const uint32_t hb = low_lx + i;
                const double c0 = sh[hb - 2U], c1 = sh[i - 1U], c2 = sh[hb - 1U],
                             c3 = sh[i], c4 = sh[hb], c5 = sh[i + 1U], c6 = sh[hb + 1U],
                             c7 = sh[i + 2U], c8 = sh[hb + 2U], c9 = sh[i + 3U],
                             c10 = sh[hb + 3U];
                e0 = double(IL0) * (c0 + c6) + double(IL1) * (c1 + c5) +
                     double(IL2) * (c2 + c4) + double(IL3) * c3;
                o0 = double(IH0) * (c0 + c8) + double(IH1) * (c1 + c7) +
                     double(IH2) * (c2 + c6) + double(IH3) * (c3 + c5) + double(IH4) * c4;
                e1 = double(IL0) * (c2 + c8) + double(IL1) * (c3 + c7) +
                     double(IL2) * (c4 + c6) + double(IL3) * c5;
                o1 = double(IH0) * (c2 + c10) + double(IH1) * (c3 + c9) +
                     double(IH2) * (c4 + c8) + double(IH3) * (c5 + c7) + double(IH4) * c6;
            } else {
                auto C = [&](long long idx) -> double {
                    const uint32_t pos = sym_idx(idx, last_x);
                    return sh[(pos & 1U) ? (low_lx + (pos >> 1U)) : (pos >> 1U)];
                };
                if (have_second)
                    synth_quad<double>(i, C, e0, o0, e1, o1);
                else
                    synth_pair<double>(i, C, e0, o0);
            }

            const uint32_t gy = 2U * pair0 + r;
            const size_t base = static_cast<size_t>(gy) * nx + static_cast<size_t>(z) * leap_z;
            const uint32_t ox = 2U * i;
            if constexpr (Mode == PWE::T) {
                const double de0 = e0 - orig[base + ox];
                if (fabs(de0) > bound) {
                    const unsigned int slot = atomicAdd(ocnt, 1U);
                    if (oidx != nullptr && slot < ocap) {
                        oidx[slot] = static_cast<uint32_t>(base + ox);
                        oerr[slot] = de0;
                    }
                }
                if (ox + 1U < lx) {
                    const double de = o0 - orig[base + ox + 1U];
                    if (fabs(de) > bound) {
                        const unsigned int slot = atomicAdd(ocnt, 1U);
                        if (oidx != nullptr && slot < ocap) {
                            oidx[slot] = static_cast<uint32_t>(base + ox + 1U);
                            oerr[slot] = de;
                        }
                    }
                }
                if (have_second && ox + 2U < lx) {
                    const double de = e1 - orig[base + ox + 2U];
                    if (fabs(de) > bound) {
                        const unsigned int slot = atomicAdd(ocnt, 1U);
                        if (oidx != nullptr && slot < ocap) {
                            oidx[slot] = static_cast<uint32_t>(base + ox + 2U);
                            oerr[slot] = de;
                        }
                    }
                }
                if (have_second && ox + 3U < lx) {
                    const double de = o1 - orig[base + ox + 3U];
                    if (fabs(de) > bound) {
                        const unsigned int slot = atomicAdd(ocnt, 1U);
                        if (oidx != nullptr && slot < ocap) {
                            oidx[slot] = static_cast<uint32_t>(base + ox + 3U);
                            oerr[slot] = de;
                        }
                    }
                }
            } else if (have_second && ox + 3U < lx && (nx & 1U) == 0U) {
                *reinterpret_cast<double2*>(dst + base + ox) = make_double2(e0, o0);
                *reinterpret_cast<double2*>(dst + base + ox + 2U) = make_double2(e1, o1);
            } else {
                dst[base + ox] = e0;
                if (ox + 1U < lx)
                    dst[base + ox + 1U] = o0;
                if (have_second && ox + 2U < lx)
                    dst[base + ox + 2U] = e1;
                if (have_second && ox + 3U < lx)
                    dst[base + ox + 3U] = o1;
            }
        }
        __syncthreads();
    }
}

// Axis-specialized normal kernels.  Keeping Z/Y/X in separate entry functions avoids
// the mega-kernel's union of live registers and lets Y run without the 32-KB shared
// allocation required only by X/Z staging. Stream ordering provides inter-kernel ordering.
template <typename T>
__global__ __launch_bounds__(TPB, 2) void idwt_3d_z_single_global(
    const T* src, T* dst, dim3 data_dims, dim3 chunk_dims) {
    idwt_z_pass<T>(src, dst, data_dims.x, data_dims.y, chunk_dims.x, chunk_dims.y, chunk_dims.z);
}

template <typename T>
__global__ __launch_bounds__(TPB,(std::is_same_v<T, float> ? 4 : CDF97_MIN_BLOCKS(T)))
    void idwt_3d_z_single_static(const T* src, T* dst, dim3 data_dims, dim3 chunk_dims, bool vector_z) {
    __shared__ alignas(128) T s_data[WALTZ::IDWT_SHARED_BYTES / sizeof(T)];
    idwt_z_level_shared<T>(src,dst, s_data, WALTZ::IDWT_SHARED_BYTES / sizeof(T),
    data_dims.x, data_dims.y, chunk_dims.x, chunk_dims.y, chunk_dims.z, vector_z);
}

__device__ __forceinline__ void idwt_pwe_record_float(float out, size_t g, const float* orig,
    double bound, float fast_bound, unsigned int* ocnt, uint32_t* oidx, float* oerr, uint32_t ocap);

// Float whole-row Y->X synthesis.  Y uses aligned float4 columns (or four-pair
// scalar chains), while X synthesizes two adjacent coefficient pairs per item.
template <PWE Mode>
__global__ __launch_bounds__(TPB, 3) void idwt_yx_float(const float* __restrict__ src,
    float* __restrict__ dst,dim3 data_dims,dim3 chunk_dims,uint32_t pairs,const float* __restrict__ orig,
    double bound, unsigned int* ocnt, uint32_t* oidx, float* oerr, uint32_t ocap) {
    __shared__ alignas(128) float rows[48U * 1024U / sizeof(float)];
    const uint32_t nx = data_dims.x;
    const uint32_t ny = data_dims.y;
    const uint32_t nz = chunk_dims.z;
    const uint32_t lx = chunk_dims.x;
    const uint32_t ly = chunk_dims.y;
    if (nx < 2U || ny < 2U || lx < 2U || ly < 2U || lx > nx || ly > ny || pairs == 0U)
        return;

    float pwe_fast_bound = 0.0f;
    if constexpr (Mode == PWE::T) {
        pwe_fast_bound =
            __uint_as_float(__float_as_uint(__double2float_rn(bound)) - 1U);
    }

    const size_t leap_z = static_cast<size_t>(nx) * ny;
    const uint32_t low_ly = ly - (ly >> 1U);
    const uint32_t low_lx = lx - (lx >> 1U);
    const long long last_y = static_cast<long long>(ly) - 1;
    const long long last_x = static_cast<long long>(lx) - 1;
    const uint32_t nty = (low_ly + pairs - 1U) / pairs;
    const uint32_t ntiles = nty * nz;

    for (uint32_t tile = blockIdx.x; tile < ntiles; tile += gridDim.x) {
        const uint32_t z = tile / nty;
        const uint32_t pt = tile - z * nty;
        const uint32_t pair0 = pt * pairs;
        const uint32_t pairn = min(pairs, low_ly - pair0);
        const uint32_t out_rows = min(2U * pairn, ly - 2U * pair0);
        const uint32_t groups = (pairn + 1U) >> 1U;
        const uint32_t chains = (groups + 1U) >> 1U;
        const uint32_t ywork = chains * lx;
        const bool vec_y = ((lx & 3U) == 0U) && ((nx & 3U) == 0U);
        if (vec_y) {
            const uint32_t vec_lx = lx >> 2U;
            const uint32_t ywork4 = groups * vec_lx;
            for (uint32_t e = threadIdx.x; e < ywork4; e += TPB) {
                const uint32_t vc = e % vec_lx;
                const uint32_t x = vc << 2U;
                const uint32_t group = e / vec_lx;
                const uint32_t i = pair0 + (group << 1U);
                const size_t xz = static_cast<size_t>(x) + static_cast<size_t>(z) * leap_z;
                auto C4 = [&](long long idx) -> float4 {
                    const uint32_t pos = sym_idx(idx, last_y);
                    const uint32_t sy = (pos & 1U) ? (low_ly + (pos >> 1U)) : (pos >> 1U);
                    return *reinterpret_cast<const float4*>(src + xz +
                                                            static_cast<size_t>(sy) * nx);
                };
                const uint32_t local = (i - pair0) << 1U;
                const long long b = 2LL * static_cast<long long>(i) - 3LL;
                const float4 c0 = C4(b + 0), c1 = C4(b + 1), c2 = C4(b + 2), c3 = C4(b + 3),
                             c4 = C4(b + 4), c5 = C4(b + 5), c6 = C4(b + 6), c7 = C4(b + 7),
                             c8 = C4(b + 8), c9 = C4(b + 9), c10 = C4(b + 10);
                float4 e0, o0, e1, o1;
                synth_quad4(c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, e0, o0, e1, o1);
                *reinterpret_cast<float4*>(rows + static_cast<size_t>(local + 0U) * lx + x) = e0;
                if (local + 1U < out_rows)
                    *reinterpret_cast<float4*>(rows + static_cast<size_t>(local + 1U) * lx + x) = o0;
                if (local + 2U < out_rows)
                    *reinterpret_cast<float4*>(rows + static_cast<size_t>(local + 2U) * lx + x) = e1;
                if (local + 3U < out_rows)
                    *reinterpret_cast<float4*>(rows + static_cast<size_t>(local + 3U) * lx + x) = o1;
            }
        } else {
            for (uint32_t e = threadIdx.x; e < ywork; e += TPB) {
                const uint32_t x = e % lx;
                const uint32_t chain = e / lx;
                const uint32_t i = pair0 + (chain << 2U);
                const size_t xz = x + static_cast<size_t>(z) * leap_z;
                auto C = [&](long long idx) -> float {
                    const uint32_t pos = sym_idx(idx, last_y);
                    const uint32_t sy = (pos & 1U) ? (low_ly + (pos >> 1U)) : (pos >> 1U);
                    return src[xz + static_cast<size_t>(sy) * nx];
                };
                const uint32_t local = (i - pair0) << 1U;
                const long long b = 2LL * static_cast<long long>(i) - 3LL;
                const float c0 = C(b + 0), c1 = C(b + 1), c2 = C(b + 2), c3 = C(b + 3),
                            c4 = C(b + 4), c5 = C(b + 5), c6 = C(b + 6), c7 = C(b + 7),
                            c8 = C(b + 8), c9 = C(b + 9), c10 = C(b + 10);
                const float e0 = float(IL0) * (c0 + c6) + float(IL1) * (c1 + c5) +
                                 float(IL2) * (c2 + c4) + float(IL3) * c3;
                const float o0 = float(IH0) * (c0 + c8) + float(IH1) * (c1 + c7) +
                                 float(IH2) * (c2 + c6) + float(IH3) * (c3 + c5) +
                                 float(IH4) * c4;
                rows[(local + 0U) * lx + x] = e0;
                if (local + 1U < out_rows)
                    rows[(local + 1U) * lx + x] = o0;
                if (i + 1U < pair0 + pairn) {
                    const float e1 = float(IL0) * (c2 + c8) + float(IL1) * (c3 + c7) +
                                     float(IL2) * (c4 + c6) + float(IL3) * c5;
                    const float o1 = float(IH0) * (c2 + c10) + float(IH1) * (c3 + c9) +
                                     float(IH2) * (c4 + c8) + float(IH3) * (c5 + c7) +
                                     float(IH4) * c6;
                    rows[(local + 2U) * lx + x] = e1;
                    if (local + 3U < out_rows)
                        rows[(local + 3U) * lx + x] = o1;
                }
                if (i + 2U < pair0 + pairn) {
                    const float c11 = C(b + 11), c12 = C(b + 12);
                    const float e2 = float(IL0) * (c4 + c10) + float(IL1) * (c5 + c9) +
                                     float(IL2) * (c6 + c8) + float(IL3) * c7;
                    const float o2 = float(IH0) * (c4 + c12) + float(IH1) * (c5 + c11) +
                                     float(IH2) * (c6 + c10) + float(IH3) * (c7 + c9) +
                                     float(IH4) * c8;
                    rows[(local + 4U) * lx + x] = e2;
                    if (local + 5U < out_rows)
                        rows[(local + 5U) * lx + x] = o2;
                    if (i + 3U < pair0 + pairn) {
                        const float c13 = C(b + 13), c14 = C(b + 14);
                        const float e3 = float(IL0) * (c6 + c12) + float(IL1) * (c7 + c11) +
                                         float(IL2) * (c8 + c10) + float(IL3) * c9;
                        const float o3 = float(IH0) * (c6 + c14) + float(IH1) * (c7 + c13) +
                                         float(IH2) * (c8 + c12) + float(IH3) * (c9 + c11) +
                                         float(IH4) * c10;
                        rows[(local + 6U) * lx + x] = e3;
                        if (local + 7U < out_rows)
                            rows[(local + 7U) * lx + x] = o3;
                    }
                }
            }
        }
        __syncthreads();

        const uint32_t xgroups = (low_lx + 1U) >> 1U;
        const uint32_t xwork = out_rows * xgroups;
        const uint32_t half = lx >> 1U;
        const uint32_t quad_hi = (half >= 4U) ? (half - 4U) : 0U;
        for (uint32_t e = threadIdx.x; e < xwork; e += TPB) {
            const uint32_t r = e / xgroups;
            const uint32_t group = e - r * xgroups;
            const uint32_t i = group << 1U;
            const float* sh = rows + r * lx;
            float e0, o0, e1 = 0.0f, o1 = 0.0f;
            const bool have_second = i + 1U < low_lx;
            if (have_second && i >= 2U && i <= quad_hi) {
                const uint32_t hb = low_lx + i;
                const float c0 = sh[hb - 2U], c1 = sh[i - 1U], c2 = sh[hb - 1U], c3 = sh[i],
                            c4 = sh[hb], c5 = sh[i + 1U], c6 = sh[hb + 1U], c7 = sh[i + 2U],
                            c8 = sh[hb + 2U], c9 = sh[i + 3U], c10 = sh[hb + 3U];
                e0 = float(IL0) * (c0 + c6) + float(IL1) * (c1 + c5) + float(IL2) * (c2 + c4) +
                     float(IL3) * c3;
                o0 = float(IH0) * (c0 + c8) + float(IH1) * (c1 + c7) + float(IH2) * (c2 + c6) +
                     float(IH3) * (c3 + c5) + float(IH4) * c4;
                e1 = float(IL0) * (c2 + c8) + float(IL1) * (c3 + c7) + float(IL2) * (c4 + c6) +
                     float(IL3) * c5;
                o1 = float(IH0) * (c2 + c10) + float(IH1) * (c3 + c9) + float(IH2) * (c4 + c8) +
                     float(IH3) * (c5 + c7) + float(IH4) * c6;
            } else {
                auto C = [&](long long idx) -> float {
                    const uint32_t pos = sym_idx(idx, last_x);
                    return sh[(pos & 1U) ? (low_lx + (pos >> 1U)) : (pos >> 1U)];
                };
                if (have_second)
                    synth_quad<float>(i, C, e0, o0, e1, o1);
                else
                    synth_pair<float>(i, C, e0, o0);
            }
            const uint32_t gy = 2U * pair0 + r;
            const size_t base = static_cast<size_t>(gy) * nx + static_cast<size_t>(z) * leap_z;
            const uint32_t ox = 2U * i;
            if constexpr (Mode == PWE::T) {
                idwt_pwe_record_float(
                    e0, base + ox, orig, bound, pwe_fast_bound, ocnt, oidx, oerr, ocap);
                if (ox + 1U < lx)
                    idwt_pwe_record_float(
                        o0, base + ox + 1U, orig, bound, pwe_fast_bound, ocnt, oidx, oerr, ocap);
                if (have_second && ox + 2U < lx)
                    idwt_pwe_record_float(
                        e1, base + ox + 2U, orig, bound, pwe_fast_bound, ocnt, oidx, oerr, ocap);
                if (have_second && ox + 3U < lx)
                    idwt_pwe_record_float(
                        o1, base + ox + 3U, orig, bound, pwe_fast_bound, ocnt, oidx, oerr, ocap);
            } else {
                if (have_second && ox + 3U < lx && (nx & 3U) == 0U) {
                    *reinterpret_cast<float4*>(dst + base + ox) = make_float4(e0, o0, e1, o1);
                } else {
                    dst[base + ox] = e0;
                    if (ox + 1U < lx)
                        dst[base + ox + 1U] = o0;
                    if (have_second && ox + 2U < lx)
                        dst[base + ox + 2U] = e1;
                    if (have_second && ox + 3U < lx)
                        dst[base + ox + 3U] = o1;
                }
            }
        }
        __syncthreads();
    }
}

// Whole-row Y->X plane inverse with optional peripheral coefficient copy.
// Keep each precision's storage, synthesis and copy strategy specialized.
template <typename T, PWE Mode = PWE::F, int KTPB = 256>
__global__ void idwt_yx_plane(const T* __restrict__ src, T* __restrict__ dst,
    dim3 data_dims, dim3 chunk_dims, uint32_t pairs, uint32_t next_lx, uint32_t next_ly, bool carry,
    const T* __restrict__ orig = nullptr, double bound = 0.0, unsigned int* ocnt = nullptr, uint32_t* oidx = nullptr,
    T* oerr = nullptr, uint32_t ocap = 0U, bool vector_y = false) {
    if constexpr (std::is_same_v<T, float>) {
        __shared__ alignas(128) float rows[32U * 1024U / sizeof(float)];
        const uint32_t nx = data_dims.x;
        const uint32_t ny = data_dims.y;
        const uint32_t nz = chunk_dims.z;
        const uint32_t lx = chunk_dims.x;
        const uint32_t ly = chunk_dims.y;
        if (nx < 2U || ny < 2U || lx < 2U || ly < 2U || lx > nx || ly > ny || pairs == 0U)
            return;

        float pwe_fast_bound = 0.0f;
        if constexpr (Mode == PWE::T) {
            pwe_fast_bound =
                __uint_as_float(__float_as_uint(__double2float_rn(bound)) - 1U);
        }

        const size_t leap_z = static_cast<size_t>(nx) * ny;
        const uint32_t low_ly = ly - (ly >> 1U);
        const uint32_t low_lx = lx - (lx >> 1U);
        const long long last_y = static_cast<long long>(ly) - 1;
        const long long last_x = static_cast<long long>(lx) - 1;
        const uint32_t nty = (low_ly + pairs - 1U) / pairs;
        const uint32_t ntiles = nty * nz;

        for (uint32_t tile = blockIdx.x; tile < ntiles; tile += gridDim.x) {
            const uint32_t z = tile / nty;
            const uint32_t pt = tile - z * nty;
            const uint32_t pair0 = pt * pairs;
            const uint32_t pairn = min(pairs, low_ly - pair0);
            const uint32_t out_rows = min(2U * pairn, ly - 2U * pair0);
            const uint32_t groups = (pairn + 1U) >> 1U;
            const uint32_t chains = (groups + 1U) >> 1U;
            const uint32_t ywork = chains * lx;
            const bool vec_y = ((lx & 3U) == 0U) && ((nx & 3U) == 0U);
            if (vec_y) {
                const uint32_t vec_lx = lx >> 2U;
                const uint32_t ywork4 = groups * vec_lx;
                for (uint32_t e = threadIdx.x; e < ywork4; e += KTPB) {
                    const uint32_t vc = e % vec_lx;
                    const uint32_t x = vc << 2U;
                    const uint32_t group = e / vec_lx;
                    const uint32_t i = pair0 + (group << 1U);
                    const size_t xz = static_cast<size_t>(x) + static_cast<size_t>(z) * leap_z;
                    auto C4 = [&](long long idx) -> float4 {
                        const uint32_t pos = sym_idx(idx, last_y);
                        const uint32_t sy = (pos & 1U) ? (low_ly + (pos >> 1U)) : (pos >> 1U);
                        return *reinterpret_cast<const float4*>(src + xz +
                                                                static_cast<size_t>(sy) * nx);
                    };
                    const uint32_t local = (i - pair0) << 1U;
                    const long long b = 2LL * static_cast<long long>(i) - 3LL;
                    const float4 c0 = C4(b + 0), c1 = C4(b + 1), c2 = C4(b + 2), c3 = C4(b + 3),
                                 c4 = C4(b + 4), c5 = C4(b + 5), c6 = C4(b + 6), c7 = C4(b + 7),
                                 c8 = C4(b + 8), c9 = C4(b + 9), c10 = C4(b + 10);
                    float4 e0, o0, e1, o1;
                    synth_quad4(c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, e0, o0, e1, o1);
                    *reinterpret_cast<float4*>(rows + static_cast<size_t>(local + 0U) * lx + x) = e0;
                    if (local + 1U < out_rows)
                        *reinterpret_cast<float4*>(rows + static_cast<size_t>(local + 1U) * lx + x) =
                            o0;
                    if (local + 2U < out_rows)
                        *reinterpret_cast<float4*>(rows + static_cast<size_t>(local + 2U) * lx + x) =
                            e1;
                    if (local + 3U < out_rows)
                        *reinterpret_cast<float4*>(rows + static_cast<size_t>(local + 3U) * lx + x) =
                            o1;
                }
            } else {
                for (uint32_t e = threadIdx.x; e < ywork; e += KTPB) {
                    const uint32_t x = e % lx;
                    const uint32_t chain = e / lx;
                    const uint32_t i = pair0 + (chain << 2U);
                    const size_t xz = x + static_cast<size_t>(z) * leap_z;
                    auto C = [&](long long idx) -> float {
                        const uint32_t pos = sym_idx(idx, last_y);
                        const uint32_t sy = (pos & 1U) ? (low_ly + (pos >> 1U)) : (pos >> 1U);
                        return src[xz + static_cast<size_t>(sy) * nx];
                    };
                    const uint32_t local = (i - pair0) << 1U;
                    const long long b = 2LL * static_cast<long long>(i) - 3LL;
                    const float c0 = C(b + 0), c1 = C(b + 1), c2 = C(b + 2), c3 = C(b + 3),
                                c4 = C(b + 4), c5 = C(b + 5), c6 = C(b + 6), c7 = C(b + 7),
                                c8 = C(b + 8), c9 = C(b + 9), c10 = C(b + 10);
                    const float e0 = float(IL0) * (c0 + c6) + float(IL1) * (c1 + c5) +
                                     float(IL2) * (c2 + c4) + float(IL3) * c3;
                    const float o0 = float(IH0) * (c0 + c8) + float(IH1) * (c1 + c7) +
                                     float(IH2) * (c2 + c6) + float(IH3) * (c3 + c5) +
                                     float(IH4) * c4;
                    rows[(local + 0U) * lx + x] = e0;
                    if (local + 1U < out_rows)
                        rows[(local + 1U) * lx + x] = o0;
                    if (i + 1U < pair0 + pairn) {
                        const float e1 = float(IL0) * (c2 + c8) + float(IL1) * (c3 + c7) +
                                         float(IL2) * (c4 + c6) + float(IL3) * c5;
                        const float o1 = float(IH0) * (c2 + c10) + float(IH1) * (c3 + c9) +
                                         float(IH2) * (c4 + c8) + float(IH3) * (c5 + c7) +
                                         float(IH4) * c6;
                        rows[(local + 2U) * lx + x] = e1;
                        if (local + 3U < out_rows)
                            rows[(local + 3U) * lx + x] = o1;
                    }
                    if (i + 2U < pair0 + pairn) {
                        const float c11 = C(b + 11), c12 = C(b + 12);
                        const float e2 = float(IL0) * (c4 + c10) + float(IL1) * (c5 + c9) +
                                         float(IL2) * (c6 + c8) + float(IL3) * c7;
                        const float o2 = float(IH0) * (c4 + c12) + float(IH1) * (c5 + c11) +
                                         float(IH2) * (c6 + c10) + float(IH3) * (c7 + c9) +
                                         float(IH4) * c8;
                        rows[(local + 4U) * lx + x] = e2;
                        if (local + 5U < out_rows)
                            rows[(local + 5U) * lx + x] = o2;
                        if (i + 3U < pair0 + pairn) {
                            const float c13 = C(b + 13), c14 = C(b + 14);
                            const float e3 = float(IL0) * (c6 + c12) + float(IL1) * (c7 + c11) +
                                             float(IL2) * (c8 + c10) + float(IL3) * c9;
                            const float o3 = float(IH0) * (c6 + c14) + float(IH1) * (c7 + c13) +
                                             float(IH2) * (c8 + c12) + float(IH3) * (c9 + c11) +
                                             float(IH4) * c10;
                            rows[(local + 6U) * lx + x] = e3;
                            if (local + 7U < out_rows)
                                rows[(local + 7U) * lx + x] = o3;
                        }
                    }
                }
            }
            __syncthreads();

            const uint32_t xgroups = (low_lx + 1U) >> 1U;
            const uint32_t xwork = out_rows * xgroups;
            const uint32_t half = lx >> 1U;
            const uint32_t quad_hi = (half >= 4U) ? (half - 4U) : 0U;
            for (uint32_t e = threadIdx.x; e < xwork; e += KTPB) {
                const uint32_t r = e / xgroups;
                const uint32_t group = e - r * xgroups;
                const uint32_t i = group << 1U;
                const float* sh = rows + r * lx;
                float e0, o0, e1 = 0.0f, o1 = 0.0f;
                const bool have_second = i + 1U < low_lx;
                if (have_second && i >= 2U && i <= quad_hi) {
                    const uint32_t hb = low_lx + i;
                    const float c0 = sh[hb - 2U], c1 = sh[i - 1U], c2 = sh[hb - 1U], c3 = sh[i],
                                c4 = sh[hb], c5 = sh[i + 1U], c6 = sh[hb + 1U], c7 = sh[i + 2U],
                                c8 = sh[hb + 2U], c9 = sh[i + 3U], c10 = sh[hb + 3U];
                    e0 = float(IL0) * (c0 + c6) + float(IL1) * (c1 + c5) + float(IL2) * (c2 + c4) +
                         float(IL3) * c3;
                    o0 = float(IH0) * (c0 + c8) + float(IH1) * (c1 + c7) + float(IH2) * (c2 + c6) +
                         float(IH3) * (c3 + c5) + float(IH4) * c4;
                    e1 = float(IL0) * (c2 + c8) + float(IL1) * (c3 + c7) + float(IL2) * (c4 + c6) +
                         float(IL3) * c5;
                    o1 = float(IH0) * (c2 + c10) + float(IH1) * (c3 + c9) + float(IH2) * (c4 + c8) +
                         float(IH3) * (c5 + c7) + float(IH4) * c6;
                } else {
                    auto C = [&](long long idx) -> float {
                        const uint32_t pos = sym_idx(idx, last_x);
                        return sh[(pos & 1U) ? (low_lx + (pos >> 1U)) : (pos >> 1U)];
                    };
                    if (have_second)
                        synth_quad<float>(i, C, e0, o0, e1, o1);
                    else
                        synth_pair<float>(i, C, e0, o0);
                }
                const uint32_t gy = 2U * pair0 + r;
                const size_t base = static_cast<size_t>(gy) * nx + static_cast<size_t>(z) * leap_z;
                const uint32_t ox = 2U * i;
                if constexpr (Mode == PWE::T) {
                    idwt_pwe_record_float(
                        e0, base + ox, orig, bound, pwe_fast_bound, ocnt, oidx, oerr, ocap);
                    if (ox + 1U < lx)
                        idwt_pwe_record_float(
                            o0, base + ox + 1U, orig, bound, pwe_fast_bound, ocnt, oidx, oerr, ocap);
                    if (have_second && ox + 2U < lx)
                        idwt_pwe_record_float(
                            e1, base + ox + 2U, orig, bound, pwe_fast_bound, ocnt, oidx, oerr, ocap);
                    if (have_second && ox + 3U < lx)
                        idwt_pwe_record_float(
                            o1, base + ox + 3U, orig, bound, pwe_fast_bound, ocnt, oidx, oerr, ocap);
                } else {
                    if (have_second && ox + 3U < lx && (nx & 3U) == 0U) {
                        *reinterpret_cast<float4*>(dst + base + ox) = make_float4(e0, o0, e1, o1);
                    } else {
                        dst[base + ox] = e0;
                        if (ox + 1U < lx)
                            dst[base + ox + 1U] = o0;
                        if (have_second && ox + 2U < lx)
                            dst[base + ox + 2U] = e1;
                        if (have_second && ox + 3U < lx)
                            dst[base + ox + 3U] = o1;
                    }
                }
            }
            __syncthreads();
        }
        // Forward only the disjoint annulus needed by the next (larger) rectangle.
        // Input and output buffers must not alias. PWE::T preserves the original
        // no-active-write behavior, but still copies this annulus when carry is enabled
        // and dst is valid. Plane intermediates should use PWE::F.
        if (carry && dst != nullptr && next_lx >= lx && next_ly >= ly && next_lx <= nx &&
            next_ly <= ny) {
            const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
            const size_t first = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            // Copy horizontal and vertical bands separately so they are disjoint. Aligned
            // rows use float4 transactions, with scalar tails for odd/non-vector widths.
            const uint32_t dx = next_lx - lx;
            if (dx != 0U && ly != 0U) {
                const bool vec_ok = ((nx & 3U) == 0U) && ((lx & 3U) == 0U) && ((dx & 3U) == 0U);
                const size_t rows_h = static_cast<size_t>(nz) * ly;
                if (vec_ok) {
                    const uint32_t nv = dx >> 2U;
                    const size_t totalv = rows_h * nv;
                    for (size_t e = first; e < totalv; e += stride) {
                        const uint32_t vc = static_cast<uint32_t>(e % nv);
                        const size_t q = e / nv;
                        const uint32_t y = static_cast<uint32_t>(q % ly);
                        const uint32_t z = static_cast<uint32_t>(q / ly);
                        const size_t off =
                            static_cast<size_t>(z) * leap_z + static_cast<size_t>(y) * nx + lx;
                        reinterpret_cast<float4*>(dst + off)[vc] =
                            reinterpret_cast<const float4*>(src + off)[vc];
                    }
                } else {
                    const size_t total = rows_h * dx;
                    for (size_t e = first; e < total; e += stride) {
                        const uint32_t x = static_cast<uint32_t>(e % dx);
                        const size_t q = e / dx;
                        const uint32_t y = static_cast<uint32_t>(q % ly);
                        const uint32_t z = static_cast<uint32_t>(q / ly);
                        const size_t off =
                            static_cast<size_t>(z) * leap_z + static_cast<size_t>(y) * nx + lx + x;
                        dst[off] = src[off];
                    }
                }
            }
            if (next_ly > ly) {
                const uint32_t dy = next_ly - ly;
                const size_t rows_v = static_cast<size_t>(nz) * dy;
                const uint32_t nv = next_lx >> 2U;
                const bool vec_ok = ((nx & 3U) == 0U) && (nv != 0U);
                if (vec_ok) {
                    const size_t totalv = rows_v * nv;
                    for (size_t e = first; e < totalv; e += stride) {
                        const uint32_t vc = static_cast<uint32_t>(e % nv);
                        const size_t q = e / nv;
                        const uint32_t y = ly + static_cast<uint32_t>(q % dy);
                        const uint32_t z = static_cast<uint32_t>(q / dy);
                        const size_t off =
                            static_cast<size_t>(z) * leap_z + static_cast<size_t>(y) * nx;
                        reinterpret_cast<float4*>(dst + off)[vc] =
                            reinterpret_cast<const float4*>(src + off)[vc];
                    }
                    const uint32_t tail = next_lx & 3U;
                    if (tail != 0U) {
                        const size_t totalt = rows_v * tail;
                        for (size_t e = first; e < totalt; e += stride) {
                            const uint32_t x = (nv << 2U) + static_cast<uint32_t>(e % tail);
                            const size_t q = e / tail;
                            const uint32_t y = ly + static_cast<uint32_t>(q % dy);
                            const uint32_t z = static_cast<uint32_t>(q / dy);
                            const size_t off =
                                static_cast<size_t>(z) * leap_z + static_cast<size_t>(y) * nx + x;
                            dst[off] = src[off];
                        }
                    }
                } else {
                    const size_t total = rows_v * next_lx;
                    for (size_t e = first; e < total; e += stride) {
                        const uint32_t x = static_cast<uint32_t>(e % next_lx);
                        const size_t q = e / next_lx;
                        const uint32_t y = ly + static_cast<uint32_t>(q % dy);
                        const uint32_t z = static_cast<uint32_t>(q / dy);
                        const size_t off =
                            static_cast<size_t>(z) * leap_z + static_cast<size_t>(y) * nx + x;
                        dst[off] = src[off];
                    }
                }
            }
        }
    } else if constexpr (std::is_same_v<T, double>) {
        static __shared__ __align__(128) double yx_double_rows[48U * 1024U / sizeof(double)];
        const uint32_t nx = data_dims.x;
        const uint32_t ny = data_dims.y;
        const uint32_t nz = chunk_dims.z;
        const uint32_t lx = chunk_dims.x;
        const uint32_t ly = chunk_dims.y;
        const uint32_t lz = chunk_dims.z;
        idwt_yx_tiled_double<PWE::F, KTPB>(src,
                             dst,
                             yx_double_rows,
                             2U * pairs * lx,
                             nx,
                             ny,
                             lx,
                             ly,
                             lz,
                             nullptr,
                             0.0,
                             nullptr,
                             nullptr,
                             nullptr,
                             0,
                             vector_y);
        if (!carry)
            return;

        const size_t leap_z = static_cast<size_t>(nx) * ny;
        const size_t right_width = next_lx - lx;
        const size_t right_count = right_width * ly * nz;
        const size_t bottom_height = next_ly - ly;
        const size_t bottom_count = static_cast<size_t>(next_lx) * bottom_height * nz;
        const size_t total = right_count + bottom_count;
        const size_t first = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
        const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
        const bool aligned = ((nx | lx | next_lx) & 1U) == 0U &&
            ((reinterpret_cast<uintptr_t>(src) | reinterpret_cast<uintptr_t>(dst)) & 15U) == 0U;
        if (aligned && total >= stride && total <= UINT32_MAX) {
            #pragma unroll
            for (int band = 0; band < 2; ++band) {
                const uint32_t width = band == 0 ? static_cast<uint32_t>(right_width) : next_lx;
                const uint32_t height = band == 0 ? ly : static_cast<uint32_t>(bottom_height);
                const uint32_t x0 = band == 0 ? lx : 0U;
                const uint32_t y0 = band == 0 ? 0U : ly;
                const uint32_t pairs_x = width >> 1U;
                const size_t count = static_cast<size_t>(pairs_x) * height * nz;
                for (size_t item = first; item < count; item += stride) {
                    const uint32_t i = static_cast<uint32_t>(item);
                    const uint32_t x = (i % pairs_x) << 1U;
                    const uint32_t q = i / pairs_x;
                    const uint32_t y = q % height;
                    const uint32_t z = q / height;
                    const size_t offset = static_cast<size_t>(z) * leap_z + static_cast<size_t>(y0 + y) * nx + x0 + x;
                    *reinterpret_cast<double2*>(dst + offset) = *reinterpret_cast<const double2*>(src + offset);
                }
            }
            return;
        }

        if (total <= UINT32_MAX) {
            const uint32_t right = static_cast<uint32_t>(right_count);
            const uint32_t width = static_cast<uint32_t>(right_width);
            const uint32_t height = static_cast<uint32_t>(bottom_height);
            for (size_t item = first; item < total; item += stride) {
                const uint32_t i = static_cast<uint32_t>(item);
                uint32_t x, y, z;
                if (i < right) {
                    x = lx + i % width;
                    const uint32_t q = i / width;
                    y = q % ly;
                    z = q / ly;
                } else {
                    const uint32_t j = i - right;
                    x = j % next_lx;
                    const uint32_t q = j / next_lx;
                    y = ly + q % height;
                    z = q / height;
                }
                const size_t offset = static_cast<size_t>(z) * leap_z + static_cast<size_t>(y) * nx + x;
                dst[offset] = src[offset];
            }
            return;
        }

        for (size_t i = first; i < total; i += stride) {
            size_t offset;
            if (i < right_count) {
                const uint32_t x = lx + static_cast<uint32_t>(i % right_width);
                const size_t q = i / right_width;
                const uint32_t y = static_cast<uint32_t>(q % ly);
                const uint32_t z = static_cast<uint32_t>(q / ly);
                offset = x + static_cast<size_t>(y) * nx + static_cast<size_t>(z) * leap_z;
            } else {
                const size_t j = i - right_count;
                const uint32_t x = static_cast<uint32_t>(j % next_lx);
                const size_t q = j / next_lx;
                const uint32_t y = ly + static_cast<uint32_t>(q % bottom_height);
                const uint32_t z = static_cast<uint32_t>(q / bottom_height);
                offset = x + static_cast<size_t>(y) * nx + static_cast<size_t>(z) * leap_z;
            }
            dst[offset] = src[offset];
        }
    }
}

// X pair helper for the halo fallback.  Keep the first coefficient product as an
// independently rounded multiply so nvcc cannot choose the opposite initial
// FMA seed for the halo fallback.  The common synth_pair remains unchanged for all
// other kernels and paths.
template <typename F>
__device__ __forceinline__ void idwt_yx_halo_synth_pair(uint32_t i,
                                                        F C,
                                                        float& out_even,
                                                        float& out_odd) {
    const long long b = 2 * static_cast<long long>(i) - 3;
    const float c0 = C(b), c1 = C(b + 1), c2 = C(b + 2), c3 = C(b + 3), c4 = C(b + 4),
                c5 = C(b + 5), c6 = C(b + 6), c7 = C(b + 7), c8 = C(b + 8);
    const float even0 = __fmul_rn(float(IL0), c0 + c6);
    out_even = even0 + float(IL1) * (c1 + c5) + float(IL2) * (c2 + c4) + float(IL3) * c3;
    const float odd0 = __fmul_rn(float(IH0), c0 + c8);
    out_odd = odd0 + float(IH1) * (c1 + c7) + float(IH2) * (c2 + c6) +
              float(IH3) * (c3 + c5) + float(IH4) * c4;
}

// Two neighbouring X coefficient pairs share seven of their eleven logical
// taps.  Load that union once while preserving the exact expression order of
// two idwt_yx_halo_synth_pair calls.
template <typename F>
__device__ __forceinline__ void idwt_yx_halo_synth_quad(uint32_t i,
                                                        F C,
                                                        float& even0,
                                                        float& odd0,
                                                        float& even1,
                                                        float& odd1) {
    const long long b = 2 * static_cast<long long>(i) - 3;
    const float c0 = C(b), c1 = C(b + 1), c2 = C(b + 2), c3 = C(b + 3), c4 = C(b + 4),
                c5 = C(b + 5), c6 = C(b + 6), c7 = C(b + 7), c8 = C(b + 8),
                c9 = C(b + 9), c10 = C(b + 10);
    const float e0 = __fmul_rn(float(IL0), c0 + c6);
    even0 = e0 + float(IL1) * (c1 + c5) + float(IL2) * (c2 + c4) + float(IL3) * c3;
    const float o0 = __fmul_rn(float(IH0), c0 + c8);
    odd0 = o0 + float(IH1) * (c1 + c7) + float(IH2) * (c2 + c6) +
           float(IH3) * (c3 + c5) + float(IH4) * c4;
    const float e1 = __fmul_rn(float(IL0), c2 + c8);
    even1 = e1 + float(IL1) * (c3 + c7) + float(IL2) * (c4 + c6) + float(IL3) * c5;
    const float o1 = __fmul_rn(float(IH0), c2 + c10);
    odd1 = o1 + float(IH1) * (c3 + c9) + float(IH2) * (c4 + c8) +
           float(IH3) * (c5 + c7) + float(IH4) * c6;
}

// Wide-X halo fallback: synthesize a small Y->X tile without materializing the full
// Y result between the two axes.  Shared rows use natural/interleaved owner
// coordinates, with four owner positions of X halo on each side.  Thus the X
// tap window uses the same synthesis math as the other inverse paths while the global
// Y reads retain the same reflected low|high source mapping.
template <uint32_t BX, uint32_t BY>
__global__ __launch_bounds__(TPB, 3) void idwt_yx_halo(
    const float* __restrict__ src,
    float* __restrict__ dst,
    dim3 data_dims,
    dim3 chunk_dims,
    const float* __restrict__ orig,
    double bound,
    unsigned int* ocnt,
    uint32_t* oidx,
    float* oerr,
    uint32_t ocap) {
    static_assert(BX >= 2U && (BX & 1U) == 0U,
                  "idwt_yx_halo BX must be a positive even output width");
    static_assert(BY >= 2U && (BY & 1U) == 0U,
                  "idwt_yx_halo BY must be a positive even output height");
    constexpr int XHALO = 4;
    constexpr uint32_t SHARED_X = BX + 2U * static_cast<uint32_t>(XHALO);
    extern __shared__ __align__(128) unsigned char halo_storage[];
    float* const rows = reinterpret_cast<float*>(halo_storage);

    const uint32_t nx = data_dims.x;
    const uint32_t ny = data_dims.y;
    const uint32_t lx = chunk_dims.x;
    const uint32_t ly = chunk_dims.y;
    const uint32_t lz = chunk_dims.z;

    if (lx == 0U || ly == 0U || nx == 0U || ny == 0U || lz == 0U)
        return;

    const size_t leap_z = static_cast<size_t>(nx) * ny;
    const uint32_t low_lx = lx - (lx >> 1U);
    const uint32_t low_ly = ly - (ly >> 1U);
    const int last_x = static_cast<int>(lx) - 1;
    const int last_y = static_cast<int>(ly) - 1;
    const uint32_t xtiles = (lx + BX - 1U) / BX;
    const uint32_t ytiles = (ly + BY - 1U) / BY;
    const size_t tiles_xy = static_cast<size_t>(xtiles) * ytiles;
    const size_t tile_count = tiles_xy * lz;

    // A capped launch can process several logical tiles per block; the final
    // barrier is required before rows[] is reused by the next tile.
    for (size_t tile = static_cast<size_t>(blockIdx.x); tile < tile_count;
         tile += static_cast<size_t>(gridDim.x)) {
        const uint32_t z = static_cast<uint32_t>(tile / tiles_xy);
        const size_t rem = tile - static_cast<size_t>(z) * tiles_xy;
        const uint32_t ty = static_cast<uint32_t>(rem / xtiles);
        const uint32_t tx = static_cast<uint32_t>(rem - static_cast<size_t>(ty) * xtiles);
        const uint32_t x0 = tx * BX;
        const uint32_t y0 = ty * BY;
        const uint32_t out_x = min(BX, lx - x0);
        const uint32_t out_y = min(BY, ly - y0);
        const uint32_t xpair0 = x0 >> 1U;
        const uint32_t xpairn = (out_x + 1U) >> 1U;
        const uint32_t ypair0 = y0 >> 1U;
        const uint32_t ypairn = (out_y + 1U) >> 1U;
        const uint32_t out_rows = min(2U * ypairn, ly - y0);

        // Each X output pair reads logical owners [2*i-3, 2*i+5].
        const int owner0 = static_cast<int>(xpair0 << 1U) - XHALO;
        const uint32_t stage_x = (xpairn << 1U) + 2U * static_cast<uint32_t>(XHALO);

        // Y synthesis.  Stage only the owner interval needed by this X tile;
        // reflected owners map back to the deinterleaved source columns.
        const uint32_t groups = (ypairn + 1U) >> 1U;
        const uint32_t chains = (groups + 1U) >> 1U;
        const uint32_t ywork = chains * stage_x;
        for (uint32_t e = threadIdx.x; e < ywork; e += TPB) {
            const uint32_t col = e % stage_x;
            const uint32_t chain = e / stage_x;
            const uint32_t i = ypair0 + (chain << 2U);
            const int owner = owner0 + static_cast<int>(col);
            const uint32_t mapped = lx == 1U ? 0U : sym_idx32(owner, last_x);
            const uint32_t sx = (mapped & 1U) ? (low_lx + (mapped >> 1U))
                                               : (mapped >> 1U);
            const size_t xz = static_cast<size_t>(z) * leap_z + sx;
            auto C = [&](long long idx) -> float {
                const uint32_t pos = ly == 1U ? 0U : sym_idx32(static_cast<int>(idx), last_y);
                const uint32_t sy = (pos & 1U) ? (low_ly + (pos >> 1U)) : (pos >> 1U);
                return src[xz + static_cast<size_t>(sy) * nx];
            };
            const uint32_t local = (i - ypair0) << 1U;
            if (ly == 1U) {
                rows[static_cast<size_t>(local) * SHARED_X + col] = src[xz];
            } else {
                const long long b = 2LL * static_cast<long long>(i) - 3LL;
                const float c0 = C(b + 0), c1 = C(b + 1), c2 = C(b + 2), c3 = C(b + 3),
                            c4 = C(b + 4), c5 = C(b + 5), c6 = C(b + 6), c7 = C(b + 7),
                            c8 = C(b + 8), c9 = C(b + 9), c10 = C(b + 10);
                const float e0 = float(IL0) * (c0 + c6) + float(IL1) * (c1 + c5) +
                                 float(IL2) * (c2 + c4) + float(IL3) * c3;
                const float o0 = float(IH0) * (c0 + c8) + float(IH1) * (c1 + c7) +
                                 float(IH2) * (c2 + c6) + float(IH3) * (c3 + c5) +
                                 float(IH4) * c4;
                rows[(static_cast<size_t>(local) + 0U) * SHARED_X + col] = e0;
                if (local + 1U < out_rows)
                    rows[(static_cast<size_t>(local) + 1U) * SHARED_X + col] = o0;
                if (i + 1U < ypair0 + ypairn) {
                    const float e1 = float(IL0) * (c2 + c8) + float(IL1) * (c3 + c7) +
                                     float(IL2) * (c4 + c6) + float(IL3) * c5;
                    const float o1 = float(IH0) * (c2 + c10) + float(IH1) * (c3 + c9) +
                                     float(IH2) * (c4 + c8) + float(IH3) * (c5 + c7) +
                                     float(IH4) * c6;
                    rows[(static_cast<size_t>(local) + 2U) * SHARED_X + col] = e1;
                    if (local + 3U < out_rows)
                        rows[(static_cast<size_t>(local) + 3U) * SHARED_X + col] = o1;
                }
                if (i + 2U < ypair0 + ypairn) {
                    const float c11 = C(b + 11), c12 = C(b + 12);
                    const float e2 = float(IL0) * (c4 + c10) + float(IL1) * (c5 + c9) +
                                     float(IL2) * (c6 + c8) + float(IL3) * c7;
                    const float o2 = float(IH0) * (c4 + c12) + float(IH1) * (c5 + c11) +
                                     float(IH2) * (c6 + c10) + float(IH3) * (c7 + c9) +
                                     float(IH4) * c8;
                    rows[(static_cast<size_t>(local) + 4U) * SHARED_X + col] = e2;
                    if (local + 5U < out_rows)
                        rows[(static_cast<size_t>(local) + 5U) * SHARED_X + col] = o2;
                    if (i + 3U < ypair0 + ypairn) {
                        const float c13 = C(b + 13), c14 = C(b + 14);
                        const float e3 = float(IL0) * (c6 + c12) + float(IL1) * (c7 + c11) +
                                         float(IL2) * (c8 + c10) + float(IL3) * c9;
                        const float o3 = float(IH0) * (c6 + c14) + float(IH1) * (c7 + c13) +
                                         float(IH2) * (c8 + c12) + float(IH3) * (c9 + c11) +
                                         float(IH4) * c10;
                        rows[(static_cast<size_t>(local) + 6U) * SHARED_X + col] = e3;
                        if (local + 7U < out_rows)
                            rows[(static_cast<size_t>(local) + 7U) * SHARED_X + col] = o3;
                    }
                }
            }
        }
        __syncthreads();

        // X synthesis: one work item emits two adjacent coefficient pairs.
        const uint32_t xgroups = (xpairn + 1U) >> 1U;
        const uint32_t xwork = out_rows * xgroups;
        for (uint32_t e = threadIdx.x; e < xwork; e += TPB) {
            const uint32_t r = e / xgroups;
            const uint32_t group = e - r * xgroups;
            const uint32_t local_i = group << 1U;
            const uint32_t i = xpair0 + local_i;
            const float* sh = rows + static_cast<size_t>(r) * SHARED_X;
            float e0, o0, e1 = 0.0f, o1 = 0.0f;
            const bool have_second = local_i + 1U < xpairn;
            if (lx == 1U) {
                e0 = sh[static_cast<int>(0U) - owner0];
                o0 = 0.0f;
            } else {
                auto C = [&](long long idx) -> float {
                    return sh[static_cast<int>(idx) - owner0];
                };
                if (have_second)
                    idwt_yx_halo_synth_quad(i, C, e0, o0, e1, o1);
                else
                    idwt_yx_halo_synth_pair(i, C, e0, o0);
            }

            const uint32_t gy = y0 + r;
            const uint32_t ox = i << 1U;
            const size_t base = static_cast<size_t>(gy) * nx + static_cast<size_t>(z) * leap_z;
            if (orig == nullptr) {
                dst[base + ox] = e0;
                if (ox + 1U < lx)
                    dst[base + ox + 1U] = o0;
                if (have_second && ox + 2U < lx)
                    dst[base + ox + 2U] = e1;
                if (have_second && ox + 3U < lx)
                    dst[base + ox + 3U] = o1;
            } else {
                const float typed_bound = static_cast<float>(bound);
                const float error0 = e0 - orig[base + ox];
                if (error0 > typed_bound || error0 < -typed_bound) {
                    const unsigned int slot = atomicAdd(ocnt, 1U);
                    if (oidx != nullptr && slot < ocap) {
                        oidx[slot] = static_cast<uint32_t>(base + ox);
                        oerr[slot] = error0;
                    }
                }
                if (ox + 1U < lx) {
                    const float error1 = o0 - orig[base + ox + 1U];
                    if (error1 > typed_bound || error1 < -typed_bound) {
                        const unsigned int slot = atomicAdd(ocnt, 1U);
                        if (oidx != nullptr && slot < ocap) {
                            oidx[slot] = static_cast<uint32_t>(base + ox + 1U);
                            oerr[slot] = error1;
                        }
                    }
                }
                if (have_second && ox + 2U < lx) {
                    const float error2 = e1 - orig[base + ox + 2U];
                    if (error2 > typed_bound || error2 < -typed_bound) {
                        const unsigned int slot = atomicAdd(ocnt, 1U);
                        if (oidx != nullptr && slot < ocap) {
                            oidx[slot] = static_cast<uint32_t>(base + ox + 2U);
                            oerr[slot] = error2;
                        }
                    }
                }
                if (have_second && ox + 3U < lx) {
                    const float error3 = o1 - orig[base + ox + 3U];
                    if (error3 > typed_bound || error3 < -typed_bound) {
                        const unsigned int slot = atomicAdd(ocnt, 1U);
                        if (oidx != nullptr && slot < ocap) {
                            oidx[slot] = static_cast<uint32_t>(base + ox + 3U);
                            oerr[slot] = error3;
                        }
                    }
                }
            }
        }
        __syncthreads();
    }
}

template <PWE Mode>
__global__ void idwt_yx_double(
    const double* __restrict__ src,
    double* __restrict__ dst,
    dim3 data_dims,
    dim3 chunk_dims,
    uint32_t pairs,
    const double* orig,
    double bound,
    unsigned int* ocnt,
    uint32_t* oidx,
    double* oerr,
    uint32_t ocap,
    bool vector_y) {
    static __shared__ __align__(128) double yx_double_rows[48U * 1024U / sizeof(double)];
    const uint32_t nx = data_dims.x;
    const uint32_t ny = data_dims.y;
    const uint32_t lx = chunk_dims.x;
    const uint32_t ly = chunk_dims.y;
    const uint32_t lz = chunk_dims.z;
    idwt_yx_tiled_double<Mode>(src,
                                      dst,
                                      yx_double_rows,
                                      2U * pairs * lx,
                                      nx,
                                      ny,
                                      lx,
                                      ly,
                                      lz,
                                      orig,
                                      bound,
                                      ocnt,
                                      oidx,
                                      oerr,
                                      ocap,
                                      vector_y);
}

// Host launcher: axis-specialized launches using the caller-provided persistent tmp scratch.
// Requires idwt3d_prealloc<T>() before launch to prepare IDWTConfig<T>.
// Optional PWE recording at the finest level: d_orig/bound/d_ocnt (pre-zeroed)
// enable detection; d_oidx/d_oerr receive the index and original-precision error.
template <typename T>
inline cudaError_t idwt3d_xy_halo_z(T* d_vol, T* d_tmp, dim3 data_dims, int levels, const T* d_orig = nullptr,
    double bound = 0.0, unsigned int* d_ocnt = nullptr, uint32_t* d_oidx = nullptr, T* d_oerr = nullptr,
                                 uint32_t ocap = 0, cudaStream_t stream = 0) {
    if (levels > CDF97_MAX_LEVELS)
        return cudaErrorInvalidValue;
    dim3 level_dims[CDF97_MAX_LEVELS];
    dim3 dims = data_dims;
    for (int lev = 0; lev < levels; ++lev) {
        level_dims[lev] = dims;
        dims.x -= dims.x >> 1U;
        dims.y -= dims.y >> 1U;
        dims.z -= dims.z >> 1U;
    }
    constexpr uint32_t capacity = WALTZ::IDWT_SHARED_BYTES / sizeof(T);
    for (int lev = levels - 1; lev >= 0; --lev) {
        const dim3 chunk_dims = level_dims[lev];
        const uint32_t lx = chunk_dims.x, ly = chunk_dims.y, lz = chunk_dims.z;
        const T* const original = lev == 0 ? d_orig : nullptr;

        // Z: data -> tmp. Limit the grid to useful tiles/items at coarse levels.
        const bool use_static = lev == 0 && lz * 16U <= capacity;
        constexpr uint32_t align_mask = std::is_same_v<T, float> ? 3U : 1U;
        const bool vector_z = (data_dims.x & align_mask) == 0U;
        size_t required_blocks;
        int full;
        if (use_static) {
            const uint32_t columns_per_tile = capacity / lz;
            const size_t tiles = static_cast<size_t>(ly) * ((static_cast<size_t>(lx) + columns_per_tile - 1U) / columns_per_tile);
            required_blocks = tiles;
            full = IDWTConfig<T>::z_static_blocks;
        } else {
            size_t items_per_column = lz - (lz >> 1U);
            if constexpr (std::is_same_v<T, float>)
                items_per_column = (items_per_column + 1U) >> 1U; // Float synthesizes two pairs per item.
            required_blocks = (static_cast<size_t>(lx) * ly * items_per_column + TPB - 1U) / TPB;
            full = IDWTConfig<T>::z_global_blocks;
        }
        const int grid = required_blocks < static_cast<size_t>(full) ? static_cast<int>(required_blocks) : full;
        if (use_static) {
            idwt_3d_z_single_static<T><<<grid, TPB, 0, stream>>>(
                d_vol, d_tmp, data_dims, chunk_dims, vector_z);
        } else {
            idwt_3d_z_single_global<T><<<grid, TPB, 0, stream>>>(
                d_vol, d_tmp, data_dims, chunk_dims);
        }

        // YX: tmp -> data. The finest level can also record PWE outliers.
        const size_t low_y = ly - (ly >> 1U);
        constexpr size_t static_capacity = 48U * 1024U / sizeof(T);
        if constexpr (std::is_same_v<T, double>) {
            if (2ULL * lx > static_capacity)
                return cudaErrorInvalidValue;
            const uint32_t pairs = min(8U, static_cast<uint32_t>(static_capacity / (2ULL * lx)));
            const size_t tiles = ((low_y + pairs - 1U) / pairs) * lz;
            const int full = IDWTConfig<T>::yx_blocks;
            const int grid = tiles < static_cast<size_t>(full) ? static_cast<int>(tiles) : full;
            const bool vector_y = (data_dims.x & 1U) == 0U && (lx & 1U) == 0U;
            if (original != nullptr) {
                idwt_yx_double<PWE::T><<<grid, TPB, 0, stream>>>(
                    d_tmp, nullptr, data_dims, chunk_dims, pairs, original, bound, d_ocnt,
                    d_oidx, d_oerr, ocap, vector_y);
            } else {
                idwt_yx_double<PWE::F><<<grid, TPB, 0, stream>>>(
                    d_tmp, d_vol, data_dims, chunk_dims, pairs, nullptr, 0.0, nullptr, nullptr,
                    nullptr, 0, vector_y);
            }
        } else if constexpr (std::is_same_v<T, float>) {
            const uint32_t pairs = min(12U,static_cast<uint32_t>(static_capacity / (2ULL * lx)));
            if (lx >= 2U && ly >= 2U && pairs != 0U) {
                const size_t tiles = ((low_y + pairs - 1U) / pairs) * lz;
                const int full = IDWTConfig<T>::yx_blocks;
                const int grid = tiles < static_cast<size_t>(full) ? static_cast<int>(tiles) : full;
                if (original != nullptr)
                    idwt_yx_float<PWE::T><<<grid, TPB, 0, stream>>>(
                        d_tmp, nullptr, data_dims, chunk_dims, pairs, original, bound, d_ocnt,
                        d_oidx, d_oerr, ocap);
                else
                    idwt_yx_float<PWE::F><<<grid, TPB, 0, stream>>>(
                        d_tmp, d_vol, data_dims, chunk_dims, pairs, nullptr, 0.0, nullptr, nullptr,
                        nullptr, 0);
            } else {
                const uint32_t xtiles = (lx + BX_128 - 1U) / BX_128;
                const uint32_t ytiles = (ly + BY_16 - 1U) / BY_16;
                const size_t tiles = static_cast<size_t>(xtiles) * ytiles * lz;
                const int full = IDWTConfig<T>::yx_halo_blocks;
                const int grid = tiles < static_cast<size_t>(full) ? static_cast<int>(tiles) : full;
                idwt_yx_halo<BX_128, BY_16><<<grid, TPB, IDWT_YX_HALO_SHARED_BYTES, stream>>>(d_tmp, d_vol, data_dims, chunk_dims,
                        original, bound, d_ocnt, d_oidx, d_oerr, ocap);
            }
        }
    }
    return cudaGetLastError();
}



__device__ __forceinline__ void idwt_pwe_record_float(float out,
                                                      size_t g,
                                                      const float* orig,
                                                      double bound,
                                                      float fast_bound,
                                                      unsigned int* ocnt,
                                                      uint32_t* oidx,
                                                      float* oerr,
                                                      uint32_t ocap) {
    const float reference = orig[g];
    const float fe = __fsub_rn(out, reference);
    if (fabsf(fe) <= fast_bound)
        return;
    if (fabsf(fe) <= static_cast<float>(bound))
        return;
    const unsigned int slot = atomicAdd(ocnt, 1U);
    if (oidx != nullptr && slot < ocap) {
        oidx[slot] = static_cast<uint32_t>(g);
        oerr[slot] = fe;
    }
}

// Variant used when a producer already fetched a coalesced vector of reference
// samples.  Keeping the reference value in a register avoids four independent
// 16-byte-strided scalar loads across a warp in the float4 inverse-Z path.
__device__ __forceinline__ void idwt_pwe_record_float_value(float out,
                                                            float reference,
                                                            size_t g,
                                                            double bound,
                                                            float fast_bound,
                                                            unsigned int* ocnt,
                                                            uint32_t* oidx,
                                                            float* oerr,
                                                            uint32_t ocap) {
    const float fe = __fsub_rn(out, reference);
    if (fabsf(fe) <= fast_bound)
        return;
    if (fabsf(fe) <= static_cast<float>(bound))
        return;
    const unsigned int slot = atomicAdd(ocnt, 1U);
    if (oidx != nullptr && slot < ocap) {
        oidx[slot] = static_cast<uint32_t>(g);
        oerr[slot] = fe;
    }
}

__device__ __forceinline__ void idwt_pwe_record_error_float(float error,
                                                            size_t g,
                                                            double bound,
                                                            float fast_bound,
                                                            unsigned int* ocnt,
                                                            uint32_t* oidx,
                                                            float* oerr,
                                                            uint32_t ocap) {
    if (fabsf(error) <= fast_bound)
        return;
    if (fabsf(error) <= static_cast<float>(bound))
        return;
    const unsigned int slot = atomicAdd(ocnt, 1U);
    if (oidx != nullptr && slot < ocap) {
        oidx[slot] = static_cast<uint32_t>(g);
        oerr[slot] = error;
    }
}

// Keep original detail bands immutable; ping-pong only the reconstructed low band.
// Scratch lengths ceil(nz/2) and ceil(nz/4) suffice because the last level writes
// directly to global memory (or records PWE). No finer-band carry copy is needed.
template <typename T>
__device__ __forceinline__ void idwt_z_columns_all(const T* src, T* dst,
    uint32_t nx, uint32_t ny, uint32_t nz, int levels_z, T* s_data, uint32_t cap, const T* orig,
    double bound, unsigned int* ocnt, uint32_t* oidx, T* oerr, uint32_t ocap, bool direct_error = false) {
    const uint32_t full_z = nz;
    const uint32_t a1 = full_z - (full_z >> 1U);
    const uint32_t a2 = a1 - (a1 >> 1U);
    const uint32_t per_col = full_z + a1 + a2;
    const uint32_t total_columns = nx * ny;
    uint32_t cols = min(cap / per_col, total_columns);
    // Four-column tiles preserve float4 alignment and regular double accesses.
    constexpr uint32_t column_alignment = 4U;
    const uint32_t aligned_cols = cols & ~(column_alignment - 1U);
    if (aligned_cols >= column_alignment)
        cols = aligned_cols;
    if (cols == 0U)
        return;

    const uint32_t max_count = (total_columns - 1U) / cols + 1U;
    const uint32_t imm_cap = full_z * cols;
    const uint32_t scratch_a_cap = a1 * cols;
    const size_t stride_z = static_cast<size_t>(nx) * ny;
    float fast_bound = 0.0f;
    if constexpr (std::is_same_v<T, float>) {
        const float bf = __double2float_rn(bound);
        fast_bound = __uint_as_float(__float_as_uint(bf) - 1U);
    }

    for (uint32_t chunkID = blockIdx.x; chunkID < max_count; chunkID += gridDim.x) {
        const uint32_t col0 = chunkID * cols;
        const uint32_t ncol = min(cols, total_columns - col0);
        const uint32_t tile_elems = ncol * full_z;
        T* const imm = s_data;
        T* const scratch_a = s_data + imm_cap;
        T* const scratch_b = scratch_a + scratch_a_cap;
        const bool vector_tile = std::is_same_v<T, float> &&
            (ncol & 3U) == 0U && ((col0 | static_cast<uint32_t>(stride_z)) & 3U) == 0U;

        if (vector_tile) {
            if constexpr (std::is_same_v<T, float>) {
                const uint32_t vec_cols = ncol >> 2U;
                const uint32_t vec_elems = full_z * vec_cols;
                float4* const sh4 = reinterpret_cast<float4*>(imm);
                for (uint32_t i = threadIdx.x; i < vec_elems; i += TPB) {
                    const uint32_t z = i / vec_cols;
                    const uint32_t vc = i - z * vec_cols;
                    sh4[i] = *reinterpret_cast<const float4*>(src + col0 + (vc << 2U) +
                                                              static_cast<size_t>(z) * stride_z);
                }
            }
        } else {
            for (uint32_t i = threadIdx.x; i < tile_elems; i += TPB) {
                const uint32_t z = i / ncol;
                const uint32_t lc = i - z * ncol;
                imm[i] = src[col0 + lc + static_cast<size_t>(z) * stride_z];
            }
        }
        __syncthreads();

        const T* low_src = imm;
        T* out_sh = (levels_z & 1) == 0 ? scratch_a : scratch_b;
        for (int lev = levels_z - 1; lev >= 0; --lev) {
            const uint32_t lz = approx_len_dev(full_z, lev);
            const uint32_t low_lz = lz - (lz >> 1U);
            const long long last = static_cast<long long>(lz) - 1;
            const uint32_t half = lz >> 1U;
            const uint32_t hi_i = half >= 3U ? half - 3U : 0U;
            const bool final = lev == 0;

            if (vector_tile) {
                if constexpr (std::is_same_v<T, float>) {
                    const uint32_t vec_cols = ncol >> 2U;
                    const uint32_t vec_pairs = low_lz * vec_cols;
                    for (uint32_t vi = threadIdx.x; vi < vec_pairs; vi += TPB) {
                        const uint32_t i = vi / vec_cols;
                        const uint32_t lc = (vi - i * vec_cols) << 2U;
                        auto low4 = [&](uint32_t k) -> float4 {
                            return *reinterpret_cast<const float4*>(low_src +
                                                                    static_cast<size_t>(k) * ncol + lc);
                        };
                        auto high4 = [&](uint32_t k) -> float4 {
                            return *reinterpret_cast<const float4*>(
                                imm + static_cast<size_t>(low_lz + k) * ncol + lc);
                        };
                        auto C4 = [&](long long idx) -> float4 {
                            const uint32_t pos = sym_idx(idx, last);
                            return (pos & 1U) ? high4(pos >> 1U) : low4(pos >> 1U);
                        };
                        float4 oe, oo;
                        if (i >= 2U && i <= hi_i) {
                            const float4 c0 = high4(i - 2U), c1 = low4(i - 1U), c2 = high4(i - 1U),
                                         c3 = low4(i), c4 = high4(i), c5 = low4(i + 1U),
                                         c6 = high4(i + 1U), c7 = low4(i + 2U), c8 = high4(i + 2U);
                            oe = idwt_even4(c0, c1, c2, c3, c4, c5, c6);
                            oo = idwt_odd4(c0, c1, c2, c3, c4, c5, c6, c7, c8);
                        } else {
                            const long long b = 2LL * static_cast<long long>(i) - 3LL;
                            oe = idwt_even4(C4(b + 0),
                                            C4(b + 1),
                                            C4(b + 2),
                                            C4(b + 3),
                                            C4(b + 4),
                                            C4(b + 5),
                                            C4(b + 6));
                            oo = idwt_odd4(C4(b + 0),
                                           C4(b + 1),
                                           C4(b + 2),
                                           C4(b + 3),
                                           C4(b + 4),
                                           C4(b + 5),
                                           C4(b + 6),
                                           C4(b + 7),
                                           C4(b + 8));
                        }

                        if (!final) {
                            *reinterpret_cast<float4*>(out_sh + static_cast<size_t>(2U * i) * ncol +
                                                       lc) = oe;
                            if (2U * i + 1U < lz)
                                *reinterpret_cast<float4*>(
                                    out_sh + static_cast<size_t>(2U * i + 1U) * ncol + lc) = oo;
                        } else if (direct_error) {
                            const size_t g0 = col0 + lc + static_cast<size_t>(2U * i) * stride_z;
                            idwt_pwe_record_error_float(
                                oe.x, g0 + 0U, bound, fast_bound, ocnt, oidx, oerr, ocap);
                            idwt_pwe_record_error_float(
                                oe.y, g0 + 1U, bound, fast_bound, ocnt, oidx, oerr, ocap);
                            idwt_pwe_record_error_float(
                                oe.z, g0 + 2U, bound, fast_bound, ocnt, oidx, oerr, ocap);
                            idwt_pwe_record_error_float(
                                oe.w, g0 + 3U, bound, fast_bound, ocnt, oidx, oerr, ocap);
                            if (2U * i + 1U < lz) {
                                const size_t g1 = g0 + stride_z;
                                idwt_pwe_record_error_float(
                                    oo.x, g1 + 0U, bound, fast_bound, ocnt, oidx, oerr, ocap);
                                idwt_pwe_record_error_float(
                                    oo.y, g1 + 1U, bound, fast_bound, ocnt, oidx, oerr, ocap);
                                idwt_pwe_record_error_float(
                                    oo.z, g1 + 2U, bound, fast_bound, ocnt, oidx, oerr, ocap);
                                idwt_pwe_record_error_float(
                                    oo.w, g1 + 3U, bound, fast_bound, ocnt, oidx, oerr, ocap);
                            }
                        } else if (orig == nullptr) {
                            *reinterpret_cast<float4*>(dst + col0 + lc +
                                                       static_cast<size_t>(2U * i) * stride_z) = oe;
                            if (2U * i + 1U < lz)
                                *reinterpret_cast<float4*>(
                                    dst + col0 + lc + static_cast<size_t>(2U * i + 1U) * stride_z) = oo;
                        } else {
                            const size_t g0 = col0 + lc + static_cast<size_t>(2U * i) * stride_z;
                            const float4 r0 = *reinterpret_cast<const float4*>(orig + g0);
                            idwt_pwe_record_float_value(
                                oe.x, r0.x, g0 + 0U, bound, fast_bound, ocnt, oidx, oerr, ocap);
                            idwt_pwe_record_float_value(
                                oe.y, r0.y, g0 + 1U, bound, fast_bound, ocnt, oidx, oerr, ocap);
                            idwt_pwe_record_float_value(
                                oe.z, r0.z, g0 + 2U, bound, fast_bound, ocnt, oidx, oerr, ocap);
                            idwt_pwe_record_float_value(
                                oe.w, r0.w, g0 + 3U, bound, fast_bound, ocnt, oidx, oerr, ocap);
                            if (2U * i + 1U < lz) {
                                const size_t g1 = g0 + stride_z;
                                const float4 r1 = *reinterpret_cast<const float4*>(orig + g1);
                                idwt_pwe_record_float_value(
                                    oo.x, r1.x, g1 + 0U, bound, fast_bound, ocnt, oidx, oerr, ocap);
                                idwt_pwe_record_float_value(
                                    oo.y, r1.y, g1 + 1U, bound, fast_bound, ocnt, oidx, oerr, ocap);
                                idwt_pwe_record_float_value(
                                    oo.z, r1.z, g1 + 2U, bound, fast_bound, ocnt, oidx, oerr, ocap);
                                idwt_pwe_record_float_value(
                                    oo.w, r1.w, g1 + 3U, bound, fast_bound, ocnt, oidx, oerr, ocap);
                            }
                        }
                    }
                }
            } else {
                const uint32_t pairs = low_lz * ncol;
                for (uint32_t e = threadIdx.x; e < pairs; e += TPB) {
                    const uint32_t i = e / ncol;
                    const uint32_t lc = e - i * ncol;
                    auto C = [&](long long idx) -> T {
                        const uint32_t pos = sym_idx(idx, last);
                        return (pos & 1U)
                                   ? imm[static_cast<size_t>(low_lz + (pos >> 1U)) * ncol + lc]
                                   : low_src[static_cast<size_t>(pos >> 1U) * ncol + lc];
                    };
                    T oe, oo;
                    if constexpr (std::is_same_v<T, double>) {
                        // Preserve the original double arithmetic order and interior fast path.
                        if (i >= 2U && i <= hi_i) {
                            const size_t lo = static_cast<size_t>(i) * ncol + lc;
                            const size_t hi = static_cast<size_t>(low_lz + i) * ncol + lc;
                            const size_t s = ncol;
                            const T l_m1 = low_src[lo - s], l_0 = low_src[lo],
                                    l_p1 = low_src[lo + s], l_p2 = low_src[lo + 2 * s];
                            const T h_m2 = imm[hi - 2 * s], h_m1 = imm[hi - s], h_0 = imm[hi],
                                    h_p1 = imm[hi + s], h_p2 = imm[hi + 2 * s];
                            oe = T(IL0) * (h_m2 + h_p1) + T(IL1) * (l_m1 + l_p1) +
                                 T(IL2) * (h_m1 + h_0) + T(IL3) * l_0;
                            oo = T(IH0) * (h_m2 + h_p2) + T(IH1) * (l_m1 + l_p2) +
                                 T(IH2) * (h_m1 + h_p1) + T(IH3) * (l_0 + l_p1) + T(IH4) * h_0;
                        } else {
                            synth_pair<T>(i, C, oe, oo);
                        }
                    } else {
                        synth_pair<T>(i, C, oe, oo);
                    }
                    if (!final) {
                        out_sh[static_cast<size_t>(2U * i) * ncol + lc] = oe;
                        if (2U * i + 1U < lz)
                            out_sh[static_cast<size_t>(2U * i + 1U) * ncol + lc] = oo;
                    } else {
                        const size_t g0 = col0 + lc + static_cast<size_t>(2U * i) * stride_z;
                        if constexpr (std::is_same_v<T, float>) {
                            if (direct_error) {
                                idwt_pwe_record_error_float(
                                    oe, g0, bound, fast_bound, ocnt, oidx, oerr, ocap);
                                if (2U * i + 1U < lz)
                                    idwt_pwe_record_error_float(
                                        oo, g0 + stride_z, bound, fast_bound, ocnt, oidx, oerr, ocap);
                            } else if (orig == nullptr) {
                                dst[g0] = oe;
                                if (2U * i + 1U < lz)
                                    dst[g0 + stride_z] = oo;
                            } else {
                                idwt_pwe_record_float(
                                    oe, g0, orig, bound, fast_bound, ocnt, oidx, oerr, ocap);
                                if (2U * i + 1U < lz)
                                    idwt_pwe_record_float(oo,
                                                          g0 + stride_z,
                                                          orig,
                                                          bound,
                                                          fast_bound,
                                                          ocnt,
                                                          oidx,
                                                          oerr,
                                                          ocap);
                            }
                        } else if (orig == nullptr) {
                            dst[g0] = oe;
                            if (2U * i + 1U < lz)
                                dst[g0 + stride_z] = oo;
                        } else {
                            auto record = [&](T out, size_t g) {
                                const T error = out - orig[g];
                                if (error > bound || error < -bound) {
                                    const unsigned int slot = atomicAdd(ocnt, 1U);
                                    if (oidx != nullptr && slot < ocap) {
                                        oidx[slot] = static_cast<uint32_t>(g);
                                        oerr[slot] = error;
                                    }
                                }
                            };
                            record(oe, g0);
                            if (2U * i + 1U < lz)
                                record(oo, g0 + stride_z);
                        }
                    }
                }
            }
            __syncthreads();
            if (!final) {
                low_src = out_sh;
                out_sh = out_sh == scratch_a ? scratch_b : scratch_a;
            }
        }
    }
}


// Dynamically sized shared-memory Z inverse.
template <typename T>
__global__ __launch_bounds__(TPB, std::is_same_v<T, float> ? 3 : CDF97_MIN_BLOCKS_D)
void idwt_z_all_dynamic(const T* src, T* dst, uint32_t nx, uint32_t ny, uint32_t nz,
                        int levels_z, const T* orig, double bound, unsigned int* ocnt,
                        uint32_t* oidx, T* oerr, uint32_t ocap, uint32_t cap,
                        bool direct_error = false) {
    extern __shared__ __align__(128) unsigned char z_shared[];
    T* const s_data = reinterpret_cast<T*>(z_shared);
    idwt_z_columns_all<T>(src, dst, nx, ny, nz, levels_z, s_data, cap,
                                  orig, bound, ocnt, oidx, oerr, ocap, direct_error);
}

// Configure dyadic/plane inverse launches and optionally allocate temporary storage.
// Call once before launching IDWT; callers with their own scratch pass false.
template <typename T>
inline cudaError_t idwt3d_prealloc(dim3 dims, bool use_dyadic, bool allocate_scratch = true,
    cudaStream_t stream = 0) {
    if (allocate_scratch) {
        const size_t bytes = static_cast<size_t>(dims.x) * dims.y * dims.z * sizeof(T);
        if (bytes > IDWTConfig<T>::cap) {
            if (IDWTConfig<T>::tmp)
                cudaFree(IDWTConfig<T>::tmp);
            cudaError_t error = cudaMalloc(&IDWTConfig<T>::tmp, bytes);
            if (error != cudaSuccess) {
                IDWTConfig<T>::tmp = nullptr;
                IDWTConfig<T>::cap = 0;
                IDWTConfig<T>::mapped_cap = 0;
                return error;
            }
            IDWTConfig<T>::cap = bytes;
            IDWTConfig<T>::mapped_cap = 0;
        }
        if (IDWTConfig<T>::mapped_cap < bytes) {
            cudaMemsetAsync(IDWTConfig<T>::tmp, 0, bytes, stream);
            IDWTConfig<T>::mapped_cap = bytes;
        }
    }
    if (use_dyadic) {
        // Cache the occupancy grids used by the dyadic inverse launchers once per type.
        // Keep the global-Z result local until every query has completed so the guard cannot
        // expose a partially initialized set of caches.
        if (IDWTConfig<T>::z_global_blocks == 0) {
            const int sm_count = WALTZ::gpu_config().sm_count();
            int blocks_per_sm = 0;
            cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                &blocks_per_sm, idwt_3d_z_single_global<T>, TPB, 0);
            const int z_global_blocks = max(blocks_per_sm, 1) * sm_count;

            blocks_per_sm = 0;
            cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                &blocks_per_sm, idwt_3d_z_single_static<T>, TPB, 0);
            const int z_static_blocks = max(blocks_per_sm, 1) * sm_count;
            IDWTConfig<T>::z_static_blocks = z_static_blocks;

            if constexpr (std::is_same_v<T, float>) {
                blocks_per_sm = 0;
                cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                    &blocks_per_sm, idwt_yx_halo<BX_128, BY_16>,
                    TPB, IDWT_YX_HALO_SHARED_BYTES);
                IDWTConfig<T>::yx_halo_blocks = max(blocks_per_sm, 1) * sm_count;

                blocks_per_sm = 0;
                cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                    &blocks_per_sm, idwt_yx_float<PWE::F>, TPB, 0);
                IDWTConfig<T>::yx_blocks = max(blocks_per_sm, 1) * sm_count;
            } else if constexpr (std::is_same_v<T, double>) {
                blocks_per_sm = 0;
                cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                    &blocks_per_sm, idwt_yx_double<PWE::F>, TPB, 0);
                IDWTConfig<T>::yx_blocks = max(blocks_per_sm, 1) * sm_count;
            }

            IDWTConfig<T>::z_global_blocks = z_global_blocks;
        }
        return cudaGetLastError();
    }

    const cudaDeviceProp& prop = WALTZ::gpu_config().properties;
    if (IDWTConfig<T>::configured_nx == dims.x && IDWTConfig<T>::configured_nz == dims.z)
        return cudaSuccess;

    // The XY kernel and launch resources do not depend on the volume dimensions.
    if (IDWTConfig<T>::plane_yx_blocks == 0) {
        int yx_blocks_per_sm = 0;
        cudaError_t yx_error = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &yx_blocks_per_sm, idwt_yx_plane<T>, 256, 0);
        if (yx_error != cudaSuccess) return yx_error;
        IDWTConfig<T>::plane_yx_blocks = max(yx_blocks_per_sm, 1) * prop.multiProcessorCount;
    }
    IDWTConfig<T>::configured_nx = dims.x;

    if constexpr (std::is_same_v<T, double>) {
        constexpr size_t MAX_SHARED_BYTES = 48U * 1024U;
        const size_t budget = min(MAX_SHARED_BYTES, static_cast<size_t>(prop.sharedMemPerBlockOptin));

        const uint32_t pairs = dims.x == 0U ? 0U : max(1U, min(8U,
            static_cast<uint32_t>(MAX_SHARED_BYTES / (2ULL * dims.x * sizeof(T)))));
        const size_t yx_smem = 2ULL * pairs * dims.x * sizeof(T);
        const uint32_t z_cap = static_cast<uint32_t>(budget / sizeof(T));
        if (yx_smem > budget || dims.z * 2U > z_cap)
            return cudaErrorInvalidValue;

        const size_t z_smem = static_cast<size_t>(z_cap) * sizeof(T);
        cudaError_t error = cudaFuncSetAttribute(idwt_z_all_dynamic<double>,
                                     cudaFuncAttributeMaxDynamicSharedMemorySize,
                                     static_cast<int>(z_smem));
        if (error != cudaSuccess)
            return error;

        int blocks_per_sm = 0;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &blocks_per_sm, idwt_z_all_dynamic<double>, TPB, z_smem);
        IDWTConfig<T>::plane_z_blocks = max(blocks_per_sm, 1) * prop.multiProcessorCount;
        IDWTConfig<T>::configured_nx = dims.x;
        IDWTConfig<T>::configured_nz = dims.z;
        IDWTConfig<T>::z_capacity = z_cap;
        return cudaGetLastError();
    } else {
        if (IDWTConfig<T>::plane_z_blocks == 0 || IDWTConfig<T>::configured_nz != dims.z) {
            int target_blocks_per_sm = 0;
            constexpr size_t BASELINE_Z_SMEM = 11000U * sizeof(T);
            cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                &target_blocks_per_sm, idwt_z_all_dynamic<float>, TPB, BASELINE_Z_SMEM);
            target_blocks_per_sm = max(1, target_blocks_per_sm);
            const size_t sm_budget = min(static_cast<size_t>(prop.sharedMemPerBlockOptin),
                static_cast<size_t>(prop.sharedMemPerMultiprocessor) / static_cast<size_t>(target_blocks_per_sm));
            uint32_t try_cols = static_cast<uint32_t>(sm_budget / (2ULL * dims.z * sizeof(T)));
            try_cols &= ~3U;
            if (try_cols < 4U)
                return cudaErrorInvalidValue;
            const uint32_t chosen_cap = 2U * dims.z * try_cols;
            const size_t chosen_smem = static_cast<size_t>(chosen_cap) * sizeof(T);

            cudaError_t error = cudaFuncSetAttribute(idwt_z_all_dynamic<float>,
                                                      cudaFuncAttributeMaxDynamicSharedMemorySize,
                                                      static_cast<int>(chosen_smem));
            if (error != cudaSuccess)
                return error;
            int blocks_per_sm = 0;
            cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                &blocks_per_sm, idwt_z_all_dynamic<float>, TPB, chosen_smem);
            IDWTConfig<T>::plane_z_blocks = max(blocks_per_sm, 1) * prop.multiProcessorCount;
            IDWTConfig<T>::z_smem_bytes = chosen_smem;
            IDWTConfig<T>::z_capacity = chosen_cap;
            IDWTConfig<T>::configured_nz = dims.z;
        }
        return cudaGetLastError();
    }
}

template <typename T>
inline cudaError_t idwt3d_plane_xy_halo_z(T* d_vol, T* d_tmp, dim3 dims, int levels_xy, int levels_z,
    const T* d_orig = nullptr, double bound = 0.0, unsigned int* d_ocnt = nullptr, uint32_t* d_oidx = nullptr,
    T* d_oerr = nullptr, uint32_t ocap = 0, bool direct_error = false, cudaStream_t stream = 0) {
    // As with the dyadic launcher, caller-provided scratch requires prealloc.
    if (d_tmp == nullptr) {
        cudaError_t e = idwt3d_prealloc<T>(dims, false, true, stream);
        if (e != cudaSuccess)
            return e;
        d_tmp = static_cast<T*>(IDWTConfig<T>::tmp);
    }

    dim3 level_dims[CDF97_MAX_LEVELS];
    dim3 chunk_dims = dims;
    for (int lev = 0; lev < levels_xy; ++lev) {
        level_dims[lev] = chunk_dims;
        chunk_dims.x -= chunk_dims.x >> 1U;
        chunk_dims.y -= chunk_dims.y >> 1U;
    }
    T* src = d_vol;
    T* dst = d_tmp;

    constexpr bool is_float = std::is_same_v<T, float>;
    constexpr uint32_t xy_tpb = 256U;
    constexpr uint32_t static_capacity = (is_float ? 32U : 48U) * 1024U / sizeof(T);
    constexpr uint32_t max_pairs = is_float ? 12U : 8U;

    for (int lev = levels_xy - 1; lev >= 0; --lev) {
        const dim3 chunk_dims = level_dims[lev];
        const dim3 next_dims = lev > 0 ? level_dims[lev - 1] : chunk_dims;
        uint32_t pairs = min(max_pairs, static_cast<uint32_t>(static_capacity / (2ULL * chunk_dims.x)));
        const uint32_t low_y = (chunk_dims.y + 1U) >> 1U;
        if (pairs > 4U && static_cast<size_t>((low_y + pairs - 1U) / pairs) * chunk_dims.z <
            static_cast<size_t>(WALTZ::gpu_config().sm_count()))
            pairs = 4U;
        if (pairs == 0U)
            return cudaErrorInvalidValue;

        const bool carry = lev > 0 && dst == d_tmp;
        const bool vector_y = (dims.x & 1U) == 0U && (chunk_dims.x & 1U) == 0U;
        idwt_yx_plane<T><<<IDWTConfig<T>::plane_yx_blocks, xy_tpb, 0, stream>>>(
            src, dst, dims, chunk_dims, pairs, next_dims.x, next_dims.y, carry,
            nullptr, 0.0, nullptr, nullptr, nullptr, 0U, vector_y);
        cudaError_t e = cudaGetLastError();
        if (e != cudaSuccess)
            return e;

        T* swap = src;
        src = dst;
        dst = swap;
    }

    const uint32_t z_cap = IDWTConfig<T>::z_capacity;
    const size_t z_smem = static_cast<size_t>(z_cap) * sizeof(T);
    idwt_z_all_dynamic<T><<<IDWTConfig<T>::plane_z_blocks, TPB, z_smem, stream>>>(
        src, d_vol, dims.x, dims.y, dims.z, levels_z,
        d_orig, bound, d_ocnt, d_oidx, d_oerr, ocap, z_cap, direct_error);
    return cudaGetLastError();
}


} // namespace CDF97

#undef TPB
