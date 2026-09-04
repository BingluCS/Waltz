#pragma once

// Cooperative GPU inverse CDF97 9/7 DWT -- the perfect-reconstruction inverse of the
// forward CDF97::dwt3d, done with the POLYPHASE convolution synthesis (no upsample
// zeros) so it is as cheap as the forward.  Operates entirely in NATURAL order (the
// z-order permutation is only the LC magnitude's business; the inverse reads the
// dequantised coeffs and writes the reconstruction, both natural).
//
// Filter derivation.  The full synthesis filters are
//   g0 (REC_LOW,  7-tap) = (-1)^k     * H_analysis   (modulated analysis high-pass)
//   g1 (REC_HIGH, 9-tap) = (-1)^(k+1) * L_analysis   (modulated analysis low-pass)
// and reconstruction is x = upsample(low) (*) g0 + upsample(high) (*) g1.  Folding
// out the inserted zeros gives ONE even-output filter (7-tap, IL*) and ONE odd-output
// filter (9-tap, IH*) that read the deinterleaved [low|high] coeffs directly:
//   even x[2i]   = sum_{k=0..6} IL[k] * coeff(2i   + k - 3)   (IL symmetric)
//   odd  x[2i+1] = sum_{k=0..8} IH[k] * coeff(2i+1 + k - 4)   (IH symmetric)
// where coeff(idx): even idx -> low[idx/2], odd idx -> high[idx/2] (symmetric edge).
// IL blends g0 (on the low taps) with g1 (on the high taps); IH likewise -- that is
// why IL/IH differ from g0/g1 numerically while computing the identical transform.
// (Verified bit-exact vs the CPU reference: round-trip recovers to ~2e-13 / double.)

#include "CDF97.hpp"

#include <cstdlib>

// even-output (low-phase) polyphase filter, symmetric: {IL0,IL1,IL2,IL3,IL2,IL1,IL0}
#ifdef WALTZ_P4_TRANSFORM
#define IL0 0.0
#define IL1 0.0
#define IL2 -0.24748737341529164
#define IL3 0.70710678118654746
#else
#define IL0 -0.0238494650195568
#define IL1 -0.0406894176097634
#define IL2 0.3774028556128310
#define IL3 0.7884856164052130
#endif
// odd-output (high-phase) polyphase filter, symmetric: {IH0,IH1,IH2,IH3,IH4,IH3,IH2,IH1,IH0}
#ifdef WALTZ_P4_TRANSFORM
#define IH0 0.016581654018824540
#define IH1 -0.047376154339498683
#define IH2 -0.12374368670764582
#define IH3 0.40092954493277239
#define IH4 1.1335628809201543
#else
#define IH0 -0.0378284555072640
#define IH1 -0.0645388826292432
#define IH2 0.1106244044184370
#define IH3 0.4180922732229480
#define IH4 -0.8526986790088940
#endif

namespace CDF97 {

using WALTZ::IDWTConfig;

__device__ __forceinline__ uint32_t approx_len_dev(uint32_t n, int lev) {
    uint32_t low = n;
    for (int i = 0; i < lev; ++i)
        low -= low >> 1;
    return low;
}

// whole-sample symmetric (mirror, no edge repeat) folding -- SPERR's symmetric_idx
__device__ __forceinline__ uint32_t sym_idx(long long idx, long long last) {
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

// Four-column companion of synth_pair for the float Z tile.  idwt_z_columns_all
// stages complete columns in shared; when the tile width is a multiple of four,
// neighboring columns are independent and can be synthesized with one float4 load
// per tap.  Component expressions intentionally retain synth_pair's tap ordering so
// this is an execution-only vectorization (the quantized coefficient stream is
// unaffected).
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

__device__ __forceinline__ float4 idwt_odd4(float4 c0,
                                            float4 c1,
                                            float4 c2,
                                            float4 c3,
                                            float4 c4,
                                            float4 c5,
                                            float4 c6,
                                            float4 c7,
                                            float4 c8) {
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

// Four-column version of the four-pair register block used by the plane XY
// inverse.  Each lane follows the scalar synth_quad expression independently;
// the vector form only changes the memory transaction granularity.
__device__ __forceinline__ void synth_quad4(float4 c0,
                                            float4 c1,
                                            float4 c2,
                                            float4 c3,
                                            float4 c4,
                                            float4 c5,
                                            float4 c6,
                                            float4 c7,
                                            float4 c8,
                                            float4 c9,
                                            float4 c10,
                                            float4& e0,
                                            float4& o0,
                                            float4& e1,
                                            float4& o1) {
    e0 = idwt_even4(c0, c1, c2, c3, c4, c5, c6);
    o0 = idwt_odd4(c0, c1, c2, c3, c4, c5, c6, c7, c8);
    e1 = idwt_even4(c2, c3, c4, c5, c6, c7, c8);
    o1 = idwt_odd4(c2, c3, c4, c5, c6, c7, c8, c9, c10);
}

// Z synthesis: coalesced across x (innermost); float register-blocks 2 pairs/thread
// (synth_quad) for more in-flight loads, double uses 1 pair (lower register pressure).
template <typename T>
__device__ void idwt_z_pass(const T* __restrict__ src,
                            T* __restrict__ dst,
                            uint32_t nx,
                            uint32_t ny,
                            uint32_t lx,
                            uint32_t ly,
                            uint32_t lz) {
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
            if (i >= 2U && 2U * i + 5U < lz) {
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

// Y synthesis: coalesced across x; same float/double blocking as Z.
template <typename T>
__device__ void idwt_y_pass(const T* __restrict__ src,
                            T* __restrict__ dst,
                            uint32_t nx,
                            uint32_t ny,
                            uint32_t lx,
                            uint32_t ly,
                            uint32_t lz) {
    if (ly < 2)
        return;
    const size_t leap_y = nx;
    const size_t leap_z = static_cast<size_t>(nx) * ny;
    const uint32_t low_ly = ly - (ly >> 1);
    const long long last = static_cast<long long>(ly) - 1;
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    const size_t g0 = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if constexpr (std::is_same_v<T, float>) { // synth_quad: MLP > spill cost (see idwt_z_pass note)
        const uint32_t dp = (low_ly + 1u) >> 1;
        const size_t total = static_cast<size_t>(lx) * dp * lz;
        const bool p2x = (lx & (lx - 1U)) == 0U;
        const bool p2dp = (dp & (dp - 1U)) == 0U;
        const int sx = p2x ? (__ffs(static_cast<int>(lx)) - 1) : 0;
        const int sdp = p2dp ? (__ffs(static_cast<int>(dp)) - 1) : 0;
        for (size_t g = g0; g < total; g += stride) {
            const uint32_t x =
                p2x ? static_cast<uint32_t>(g) & (lx - 1U) : static_cast<uint32_t>(g % lx);
            const size_t r = p2x ? (g >> sx) : (g / lx);
            const uint32_t i = 2u * static_cast<uint32_t>(p2dp ? (r & (dp - 1U)) : (r % dp));
            const uint32_t z = static_cast<uint32_t>(p2dp ? (r >> sdp) : (r / dp));
            const size_t xz = x + static_cast<size_t>(z) * leap_z;
            if (i >= 2U && 2U * i + 5U < ly) {
                // Interior synthesis has no reflected taps.  Keep the exact
                // synth_quad arithmetic while replacing ten sym_idx calls with
                // direct deinterleaved low/high row addresses.
                auto Ci = [&](int idx) -> T {
                    const uint32_t u = static_cast<uint32_t>(idx);
                    const uint32_t row = (u & 1U) ? (low_ly + (u >> 1U)) : (u >> 1U);
                    return src[xz + static_cast<size_t>(row) * leap_y];
                };
                T e0, o0, e1, o1;
                synth_quad<T>(static_cast<int>(i), Ci, e0, o0, e1, o1);
                dst[xz + static_cast<size_t>(2U * i) * leap_y] = e0;
                dst[xz + static_cast<size_t>(2U * i + 1U) * leap_y] = o0;
                if (2U * i + 2U < ly)
                    dst[xz + static_cast<size_t>(2U * i + 2U) * leap_y] = e1;
                if (2U * i + 3U < ly)
                    dst[xz + static_cast<size_t>(2U * i + 3U) * leap_y] = o1;
                continue;
            }
            auto C = [&](long long idx) -> T {
                const uint32_t pos = sym_idx(idx, last);
                return src[xz +
                           static_cast<size_t>((pos & 1u) ? (low_ly + (pos >> 1)) : (pos >> 1)) *
                               leap_y];
            };
            if (i + 1u < low_ly) {
                T e0, o0, e1, o1;
                synth_quad<T>(i, C, e0, o0, e1, o1);
                dst[xz + static_cast<size_t>(2 * i) * leap_y] = e0;
                dst[xz + static_cast<size_t>(2 * i + 1) * leap_y] = o0;
                dst[xz + static_cast<size_t>(2 * i + 2) * leap_y] = e1;
                if (2 * i + 3 < ly)
                    dst[xz + static_cast<size_t>(2 * i + 3) * leap_y] = o1;
            } else {
                T e0, o0;
                synth_pair<T>(i, C, e0, o0);
                dst[xz + static_cast<size_t>(2 * i) * leap_y] = e0;
                if (2 * i + 1 < ly)
                    dst[xz + static_cast<size_t>(2 * i + 1) * leap_y] = o0;
            }
        }
    } else {
        const size_t total = static_cast<size_t>(lx) * low_ly * lz;
        for (size_t g = g0; g < total; g += stride) {
            const uint32_t x = static_cast<uint32_t>(g % lx);
            const size_t r = g / lx;
            const uint32_t i = static_cast<uint32_t>(r % low_ly);
            const uint32_t z = static_cast<uint32_t>(r / low_ly);
            const size_t xz = x + static_cast<size_t>(z) * leap_z;
            auto C = [&](long long idx) -> T {
                const uint32_t pos = sym_idx(idx, last);
                return src[xz +
                           static_cast<size_t>((pos & 1u) ? (low_ly + (pos >> 1)) : (pos >> 1)) *
                               leap_y];
            };
            T e0, o0;
            synth_pair<T>(i, C, e0, o0);
            dst[xz + static_cast<size_t>(2 * i) * leap_y] = e0;
            if (2 * i + 1 < ly)
                dst[xz + static_cast<size_t>(2 * i + 1) * leap_y] = o0;
        }
    }
}

// SHARED-staged single-level Z synthesis (the "stage + reorder-free contiguous read"
// idea).  A tile of TX=cap/lz consecutive x-columns x the full z-line (at fixed y) is
// staged once into shared, then the synthesis reads the CONTIGUOUS low/high windows from
// shared (interior fast path, no per-tap deinterleave) -- so the deinterleaved global
// read never hits L1 latency.  One global load + one global store per element (NO extra
// pass), the reorder is absorbed into shared.  Same s_data the X pass already needs, so
// no extra shared / occupancy cost.  TX>=1 required (lz<=cap); caller falls back if not.
template <typename T, bool Vec4 = false>
__device__ void idwt_z_level_shared(const T* src,
                                    T* dst,
                                    T* s_data,
                                    uint32_t cap,
                                    uint32_t nx,
                                    uint32_t ny,
                                    uint32_t lx,
                                    uint32_t ly,
                                    uint32_t lz) {
    if (lz < 2)
        return;
    const size_t leap_z = static_cast<size_t>(nx) * ny;
    const uint32_t low_lz = lz - (lz >> 1);
    const long long last = static_cast<long long>(lz) - 1;
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
        // Optional vector synthesis: four adjacent x-columns share the same
        // z-window, so float4 shared loads/stores remove the scalar transaction
        // overhead.  Keep it explicitly gated because some older GPUs trade the
        // lower instruction count for higher register pressure.
        const bool vector_tile =
            Vec4 && sizeof(T) == sizeof(float) && txn >= 4U && (txn & 3U) == 0U;
        if (vector_tile) {
            if constexpr (sizeof(T) == sizeof(float)) {
                if (vector_tile) {
                    const uint32_t nv = txn >> 2U;
                    for (uint32_t e = threadIdx.x; e < low_lz * nv; e += TPB) {
                        const uint32_t i = e / nv, vc = e - i * nv;
                        const uint32_t tx = vc << 2U;
                        float4 oe, oo;
                        auto C4 = [&](long long idx) -> float4 {
                            const uint32_t pos = sym_idx(idx, last);
                            const uint32_t k = (pos & 1u) ? (low_lz + (pos >> 1)) : (pos >> 1);
                            return *reinterpret_cast<const float4*>(
                                s_data + static_cast<size_t>(k) * txn + tx);
                        };
                        if (i >= 2u && i <= hi_i) {
                            const float4 c0 = *reinterpret_cast<const float4*>(
                                s_data + static_cast<size_t>(low_lz + i - 2U) * txn + tx);
                            const float4 c1 = *reinterpret_cast<const float4*>(
                                s_data + static_cast<size_t>(i - 1U) * txn + tx);
                            const float4 c2 = *reinterpret_cast<const float4*>(
                                s_data + static_cast<size_t>(low_lz + i - 1U) * txn + tx);
                            const float4 c3 = *reinterpret_cast<const float4*>(
                                s_data + static_cast<size_t>(i) * txn + tx);
                            const float4 c4 = *reinterpret_cast<const float4*>(
                                s_data + static_cast<size_t>(low_lz + i) * txn + tx);
                            const float4 c5 = *reinterpret_cast<const float4*>(
                                s_data + static_cast<size_t>(i + 1U) * txn + tx);
                            const float4 c6 = *reinterpret_cast<const float4*>(
                                s_data + static_cast<size_t>(low_lz + i + 1U) * txn + tx);
                            const float4 c7 = *reinterpret_cast<const float4*>(
                                s_data + static_cast<size_t>(i + 2U) * txn + tx);
                            const float4 c8 = *reinterpret_cast<const float4*>(
                                s_data + static_cast<size_t>(low_lz + i + 2U) * txn + tx);
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
                        *reinterpret_cast<float4*>(dst + base + tx +
                                                   static_cast<size_t>(2U * i) * leap_z) = oe;
                        if (2U * i + 1U < lz)
                            *reinterpret_cast<float4*>(
                                dst + base + tx + static_cast<size_t>(2U * i + 1U) * leap_z) = oo;
                    }
                } else {
                    for (uint32_t e = threadIdx.x; e < low_lz * txn; e += TPB) {
                        const uint32_t i = e / txn, tx = e - i * txn;
                        const T* col = s_data + tx;
                        T oe, oo;
                        if (i >= 2u && i <= hi_i) {
                            const size_t lo = static_cast<size_t>(i) * txn,
                                         hi = static_cast<size_t>(low_lz + i) * txn, s = txn;
                            const T l_m1 = col[lo - s], l_0 = col[lo], l_p1 = col[lo + s],
                                    l_p2 = col[lo + 2 * s];
                            const T h_m2 = col[hi - 2 * s], h_m1 = col[hi - s], h_0 = col[hi],
                                    h_p1 = col[hi + s], h_p2 = col[hi + 2 * s];
                            oe = T(IL0) * (h_m2 + h_p1) + T(IL1) * (l_m1 + l_p1) +
                                 T(IL2) * (h_m1 + h_0) + T(IL3) * l_0;
                            oo = T(IH0) * (h_m2 + h_p2) + T(IH1) * (l_m1 + l_p2) +
                                 T(IH2) * (h_m1 + h_p1) + T(IH3) * (l_0 + l_p1) + T(IH4) * h_0;
                        } else {
                            auto C = [&](long long idx) -> T {
                                const uint32_t pos = sym_idx(idx, last);
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
                }
            } else {
                for (uint32_t e = threadIdx.x; e < low_lz * txn; e += TPB) {
                    const uint32_t i = e / txn, tx = e - i * txn;
                    const T* col = s_data + tx;
                    T oe, oo;
                    if (i >= 2u && i <= hi_i) {
                        const size_t lo = static_cast<size_t>(i) * txn,
                                     hi = static_cast<size_t>(low_lz + i) * txn, s = txn;
                        const T l_m1 = col[lo - s], l_0 = col[lo], l_p1 = col[lo + s],
                                l_p2 = col[lo + 2 * s];
                        const T h_m2 = col[hi - 2 * s], h_m1 = col[hi - s], h_0 = col[hi],
                                h_p1 = col[hi + s], h_p2 = col[hi + 2 * s];
                        oe = T(IL0) * (h_m2 + h_p1) + T(IL1) * (l_m1 + l_p1) +
                             T(IL2) * (h_m1 + h_0) + T(IL3) * l_0;
                        oo = T(IH0) * (h_m2 + h_p2) + T(IH1) * (l_m1 + l_p2) +
                             T(IH2) * (h_m1 + h_p1) + T(IH3) * (l_0 + l_p1) + T(IH4) * h_0;
                    } else {
                        auto C = [&](long long idx) -> T {
                            const uint32_t pos = sym_idx(idx, last);
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
                    auto C = [&](long long idx) -> T {
                        const uint32_t pos = sym_idx(idx, last);
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

// (Note: a SHARED-staged Y variant was tried and removed -- per-axis sweep showed Y's
// deinterleaved clusters (~512KB apart) stay resident in L2, so staging only adds sync
// overhead with no latency to hide.  Y stays on the coalesced global idwt_y_pass.)

// X synthesis: in place, staging a TILE of whole x-rows in shared (like the forward
// dwt_x_rows) so each shared load is amortised over many rows.  `s_data` holds `cap`
// elements; rows_per_tile = cap / lx rows are processed per block iteration.
// Optional fused outlier detection (orig != nullptr, finest level only): the freshly
// synthesized value is still in registers, so |x_hat - orig| > bound is counted here
// for free instead of re-reading the whole reconstruction in a separate pass.
template <typename T>
__device__ void idwt_x_tiled(const T* src,
                             T* dst,
                             T* s_data,
                             uint32_t cap,
                             uint32_t nx,
                             uint32_t ny,
                             uint32_t lx,
                             uint32_t ly,
                             uint32_t lz,
                             const T* orig = nullptr,
                             double bound = 0.0,
                             unsigned int* ocnt = nullptr,
                             uint32_t* oidx = nullptr,
                             T* oerr = nullptr,
                             uint32_t ocap = 0) {
    if (lx < 2)
        return;
    const size_t leap_z = static_cast<size_t>(nx) * ny;
    const uint32_t low_lx = lx - (lx >> 1);
    const long long last = static_cast<long long>(lx) - 1;
    const uint32_t num_rows = ly * lz;
    const uint32_t rpt = cap / lx; // rows per tile
    if (rpt == 0)
        return;
    const uint32_t num_tiles = (num_rows + rpt - 1) / rpt;
    // Finest-level fast path (mirrors the forward dwt_x_rows): when the active region is
    // the full contiguous volume, the tile's rows are contiguous in memory, so stage AND
    // write flat (src[base0+e]) -- skips the per-element row/col split with its ~2 integer
    // divisions.  Those divisions (not the synth) were the bulk of the X inverse gap.
    const bool contig = (lx == nx) && (static_cast<size_t>(nx) * ly == leap_z);
    for (uint32_t t = blockIdx.x; t < num_tiles; t += gridDim.x) {
        const uint32_t row0 = t * rpt;
        const uint32_t rows = min(rpt, num_rows - row0);
        const uint32_t nelem = rows * lx;
        const size_t base0 = static_cast<size_t>(row0) * lx; // contig only
        if (contig) {
            for (uint32_t e = threadIdx.x; e < nelem; e += TPB) // flat coalesced stage
                s_data[e] = src[base0 + e];
        } else {
            for (uint32_t e = threadIdx.x; e < nelem; e += TPB) {
                const uint32_t r = e / lx, k = e - r * lx, row = row0 + r;
                s_data[e] = src[static_cast<size_t>(row % ly) * nx +
                                static_cast<size_t>(row / ly) * leap_z + k];
            }
        }
        __syncthreads();
        // Deinterleaved low|high are contiguous (stride 1) within each staged row, so the
        // interior pairs read the low/high windows directly -- no per-tap (pos&1) gather,
        // matching the forward's contiguous reads.  This was the dominant inverse gap (X
        // at the finest level was 2.1x the forward before this).  Boundary pairs: slow path.
        const uint32_t half = lx >> 1;
        const uint32_t hi_i = (half >= 3u) ? (half - 3u) : 0u;
        const uint32_t npairs = rows * low_lx;
        for (uint32_t e = threadIdx.x; e < npairs; e += TPB) {
            const uint32_t r = e / low_lx, i = e - r * low_lx;
            const T* sh = s_data + r * lx; // sh[k] = position k (low at [0,low_lx), high after)
            T oe, oo;
            if (i >= 2u && i <= hi_i) { // contiguous window
                const uint32_t hb = low_lx + i;
                const T l_m1 = sh[i - 1], l_0 = sh[i], l_p1 = sh[i + 1], l_p2 = sh[i + 2];
                const T h_m2 = sh[hb - 2], h_m1 = sh[hb - 1], h_0 = sh[hb], h_p1 = sh[hb + 1],
                        h_p2 = sh[hb + 2];
                oe = T(IL0) * (h_m2 + h_p1) + T(IL1) * (l_m1 + l_p1) + T(IL2) * (h_m1 + h_0) +
                     T(IL3) * l_0;
                oo = T(IH0) * (h_m2 + h_p2) + T(IH1) * (l_m1 + l_p2) + T(IH2) * (h_m1 + h_p1) +
                     T(IH3) * (l_0 + l_p1) + T(IH4) * h_0;
            } else { // boundary (sym_idx gather)
                auto C = [&](long long idx) -> T {
                    const uint32_t pos = sym_idx(idx, last);
                    return sh[(pos & 1u) ? (low_lx + (pos >> 1)) : (pos >> 1)];
                };
                synth_pair<T>(i, C, oe, oo);
            }
            const uint32_t row = row0 + r;
            const size_t base =
                contig
                    ? (base0 + static_cast<size_t>(r) * lx)
                    : (static_cast<size_t>(row % ly) * nx + static_cast<size_t>(row / ly) * leap_z);
            // Compression-side PWE validation consumes the finest reconstruction only
            // to compare it with `orig`; no later compression stage reads `dst`.
            // Avoid the final full-volume write in that path.  Decompression passes
            // orig == nullptr and therefore still materializes every output value.
            if (orig == nullptr) {
                dst[base + 2 * i] = oe;
                if (2 * i + 1 < lx)
                    dst[base + 2 * i + 1] = oo;
            } else { // fused outlier detect + record (values still in registers)
                const T typed_bound = static_cast<T>(bound);
                const T e0 = oe - orig[base + 2 * i];
                if (e0 > typed_bound || e0 < -typed_bound) {
                    const unsigned int slot = atomicAdd(ocnt, 1u);
                    if (oidx != nullptr && slot < ocap) {
                        oidx[slot] = static_cast<uint32_t>(base + 2 * i);
                        oerr[slot] = e0;
                    }
                }
                if (2 * i + 1 < lx) {
                    const T e1 = oo - orig[base + 2 * i + 1];
                    if (e1 > typed_bound || e1 < -typed_bound) {
                        const unsigned int slot = atomicAdd(ocnt, 1u);
                        if (oidx != nullptr && slot < ocap) {
                            oidx[slot] = static_cast<uint32_t>(base + 2 * i + 1);
                            oerr[slot] = e1;
                        }
                    }
                }
            }
        }
        __syncthreads();
    }
}

// Cooperative inverse dyadic 3D DWT (coarse-to-fine, Z->Y->X each level).  vol holds
// the coeffs in place; tmp is scratch; dynamic shared >= max dim * sizeof(T).
// Finest-level double Y+X synthesis fused through the X tile's existing shared
// storage. The ordinary path materializes the complete Y result in `vol` and
// immediately reloads it for X. Here each block synthesizes a small band of Y rows into shared,
// then performs the identical polyphase X convolution directly from those rows.
// `synth_pair<double>` and the X interior expression deliberately match the
// unfused double path's operation order, so the coefficient basis and PWE stream
// do not change.  Compression (orig != nullptr) records PWE without materializing
// the final field; decompression still writes every reconstructed value.
__device__ __forceinline__ void idwt_yx_tiled_double(const double* __restrict__ src,
                                                     double* __restrict__ dst,
                                                     double* rows,
                                                     uint32_t cap,
                                                     uint32_t nx,
                                                     uint32_t ny,
                                                     uint32_t lx,
                                                     uint32_t ly,
                                                     uint32_t lz,
                                                     const double* orig,
                                                     double bound,
                                                     unsigned int* ocnt,
                                                     uint32_t* oidx,
                                                     double* oerr,
                                                     uint32_t ocap) {
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

        // Y synthesis.  One owner produces the adjacent even/odd output rows
        // from a shared 9-tap coefficient window, exactly as idwt_y_pass<double>.
        const uint32_t ywork = pairn * lx;
        for (uint32_t e = threadIdx.x; e < ywork; e += TPB) {
            const uint32_t x = e % lx;
            const uint32_t i = pair0 + e / lx;
            const size_t xz = x + static_cast<size_t>(z) * leap_z;
            auto C = [&](long long idx) -> double {
                const uint32_t pos = sym_idx(idx, last_y);
                return src[xz +
                           static_cast<size_t>((pos & 1U) ? (low_ly + (pos >> 1U)) : (pos >> 1U)) *
                               nx];
            };
            double ev, ov;
            synth_pair<double>(i, C, ev, ov);
            const uint32_t local = (i - pair0) << 1U;
            rows[local * lx + x] = ev;
            if (local + 1U < out_rows)
                rows[(local + 1U) * lx + x] = ov;
        }
        __syncthreads();

        // X synthesis directly from the shared Y rows.  The interior and edge
        // branches are the same as idwt_x_tiled<double>.
        const uint32_t xwork = out_rows * low_lx;
        const uint32_t half = lx >> 1U;
        const uint32_t hi_i = (half >= 3U) ? (half - 3U) : 0U;
        for (uint32_t e = threadIdx.x; e < xwork; e += TPB) {
            const uint32_t r = e / low_lx;
            const uint32_t i = e - r * low_lx;
            const double* sh = rows + r * lx;
            double oe, oo;
            if (i >= 2U && i <= hi_i) {
                const uint32_t hb = low_lx + i;
                const double l_m1 = sh[i - 1], l_0 = sh[i], l_p1 = sh[i + 1], l_p2 = sh[i + 2];
                const double h_m2 = sh[hb - 2], h_m1 = sh[hb - 1], h_0 = sh[hb], h_p1 = sh[hb + 1],
                             h_p2 = sh[hb + 2];
                oe = double(IL0) * (h_m2 + h_p1) + double(IL1) * (l_m1 + l_p1) +
                     double(IL2) * (h_m1 + h_0) + double(IL3) * l_0;
                oo = double(IH0) * (h_m2 + h_p2) + double(IH1) * (l_m1 + l_p2) +
                     double(IH2) * (h_m1 + h_p1) + double(IH3) * (l_0 + l_p1) + double(IH4) * h_0;
            } else {
                auto C = [&](long long idx) -> double {
                    const uint32_t pos = sym_idx(idx, last_x);
                    return sh[(pos & 1U) ? (low_lx + (pos >> 1U)) : (pos >> 1U)];
                };
                synth_pair<double>(i, C, oe, oo);
            }

            const uint32_t gy = 2U * pair0 + r;
            const size_t base = static_cast<size_t>(gy) * nx + static_cast<size_t>(z) * leap_z;
            if (orig == nullptr) {
                dst[base + 2U * i] = oe;
                if (2U * i + 1U < lx)
                    dst[base + 2U * i + 1U] = oo;
            } else {
                const double de0 = oe - orig[base + 2U * i];
                if (fabs(de0) > bound) {
                    const unsigned int slot = atomicAdd(ocnt, 1U);
                    if (oidx != nullptr && slot < ocap) {
                        oidx[slot] = static_cast<uint32_t>(base + 2U * i);
                        oerr[slot] = de0;
                    }
                }
                if (2U * i + 1U < lx) {
                    const double de1 = oo - orig[base + 2U * i + 1U];
                    if (fabs(de1) > bound) {
                        const unsigned int slot = atomicAdd(ocnt, 1U);
                        if (oidx != nullptr && slot < ocap) {
                            oidx[slot] = static_cast<uint32_t>(base + 2U * i + 1U);
                            oerr[slot] = de1;
                        }
                    }
                }
            }
        }
        __syncthreads();
    }
}

// Optional fused outlier detection: when `orig` is non-null, the finest-level X pass
// (the one that writes the final reconstruction) also counts |x_hat - orig| > bound
// into *ocnt -- saving the separate full-volume outlier pass (one 1x read of x_hat).
template <typename T>
__global__ __launch_bounds__(TPB, CDF97_MIN_BLOCKS(T)) void idwt3d(T* vol,
                                                                   T* tmp,
                                                                   uint32_t nx,
                                                                   uint32_t ny,
                                                                   uint32_t nz,
                                                                   int levels,
                                                                   const T* orig,
                                                                   double bound,
                                                                   unsigned int* ocnt,
                                                                   uint32_t* oidx,
                                                                   T* oerr,
                                                                   uint32_t ocap) {
    cooperative_groups::grid_group grid = cooperative_groups::this_grid();
    __shared__ alignas(128) T s_data[WALTZ::IDWT_SHARED_BYTES / sizeof(T)];
    for (int lev = levels - 1; lev >= 0; --lev) {
        const uint32_t lx = approx_len_dev(nx, lev);
        const uint32_t ly = approx_len_dev(ny, lev);
        const uint32_t lz = approx_len_dev(nz, lev);
        // Stage only finest-level Z in shared memory. Its deinterleaved clusters
        // can be farther apart than cache capacity, while deeper subcubes and Y
        // accesses generally remain cache-friendly enough that synchronization
        // costs more than staging saves.
        // Needs TX=cap/lz >= 16 to coalesce the staging load (long lines stay global).
        if constexpr (std::is_same_v<T, float>) {
            if (lev == 0 && lz * 16u <= WALTZ::IDWT_SHARED_BYTES / sizeof(T))
                idwt_z_level_shared<T>(vol, tmp, s_data, WALTZ::IDWT_SHARED_BYTES / sizeof(T), nx, ny, lx, ly, lz);
            else
                idwt_z_pass<T>(vol, tmp, nx, ny, lx, ly, lz);
        } else {
            idwt_z_pass<T>(vol, tmp, nx, ny, lx, ly, lz);
        }
        grid.sync();
        if constexpr (sizeof(T) == 8U) {
            if (lev == 0 && lx * 2U <= WALTZ::IDWT_SHARED_BYTES / sizeof(T) && ly >= 2U) {
                idwt_yx_tiled_double(reinterpret_cast<const double*>(tmp),
                                     reinterpret_cast<double*>(vol),
                                     reinterpret_cast<double*>(s_data),
                                     WALTZ::IDWT_SHARED_BYTES / sizeof(T),
                                     nx,
                                     ny,
                                     lx,
                                     ly,
                                     lz,
                                     reinterpret_cast<const double*>(orig),
                                     bound,
                                     ocnt,
                                     oidx,
                                     oerr,
                                     ocap);
            } else {
                idwt_y_pass<T>(tmp, vol, nx, ny, lx, ly, lz);
                grid.sync();
                idwt_x_tiled<T>(vol,
                                vol,
                                s_data,
                                WALTZ::IDWT_SHARED_BYTES / sizeof(T),
                                nx,
                                ny,
                                lx,
                                ly,
                                lz,
                                lev == 0 ? orig : nullptr,
                                bound,
                                ocnt,
                                oidx,
                                oerr,
                                ocap);
            }
        } else {
            idwt_y_pass<T>(tmp, vol, nx, ny, lx, ly, lz); // Y stays global for float
            grid.sync();
            idwt_x_tiled<T>(vol,
                            vol,
                            s_data,
                            WALTZ::IDWT_SHARED_BYTES / sizeof(T),
                            nx,
                            ny,
                            lx,
                            ly,
                            lz,
                            lev == 0 ? orig : nullptr,
                            bound,
                            ocnt,
                            oidx,
                            oerr,
                            ocap);
        }
        grid.sync();
    }
}

// Axis-specialized normal kernels.  Keeping Z/Y/X in separate entry functions avoids
// the mega-kernel's union of live registers and lets Y run without the 32-KB shared
// allocation required only by X/Z staging.  Stream ordering replaces grid.sync().
template <typename T>
__global__ __launch_bounds__(TPB, 2) void idwt_z_axis(
    const T* src, T* dst, uint32_t nx, uint32_t ny, uint32_t lx, uint32_t ly, uint32_t lz) {
    idwt_z_pass<T>(src, dst, nx, ny, lx, ly, lz);
}

template <typename T>
__global__ __launch_bounds__(TPB, CDF97_MIN_BLOCKS(T)) void idwt_z_axis_shared(
    const T* src, T* dst, uint32_t nx, uint32_t ny, uint32_t lx, uint32_t ly, uint32_t lz) {
    __shared__ alignas(128) T s_data[WALTZ::IDWT_SHARED_BYTES / sizeof(T)];
    idwt_z_level_shared<T>(src, dst, s_data, WALTZ::IDWT_SHARED_BYTES / sizeof(T), nx, ny, lx, ly, lz);
}

// Opt-in float4 synthesis companion.  The shared staging/layout is identical
// to idwt_z_axis_shared; only the per-column convolution is vectorized.
__global__ __launch_bounds__(TPB, 3) void idwt_z_axis_shared_vec4(
    const float* src, float* dst, uint32_t nx, uint32_t ny, uint32_t lx, uint32_t ly, uint32_t lz) {
    __shared__ alignas(128) float s_data[WALTZ::IDWT_SHARED_BYTES / sizeof(float)];
    idwt_z_level_shared<float, true>(src, dst, s_data, WALTZ::IDWT_SHARED_BYTES / sizeof(float), nx, ny, lx, ly, lz);
}

template <typename T>
__global__ __launch_bounds__(TPB, 2) void idwt_y_axis(
    const T* src, T* dst, uint32_t nx, uint32_t ny, uint32_t lx, uint32_t ly, uint32_t lz) {
    idwt_y_pass<T>(src, dst, nx, ny, lx, ly, lz);
}

template <typename T>
__global__ __launch_bounds__(TPB, CDF97_MIN_BLOCKS(T)) void idwt_x_axis(T* vol,
                                                                        uint32_t nx,
                                                                        uint32_t ny,
                                                                        uint32_t lx,
                                                                        uint32_t ly,
                                                                        uint32_t lz,
                                                                        const T* orig,
                                                                        double bound,
                                                                        unsigned int* ocnt,
                                                                        uint32_t* oidx,
                                                                        T* oerr,
                                                                        uint32_t ocap) {
    __shared__ alignas(128) T s_data[WALTZ::IDWT_SHARED_BYTES / sizeof(T)];
    idwt_x_tiled<T>(
        vol, vol, s_data, WALTZ::IDWT_SHARED_BYTES / sizeof(T), nx, ny, lx, ly, lz, orig, bound, ocnt, oidx, oerr, ocap);
}

// Finest-level float Y+X fusion.  Y synthesizes 10 adjacent pairs (up to 20 rows)
// directly into the existing 32-KB shared tile; X consumes those rows without the
// former full-volume Y store + X reload and writes the final reconstruction.
__global__ __launch_bounds__(TPB, 3) void idwt_yx_axis_fused_float(const float* __restrict__ src,
                                                                   float* __restrict__ dst,
                                                                   uint32_t nx,
                                                                   uint32_t ny,
                                                                   uint32_t lx,
                                                                   uint32_t ly,
                                                                   uint32_t lz,
                                                                   const float* orig,
                                                                   double bound,
                                                                   unsigned int* ocnt,
                                                                   uint32_t* oidx,
                                                                   float* oerr,
                                                                   uint32_t ocap) {
    // A 24-row tile is near the portable 48-KB static shared limit while
    // cutting the per-tile barrier/loop overhead versus the original 16-row
    // tile.  The arithmetic and boundary handling are unchanged.
    constexpr uint32_t PAIRS = 12;
    constexpr uint32_t MAX_ROWS = 2 * PAIRS;
    __shared__ alignas(128) float rows[MAX_ROWS * 512];
    if (lx > 512 || lx < 2 || ly < 2)
        return;

    const size_t leap_z = static_cast<size_t>(nx) * ny;
    const uint32_t low_ly = ly - (ly >> 1U);
    const uint32_t low_lx = lx - (lx >> 1U);
    const long long last_y = static_cast<long long>(ly) - 1;
    const long long last_x = static_cast<long long>(lx) - 1;
    const uint32_t nty = (low_ly + PAIRS - 1U) / PAIRS;
    const uint32_t ntiles = nty * lz;

    for (uint32_t tile = blockIdx.x; tile < ntiles; tile += gridDim.x) {
        const uint32_t z = tile / nty;
        const uint32_t pt = tile - z * nty;
        const uint32_t pair0 = pt * PAIRS;
        const uint32_t pairn = min(PAIRS, low_ly - pair0);
        const uint32_t out_rows = min(2U * pairn, ly - 2U * pair0);
        const uint32_t groups = (pairn + 1U) >> 1U;

        // Two Y pairs per thread item (11 source loads -> four synthesized values).
        const uint32_t ywork = groups * lx;
        for (uint32_t e = threadIdx.x; e < ywork; e += TPB) {
            const uint32_t x = e % lx;
            const uint32_t group = e / lx;
            const uint32_t i = pair0 + (group << 1U);
            const size_t xz = x + static_cast<size_t>(z) * leap_z;
            auto C = [&](long long idx) -> float {
                const uint32_t pos = sym_idx(idx, last_y);
                return src[xz +
                           static_cast<size_t>((pos & 1U) ? (low_ly + (pos >> 1U)) : (pos >> 1U)) *
                               nx];
            };
            const uint32_t local = (i - pair0) << 1U;
            if (i + 1U < pair0 + pairn) {
                float e0, o0, e1, o1;
                synth_quad<float>(i, C, e0, o0, e1, o1);
                rows[(local + 0U) * lx + x] = e0;
                if (local + 1U < out_rows)
                    rows[(local + 1U) * lx + x] = o0;
                rows[(local + 2U) * lx + x] = e1;
                if (local + 3U < out_rows)
                    rows[(local + 3U) * lx + x] = o1;
            } else {
                float ev, ov;
                synth_pair<float>(i, C, ev, ov);
                rows[local * lx + x] = ev;
                if (local + 1U < out_rows)
                    rows[(local + 1U) * lx + x] = ov;
            }
        }
        __syncthreads();

        // X synthesis from shared Y rows (same interior/boundary arithmetic as
        // idwt_x_tiled), followed by the optional fused PWE check.
        const uint32_t xwork = out_rows * low_lx;
        const uint32_t half = lx >> 1U;
        const uint32_t hi_i = (half >= 3U) ? (half - 3U) : 0U;
        for (uint32_t e = threadIdx.x; e < xwork; e += TPB) {
            const uint32_t r = e / low_lx, i = e - r * low_lx;
            const float* sh = rows + r * lx;
            float oe, oo;
            if (i >= 2U && i <= hi_i) {
                const uint32_t hb = low_lx + i;
                const float l_m1 = sh[i - 1], l_0 = sh[i], l_p1 = sh[i + 1], l_p2 = sh[i + 2];
                const float h_m2 = sh[hb - 2], h_m1 = sh[hb - 1], h_0 = sh[hb], h_p1 = sh[hb + 1],
                            h_p2 = sh[hb + 2];
                oe = float(IL0) * (h_m2 + h_p1) + float(IL1) * (l_m1 + l_p1) +
                     float(IL2) * (h_m1 + h_0) + float(IL3) * l_0;
                oo = float(IH0) * (h_m2 + h_p2) + float(IH1) * (l_m1 + l_p2) +
                     float(IH2) * (h_m1 + h_p1) + float(IH3) * (l_0 + l_p1) + float(IH4) * h_0;
            } else {
                auto C = [&](long long idx) -> float {
                    const uint32_t pos = sym_idx(idx, last_x);
                    return sh[(pos & 1U) ? (low_lx + (pos >> 1U)) : (pos >> 1U)];
                };
                synth_pair<float>(i, C, oe, oo);
            }
            const uint32_t gy = 2U * pair0 + r;
            const size_t base = static_cast<size_t>(gy) * nx + static_cast<size_t>(z) * leap_z;
            dst[base + 2U * i] = oe;
            if (2U * i + 1U < lx)
                dst[base + 2U * i + 1U] = oo;
            if (orig != nullptr) {
                const float typed_bound = static_cast<float>(bound);
                const float e0 = oe - orig[base + 2U * i];
                if (e0 > typed_bound || e0 < -typed_bound) {
                    const unsigned int slot = atomicAdd(ocnt, 1U);
                    if (oidx != nullptr && slot < ocap) {
                        oidx[slot] = static_cast<uint32_t>(base + 2U * i);
                        oerr[slot] = e0;
                    }
                }
                if (2U * i + 1U < lx) {
                    const float e1 = oo - orig[base + 2U * i + 1U];
                    if (e1 > typed_bound || e1 < -typed_bound) {
                        const unsigned int slot = atomicAdd(ocnt, 1U);
                        if (oidx != nullptr && slot < ocap) {
                            oidx[slot] = static_cast<uint32_t>(base + 2U * i + 1U);
                            oerr[slot] = e1;
                        }
                    }
                }
            }
        }
        __syncthreads();
    }
}

// Double-precision counterpart of the float Y+X fused axis kernel.  A dynamic
// shared tile lets the launcher retain enough adjacent Y pairs to amortize the
// two block barriers without imposing the float path's fixed 512-column limit.
// The device helper is shared with the cooperative double inverse, so the
// arithmetic and boundary treatment remain identical to the established path.
inline uint32_t idwt_double_yx_pairs(uint32_t lx) {
    constexpr size_t target_smem = 48U * 1024U;
    constexpr uint32_t max_pairs = 8U;
    if (lx == 0U)
        return 0U;
    const size_t pair_bytes = 2ULL * lx * sizeof(double);
    const uint32_t pairs = static_cast<uint32_t>(target_smem / pair_bytes);
    return max(1U, min(max_pairs, pairs));
}

__global__ __launch_bounds__(TPB, CDF97_MIN_BLOCKS_D) void idwt_yx_axis_fused_double(
    const double* __restrict__ src,
    double* __restrict__ dst,
    uint32_t nx,
    uint32_t ny,
    uint32_t lx,
    uint32_t ly,
    uint32_t lz,
    uint32_t pairs,
    const double* orig,
    double bound,
    unsigned int* ocnt,
    uint32_t* oidx,
    double* oerr,
    uint32_t ocap) {
    extern __shared__ __align__(128) double yx_double_rows[];
    idwt_yx_tiled_double(src,
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
                         ocap);
}

// Persistent launcher state (tmp scratch + cached cooperative grid sizes), shared by the
// dyadic and plane inverse launchers.  Exposed so idwt3d_prealloc can pay the one-time
// costs (large cudaMalloc + first-touch page mapping + occupancy/device queries) up
// front, outside any timed region.
template <typename T> struct IdwtScratch {
    static inline void* tmp = nullptr;
    static inline size_t cap = 0;
    static inline size_t mapped_cap = 0;
    static inline int grid_dyadic = 0;
    static inline int grid_plane = 0;
    static inline int grid_z = 0;
    static inline int grid_z_shared = 0;
    static inline int grid_z_shared_vec4 = 0;
    static inline int grid_z_columns = 0;
    static inline int grid_y = 0;
    static inline int grid_x = 0;
    static inline int grid_yx = 0;
    static inline int grid_yx_ext = 0;
};

template <typename T> inline cudaError_t idwt_ensure_tmp(size_t bytes) {
    if (bytes > IdwtScratch<T>::cap) {
        if (IdwtScratch<T>::tmp)
            cudaFree(IdwtScratch<T>::tmp);
        cudaError_t em = cudaMalloc(&IdwtScratch<T>::tmp, bytes);
        if (em != cudaSuccess) {
            IdwtScratch<T>::tmp = nullptr;
            IdwtScratch<T>::cap = 0;
            IdwtScratch<T>::mapped_cap = 0;
            return em;
        }
        IdwtScratch<T>::cap = bytes;
        IdwtScratch<T>::mapped_cap = 0;
    }
    return cudaSuccess;
}

template <typename T, typename K> inline int idwt_coop_grid(K kernel, int& cache) {
    if (cache == 0) {
        int blocks_per_sm = 0;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_per_sm, kernel, TPB, 0);
        const int sm_count = WALTZ::gpu_config().sm_count();
        cache = blocks_per_sm * sm_count;
        if (cache <= 0)
            cache = sm_count;
    }
    return cache;
}

template <typename K> inline int idwt_axis_grid(K kernel, int& cache) {
    if (cache == 0) {
        int blocks_per_sm = 0;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_per_sm, kernel, TPB, 0);
        const int sm_count = WALTZ::gpu_config().sm_count();
        cache = blocks_per_sm * sm_count;
        if (cache <= 0)
            cache = sm_count;
    }
    return cache;
}

// (defined further down; forward declaration so idwt3d_prealloc can query its occupancy)
template <typename T>
__global__ __launch_bounds__(TPB, CDF97_MIN_BLOCKS(T)) void idwt3d_plane(T* vol,
                                                                         T* tmp,
                                                                         uint32_t nx,
                                                                         uint32_t ny,
                                                                         uint32_t nz,
                                                                         int levels_xy,
                                                                         int levels_z,
                                                                         const T* orig,
                                                                         double bound,
                                                                         unsigned int* ocnt,
                                                                         uint32_t* oidx,
                                                                         T* oerr,
                                                                         uint32_t ocap);

template <int KTPB = TPB>
__global__
    __launch_bounds__(KTPB,
                      3) void idwt_yx_axis_fused_plane_float_ext(const float* __restrict__ src,
                                                                 float* __restrict__ dst,
                                                                 uint32_t nx_arg,
                                                                 uint32_t ny_arg,
                                                                 uint32_t nz_arg,
                                                                 uint32_t lx_arg,
                                                                 uint32_t ly_arg,
                                                                 uint32_t pairs_arg,
                                                                 uint32_t next_lx,
                                                                 uint32_t next_ly,
                                                                 int carry,
                                                                 const float* orig,
                                                                 double bound,
                                                                 unsigned int* ocnt,
                                                                 uint32_t* oidx,
                                                                 float* oerr,
                                                                 uint32_t ocap);

__global__
    __launch_bounds__(TPB,
                      CDF97_MIN_BLOCKS_F) void idwt_z_columns_all_axis_float(const float* src,
                                                                             float* dst,
                                                                             uint32_t nx,
                                                                             uint32_t ny,
                                                                             uint32_t nz,
                                                                             int levels_z,
                                                                             const float* orig,
                                                                             double bound,
                                                                             unsigned int* ocnt,
                                                                             uint32_t* oidx,
                                                                             float* oerr,
                                                                             uint32_t ocap);

// One-time setup for the inverse launchers: allocate + page-map the tmp scratch and,
// for the ordinary multi-kernel path, cache the axis-launch occupancy grids.  The
// cooperative grid is selected by the outer pipeline together with the forward grid.
template <typename T>
inline cudaError_t idwt3d_prealloc(dim3 dims,
                                   bool use_dyadic,
                                   bool use_cooperative,
                                   cudaStream_t stream = 0) {
    const size_t bytes = static_cast<size_t>(dims.x) * dims.y * dims.z * sizeof(T);
    cudaError_t em = idwt_ensure_tmp<T>(bytes);
    if (em != cudaSuccess)
        return em;
    if (IdwtScratch<T>::mapped_cap < bytes) {
        cudaMemsetAsync(IdwtScratch<T>::tmp, 0, bytes, stream);
        IdwtScratch<T>::mapped_cap = bytes;
    }
    if (use_dyadic && !use_cooperative) {
        if constexpr (std::is_same_v<T, float>) {
            idwt_axis_grid(idwt_z_axis<T>, IdwtScratch<T>::grid_z);
            idwt_axis_grid(idwt_z_axis_shared<T>, IdwtScratch<T>::grid_z_shared);
            idwt_axis_grid(idwt_z_axis_shared_vec4, IdwtScratch<T>::grid_z_shared_vec4);
            if (std::getenv("WALTZ_IDWT_Z_COLUMNS") != nullptr)
                idwt_axis_grid(idwt_z_columns_all_axis_float, IdwtScratch<T>::grid_z_columns);
            idwt_axis_grid(idwt_y_axis<T>, IdwtScratch<T>::grid_y);
            idwt_axis_grid(idwt_x_axis<T>, IdwtScratch<T>::grid_x);
            idwt_axis_grid(idwt_yx_axis_fused_float, IdwtScratch<T>::grid_yx);
            // Use fewer row pairs for wider active X lines to reduce shared
            // footprint; this resource rule applies to every volume.
            const uint32_t ext_pairs = dims.x >= 512U ? 6U : 8U;
            const size_t ext_smem = 2ULL * ext_pairs * dims.x * sizeof(float);
            cudaFuncSetAttribute(idwt_yx_axis_fused_plane_float_ext<TPB>,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 static_cast<int>(ext_smem));
            idwt_axis_grid(idwt_yx_axis_fused_plane_float_ext<TPB>,
                           IdwtScratch<T>::grid_yx_ext);
        } else {
            idwt_axis_grid(idwt_z_axis<T>, IdwtScratch<T>::grid_z);
            idwt_axis_grid(idwt_z_axis_shared<T>, IdwtScratch<T>::grid_z_shared);
            idwt_axis_grid(idwt_y_axis<T>, IdwtScratch<T>::grid_y);
            idwt_axis_grid(idwt_x_axis<T>, IdwtScratch<T>::grid_x);
            const uint32_t pairs = idwt_double_yx_pairs(dims.x);
            const size_t yx_smem = 2ULL * pairs * dims.x * sizeof(double);
            cudaError_t ea = cudaFuncSetAttribute(idwt_yx_axis_fused_double,
                                                  cudaFuncAttributeMaxDynamicSharedMemorySize,
                                                  static_cast<int>(yx_smem));
            if (ea != cudaSuccess)
                return ea;
            idwt_axis_grid(idwt_yx_axis_fused_double, IdwtScratch<T>::grid_yx);
        }
    }
    return cudaGetLastError();
}

// Host launcher: axis-specialized launches + persistent tmp scratch (grown on demand).
// If out_kernel_ms != nullptr, the complete Z/Y/X launch sequence is timed.
// Optional fused outlier detect+record: pass d_orig/bound/d_ocnt (pre-zeroed) and the
// finest-level X pass counts |x_hat - orig| > bound while writing the reconstruction;
// With d_oidx/d_oerr it also records the row-major uint32_t index and the exact
// reconstruction error in the original scalar type for the decoder.
template <typename T>
inline cudaError_t idwt3d_dyadic(T* d_vol,
                                 dim3 dims,
                                 int levels,
                                 cudaStream_t stream = 0,
                                 float* out_kernel_ms = nullptr,
                                 int grid_blocks = 0,
                                 const T* d_orig = nullptr,
                                 double bound = 0.0,
                                 unsigned int* d_ocnt = nullptr,
                                 uint32_t* d_oidx = nullptr,
                                 T* d_oerr = nullptr,
                                 uint32_t ocap = 0,
                                 T* d_tmp_override = nullptr,
                                 bool use_cooperative = false) {
    if (levels <= 0)
        return cudaSuccess;
    T* d_tmp = d_tmp_override;
    if (d_tmp == nullptr) {
        const size_t bytes = static_cast<size_t>(dims.x) * dims.y * dims.z * sizeof(T);
        cudaError_t em = idwt_ensure_tmp<T>(bytes);
        if (em != cudaSuccess)
            return em;
        d_tmp = static_cast<T*>(IdwtScratch<T>::tmp);
    }
    if (use_cooperative) {
        int g = grid_blocks;
        if (g <= 0)
            g = idwt_coop_grid<T>(idwt3d<T>, IdwtScratch<T>::grid_dyadic);
        void* args[] = {&d_vol,
                        &d_tmp,
                        &dims.x,
                        &dims.y,
                        &dims.z,
                        &levels,
                        &d_orig,
                        &bound,
                        &d_ocnt,
                        &d_oidx,
                        &d_oerr,
                        &ocap};
        if (out_kernel_ms == nullptr)
            return cudaLaunchCooperativeKernel(
                reinterpret_cast<void*>(idwt3d<T>), dim3(g), dim3(TPB), args, 0, stream);
        static cudaEvent_t d0 = nullptr, d1 = nullptr;
        if (d0 == nullptr) {
            cudaEventCreate(&d0);
            cudaEventCreate(&d1);
        }
        cudaEventRecord(d0, stream);
        cudaError_t e = cudaLaunchCooperativeKernel(
            reinterpret_cast<void*>(idwt3d<T>), dim3(g), dim3(TPB), args, 0, stream);
        cudaEventRecord(d1, stream);
        cudaEventSynchronize(d1);
        cudaEventElapsedTime(out_kernel_ms, d0, d1);
        return e;
    }
    // time only GPU work; occupancy/property setup was already paid by prealloc.
    static cudaEvent_t k0 = nullptr, k1 = nullptr;
    if (out_kernel_ms != nullptr) {
        if (k0 == nullptr) {
            cudaEventCreate(&k0);
            cudaEventCreate(&k1);
        }
        cudaEventRecord(k0, stream);
    }
    auto limited_grid = [&](int g) {
        return (grid_blocks > 0 && grid_blocks < g) ? grid_blocks : g;
    };
    // Axis kernels are ordinary stream-ordered launches (no grid-wide barrier),
    // and each one already uses a grid-stride loop.  At coarse dyadic levels the
    // active cube is much smaller than the cached occupancy grid, so launching
    // every resident CTA only to execute an empty loop adds a measurable fixed
    // cost.  Cap the grid to the actual number of producer items while retaining
    // the caller's explicit grid limit.  The cap is purely scheduling; all
    // arithmetic and coefficient order stay in the device kernels unchanged.
    auto work_grid = [&](int full, size_t work_items) {
        const size_t need = std::max<size_t>(
            1U, (work_items + static_cast<size_t>(TPB) - 1U) / static_cast<size_t>(TPB));
        const int capped = need < static_cast<size_t>(full) ? static_cast<int>(need) : full;
        return limited_grid(max(1, capped));
    };
    auto x_work_items = [&](uint32_t lx, uint32_t ly, uint32_t lz) {
        if (lx == 0U)
            return static_cast<size_t>(0);
        const uint32_t rpt = WALTZ::IDWT_SHARED_BYTES / sizeof(T) / lx;
        if (rpt == 0U)
            return static_cast<size_t>(0);
        const size_t tiles = (static_cast<size_t>(ly) * lz + rpt - 1U) / rpt;
        // idwt_x_tiled assigns one logical tile to a CTA; a CTA's internal
        // threads then walk all rows/elements, so one item per tile is exact.
        return tiles * static_cast<size_t>(TPB);
    };
    for (int lev = levels - 1; lev >= 0; --lev) {
        const uint32_t lx = [&] {
            uint32_t v = dims.x;
            for (int i = 0; i < lev; ++i)
                v -= v >> 1;
            return v;
        }();
        const uint32_t ly = [&] {
            uint32_t v = dims.y;
            for (int i = 0; i < lev; ++i)
                v -= v >> 1;
            return v;
        }();
        const uint32_t lz = [&] {
            uint32_t v = dims.z;
            for (int i = 0; i < lev; ++i)
                v -= v >> 1;
            return v;
        }();
        if constexpr (std::is_same_v<T, float>) {
            const bool use_z_columns = lev == 0 &&
                                       std::getenv("WALTZ_IDWT_Z_COLUMNS") != nullptr &&
                                       d_orig == nullptr && lz * 2U <= WALTZ::IDWT_SHARED_BYTES / sizeof(T);
            const bool use_z_vec4 = lev == 0 && (lx & 3U) == 0U &&
                                    lz * 16U <= WALTZ::IDWT_SHARED_BYTES / sizeof(T);
            if (use_z_columns) {
                const int full =
                    idwt_axis_grid(idwt_z_columns_all_axis_float, IdwtScratch<T>::grid_z_columns);
                const size_t cols = WALTZ::IDWT_SHARED_BYTES / sizeof(T) / (2U * lz);
                const size_t nt = (static_cast<size_t>(lx) * ly + cols - 1U) / cols;
                const int g = work_grid(full, nt * static_cast<size_t>(TPB));
                idwt_z_columns_all_axis_float<<<g, TPB, 0, stream>>>(d_vol,
                                                                     d_tmp,
                                                                     dims.x,
                                                                     dims.y,
                                                                     lz,
                                                                     1,
                                                                     nullptr,
                                                                     0.0,
                                                                     nullptr,
                                                                     nullptr,
                                                                     nullptr,
                                                                     0);
            } else if (use_z_vec4) {
                const int full =
                    idwt_axis_grid(idwt_z_axis_shared_vec4, IdwtScratch<T>::grid_z_shared_vec4);
                const uint32_t tx = WALTZ::IDWT_SHARED_BYTES / sizeof(T) / lz;
                const size_t nt =
                    static_cast<size_t>(ly) * ((static_cast<size_t>(lx) + tx - 1U) / tx);
                const int g = work_grid(full, nt * static_cast<size_t>(TPB));
                idwt_z_axis_shared_vec4<<<g, TPB, 0, stream>>>(
                    d_vol, d_tmp, dims.x, dims.y, lx, ly, lz);
            } else if (lev == 0 && lz * 16u <= WALTZ::IDWT_SHARED_BYTES / sizeof(T)) {
                const int full =
                    idwt_axis_grid(idwt_z_axis_shared<T>, IdwtScratch<T>::grid_z_shared);
                const uint32_t tx = WALTZ::IDWT_SHARED_BYTES / sizeof(T) / lz;
                const size_t nt =
                    static_cast<size_t>(ly) * ((static_cast<size_t>(lx) + tx - 1U) / tx);
                const int g = work_grid(full, nt * static_cast<size_t>(TPB));
                idwt_z_axis_shared<T>
                    <<<g, TPB, 0, stream>>>(d_vol, d_tmp, dims.x, dims.y, lx, ly, lz);
            } else {
                const int full = idwt_axis_grid(idwt_z_axis<T>, IdwtScratch<T>::grid_z);
                const size_t dp = (std::is_same_v<T, float>)
                                      ? static_cast<size_t>((lz - (lz >> 1U) + 1U) >> 1U)
                                      : static_cast<size_t>(lz - (lz >> 1U));
                const int g = work_grid(full, static_cast<size_t>(lx) * ly * dp);
                idwt_z_axis<T><<<g, TPB, 0, stream>>>(d_vol, d_tmp, dims.x, dims.y, lx, ly, lz);
            }
        } else {
            if (lev == 0 && lz * 16U <= WALTZ::IDWT_SHARED_BYTES / sizeof(T)) {
                const int full =
                    idwt_axis_grid(idwt_z_axis_shared<T>, IdwtScratch<T>::grid_z_shared);
                const uint32_t tx = WALTZ::IDWT_SHARED_BYTES / sizeof(T) / lz;
                const size_t nt =
                    static_cast<size_t>(ly) * ((static_cast<size_t>(lx) + tx - 1U) / tx);
                const int g = work_grid(full, nt * static_cast<size_t>(TPB));
                idwt_z_axis_shared<T>
                    <<<g, TPB, 0, stream>>>(d_vol, d_tmp, dims.x, dims.y, lx, ly, lz);
            } else {
                const int full = idwt_axis_grid(idwt_z_axis<T>, IdwtScratch<T>::grid_z);
                const size_t dp = static_cast<size_t>(lz - (lz >> 1U));
                const int g = work_grid(full, static_cast<size_t>(lx) * ly * dp);
                idwt_z_axis<T><<<g, TPB, 0, stream>>>(d_vol, d_tmp, dims.x, dims.y, lx, ly, lz);
            }
        }
        // The fused Y+X kernel is valid for every active dyadic level whose
        // x-line fits the fixed 16-row shared tile.  Historically it was
        // restricted to the finest level because that was the first path
        // benchmarked; at coarser levels the same tensor-product operation
        // applies and eliminating the intermediate Y global write/read saves
        // another full active-cube pass.  Keep an opt-out for shape-specific
        // A/B runs while making the safe power-of-two/<=512 case available.
        const bool fuse_yx_all = std::is_same_v<T, float> && lx <= 512U && ly >= 2U &&
                                 (lev == 0 || std::getenv("WALTZ_IDWT_YX_ALL_OFF") == nullptr);
        // The generalized shared-tile implementation chains four Y pairs and
        // uses vector stores.  It has no PWE side channel, so use it for the
        // decode/reconstruction path (d_orig == nullptr); compression keeps
        // the PWE-capable kernel below.
        const bool use_yx_ext = std::is_same_v<T, float> && lev == 0 && lx <= 512U && ly >= 2U;
        if constexpr (sizeof(T) == 8U) {
            const uint32_t pairs = idwt_double_yx_pairs(lx);
            const size_t nt =
                ((static_cast<size_t>(ly - (ly >> 1U)) + pairs - 1U) / pairs) * lz;
            const int full =
                idwt_axis_grid(idwt_yx_axis_fused_double, IdwtScratch<T>::grid_yx);
            const int g = work_grid(full, nt * static_cast<size_t>(TPB));
            const size_t smem = 2ULL * pairs * lx * sizeof(double);
            idwt_yx_axis_fused_double<<<g, TPB, smem, stream>>>(
                reinterpret_cast<const double*>(d_tmp),
                reinterpret_cast<double*>(d_vol),
                dims.x,
                dims.y,
                lx,
                ly,
                lz,
                pairs,
                reinterpret_cast<const double*>(lev == 0 ? d_orig : nullptr),
                bound,
                d_ocnt,
                d_oidx,
                reinterpret_cast<double*>(d_oerr),
                ocap);
        } else if (use_yx_ext) {
            const uint32_t pairs = lx >= 512U ? 6U : 8U;
            const int full = idwt_axis_grid(idwt_yx_axis_fused_plane_float_ext<TPB>,
                                            IdwtScratch<T>::grid_yx_ext);
            const size_t low = static_cast<size_t>(ly - (ly >> 1U));
            const size_t nt = ((low + pairs - 1U) / pairs) * static_cast<size_t>(lz);
            const int g = work_grid(full, nt * static_cast<size_t>(TPB));
            const size_t smem = 2ULL * pairs * lx * sizeof(float);
            idwt_yx_axis_fused_plane_float_ext<<<g, TPB, smem, stream>>>(
                reinterpret_cast<const float*>(d_tmp),
                reinterpret_cast<float*>(d_vol),
                dims.x,
                dims.y,
                dims.z,
                lx,
                ly,
                pairs,
                0U,
                0U,
                0,
                reinterpret_cast<const float*>(lev == 0 ? d_orig : nullptr),
                bound,
                d_ocnt,
                d_oidx,
                reinterpret_cast<float*>(d_oerr),
                ocap);
        } else if (fuse_yx_all) {
            const int full = idwt_axis_grid(idwt_yx_axis_fused_float, IdwtScratch<T>::grid_yx);
            const size_t low = static_cast<size_t>(ly - (ly >> 1U));
            const size_t nt = ((low + 11U) / 12U) * static_cast<size_t>(lz);
            const int g = work_grid(full, nt * static_cast<size_t>(TPB));
            idwt_yx_axis_fused_float<<<g, TPB, 0, stream>>>(
                reinterpret_cast<const float*>(d_tmp),
                reinterpret_cast<float*>(d_vol),
                dims.x,
                dims.y,
                lx,
                ly,
                lz,
                reinterpret_cast<const float*>(lev == 0 ? d_orig : nullptr),
                bound,
                d_ocnt,
                d_oidx,
                reinterpret_cast<float*>(d_oerr),
                ocap);
        } else {
            const int full_y = idwt_axis_grid(idwt_y_axis<T>, IdwtScratch<T>::grid_y);
            const size_t low_y = static_cast<size_t>(ly - (ly >> 1U));
            const size_t dp_y = (std::is_same_v<T, float>) ? ((low_y + 1U) >> 1U) : low_y;
            const int gy = work_grid(full_y, static_cast<size_t>(lx) * dp_y * lz);
            idwt_y_axis<T><<<gy, TPB, 0, stream>>>(d_tmp, d_vol, dims.x, dims.y, lx, ly, lz);
            const int full_x = idwt_axis_grid(idwt_x_axis<T>, IdwtScratch<T>::grid_x);
            const int gx = work_grid(full_x, x_work_items(lx, ly, lz));
            idwt_x_axis<T><<<gx, TPB, 0, stream>>>(d_vol,
                                                   dims.x,
                                                   dims.y,
                                                   lx,
                                                   ly,
                                                   lz,
                                                   lev == 0 ? d_orig : nullptr,
                                                   bound,
                                                   d_ocnt,
                                                   d_oidx,
                                                   d_oerr,
                                                   ocap);
        }
    }
    cudaError_t e = cudaGetLastError();
    if (out_kernel_ms != nullptr) {
        cudaEventRecord(k1, stream);
        cudaEventSynchronize(k1);
        cudaEventElapsedTime(out_kernel_ms, k0, k1);
    }
    return e;
}


// Contiguous copy of `count` elements src -> dst (grid-stride).  The Z inverse below is
// a single out-of-place pass per level, so the synthesized [0,lz) block is copied back
// into vol -- this preserves the frozen coarser-level detail at z >= lz (untouched here).
template <typename T>
__device__ __forceinline__ void idwt_plane_copy(T* dst, const T* src, size_t count) {
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < count;
         i += stride)
        dst[i] = src[i];
}

// All levels_z of the inverse Z transform in SHARED, in place on `vol` -- the exact
// mirror of the forward plane_z_columns_all.  A tile of z-columns (each = fixed (x,y),
// all z) is loaded once (coalesced over the column index), synthesized coarse-to-fine
// through every level via shared ping-pong (carrying the still-frozen finer detail at
// z >= lz across each swap, just like the forward carries [active_z, full_z)), then
// written back once.  This replaces the per-level global pass + copy-back, killing the
// huge-z-stride (leap_z) redundant global traffic.  Needs full_z*2 <= cap; caller checks.
template <typename T>
__device__ __forceinline__ void idwt_z_columns_all(const T* src,
                                                   T* dst,
                                                   uint32_t nx,
                                                   uint32_t ny,
                                                   uint32_t nz,
                                                   int levels_z,
                                                   T* s_data,
                                                   uint32_t cap,
                                                   const T* orig,
                                                   double bound,
                                                   unsigned int* ocnt,
                                                   uint32_t* oidx,
                                                   T* oerr,
                                                   uint32_t ocap) {
    const uint32_t full_z = nz;
    const uint32_t total_columns = nx * ny; // == leap_z
    uint32_t cols = cap / (full_z * 2U);
    cols = min(cols, total_columns);
    // Keep each complete float tile four-column aligned so the long-stride
    // staging traffic is emitted as 128-bit transactions.  The scalar tail
    // remains fully general (odd dimensions and the final partial tile are
    // common in user data).  This mirrors the forward plane-Z staging path.
    if constexpr (sizeof(T) == sizeof(float)) {
        const uint32_t aligned_cols = cols & ~3U;
        if (aligned_cols >= 4U)
            cols = aligned_cols;
    }
    if (cols == 0)
        return;
    const uint32_t max_count = (total_columns - 1U) / cols + 1U;
    const uint32_t buf = cols * full_z;
    const size_t stride_z = static_cast<size_t>(nx) * ny;
    float pwe_fast_bound = 0.0f;
    if constexpr (std::is_same_v<T, float>) {
        const float bf = __double2float_rn(bound);
        pwe_fast_bound = __uint_as_float(__float_as_uint(bf) - 1U);
    }

    for (uint32_t chunkID = blockIdx.x; chunkID < max_count; chunkID += gridDim.x) {
        const uint32_t col0 = chunkID * cols;
        const uint32_t ncol = min(cols, total_columns - col0);
        const uint32_t tile_elems = ncol * full_z;
        T* s_src = s_data;
        T* s_dst = s_data + buf;

        const bool vector_tile = sizeof(T) == sizeof(float) && ncol == cols && (ncol & 3U) == 0U &&
                                 ((col0 | static_cast<uint32_t>(stride_z)) & 3U) == 0U;
        if constexpr (sizeof(T) == sizeof(float)) {
            if (vector_tile) {
                const uint32_t vec_cols = ncol >> 2U;
                const uint32_t vec_elems = full_z * vec_cols;
                float4* const sh4 = reinterpret_cast<float4*>(s_src);
                for (uint32_t i = threadIdx.x; i < vec_elems; i += TPB) {
                    const uint32_t z = i / vec_cols;
                    const uint32_t vc = i - z * vec_cols;
                    sh4[i] = *reinterpret_cast<const float4*>(src + (col0 + (vc << 2U)) +
                                                              static_cast<size_t>(z) * stride_z);
                }
            } else {
                for (uint32_t i = threadIdx.x; i < tile_elems; i += TPB) {
                    const uint32_t z = i / ncol, lc = i - z * ncol;
                    s_src[i] = src[(col0 + lc) + static_cast<size_t>(z) * stride_z];
                }
            }
        } else {
            for (uint32_t i = threadIdx.x; i < tile_elems; i += TPB) {
                const uint32_t z = i / ncol, lc = i - z * ncol;
                s_src[i] = src[(col0 + lc) + static_cast<size_t>(z) * stride_z];
            }
        }
        __syncthreads();

        for (int lev = levels_z - 1; lev >= 0; --lev) { // coarse-to-fine (expand)
            const uint32_t lz = approx_len_dev(full_z, lev);
            const uint32_t low_lz = lz - (lz >> 1);
            const long long last = static_cast<long long>(lz) - 1;
            // carry the still-frozen finer detail [lz, full_z) across the ping-pong swap
            for (uint32_t i = threadIdx.x + lz * ncol; i < tile_elems; i += TPB)
                s_dst[i] = s_src[i];
            // Deinterleaved low|high are each contiguous (stride ncol) in shared, so interior
            // pairs read the low/high windows directly (no per-tap deinterleave); boundary
            // pairs take the sym_idx slow path.  (This beat both a 1-region read and a fully
            // uniform branchless variant -- see waltz-idwt-perf memory.)
            const uint32_t half = lz >> 1;
            const uint32_t hi_i = (half >= 3u) ? (half - 3u) : 0u;
            const uint32_t pairs = low_lz * ncol;
            if constexpr (sizeof(T) == sizeof(float)) {
                // Full-width tiles are staged four-column aligned.  Vectorize the
                // synthesis itself as well as the staging/writeback; the scalar path
                // below remains for the tail tile and odd physical widths.
                if (ncol >= 4U && (ncol & 3U) == 0U) {
                    const uint32_t vec_cols = ncol >> 2U;
                    const uint32_t vec_pairs = low_lz * vec_cols;
                    for (uint32_t vi = threadIdx.x; vi < vec_pairs; vi += TPB) {
                        const uint32_t i = vi / vec_cols;
                        const uint32_t lc = (vi - i * vec_cols) << 2U;
                        float4 oe, oo;
                        auto C4 = [&](long long idx) -> float4 {
                            const uint32_t pos = sym_idx(idx, last);
                            const uint32_t k = (pos & 1u) ? (low_lz + (pos >> 1)) : (pos >> 1);
                            return *reinterpret_cast<const float4*>(
                                s_src + static_cast<size_t>(k) * ncol + lc);
                        };
                        if (i >= 2u && i <= hi_i) {
                            // Logical taps b..b+8 alternate low/high; the shared
                            // layout stores all low rows first, then all high rows.
                            const float4 c0 = *reinterpret_cast<const float4*>(
                                s_src + static_cast<size_t>(low_lz + i - 2U) * ncol + lc);
                            const float4 c1 = *reinterpret_cast<const float4*>(
                                s_src + static_cast<size_t>(i - 1U) * ncol + lc);
                            const float4 c2 = *reinterpret_cast<const float4*>(
                                s_src + static_cast<size_t>(low_lz + i - 1U) * ncol + lc);
                            const float4 c3 = *reinterpret_cast<const float4*>(
                                s_src + static_cast<size_t>(i) * ncol + lc);
                            const float4 c4 = *reinterpret_cast<const float4*>(
                                s_src + static_cast<size_t>(low_lz + i) * ncol + lc);
                            const float4 c5 = *reinterpret_cast<const float4*>(
                                s_src + static_cast<size_t>(i + 1U) * ncol + lc);
                            const float4 c6 = *reinterpret_cast<const float4*>(
                                s_src + static_cast<size_t>(low_lz + i + 1U) * ncol + lc);
                            const float4 c7 = *reinterpret_cast<const float4*>(
                                s_src + static_cast<size_t>(i + 2U) * ncol + lc);
                            const float4 c8 = *reinterpret_cast<const float4*>(
                                s_src + static_cast<size_t>(low_lz + i + 2U) * ncol + lc);
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
                        *reinterpret_cast<float4*>(s_dst + static_cast<size_t>(2U * i) * ncol +
                                                   lc) = oe;
                        if (2U * i + 1U < lz)
                            *reinterpret_cast<float4*>(
                                s_dst + static_cast<size_t>(2U * i + 1U) * ncol + lc) = oo;
                    }
                } else {
                    for (uint32_t e = threadIdx.x; e < pairs; e += TPB) {
                        const uint32_t i = e / ncol, lc = e - i * ncol;
                        const T* col = s_src + lc;
                        T oe, oo;
                        if (i >= 2u && i <= hi_i) {
                            const size_t lo = static_cast<size_t>(i) * ncol;
                            const size_t hi = static_cast<size_t>(low_lz + i) * ncol;
                            const size_t s = ncol;
                            const T l_m1 = col[lo - s], l_0 = col[lo], l_p1 = col[lo + s],
                                    l_p2 = col[lo + 2 * s];
                            const T h_m2 = col[hi - 2 * s], h_m1 = col[hi - s], h_0 = col[hi],
                                    h_p1 = col[hi + s], h_p2 = col[hi + 2 * s];
                            oe = T(IL0) * (h_m2 + h_p1) + T(IL1) * (l_m1 + l_p1) +
                                 T(IL2) * (h_m1 + h_0) + T(IL3) * l_0;
                            oo = T(IH0) * (h_m2 + h_p2) + T(IH1) * (l_m1 + l_p2) +
                                 T(IH2) * (h_m1 + h_p1) + T(IH3) * (l_0 + l_p1) + T(IH4) * h_0;
                        } else {
                            auto C = [&](long long idx) -> T {
                                const uint32_t pos = sym_idx(idx, last);
                                return col[static_cast<size_t>((pos & 1u) ? (low_lz + (pos >> 1))
                                                                          : (pos >> 1)) *
                                           ncol];
                            };
                            synth_pair<T>(i, C, oe, oo);
                        }
                        s_dst[static_cast<size_t>(2 * i) * ncol + lc] = oe;
                        if (2 * i + 1 < lz)
                            s_dst[static_cast<size_t>(2 * i + 1) * ncol + lc] = oo;
                    }
                }
            } else {
                for (uint32_t e = threadIdx.x; e < pairs; e += TPB) {
                    const uint32_t i = e / ncol, lc = e - i * ncol;
                    const T* col = s_src + lc;
                    T oe, oo;
                    if (i >= 2u && i <= hi_i) {
                        const size_t lo = static_cast<size_t>(i) * ncol;
                        const size_t hi = static_cast<size_t>(low_lz + i) * ncol;
                        const size_t s = ncol;
                        const T l_m1 = col[lo - s], l_0 = col[lo], l_p1 = col[lo + s],
                                l_p2 = col[lo + 2 * s];
                        const T h_m2 = col[hi - 2 * s], h_m1 = col[hi - s], h_0 = col[hi],
                                h_p1 = col[hi + s], h_p2 = col[hi + 2 * s];
                        oe = T(IL0) * (h_m2 + h_p1) + T(IL1) * (l_m1 + l_p1) +
                             T(IL2) * (h_m1 + h_0) + T(IL3) * l_0;
                        oo = T(IH0) * (h_m2 + h_p2) + T(IH1) * (l_m1 + l_p2) +
                             T(IH2) * (h_m1 + h_p1) + T(IH3) * (l_0 + l_p1) + T(IH4) * h_0;
                    } else {
                        auto C = [&](long long idx) -> T {
                            const uint32_t pos = sym_idx(idx, last);
                            return col[static_cast<size_t>((pos & 1u) ? (low_lz + (pos >> 1))
                                                                      : (pos >> 1)) *
                                       ncol];
                        };
                        synth_pair<T>(i, C, oe, oo);
                    }
                    s_dst[static_cast<size_t>(2 * i) * ncol + lc] = oe;
                    if (2 * i + 1 < lz)
                        s_dst[static_cast<size_t>(2 * i + 1) * ncol + lc] = oo;
                }
            }
            __syncthreads();
            T* sw = s_src;
            s_src = s_dst;
            s_dst = sw;
        }

        if constexpr (sizeof(T) == sizeof(float)) {
            if (vector_tile && orig == nullptr) {
                const uint32_t vec_cols = ncol >> 2U;
                const uint32_t vec_elems = full_z * vec_cols;
                const float4* const sh4 = reinterpret_cast<const float4*>(s_src);
                for (uint32_t i = threadIdx.x; i < vec_elems; i += TPB) {
                    const uint32_t z = i / vec_cols;
                    const uint32_t vc = i - z * vec_cols;
                    *reinterpret_cast<float4*>(dst + (col0 + (vc << 2U)) +
                                               static_cast<size_t>(z) * stride_z) = sh4[i];
                }
            } else {
                for (uint32_t i = threadIdx.x; i < tile_elems; i += TPB) {
                    const uint32_t z = i / ncol, lc = i - z * ncol;
                    const size_t g = (col0 + lc) + static_cast<size_t>(z) * stride_z;
                    const T out = s_src[i];
                    // On the compression validation path this is the final inverse stage:
                    // only the PWE comparison/records are consumed afterwards.  Skip the
                    // reconstructed-volume store there; decompression (orig == nullptr)
                    // must and does retain the ordinary writeback.
                    if (orig == nullptr) {
                        dst[g] = out;
                    } else {
                        if constexpr (std::is_same_v<T, float>) {
                            const float fe =
                                __fsub_rn(static_cast<float>(out), static_cast<float>(orig[g]));
                            if (fabsf(fe) <= pwe_fast_bound)
                                continue;
                        }
                        const T e = out - orig[g];
                        const T typed_bound = static_cast<T>(bound);
                        if (e > typed_bound || e < -typed_bound) {
                            const unsigned int slot = atomicAdd(ocnt, 1U);
                            if (oidx != nullptr && slot < ocap) {
                                oidx[slot] = static_cast<uint32_t>(g);
                                oerr[slot] = e;
                            }
                        }
                    }
                }
            }
        } else {
            for (uint32_t i = threadIdx.x; i < tile_elems; i += TPB) {
                const uint32_t z = i / ncol, lc = i - z * ncol;
                const size_t g = (col0 + lc) + static_cast<size_t>(z) * stride_z;
                const T out = s_src[i];
                // On the compression validation path this is the final inverse stage:
                // only the PWE comparison/records are consumed afterwards.  Skip the
                // reconstructed-volume store there; decompression (orig == nullptr)
                // must and does retain the ordinary writeback.
                if (orig == nullptr) {
                    dst[g] = out;
                } else {
                    if constexpr (std::is_same_v<T, float>) {
                        const float fe =
                            __fsub_rn(static_cast<float>(out), static_cast<float>(orig[g]));
                        if (fabsf(fe) <= pwe_fast_bound)
                            continue;
                    }
                    const T e = out - orig[g];
                    const T typed_bound = static_cast<T>(bound);
                    if (e > typed_bound || e < -typed_bound) {
                        const unsigned int slot = atomicAdd(ocnt, 1U);
                        if (oidx != nullptr && slot < ocap) {
                            oidx[slot] = static_cast<uint32_t>(g);
                            oerr[slot] = e;
                        }
                    }
                }
            }
        }
        __syncthreads();
    }
}

__global__ __launch_bounds__(TPB, CDF97_MIN_BLOCKS_D) void idwt_yx_axis_fused_plane_double(
    const double* __restrict__ src,
    double* __restrict__ dst,
    uint32_t nx,
    uint32_t ny,
    uint32_t nz,
    uint32_t lx,
    uint32_t ly,
    uint32_t pairs,
    uint32_t next_lx,
    uint32_t next_ly,
    bool carry) {
    extern __shared__ __align__(128) double plane_double_rows[];
    idwt_yx_tiled_double(src,
                         dst,
                         plane_double_rows,
                         2U * pairs * lx,
                         nx,
                         ny,
                         lx,
                         ly,
                         nz,
                         nullptr,
                         0.0,
                         nullptr,
                         nullptr,
                         nullptr,
                         0U);
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

__global__ __launch_bounds__(TPB, CDF97_MIN_BLOCKS_D) void idwt_z_columns_all_axis_double(
    const double* src,
    double* dst,
    uint32_t nx,
    uint32_t ny,
    uint32_t nz,
    int levels_z,
    const double* orig,
    double bound,
    unsigned int* ocnt,
    uint32_t* oidx,
    double* oerr,
    uint32_t ocap,
    uint32_t cap) {
    extern __shared__ __align__(128) double plane_double_z[];
    idwt_z_columns_all<double>(src,
                               dst,
                               nx,
                               ny,
                               nz,
                               levels_z,
                               plane_double_z,
                               cap,
                               orig,
                               bound,
                               ocnt,
                               oidx,
                               oerr,
                               ocap);
}

inline cudaError_t idwt3d_plane_hybrid_double(double* d_vol,
                                               double* d_tmp,
                                               dim3 dims,
                                               int levels_xy,
                                               int levels_z,
                                               cudaStream_t stream = 0,
                                               const double* d_orig = nullptr,
                                               double bound = 0.0,
                                               unsigned int* d_ocnt = nullptr,
                                               uint32_t* d_oidx = nullptr,
                                               double* d_oerr = nullptr,
                                               uint32_t ocap = 0) {
    if (d_vol == nullptr || levels_xy < 1 || levels_z < 1)
        return cudaErrorInvalidValue;
    if (d_tmp == nullptr) {
        const size_t bytes = static_cast<size_t>(dims.x) * dims.y * dims.z * sizeof(double);
        cudaError_t e = idwt_ensure_tmp<double>(bytes);
        if (e != cudaSuccess)
            return e;
        d_tmp = static_cast<double*>(IdwtScratch<double>::tmp);
    }
    cudaError_t e = idwt3d_plane_prealloc<double>(dims);
    if (e != cudaSuccess)
        return e;

    auto active_len = [](uint32_t n, int lev) {
        uint32_t low = n;
        for (int i = 0; i < lev; ++i)
            low -= low >> 1U;
        return low;
    };
    double* src = d_vol;
    double* dst = d_tmp;
    for (int lev = levels_xy - 1; lev >= 0; --lev) {
        const uint32_t lx = active_len(dims.x, lev);
        const uint32_t ly = active_len(dims.y, lev);
        const uint32_t pairs = idwt_double_yx_pairs(lx);
        const uint32_t next_lx = lev > 0 ? active_len(dims.x, lev - 1) : lx;
        const uint32_t next_ly = lev > 0 ? active_len(dims.y, lev - 1) : ly;
        const bool carry = lev > 0 && dst == d_tmp;
        const size_t smem = 2ULL * pairs * lx * sizeof(double);
        idwt_yx_axis_fused_plane_double<<<IDWTConfig<double>::yx_blocks, TPB, smem, stream>>>(
            src, dst, dims.x, dims.y, dims.z, lx, ly, pairs, next_lx, next_ly, carry);
        double* swap = src;
        src = dst;
        dst = swap;
    }
    e = cudaGetLastError();
    if (e != cudaSuccess)
        return e;

    const size_t z_smem = static_cast<size_t>(IDWTConfig<double>::z_capacity) * sizeof(double);
    idwt_z_columns_all_axis_double<<<IDWTConfig<double>::z_blocks, TPB, z_smem, stream>>>(
        src,
        d_vol,
        dims.x,
        dims.y,
        dims.z,
        levels_z,
        d_orig,
        bound,
        d_ocnt,
        d_oidx,
        d_oerr,
        ocap,
        IDWTConfig<double>::z_capacity);
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

// Float plane inverse-Z without frozen-detail ping-pong copies.  The original
// coefficient bands remain immutable in the first shared region.  Reconstructed
// low prefixes alternate through ceil(z/2) and ceil(z/4) scratch regions; the
// finest level writes directly to global memory (or the PWE recorder).  This
// uses about 1.75*z values per column instead of 2*z and removes every carry.
__device__ __forceinline__ void idwt_z_columns_all_float_nocarry(const float* src,
                                                                 float* dst,
                                                                 uint32_t nx,
                                                                 uint32_t ny,
                                                                 uint32_t nz,
                                                                 int levels_z,
                                                                 float* s_data,
                                                                 uint32_t cap,
                                                                 const float* orig,
                                                                 double bound,
                                                                 unsigned int* ocnt,
                                                                 uint32_t* oidx,
                                                                 float* oerr,
                                                                 uint32_t ocap,
                                                                 bool direct_error = false) {
    const uint32_t full_z = nz;
    const uint32_t a1 = full_z - (full_z >> 1U);
    const uint32_t a2 = a1 - (a1 >> 1U);
    const uint32_t per_col = full_z + a1 + a2;
    const uint32_t total_columns = nx * ny;
    uint32_t cols = min(cap / per_col, total_columns);
    const uint32_t aligned_cols = cols & ~3U;
    if (aligned_cols >= 4U)
        cols = aligned_cols;
    if (cols == 0U)
        return;

    const uint32_t max_count = (total_columns - 1U) / cols + 1U;
    const uint32_t imm_cap = full_z * cols;
    const uint32_t scratch_a_cap = a1 * cols;
    const size_t stride_z = static_cast<size_t>(nx) * ny;
    const float bf = __double2float_rn(bound);
    const float fast_bound = __uint_as_float(__float_as_uint(bf) - 1U);

    for (uint32_t chunkID = blockIdx.x; chunkID < max_count; chunkID += gridDim.x) {
        const uint32_t col0 = chunkID * cols;
        const uint32_t ncol = min(cols, total_columns - col0);
        const uint32_t tile_elems = ncol * full_z;
        float* const imm = s_data;
        float* const scratch_a = s_data + imm_cap;
        float* const scratch_b = scratch_a + scratch_a_cap;
        const bool vector_tile =
            (ncol & 3U) == 0U && ((col0 | static_cast<uint32_t>(stride_z)) & 3U) == 0U;

        if (vector_tile) {
            const uint32_t vec_cols = ncol >> 2U;
            const uint32_t vec_elems = full_z * vec_cols;
            float4* const sh4 = reinterpret_cast<float4*>(imm);
            for (uint32_t i = threadIdx.x; i < vec_elems; i += TPB) {
                const uint32_t z = i / vec_cols;
                const uint32_t vc = i - z * vec_cols;
                sh4[i] = *reinterpret_cast<const float4*>(src + col0 + (vc << 2U) +
                                                          static_cast<size_t>(z) * stride_z);
            }
        } else {
            for (uint32_t i = threadIdx.x; i < tile_elems; i += TPB) {
                const uint32_t z = i / ncol;
                const uint32_t lc = i - z * ncol;
                imm[i] = src[col0 + lc + static_cast<size_t>(z) * stride_z];
            }
        }
        __syncthreads();

        const float* low_src = imm;
        float* out_sh = (levels_z & 1) == 0 ? scratch_a : scratch_b;
        for (int lev = levels_z - 1; lev >= 0; --lev) {
            const uint32_t lz = approx_len_dev(full_z, lev);
            const uint32_t low_lz = lz - (lz >> 1U);
            const long long last = static_cast<long long>(lz) - 1;
            const uint32_t half = lz >> 1U;
            const uint32_t hi_i = half >= 3U ? half - 3U : 0U;
            const bool final = lev == 0;

            if (vector_tile) {
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
            } else {
                const uint32_t pairs = low_lz * ncol;
                for (uint32_t e = threadIdx.x; e < pairs; e += TPB) {
                    const uint32_t i = e / ncol;
                    const uint32_t lc = e - i * ncol;
                    auto C = [&](long long idx) -> float {
                        const uint32_t pos = sym_idx(idx, last);
                        return (pos & 1U)
                                   ? imm[static_cast<size_t>(low_lz + (pos >> 1U)) * ncol + lc]
                                   : low_src[static_cast<size_t>(pos >> 1U) * ncol + lc];
                    };
                    float oe, oo;
                    synth_pair<float>(i, C, oe, oo);
                    if (!final) {
                        out_sh[static_cast<size_t>(2U * i) * ncol + lc] = oe;
                        if (2U * i + 1U < lz)
                            out_sh[static_cast<size_t>(2U * i + 1U) * ncol + lc] = oo;
                    } else {
                        const size_t g0 = col0 + lc + static_cast<size_t>(2U * i) * stride_z;
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

// Generalized XY fused inverse used by the plane IDWT. `lx`/`ly` describe the
// currently active rectangle while
// `nx`/`ny` remain the physical row/slice strides.  This matters for ordinary dyadic
// plane layout: at coarse levels the active rectangle is only a prefix of every
// physical slice, but the untouched detail coefficients outside it must stay in place
// for the next (larger) inverse level.  When `carry` is set, the kernel also forwards
// the newly exposed annulus of `next_lx` x `next_ly` from src to dst.  That lets host
// code ping-pong the two buffers without a separate full-volume copy.
template <int KTPB>
__device__ __forceinline__ void
idwt_yx_axis_fused_plane_float_ext_body(const float* __restrict__ src,
                                        float* __restrict__ dst,
                                        uint32_t nx_arg,
                                        uint32_t ny_arg,
                                        uint32_t nz_arg,
                                        uint32_t lx_arg,
                                        uint32_t ly_arg,
                                        uint32_t pairs_arg,
                                        uint32_t next_lx,
                                        uint32_t next_ly,
                                        int carry,
                                        const float* orig,
                                        double bound,
                                        unsigned int* ocnt,
                                        uint32_t* oidx,
                                        float* oerr,
                                        uint32_t ocap,
                                        float* rows,
                                        uint32_t block_index,
                                        uint32_t grid_size) {
    const uint32_t nx = nx_arg;
    const uint32_t ny = ny_arg;
    const uint32_t nz = nz_arg;
    const uint32_t lx = lx_arg;
    const uint32_t ly = ly_arg;
    const uint32_t pairs_per_tile = pairs_arg;
    if (nx < 2U || ny < 2U || lx < 2U || ly < 2U || lx > nx || ly > ny || pairs_per_tile == 0U)
        return;

    const float pwe_fast_bound =
        orig != nullptr ? __uint_as_float(__float_as_uint(__double2float_rn(bound)) - 1U) : 0.0f;

    const size_t leap_z = static_cast<size_t>(nx) * ny;
    const uint32_t low_ly = ly - (ly >> 1U);
    const uint32_t low_lx = lx - (lx >> 1U);
    const long long last_y = static_cast<long long>(ly) - 1;
    const long long last_x = static_cast<long long>(lx) - 1;
    const uint32_t nty = (low_ly + pairs_per_tile - 1U) / pairs_per_tile;
    const uint32_t ntiles = nty * nz;

    for (uint32_t tile = block_index; tile < ntiles; tile += grid_size) {
        const uint32_t z = tile / nty;
        const uint32_t pt = tile - z * nty;
        const uint32_t pair0 = pt * pairs_per_tile;
        const uint32_t pairn = min(pairs_per_tile, low_ly - pair0);
        const uint32_t out_rows = min(2U * pairn, ly - 2U * pair0);
        const uint32_t groups = (pairn + 1U) >> 1U;
        const uint32_t chains = (groups + 1U) >> 1U;
        // Y synthesis stays in shared; unlike the old full-size kernel, rows are
        // packed with the active stride lx (not the physical nx).
        // One thread chains two adjacent synth_quad groups.  Their logical
        // windows overlap by seven taps, so four pairs require 15 source loads
        // instead of 22 while retaining each output expression's operation order.
        const uint32_t ywork = chains * lx;
        const bool vec_y = ((lx & 3U) == 0U) && ((nx & 3U) == 0U);
        if (vec_y) {
            // Four adjacent X columns share every Y tap.  The source rows are
            // physically contiguous in X, so this path turns the strided
            // scalar gathers into aligned 128-bit transactions while retaining
            // the exact per-lane filter arithmetic.
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
                                 float(IH2) * (c2 + c6) + float(IH3) * (c3 + c5) + float(IH4) * c4;
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
                // Logical window 2*i-3 .. 2*i+7 in deinterleaved low|high
                // shared layout.  The expression order exactly matches two
                // adjacent synth_pair calls, but the overlapping taps load once.
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
            if (orig != nullptr) {
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
            } else if (have_second && ox + 3U < lx && (nx & 3U) == 0U) {
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
        __syncthreads();
    }

    // Forward only the annulus needed by the next (larger) rectangle.  The active
    // rectangle is disjoint from this region, so this can run after the tile loop
    // without another global barrier; source and destination are ping-pong buffers.
    if (carry && next_lx > 0U && next_ly > 0U && next_lx <= nx && next_ly <= ny) {
        const size_t stride = static_cast<size_t>(grid_size) * KTPB;
        const size_t first = static_cast<size_t>(block_index) * KTPB + threadIdx.x;
        // Copy the horizontal annulus (old y range, newly exposed x range) and the
        // vertical annulus as separate row sweeps.  Besides removing the per-element
        // branch/modulo from the original flat loop, aligned rows use float4 global
        // transactions.  The scalar tails retain full odd-dimension correctness.
        const uint32_t dx = next_lx > lx ? next_lx - lx : 0U;
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
}

template <int KTPB>
__global__ __launch_bounds__(KTPB, 3) void idwt_yx_axis_fused_plane_float_ext(
    const float* __restrict__ src,
    float* __restrict__ dst,
    uint32_t nx_arg,
    uint32_t ny_arg,
    uint32_t nz_arg,
    uint32_t lx_arg,
    uint32_t ly_arg,
    uint32_t pairs_arg,
    uint32_t next_lx,
    uint32_t next_ly,
    int carry,
    const float* orig,
    double bound,
    unsigned int* ocnt,
    uint32_t* oidx,
    float* oerr,
    uint32_t ocap) {
    extern __shared__ __align__(128) float rows[];
    idwt_yx_axis_fused_plane_float_ext_body<KTPB>(src,
                                                  dst,
                                                  nx_arg,
                                                  ny_arg,
                                                  nz_arg,
                                                  lx_arg,
                                                  ly_arg,
                                                  pairs_arg,
                                                  next_lx,
                                                  next_ly,
                                                  carry,
                                                  orig,
                                                  bound,
                                                  ocnt,
                                                  oidx,
                                                  oerr,
                                                  ocap,
                                                  rows,
                                                  blockIdx.x,
                                                  gridDim.x);
}

__global__
__launch_bounds__(TPB, CDF97_MIN_BLOCKS_F) void idwt_z_columns_all_axis_float(const float* src,
                                                                              float* dst,
                                                                              uint32_t nx,
                                                                              uint32_t ny,
                                                                              uint32_t nz,
                                                                              int levels_z,
                                                                              const float* orig,
                                                                              double bound,
                                                                              unsigned int* ocnt,
                                                                              uint32_t* oidx,
                                                                              float* oerr,
                                                                              uint32_t ocap) {
    __shared__ alignas(128) float s_data[WALTZ::IDWT_SHARED_BYTES / sizeof(float)];
    idwt_z_columns_all<float>(src,
                              dst,
                              nx,
                              ny,
                              nz,
                              levels_z,
                              s_data,
                              WALTZ::IDWT_SHARED_BYTES / sizeof(float),
                              orig,
                              bound,
                              ocnt,
                              oidx,
                              oerr,
                              ocap);
}

// Dynamically sized shared-memory Z inverse.
__global__ __launch_bounds__(TPB, 3) void idwt_z_columns_all_axis_float_dynamic(const float* src,
                                                                                float* dst,
                                                                                uint32_t nx,
                                                                                uint32_t ny,
                                                                                uint32_t nz,
                                                                                int levels_z,
                                                                                const float* orig,
                                                                                double bound,
                                                                                unsigned int* ocnt,
                                                                                uint32_t* oidx,
                                                                                float* oerr,
                                                                                uint32_t ocap,
                                                                                uint32_t cap,
                                                                                bool direct_error) {
    extern __shared__ __align__(128) float s_data[];
    idwt_z_columns_all_float_nocarry(src,
                                     dst,
                                     nx,
                                     ny,
                                     nz,
                                     levels_z,
                                     s_data,
                                     cap,
                                     orig,
                                     bound,
                                     ocnt,
                                     oidx,
                                     oerr,
                                     ocap,
                                     direct_error);
}

template <typename T> inline cudaError_t idwt3d_plane_prealloc(dim3 dims) {
    const cudaDeviceProp& prop = WALTZ::gpu_config().properties;
    if (IDWTConfig<T>::configured_nx == dims.x && IDWTConfig<T>::configured_nz == dims.z)
        return cudaSuccess;

    if constexpr (std::is_same_v<T, double>) {
        constexpr size_t MAX_SHARED_BYTES = 48U * 1024U;
        const size_t budget = min(MAX_SHARED_BYTES, static_cast<size_t>(prop.sharedMemPerBlockOptin));

        const uint32_t pairs = idwt_double_yx_pairs(dims.x);
        const size_t yx_smem = 2ULL * pairs * dims.x * sizeof(T);
        const uint32_t z_cap = static_cast<uint32_t>(budget / sizeof(T));
        if (yx_smem > budget || dims.z * 2U > z_cap)
            return cudaErrorInvalidValue;

        cudaError_t error = cudaFuncSetAttribute(idwt_yx_axis_fused_plane_double,
                                                  cudaFuncAttributeMaxDynamicSharedMemorySize,
                                                  static_cast<int>(yx_smem));
        if (error != cudaSuccess)
            return error;
        const size_t z_smem = static_cast<size_t>(z_cap) * sizeof(T);
        error = cudaFuncSetAttribute(idwt_z_columns_all_axis_double,
                                     cudaFuncAttributeMaxDynamicSharedMemorySize,
                                     static_cast<int>(z_smem));
        if (error != cudaSuccess)
            return error;

        int blocks_per_sm = 0;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &blocks_per_sm, idwt_yx_axis_fused_plane_double, TPB, yx_smem);
        IDWTConfig<T>::yx_blocks = max(1, blocks_per_sm) * prop.multiProcessorCount;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &blocks_per_sm, idwt_z_columns_all_axis_double, TPB, z_smem);
        IDWTConfig<T>::z_blocks = max(1, blocks_per_sm) * prop.multiProcessorCount;
        IDWTConfig<T>::configured_nx = dims.x;
        IDWTConfig<T>::configured_nz = dims.z;
        IDWTConfig<T>::z_capacity = z_cap;
        return cudaGetLastError();
    } else {
        int blocks_per_sm = 0;
        if (IDWTConfig<T>::yx_blocks == 0 || IDWTConfig<T>::configured_nx != dims.x) {
            uint32_t chosen_pairs = WALTZ::IDWT_SHARED_BYTES / sizeof(T) / (2U * dims.x);
            size_t chosen_smem = static_cast<size_t>(2U * chosen_pairs) * dims.x * sizeof(T);
            const bool xy_tpb256 = dims.x >= 1024U;
            cudaError_t error;
            if (xy_tpb256) {
                error = cudaFuncSetAttribute(idwt_yx_axis_fused_plane_float_ext<256>,
                                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                                             static_cast<int>(chosen_smem));
                if (error != cudaSuccess)
                    return error;
                cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                    &blocks_per_sm, idwt_yx_axis_fused_plane_float_ext<256>, 256, chosen_smem);
            } else {
                error = cudaFuncSetAttribute(idwt_yx_axis_fused_plane_float_ext<TPB>,
                                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                                             static_cast<int>(chosen_smem));
                if (error != cudaSuccess)
                    return error;
                cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                    &blocks_per_sm, idwt_yx_axis_fused_plane_float_ext<TPB>, TPB, chosen_smem);
            }
            IDWTConfig<T>::yx_blocks = max(1, blocks_per_sm) * prop.multiProcessorCount;
            IDWTConfig<T>::yx_smem_bytes = chosen_smem;
            IDWTConfig<T>::yx_capacity = 2U * chosen_pairs * dims.x;
            IDWTConfig<T>::configured_nx = dims.x;
        }
        if (IDWTConfig<T>::z_blocks == 0 || IDWTConfig<T>::configured_nz != dims.z) {
            int target_blocks_per_sm = 0;
            constexpr size_t BASELINE_Z_SMEM = 11000U * sizeof(T);
            cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                &target_blocks_per_sm, idwt_z_columns_all_axis_float_dynamic, TPB, BASELINE_Z_SMEM);
            target_blocks_per_sm = max(1, target_blocks_per_sm);
            const size_t sm_budget = min(static_cast<size_t>(prop.sharedMemPerBlockOptin),
                static_cast<size_t>(prop.sharedMemPerMultiprocessor) / static_cast<size_t>(target_blocks_per_sm));
            uint32_t try_cols = static_cast<uint32_t>(sm_budget / (2ULL * dims.z * sizeof(T)));
            try_cols &= ~3U;
            if (try_cols < 4U)
                return cudaErrorInvalidValue;
            const uint32_t chosen_cap = 2U * dims.z * try_cols;
            const size_t chosen_smem = static_cast<size_t>(chosen_cap) * sizeof(T);

            /*
            // Disabled after SCALE/Hurricane A/B testing showed that the first
            // occupancy-budgeted candidate is already accepted on the target GPU.
            uint32_t chosen_cap = 11000U;
            size_t chosen_smem = static_cast<size_t>(chosen_cap) * sizeof(T);
            for (; try_cols >= 4U; try_cols -= 4U) {
                const uint32_t try_cap = 2U * dims.z * try_cols;
                const size_t try_smem = static_cast<size_t>(try_cap) * sizeof(T);
                cudaError_t error = cudaFuncSetAttribute(idwt_z_columns_all_axis_float_dynamic,
                                                          cudaFuncAttributeMaxDynamicSharedMemorySize,
                                                          static_cast<int>(try_smem));
                if (error != cudaSuccess)
                    continue;
                int try_blocks_per_sm = 0;
                cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                    &try_blocks_per_sm, idwt_z_columns_all_axis_float_dynamic, TPB, try_smem);
                if (try_blocks_per_sm >= target_blocks_per_sm) {
                    chosen_cap = try_cap;
                    chosen_smem = try_smem;
                    break;
                }
            }
            */
            cudaError_t error = cudaFuncSetAttribute(idwt_z_columns_all_axis_float_dynamic,
                                                      cudaFuncAttributeMaxDynamicSharedMemorySize,
                                                      static_cast<int>(chosen_smem));
            if (error != cudaSuccess)
                return error;
            cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                &blocks_per_sm, idwt_z_columns_all_axis_float_dynamic, TPB, chosen_smem);
            IDWTConfig<T>::z_blocks = max(1, blocks_per_sm) * prop.multiProcessorCount;
            IDWTConfig<T>::z_smem_bytes = chosen_smem;
            IDWTConfig<T>::z_capacity = chosen_cap;
            IDWTConfig<T>::configured_nz = dims.z;
        }
        return cudaGetLastError();
    }
}

inline cudaError_t idwt3d_plane_hybrid_float(float* d_vol,
                                             float* d_tmp,
                                             dim3 dims,
                                             int levels_xy,
                                             int levels_z,
                                             cudaStream_t stream = 0,
                                             const float* d_orig = nullptr,
                                             double bound = 0.0,
                                             unsigned int* d_ocnt = nullptr,
                                             uint32_t* d_oidx = nullptr,
                                             float* d_oerr = nullptr,
                                             uint32_t ocap = 0,
                                             bool direct_error = false) {
    if (levels_xy < 1 || levels_z < 1 || d_vol == nullptr)
        return cudaErrorInvalidValue;
    if (d_tmp == nullptr) {
        const size_t bytes = static_cast<size_t>(dims.x) * dims.y * dims.z * sizeof(float);
        cudaError_t e = idwt_ensure_tmp<float>(bytes);
        if (e != cudaSuccess)
            return e;
        d_tmp = static_cast<float*>(IdwtScratch<float>::tmp);
    }
    cudaError_t e = idwt3d_plane_prealloc<float>(dims);
    if (e != cudaSuccess)
        return e;

    const int z_grid = IDWTConfig<float>::z_blocks;
    const size_t z_smem = IDWTConfig<float>::z_smem_bytes;
    const uint32_t z_cap = IDWTConfig<float>::z_capacity;
    const bool xy_tpb256 = dims.x >= 1024U;

    // Host mirror of approx_len_dev (low = n - floor(n/2), repeated lev times).
    auto active_len = [](uint32_t n, int lev) {
        uint32_t low = n;
        for (int i = 0; i < lev; ++i)
            low -= low >> 1U;
        return low;
    };
    float* src = d_vol;
    float* dst = d_tmp;
    const uint32_t cap = IDWTConfig<float>::yx_capacity != 0U
                             ? IDWTConfig<float>::yx_capacity
                             : WALTZ::IDWT_SHARED_BYTES / sizeof(float);

    // Coarse-to-fine XY levels. The same runtime-extent kernel handles every
    // volume and forwards the untouched annulus between ping-pong buffers.
    for (int lev = levels_xy - 1; lev >= 1; --lev) {
        const uint32_t lx = active_len(dims.x, lev);
        const uint32_t ly = active_len(dims.y, lev);
        const uint32_t next_lx = active_len(dims.x, lev - 1);
        const uint32_t next_ly = active_len(dims.y, lev - 1);
        const uint32_t pairs = cap / (2U * lx);
        if (pairs == 0U)
            return cudaErrorInvalidValue;
        const size_t smem = static_cast<size_t>(2U * pairs) * lx * sizeof(float);
        const int carry = dst == d_tmp ? 1 : 0;
        // Keep the occupancy-sized grid even when the synthesis tile count is
        // smaller: the same kernel also grid-strides over the carried annulus.
        if (xy_tpb256)
            idwt_yx_axis_fused_plane_float_ext<256>
                <<<IDWTConfig<float>::yx_blocks, 256, smem, stream>>>(src,
                                                                 dst,
                                                                 dims.x,
                                                                 dims.y,
                                                                 dims.z,
                                                                 lx,
                                                                 ly,
                                                                 pairs,
                                                                 next_lx,
                                                                 next_ly,
                                                                 carry,
                                                                 nullptr,
                                                                 0.0,
                                                                 nullptr,
                                                                 nullptr,
                                                                 nullptr,
                                                                 0U);
        else
            idwt_yx_axis_fused_plane_float_ext<<<IDWTConfig<float>::yx_blocks,
                                                  TPB,
                                                  smem,
                                                  stream>>>(src,
                                                           dst,
                                                           dims.x,
                                                           dims.y,
                                                           dims.z,
                                                           lx,
                                                           ly,
                                                           pairs,
                                                           next_lx,
                                                           next_ly,
                                                           carry,
                                                           nullptr,
                                                           0.0,
                                                           nullptr,
                                                           nullptr,
                                                           nullptr,
                                                           0U);
        e = cudaGetLastError();
        if (e != cudaSuccess)
            return e;
        float* sw = src;
        src = dst;
        dst = sw;
    }

    // Finest XY level.  Keep the result in whichever ping-pong buffer is next;
    // the all-Z kernel is tile-staged and is safe with src == dst, so no full
    // volume copy is required when the parity leaves d_vol as the destination.
    const uint32_t pairs = cap / (2U * dims.x);
    if (pairs == 0U)
        return cudaErrorInvalidValue;
    const size_t smem = static_cast<size_t>(2U * pairs) * dims.x * sizeof(float);
    if (xy_tpb256)
        idwt_yx_axis_fused_plane_float_ext<256>
            <<<IDWTConfig<float>::yx_blocks, 256, smem, stream>>>(src,
                                                                 dst,
                                                                 dims.x,
                                                                 dims.y,
                                                                 dims.z,
                                                                 dims.x,
                                                                 dims.y,
                                                                 pairs,
                                                                 0U,
                                                                 0U,
                                                                 0,
                                                                 nullptr,
                                                                 0.0,
                                                                 nullptr,
                                                                 nullptr,
                                                                 nullptr,
                                                                 0U);
    else
        idwt_yx_axis_fused_plane_float_ext
            <<<IDWTConfig<float>::yx_blocks, TPB, smem, stream>>>(
                src,
                dst,
                dims.x,
                dims.y,
                dims.z,
                dims.x,
                dims.y,
                pairs,
                0U,
                0U,
                0,
                nullptr,
                0.0,
                nullptr,
                nullptr,
                nullptr,
                0U);
    e = cudaGetLastError();
    if (e != cudaSuccess)
        return e;

    idwt_z_columns_all_axis_float_dynamic<<<z_grid, TPB, z_smem, stream>>>(dst,
                                                                           d_vol,
                                                                           dims.x,
                                                                           dims.y,
                                                                           dims.z,
                                                                           levels_z,
                                                                           d_orig,
                                                                           bound,
                                                                           d_ocnt,
                                                                           d_oidx,
                                                                           d_oerr,
                                                                           ocap,
                                                                           z_cap,
                                                                           direct_error);
    return cudaGetLastError();
}

// Cooperative inverse of the non-dyadic (wavelet-packet) dwt3d_plane.  The forward did
// Phase 1 (Z to full depth, levels_z levels) THEN Phase 2 (every XY plane dyadically,
// levels_xy levels, X then Y).  The inverse runs in reverse: undo XY first (coarse-to-
// fine, Y-inverse then X-inverse, over all nz planes), then undo Z (coarse-to-fine).
template <typename T>
__global__ __launch_bounds__(TPB, CDF97_MIN_BLOCKS(T)) void idwt3d_plane(T* vol,
                                                                         T* tmp,
                                                                         uint32_t nx,
                                                                         uint32_t ny,
                                                                         uint32_t nz,
                                                                         int levels_xy,
                                                                         int levels_z,
                                                                         const T* orig,
                                                                         double bound,
                                                                         unsigned int* ocnt,
                                                                         uint32_t* oidx,
                                                                         T* oerr,
                                                                         uint32_t ocap) {
    cooperative_groups::grid_group grid = cooperative_groups::this_grid();
    __shared__ alignas(128) T s_data[WALTZ::IDWT_SHARED_BYTES / sizeof(T)];

    auto xy_undo = [&]() {
        // ---- Inverse Phase 2: undo dyadic XY (full nz depth), coarse-to-fine, Y then X ----
        for (int lev = levels_xy - 1; lev >= 0; --lev) {
            const uint32_t lx = approx_len_dev(nx, lev);
            const uint32_t ly = approx_len_dev(ny, lev);
            idwt_y_pass<T>(vol, tmp, nx, ny, lx, ly, nz); // Y-inverse: vol -> tmp (global)
            grid.sync();
            idwt_x_tiled<T>(
                tmp, vol, s_data, WALTZ::IDWT_SHARED_BYTES / sizeof(T), nx, ny, lx, ly, nz); // X-inverse: tmp -> vol
            grid.sync();
        }
    };

    auto z_undo = [&]() {
        // ---- Inverse Phase 1: undo full-depth Z (full nx,ny extent) ----
        if (levels_z > 0) {
            if (nz * 2U <= WALTZ::IDWT_SHARED_BYTES / sizeof(T)) {
                // Fast path: all z-levels in shared, one load/store per column (mirrors fwd).
                idwt_z_columns_all<T>(vol,
                                      vol,
                                      nx,
                                      ny,
                                      nz,
                                      levels_z,
                                      s_data,
                                      WALTZ::IDWT_SHARED_BYTES / sizeof(T),
                                      orig,
                                      bound,
                                      ocnt,
                                      oidx,
                                      oerr,
                                      ocap);
                grid.sync();
            } else {
                // Fallback (very tall z): per-level global pass + copy-back, coarse-to-fine.
                const size_t leap_z = static_cast<size_t>(nx) * ny;
                for (int lev = levels_z - 1; lev >= 0; --lev) {
                    const uint32_t lz = approx_len_dev(nz, lev);
                    idwt_z_pass<T>(vol, tmp, nx, ny, nx, ny, lz); // vol -> tmp ([0,lz) block)
                    grid.sync();
                    idwt_plane_copy<T>(vol, tmp, static_cast<size_t>(lz) * leap_z); // keep z>=lz
                    grid.sync();
                }
            }
        }
    };

    xy_undo();
    z_undo();
}

// Host launcher for the plane (wavelet-packet) inverse.  Mirrors idwt3d_dyadic: persistent
// tmp scratch + cached occupancy grid; optional kernel-only timing via out_kernel_ms.
template <typename T>
inline cudaError_t idwt3d_plane_levels(T* d_vol,
                                       dim3 dims,
                                       int levels_xy,
                                       int levels_z,
                                       cudaStream_t stream = 0,
                                       float* out_kernel_ms = nullptr,
                                       int grid_blocks = 0) {
    if (levels_xy <= 0 && levels_z <= 0)
        return cudaSuccess;
    const size_t bytes = static_cast<size_t>(dims.x) * dims.y * dims.z * sizeof(T);
    cudaError_t em = idwt_ensure_tmp<T>(bytes);
    if (em != cudaSuccess)
        return em;
    T* d_tmp = static_cast<T*>(IdwtScratch<T>::tmp);
    int s_grid = grid_blocks;
    if (s_grid <= 0)
        s_grid = idwt_coop_grid<T>(idwt3d_plane<T>, IdwtScratch<T>::grid_plane);

    const T* d_orig = nullptr;
    double bound = 0.0;
    unsigned int* d_ocnt = nullptr;
    uint32_t* d_oidx = nullptr;
    T* d_oerr = nullptr;
    uint32_t ocap = 0;
    void* args[] = {&d_vol,
                    &d_tmp,
                    &dims.x,
                    &dims.y,
                    &dims.z,
                    &levels_xy,
                    &levels_z,
                    &d_orig,
                    &bound,
                    &d_ocnt,
                    &d_oidx,
                    &d_oerr,
                    &ocap};
    if (out_kernel_ms == nullptr)
        return cudaLaunchCooperativeKernel(
            reinterpret_cast<void*>(idwt3d_plane<T>), dim3(s_grid), dim3(TPB), args, 0, stream);
    static cudaEvent_t k0 = nullptr, k1 = nullptr;
    if (k0 == nullptr) {
        cudaEventCreate(&k0);
        cudaEventCreate(&k1);
    }
    cudaEventRecord(k0, stream);
    cudaError_t e = cudaLaunchCooperativeKernel(
        reinterpret_cast<void*>(idwt3d_plane<T>), dim3(s_grid), dim3(TPB), args, 0, stream);
    cudaEventRecord(k1, stream);
    cudaEventSynchronize(k1);
    cudaEventElapsedTime(out_kernel_ms, k0, k1);
    return e;
}

} // namespace CDF97
