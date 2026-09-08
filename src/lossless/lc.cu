/*
This file is part of the LC framework for synthesizing high-speed parallel lossless and
error-bounded lossy data compression and decompression algorithms for CPUs and GPUs.

BSD 3-Clause License

Copyright (c) 2021-2025, Noushin Azami, Alex Fallin, Brandon Burtchell, Andrew Rodriguez, Benila
Jerald, Yiqian Liu, Anju Mongandampulath Akathoott, and Martin Burtscher All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

URL: The latest version of this code is available at https://github.com/burtscher/LC-framework.

Sponsor: This code is based upon work supported by the U.S. Department of Energy, Office of Science,
Office of Advanced Scientific Research (ASCR), under contract DE-SC0022223.
*/

#define NDEBUG

#include <cassert>
#include <cmath>
#include <cstdlib>
#include <cuda.h>
#include <stdexcept>
#include <string>

using byte = unsigned char;
static const int CS = 1024 * 16;
static const int TPB = 512; // threads per block [must be power of 2 and at least 128]
#if defined(__AMDGCN_WAVEFRONT_SIZE) && (__AMDGCN_WAVEFRONT_SIZE == 64)
#define WS 64
#else
#define WS 32
#endif

#include "include/macros.h"
#include "include/max_scan.h"
#include "include/prefix_sum.h"
#include "include/sum_reduction.h"
#include "components/d_BIT_2.h"
#include "components/d_BIT_4.h"
#include "components/d_RZE_1.h"
#include "utils/Config.hpp"

// copy (len) bytes from shared memory (source) to global memory (destination)
// source must we word aligned
static inline __device__ void
s2g(void* const __restrict__ destination, const void* const __restrict__ source, const int len) {
    const int tid = threadIdx.x;
    const byte* const __restrict__ input = (byte*)source;
    byte* const __restrict__ output = (byte*)destination;
    if (len < 128) {
        if (tid < len)
            output[tid] = input[tid];
    } else {
        const int nonaligned = (int)(size_t)output;
        const int wordaligned = (nonaligned + 3) & ~3;
        const int linealigned = (nonaligned + 127) & ~127;
        const int bcnt = wordaligned - nonaligned;
        const int wcnt = (linealigned - wordaligned) / 4;
        const int* const __restrict__ in_w = (int*)input;
        if (bcnt == 0) {
            int* const __restrict__ out_w = (int*)output;
            if (tid < wcnt)
                out_w[tid] = in_w[tid];
            for (int i = tid + wcnt; i < len / 4; i += TPB) {
                out_w[i] = in_w[i];
            }
            if (tid < (len & 3)) {
                const int i = len - 1 - tid;
                output[i] = input[i];
            }
        } else {
            const int shift = bcnt * 8;
            const int rlen = len - bcnt;
            int* const __restrict__ out_w = (int*)&output[bcnt];
            if (tid < bcnt)
                output[tid] = input[tid];
            if (tid < wcnt)
                out_w[tid] = __funnelshift_r(in_w[tid], in_w[tid + 1], shift);
            for (int i = tid + wcnt; i < rlen / 4; i += TPB) {
                out_w[i] = __funnelshift_r(in_w[i], in_w[i + 1], shift);
            }
            if (tid < (rlen & 3)) {
                const int i = len - 1 - tid;
                output[i] = input[i];
            }
        }
    }
}

static __device__ unsigned long long g_chunk_counter;

static __global__ void d_reset() {
    g_chunk_counter = 0LL;
}

static inline __device__ void propagate_carry(const int value,
                                              const long long chunkID,
                                              volatile long long* const __restrict__ fullcarry,
                                              long long* const __restrict__ s_fullc) {
    if (threadIdx.x == TPB - 1) { // last thread
        fullcarry[chunkID] = (chunkID == 0) ? (long long)value : (long long)-value;
    }

    if (chunkID != 0) {
        if (threadIdx.x + WS >= TPB) { // last warp
            const int lane = threadIdx.x % WS;
            const long long cidm1ml = chunkID - 1 - lane;
            long long val = -1;
            __syncwarp(); // not optional
            do {
                if (cidm1ml >= 0) {
                    val = fullcarry[cidm1ml];
                }
            } while ((__any_sync(0xFFFFFFFFu, val == 0)) || (__all_sync(0xFFFFFFFFu, val <= 0)));
#if defined(WS) && (WS == 64)
            const long long mask = __ballot_sync(0xFFFFFFFFu, val > 0);
            const int pos = __ffsll(mask) - 1;
#else
            const int mask = __ballot_sync(0xFFFFFFFFu, val > 0);
            const int pos = __ffs(mask) - 1;
#endif
            long long partc = (lane < pos) ? -val : 0;
            partc += __shfl_xor_sync(0xFFFFFFFFu, partc, 1);
            partc += __shfl_xor_sync(0xFFFFFFFFu, partc, 2);
            partc += __shfl_xor_sync(0xFFFFFFFFu, partc, 4);
            partc += __shfl_xor_sync(0xFFFFFFFFu, partc, 8);
            partc += __shfl_xor_sync(0xFFFFFFFFu, partc, 16);
#if defined(WS) && (WS == 64)
            partc += __shfl_xor_sync(0xFFFFFFFFu, partc, 32);
#endif
            if (lane == pos) {
                const long long fullc = partc + val;
                fullcarry[chunkID] = fullc + value;
                *s_fullc = fullc;
            }
        }
    }
}

template <int BitMode> // 0 = RZE only, 2 = BIT_2+RZE, 4 = BIT_4+RZE
static __global__ void d_encode(const byte* const __restrict__ input,
                                const long long insize,
                                byte* const __restrict__ output,
                                long long* const __restrict__ outsize,
                                long long* const __restrict__ fullcarry) {
    // allocate shared memory buffer
    __shared__ long long chunk[3 * (CS / sizeof(long long))];

    // split into 3 shared memory buffers
    byte* in = (byte*)&chunk[0 * (CS / sizeof(long long))];
    byte* out = (byte*)&chunk[1 * (CS / sizeof(long long))];
    byte* const temp = (byte*)&chunk[2 * (CS / sizeof(long long))];

    // initialize
    const int tid = threadIdx.x;
    const long long last = 3 * (CS / sizeof(long long)) - 2 - WS;
    const long long chunks = (insize + CS - 1) / CS; // round up
    long long* const head_out = (long long*)output;
    unsigned short* const size_out = (unsigned short*)&head_out[1];
    byte* const data_out = (byte*)&size_out[chunks];

    // loop over chunks
    do {
        // assign work dynamically
        if (tid == 0)
            chunk[last] = atomicAdd(&g_chunk_counter, 1LL);
        __syncthreads(); // chunk[last] produced, chunk consumed

        // terminate if done
        const long long chunkID = chunk[last];
        const long long base = chunkID * CS;
        if (base >= insize)
            break;

        // load chunk
        const int osize = (int)min((long long)CS, insize - base);
        long long* const input_l = (long long*)&input[base];
        long long* const out_l = (long long*)out;
        for (int i = tid; i < osize / 8; i += TPB) {
            out_l[i] = input_l[i];
        }
        const int extra = osize % 8;
        if (tid < extra)
            out[(long long)osize - (long long)extra + (long long)tid] =
                input[base + (long long)osize - (long long)extra + (long long)tid];

        // encode chunk
        __syncthreads(); // chunk produced, chunk[last] consumed
        int csize = osize;
        bool good = true;
        if (BitMode == 4 && good) {
            byte* tmp = in;
            in = out;
            out = tmp;
            good = d_BIT_4(csize, in, out, temp);
            __syncthreads();
        } else if (BitMode == 2 && good) {
            byte* tmp = in;
            in = out;
            out = tmp;
            good = d_BIT_2(csize, in, out, temp);
            __syncthreads();
        }
        if (good) {
            byte* tmp = in;
            in = out;
            out = tmp;
            good = d_RZE_1(csize, in, out, temp);
            __syncthreads();
        }

        // handle carry
        if (!good || (csize >= osize))
            csize = osize;
        propagate_carry(csize, chunkID, fullcarry, (long long*)temp);

        // reload chunk if incompressible
        if (tid == 0)
            size_out[chunkID] = csize;
        if (csize == osize) {
            // store original data
            long long* const out_l = (long long*)out;
            for (long long i = tid; i < osize / 8; i += TPB) {
                out_l[i] = input_l[i];
            }
            const int extra = osize % 8;
            if (tid < extra)
                out[(long long)osize - (long long)extra + (long long)tid] =
                    input[base + (long long)osize - (long long)extra + (long long)tid];
        }
        __syncthreads(); // "out" done, temp produced

        // store chunk
        const long long offs = (chunkID == 0) ? 0 : *((long long*)temp);
        s2g(&data_out[offs], out, csize);

        // finalize if last chunk
        if ((tid == 0) && (base + CS >= insize)) {
            // output header
            head_out[0] = insize;
            // compute compressed size
            *outsize = &data_out[fullcarry[chunkID]] - output;
        }
    } while (true);
}

// ---------------------------------------------------------------------------
// FUSED magnitude LC + sign compact/pack.  The 3-state sign bytes are index-aligned
// with the uint16 magnitudes (both z-ordered), so each CS-byte mag chunk (CS/2
// elements) owns the sign bytes [chunkID*CS/2, +CS/2).  After the chunk's mag encode
// finishes (shared `temp` is free again), the SAME block ranks its non-2 signs
// (block_prefix_sum) and scatters the negative bits, using a second carry chain
// (signcarry, value+1 sentinel for zero-count chunks).  Kills the separate sign
// kernels: the 8KB sign read rides along with the 16KB mag chunk.
// ---------------------------------------------------------------------------
static __device__ __forceinline__ long long
warp_sign_scan(const unsigned short* const __restrict__ mag,
               const int esize,
               unsigned int& lane_nzmask,
               long long& warp_excl,
               long long* const scratch) {
    static_assert(WS == 32, "warp sign path requires 32-lane CUDA warps");
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int warp_base = warp * 512;
    lane_nzmask = 0;
#pragma unroll
    for (int j = 0; j < 16; ++j) {
        const int idx = warp_base + j * 32 + lane;
        if ((idx < esize) && (mag[idx] != 0))
            lane_nzmask |= 1u << j;
    }

    unsigned int warp_total = __popc(lane_nzmask);
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        warp_total += __shfl_down_sync(0xFFFFFFFFu, warp_total, offset);

    // Only lane 0 contributes each warp's total to the block scan.  Its inclusive
    // result minus warp_total is the number of nonzeros in all preceding warps.
    const long long incl =
        block_prefix_sum<long long>(lane == 0 ? static_cast<long long>(warp_total) : 0LL, scratch);
    const long long lane0_excl = lane == 0 ? incl - static_cast<long long>(warp_total) : 0LL;
    warp_excl = __shfl_sync(0xFFFFFFFFu, lane0_excl, 0);
    return incl;
}

template <int BitMode, bool ZeroFast, bool WarpSign>
static __global__ void d_encode_with_sign(const byte* const __restrict__ input,
                                          const long long insize,
                                          byte* const __restrict__ output,
                                          long long* const __restrict__ outsize,
                                          long long* const __restrict__ fullcarry,
                                          const uint32_t* const __restrict__ sign_bm,
                                          byte* const __restrict__ packed,
                                          long long* const __restrict__ signcarry,
                                          long long* const __restrict__ nnz_out) {
    __shared__ long long chunk[3 * (CS / sizeof(long long))];
    byte* in = (byte*)&chunk[0 * (CS / sizeof(long long))];
    byte* out = (byte*)&chunk[1 * (CS / sizeof(long long))];
    byte* const temp = (byte*)&chunk[2 * (CS / sizeof(long long))];

    const int tid = threadIdx.x;
    const long long last = 3 * (CS / sizeof(long long)) - 2 - WS;
    const long long chunks = (insize + CS - 1) / CS;
    long long* const head_out = (long long*)output;
    unsigned short* const size_out = (unsigned short*)&head_out[1];
    byte* const data_out = (byte*)&size_out[chunks];

    do {
        if (tid == 0)
            chunk[last] = atomicAdd(&g_chunk_counter, 1LL);
        __syncthreads();
        const long long chunkID = chunk[last];
        const long long base = chunkID * CS;
        if (base >= insize)
            break;

        // ---- load the magnitude chunk ----
        const int osize = (int)min((long long)CS, insize - base);
        long long* const input_l = (long long*)&input[base];
        long long* const out_l = (long long*)out;
        for (int i = tid; i < osize / 8; i += TPB)
            out_l[i] = input_l[i];
        const int extra = osize % 8;
        if (tid < extra)
            out[(long long)osize - (long long)extra + (long long)tid] =
                input[base + (long long)osize - (long long)extra + (long long)tid];
        __syncthreads();

        // ---- sign phase 1 (EARLY, raw u16 mags still in `out`): per-thread nonzero MASK +
        //      local prefix, everything kept in REGISTERS so the encode can clobber shared.
        //      The carry wait + scatter run LATE (after the encode) -- doing the lookback
        //      here puts it on the critical path before any encode work (measured +0.6 ms).
        const int esize = osize >> 1;
        const int EPT = (CS / 2) / TPB; // 16 elements/thread in either sign mapping
        unsigned int nzmask = 0;
        long long incl = 0;
        long long excl = 0;
        if constexpr (WarpSign) {
            incl =
                warp_sign_scan((const unsigned short*)out, esize, nzmask, excl, (long long*)temp);
        } else {
            const unsigned short* const s_m = (const unsigned short*)out;
            const int estart = tid * EPT;
#pragma unroll
            for (int j = 0; j < EPT; ++j) {
                const int idx = estart + j;
                if ((idx < esize) && (s_m[idx] != 0))
                    nzmask |= 1u << j;
            }
            incl = block_prefix_sum<long long>((long long)__popc(nzmask), (long long*)temp);
            excl = incl - __popc(nzmask);
        }
        long long chunk_total;
        {
            long long* const s_aux = (long long*)temp + WS;
            if (tid == TPB - 1)
                s_aux[0] = incl;
            __syncthreads();
            chunk_total = s_aux[0];
            __syncthreads(); // temp (pbuf/aux) consumed before the encode stages reuse it
        }

        // ---- magnitude encode: identical to d_encode ----
        // BIT_2 is a permutation, so a full raw-zero chunk remains all-zero.  RZE_1's
        // canonical representation for that 16KB chunk is exactly the two-byte little-
        // endian old-size trailer {0x00, 0x40}.  The sign prefix above already gives us
        // the exact zero test for free.  Skipping both stages also preserves the in/out
        // buffer orientation (the normal BIT_2+RZE_1 path swaps twice).
        bool zero_chunk = false;
        if constexpr (ZeroFast) {
            if constexpr (WarpSign) {
                // Keep the fast-path gate independent of the sparse lane-0 prefix used for
                // ranking.  Requiring both the block-wide OR and its total prevents a prefix
                // bookkeeping error from ever turning a nonzero chunk into canonical zero.
                const int block_has_nz = __syncthreads_or(nzmask != 0);
                zero_chunk = !block_has_nz && (chunk_total == 0) && (osize == CS);
            } else {
                zero_chunk = (chunk_total == 0) && (osize == CS);
            }
        }
        int csize = osize;
        bool good = true;
        if (zero_chunk) {
            csize = 2;
            if (tid == 0) {
                out[0] = static_cast<byte>(osize);
                out[1] = static_cast<byte>(osize >> 8);
            }
        } else {
            if (BitMode == 4 && good) {
                byte* tmp = in;
                in = out;
                out = tmp;
                good = d_BIT_4(csize, in, out, temp);
                __syncthreads();
            } else if (BitMode == 2 && good) {
                byte* tmp = in;
                in = out;
                out = tmp;
                good = d_BIT_2(csize, in, out, temp);
                __syncthreads();
            }
            if (good) {
                byte* tmp = in;
                in = out;
                out = tmp;
                good = d_RZE_1(csize, in, out, temp);
                __syncthreads();
            }
        }
        if (!good || (csize >= osize))
            csize = osize;
        propagate_carry(csize, chunkID, fullcarry, (long long*)temp);
        if (tid == 0)
            size_out[chunkID] = csize;
        if (csize == osize) {
            long long* const out_l2 = (long long*)out;
            for (long long i = tid; i < osize / 8; i += TPB)
                out_l2[i] = input_l[i];
            const int extra2 = osize % 8;
            if (tid < extra2)
                out[(long long)osize - (long long)extra2 + (long long)tid] =
                    input[base + (long long)osize - (long long)extra2 + (long long)tid];
        }
        __syncthreads(); // "out" done, temp produced
        const long long offs = (chunkID == 0) ? 0 : *((long long*)temp);
        s2g(&data_out[offs], out, csize);
        if ((tid == 0) && (base + CS >= insize)) {
            head_out[0] = insize;
            *outsize = &data_out[fullcarry[chunkID]] - output;
        }

        // ---- sign phase 2 (LATE): global rank via the sign carry chain, then scatter the
        //      negative bits (mask/prefix saved in registers; bitmap read per nonzero).
        __syncthreads(); // temp free again (offs consumed)
        {
            long long* const s_aux = (long long*)temp + WS;
            propagate_carry(
                (int)(chunk_total + 1), chunkID, signcarry, &s_aux[1]); // +1: zero-count sentinel
            __syncthreads();
            const long long R = (chunkID == 0) ? 0 : (s_aux[1] - chunkID);
            const long long ebase = base >> 1;
            if constexpr (WarpSign) {
                const int warp = tid >> 5;
                const int lane = tid & 31;
                const int warp_base = warp * 512;
                const unsigned int lower_lanes = lane == 0 ? 0u : ((1u << lane) - 1u);
                int group_excl = 0;
#pragma unroll
                for (int j = 0; j < EPT; ++j) {
                    const bool nz = ((nzmask >> j) & 1u) != 0;
                    const unsigned int group_mask = __ballot_sync(0xFFFFFFFFu, nz);
                    if (nz) {
                        const int lane_excl = __popc(group_mask & lower_lanes);
                        const long long rank = R + excl + group_excl + lane_excl;
                        const long long g = ebase + warp_base + j * 32 + lane;
                        if ((sign_bm[g >> 5] >> (g & 31)) & 1u)
                            atomicOr(((unsigned int*)packed) + (rank >> 5), 1u << (rank & 31));
                    }
                    group_excl += __popc(group_mask);
                }
            } else {
                const int estart = tid * EPT;
                long long rank = R + excl;
#pragma unroll
                for (int j = 0; j < EPT; ++j) {
                    if ((nzmask >> j) & 1u) {
                        const long long g = ebase + estart + j;
                        if ((sign_bm[g >> 5] >> (g & 31)) & 1u)
                            atomicOr(((unsigned int*)packed) + (rank >> 5), 1u << (rank & 31));
                        ++rank;
                    }
                }
            }
            if ((tid == 0) && (base + CS >= insize))
                *nnz_out = signcarry[chunkID] - chunks;
            __syncthreads(); // s_aux + s2g's `out` reads complete before the next chunk
        }
    } while (true);
}

// ---------------------------------------------------------------------------
// Sign compaction + bit-packing (3-state sign: 0=pos, 1=neg, 2=zero), TWO-PASS.
// Three fully parallel launches over the (u16 mags + 1-bit negative bitmap):
//   1. d_signbm_count  : per (CS/2)-element chunk, count the nonzero magnitudes
//   2. d_sign_scan     : single-block exclusive scan over the chunk counts (~9k chunks)
//   3. d_signbm_scatter: re-read each chunk's mags (shared-staged), rank its nonzeros
//                        (block_prefix_sum), add the scanned offset, atomicOr the
//                        negative bits read from the bitmap.
// Bit m (m-th nonzero) -> byte m/8, bit m%8 -- layout unchanged, decoder unchanged.
// ---------------------------------------------------------------------------
static __global__ void d_signbm_count(const unsigned short* const __restrict__ mag,
                                      const long long n,
                                      long long* const __restrict__ cnts) {
    __shared__ int s_cnt;
    const long long chunkID = blockIdx.x;
    const long long ebase = chunkID * (CS / 2);
    const int esize = (int)min((long long)(CS / 2), n - ebase);
    if (threadIdx.x == 0)
        s_cnt = 0;
    __syncthreads();
    int cnt = 0;
    for (int i = threadIdx.x; i < esize; i += TPB)
        if (mag[ebase + i] != 0)
            ++cnt;
    if (cnt != 0)
        atomicAdd(&s_cnt, cnt);
    __syncthreads();
    if (threadIdx.x == 0)
        cnts[chunkID] = s_cnt;
}

// single-block exclusive scan (in place) over nchunks counts; total -> *nnz_out
static __global__ void d_sign_scan(long long* const __restrict__ cnts,
                                   const long long nchunks,
                                   long long* const __restrict__ nnz_out) {
    __shared__ long long s_buf[1024];
    __shared__ long long s_carry;
    if (threadIdx.x == 0)
        s_carry = 0;
    __syncthreads();
    for (long long t0 = 0; t0 < nchunks; t0 += blockDim.x) {
        const long long i = t0 + threadIdx.x;
        const long long v = (i < nchunks) ? cnts[i] : 0;
        s_buf[threadIdx.x] = v;
        __syncthreads();
        for (int off = 1; off < (int)blockDim.x; off <<= 1) { // Hillis-Steele inclusive scan
            const long long add = (threadIdx.x >= off) ? s_buf[threadIdx.x - off] : 0;
            __syncthreads();
            s_buf[threadIdx.x] += add;
            __syncthreads();
        }
        if (i < nchunks)
            cnts[i] = s_carry + s_buf[threadIdx.x] - v; // exclusive prefix
        __syncthreads();
        if (threadIdx.x == 0)
            s_carry += s_buf[blockDim.x - 1];
        __syncthreads();
    }
    if (threadIdx.x == 0)
        *nnz_out = s_carry;
}

static __global__ void d_signbm_scatter(const unsigned short* const __restrict__ mag,
                                        const long long n,
                                        const uint32_t* const __restrict__ sign_bm,
                                        byte* const __restrict__ out,
                                        const long long* const __restrict__ offs) {
    __shared__ unsigned short s_mag[CS / 2]; // coalesced stage (per-thread EPT ranges would stride)
    __shared__ long long s_pbuf[WS];         // block_prefix_sum scratch
    const long long chunkID = blockIdx.x;
    const long long ebase = chunkID * (CS / 2);
    const int esize = (int)min((long long)(CS / 2), n - ebase);
    const int tid = threadIdx.x;
    const int EPT = (CS / 2) / TPB; // contiguous elements per thread

    for (int i = tid; i < esize; i += TPB)
        s_mag[i] = mag[ebase + i];
    __syncthreads();

    const int start = tid * EPT;
    int cnt = 0;
#pragma unroll
    for (int j = 0; j < EPT; ++j) {
        const int idx = start + j;
        if ((idx < esize) && (s_mag[idx] != 0))
            ++cnt;
    }
    const long long incl = block_prefix_sum<long long>((long long)cnt, s_pbuf);
    const long long excl = incl - cnt; // this thread's first rank within the chunk

    long long rank = offs[chunkID] + excl; // global start rank from the scanned counts
#pragma unroll
    for (int j = 0; j < EPT; ++j) {
        const int idx = start + j;
        if (idx < esize) {
            if (s_mag[idx] != 0) {
                const long long g = ebase + idx;
                if ((sign_bm[g >> 5] >> (g & 31)) & 1u)
                    atomicOr(((unsigned int*)out) + (rank >> 5), 1u << (rank & 31));
                ++rank;
            }
        }
    }
}

template <bool PackedAtBlobTail>
static __global__ void d_signbm_unpack(const unsigned short* const __restrict__ mag,
                                       const long long n,
                                       const byte* const __restrict__ packed_or_blob,
                                       const unsigned long long* const __restrict__ total_bytes,
                                       const unsigned long long* const __restrict__ sign_bytes,
                                       size_t tail_guard,
                                       uint32_t* const __restrict__ sign_bm,
                                       const long long* const __restrict__ offs) {
    const byte* packed = packed_or_blob;
    if constexpr (PackedAtBlobTail)
        packed += *total_bytes - tail_guard - *sign_bytes;
    const long long chunkID = blockIdx.x;
    const long long ebase = chunkID * (CS / 2);
    const int esize = (int)min((long long)(CS / 2), n - ebase);
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    __shared__ uint32_t warp_prefix[TPB / 32];
    __shared__ uint32_t tile_rank, running_rank;
    if (tid == 0)
        running_rank = (uint32_t)offs[chunkID];
    __syncthreads();
    for (int tile = 0; tile < esize; tile += TPB) {
        const int i = tile + tid;
        const bool nz = i < esize && mag[ebase + i] != 0;
        const uint32_t active = __ballot_sync(0xffffffffu, nz);
        if (lane == 0)
            warp_prefix[warp] = __popc(active);
        __syncthreads();
        if (tid == 0) {
            const uint32_t start = running_rank;
            uint32_t sum = 0;
#pragma unroll
            for (int w = 0; w < TPB / 32; ++w) {
                const uint32_t wc = warp_prefix[w];
                warp_prefix[w] = sum;
                sum += wc;
            }
            tile_rank = start;
            running_rank = start + sum;
        }
        __syncthreads();
        bool neg = false;
        if (nz) {
            const uint32_t lower = active & (lane == 0 ? 0u : ((1u << lane) - 1u));
            const uint32_t rank = tile_rank + warp_prefix[warp] + __popc(lower);
            neg = ((packed[rank >> 3] >> (rank & 7)) & 1u) != 0;
        }
        const uint32_t negmask = __ballot_sync(0xffffffffu, neg);
        if (lane == 0 && tile + warp * 32 < esize)
            sign_bm[(ebase + tile + warp * 32) >> 5] = negmask;
        __syncthreads();
    }
}

// ---------------------------------------------------------------------------
// Waltz device-resident entry point (declared in include/lossless/lc.hpp).
// ---------------------------------------------------------------------------
#include "lossless/lc.hpp"

// Persistent scratch shared by both entry points (grown on demand, never freed --
// cudaMalloc/cudaFree per call forces a device sync and dominates the per-call cost
// at constant insize).  Single-stream use only.
static int g_lc_blocks = 0;
static byte* g_lc_encoded = nullptr;
static long long g_lc_enc_cap = 0;
static long long* g_lc_encsize = nullptr;
static long long* g_lc_host_results = nullptr;
static long long* g_lc_fullcarry = nullptr;
static long long g_lc_carry_cap = 0;
// separate scratch for the sign path (so it can run on its own stream concurrently)
static long long* g_sign_fullcarry = nullptr;
static long long g_sign_carry_cap = 0;
static long long* g_sign_nnz = nullptr;

static bool lc_zero_fast_enabled() {
    const char* const env = std::getenv("WALTZ_LC_ZERO_FAST");
    return (env == nullptr) || (std::atoi(env) != 0);
}

// Keep the decoder's zero-chunk short cut independently selectable while it is
// being validated.  The encoder and decoder do not have to use the same device
// code path: the canonical two-byte representation emitted by the encoder is
// still accepted by the regular iRZE+iBIT decoder.  In particular, making this
// opt-in (rather than inheriting WALTZ_LC_ZERO_FAST) gives us a safe way to
// distinguish an encoder-side stream/carry issue from a decoder-side shortcut
// issue under compute-sanitizer.
static bool lc_zero_fast_decode_enabled() {
    const char* const env = std::getenv("WALTZ_LC_ZERO_FAST_DECODE");
    return env != nullptr && (std::atoi(env) != 0);
}

static bool lc_warp_sign_enabled() {
    if (std::getenv("WALTZ_LC_WARP_SIGN_OFF") != nullptr)
        return false;
    const char* const env = std::getenv("WALTZ_LC_WARP_SIGN");
    return env == nullptr || std::atoi(env) != 0;
}

static bool lc_warp_sign_decode_enabled() {
    const char* const env = std::getenv("WALTZ_LC_WARP_SIGN_DECODE");
    return env == nullptr ? lc_warp_sign_enabled() : (std::atoi(env) != 0);
}

static void ensure_sign_scratch(long long chunks) {
    if (chunks > g_sign_carry_cap) {
        if (g_sign_fullcarry)
            cudaFree(g_sign_fullcarry);
        cudaMalloc((void**)&g_sign_fullcarry, chunks * sizeof(long long));
        g_sign_carry_cap = chunks;
    }
    if (g_sign_nnz == nullptr)
        cudaMalloc((void**)&g_sign_nnz, sizeof(long long));
}

// Enqueue the sign compact+pack kernels on `stream` (no sync; caller reads g_sign_nnz).
static void launch_sign_pack(const unsigned short* d_mag,
                             const uint32_t* d_sign_bm,
                             unsigned char* d_packed,
                             long long n_elems,
                             cudaStream_t stream) {
    const long long chunks = (n_elems + (CS / 2) - 1) / (CS / 2);
    ensure_sign_scratch(chunks);
    const long long packbytes = ((n_elems + 63) / 64) * 8;
    cudaMemsetAsync(d_packed, 0, packbytes, stream);
    d_signbm_count<<<static_cast<int>(chunks), TPB, 0, stream>>>(d_mag, n_elems, g_sign_fullcarry);
    d_sign_scan<<<1, 1024, 0, stream>>>(g_sign_fullcarry, chunks, g_sign_nnz);
    d_signbm_scatter<<<static_cast<int>(chunks), TPB, 0, stream>>>(
        d_mag, n_elems, d_sign_bm, d_packed, g_sign_fullcarry);
}

// Grow/setup all LC scratch for a given input size.
static void ensure_lc_scratch(long long insize) {
    if (g_lc_blocks == 0) { // device-derived block count: query once, cache
        const cudaDeviceProp& prop = WALTZ::gpu_config().properties;
        const int blocks_per_sm = prop.maxThreadsPerMultiProcessor / TPB;
        g_lc_blocks = prop.multiProcessorCount * blocks_per_sm;
    }
    const long long chunks = (insize + CS - 1) / CS;
    const long long maxsize = 2 * (long long)sizeof(long long) + chunks * (long long)sizeof(short) +
                              chunks * (long long)CS;

    if (maxsize > g_lc_enc_cap) { // grow output scratch only when needed
        if (g_lc_encoded)
            cudaFree(g_lc_encoded);
        cudaMalloc((void**)&g_lc_encoded, maxsize);
        g_lc_enc_cap = maxsize;
    }
    if (g_lc_encsize == nullptr)
        cudaMalloc((void**)&g_lc_encsize, sizeof(long long));
    if (g_lc_host_results == nullptr)
        cudaMallocHost((void**)&g_lc_host_results, 2 * sizeof(long long));
    if (chunks > g_lc_carry_cap) {
        if (g_lc_fullcarry)
            cudaFree(g_lc_fullcarry);
        cudaMalloc((void**)&g_lc_fullcarry, chunks * sizeof(long long));
        g_lc_carry_cap = chunks;
    }
}

template <int BitMode>
static long long lc_compress_impl(const void* d_in, long long insize, cudaStream_t stream) {
    if (insize <= 0)
        return 0;
    ensure_lc_scratch(insize);
    const long long chunks = (insize + CS - 1) / CS;

    d_reset<<<1, 1, 0, stream>>>();
    cudaMemsetAsync(g_lc_fullcarry, 0, chunks * sizeof(long long), stream);
    d_encode<BitMode><<<g_lc_blocks, TPB, 0, stream>>>(
        static_cast<const byte*>(d_in), insize, g_lc_encoded, g_lc_encsize, g_lc_fullcarry);

    long long encsize = 0;
    cudaMemcpyAsync(&encsize, g_lc_encsize, sizeof(long long), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    return encsize;
}

long long
WALTZ::lossless::lc_compress_bit4_rze1(const void* d_in, long long insize, cudaStream_t stream) {
    return lc_compress_impl<4>(d_in, insize, stream);
}

long long
WALTZ::lossless::lc_compress_bit2_rze1(const void* d_in, long long insize, cudaStream_t stream) {
    return lc_compress_impl<2>(d_in, insize, stream);
}

long long
WALTZ::lossless::lc_compress_rze1(const void* d_in, long long insize, cudaStream_t stream) {
    return lc_compress_impl<0>(d_in, insize, stream);
}

const void* WALTZ::lossless::lc_encoded_buf() {
    return g_lc_encoded;
}

// ---------------------------------------------------------------------------
// LC DECODER (BIT_2+RZE_1 inverse).  g2s + d_decode are taken from the LC framework's
// decompressor-framework.cu with the generated decoder body fixed to our pipeline:
// inverse stages run in REVERSE encode order (iRZE_1 first, then iBIT_2).
// ---------------------------------------------------------------------------

// copy (len) bytes from global memory (source) to shared memory (destination) using
// separate shared memory buffer (temp); destination/temp word aligned
static inline __device__ void g2s(void* const __restrict__ destination,
                                  const void* const __restrict__ source,
                                  const int len,
                                  void* const __restrict__ temp) {
    const int tid = threadIdx.x;
    const byte* const __restrict__ input = (byte*)source;
    if (len < 128) {
        byte* const __restrict__ output = (byte*)destination;
        if (tid < len)
            output[tid] = input[tid];
    } else {
        const int nonaligned = (int)(size_t)input;
        const int wordaligned = (nonaligned + 3) & ~3;
        const int linealigned = (nonaligned + 127) & ~127;
        const int bcnt = wordaligned - nonaligned;
        const int wcnt = (linealigned - wordaligned) / 4;
        int* const __restrict__ out_w = (int*)destination;
        if (bcnt == 0) {
            const int* const __restrict__ in_w = (int*)input;
            byte* const __restrict__ out = (byte*)destination;
            if (tid < wcnt)
                out_w[tid] = in_w[tid];
            for (int i = tid + wcnt; i < len / 4; i += TPB) {
                out_w[i] = in_w[i];
            }
            if (tid < (len & 3)) {
                const int i = len - 1 - tid;
                out[i] = input[i];
            }
        } else {
            const int offs = 4 - bcnt;
            const int shift = offs * 8;
            const int rlen = len - bcnt;
            const int* const __restrict__ in_w = (int*)&input[bcnt];
            byte* const __restrict__ buffer = (byte*)temp;
            byte* const __restrict__ buf = (byte*)&buffer[offs];
            int* __restrict__ buf_w = (int*)&buffer[4];
            if (tid < bcnt)
                buf[tid] = input[tid];
            if (tid < wcnt)
                buf_w[tid] = in_w[tid];
            for (int i = tid + wcnt; i < rlen / 4; i += TPB) {
                buf_w[i] = in_w[i];
            }
            if (tid < (rlen & 3)) {
                const int i = len - 1 - tid;
                buf[i] = input[i];
            }
            __syncthreads();
            buf_w = (int*)buffer;
            for (int i = tid; i < (len + 3) / 4; i += TPB) {
                out_w[i] = __funnelshift_r(buf_w[i], buf_w[i + 1], shift);
            }
        }
    }
}

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 800)
static __global__ __launch_bounds__(TPB, 3)
#else
static __global__ __launch_bounds__(TPB, 2)
#endif
    void d_decode(const byte* const __restrict__ input,
                  byte* const __restrict__ output,
                  long long* const __restrict__ g_outsize) {
    __shared__ long long chunk[3 * (CS / sizeof(long long))];
    const int last = 3 * (CS / sizeof(long long)) - 2 - WS;

    long long* const head_in = (long long*)input;
    const long long outsize = head_in[0];
    const long long chunks = (outsize + CS - 1) / CS;
    unsigned short* const size_in = (unsigned short*)&head_in[1];
    byte* const data_in = (byte*)&size_in[chunks];

    const int tid = threadIdx.x;
    long long prevChunkID = 0;
    long long prevOffset = 0;
    do {
        if (tid == 0)
            chunk[last] = atomicAdd(&g_chunk_counter, 1LL);
        __syncthreads();
        const long long chunkID = chunk[last];
        const long long base = chunkID * CS;
        if (base >= outsize)
            break;

        // sum of all prior csizes (resume from previous iteration)
        long long sum = 0;
        for (long long i = prevChunkID + tid; i < chunkID; i += TPB) {
            sum += (long long)size_in[i];
        }
        int csize = (int)size_in[chunkID];
        const long long offs = prevOffset + block_sum_reduction(sum, (long long*)&chunk[last + 1]);
        prevChunkID = chunkID;
        prevOffset = offs;

        byte* in = (byte*)&chunk[0 * (CS / sizeof(long long))];
        byte* out = (byte*)&chunk[1 * (CS / sizeof(long long))];
        byte* temp = (byte*)&chunk[2 * (CS / sizeof(long long))];

        g2s(in, &data_in[offs], csize, out);
        byte* tmp = in;
        in = out;
        out = tmp;
        __syncthreads();

        const int osize = (int)min((long long)CS, outsize - base);
        if (csize < osize) {
            byte* tmp2;
            tmp2 = in;
            in = out;
            out = tmp2;
            d_iRZE_1(csize, in, out, temp);
            __syncthreads();
            tmp2 = in;
            in = out;
            out = tmp2;
            d_iBIT_2(csize, in, out, temp);
            __syncthreads();
        }

        if (csize != osize) {
            printf(
                "ERROR: csize %d doesn't match osize %d in chunk %lld\n\n", csize, osize, chunkID);
            __trap();
        }
        long long* const output_l = (long long*)&output[base];
        long long* const out_l = (long long*)out;
        for (int i = tid; i < osize / 8; i += TPB) {
            output_l[i] = out_l[i];
        }
        const int extra = osize % 8;
        if (tid < extra)
            output[base + osize - extra + tid] = out[osize - extra + tid];
    } while (true);

    if ((blockIdx.x == 0) && (tid == 0)) {
        *g_outsize = outsize;
    }
}

// ---------------------------------------------------------------------------
// FUSED decoder: LC-decode each magnitude chunk AND unpack its sign bytes while the
// decoded u16 mags are still in shared (`out`).  Mirrors d_encode_with_sign: rank the
// nonzeros with block_prefix_sum (scratch carved from `temp`, free after the inverse
// stages), get the global rank via a second carry chain, read the packed negative bits,
// stage the 3-state sign bytes in `temp` and flush them coalesced.  Replaces the
// standalone sign-unpack passes (two extra full reads of the magnitudes).
// ---------------------------------------------------------------------------
template <bool ZeroFast, bool WarpSign>
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 800)
static __global__ __launch_bounds__(TPB, 3)
#else
static __global__ __launch_bounds__(TPB, 2)
#endif
    void d_decode_with_sign(const byte* const __restrict__ input,
                            byte* const __restrict__ output,
                            long long* const __restrict__ g_outsize,
                            const byte* const __restrict__ packed,
                            uint32_t* const __restrict__ sign_bm,
                            long long* const __restrict__ signcarry) {
    __shared__ long long chunk[3 * (CS / sizeof(long long))];
    const int last = 3 * (CS / sizeof(long long)) - 2 - WS;

    long long* const head_in = (long long*)input;
    const long long outsize = head_in[0];
    const long long chunks = (outsize + CS - 1) / CS;
    unsigned short* const size_in = (unsigned short*)&head_in[1];
    byte* const data_in = (byte*)&size_in[chunks];

    const int tid = threadIdx.x;
    long long prevChunkID = 0;
    long long prevOffset = 0;
    do {
        if (tid == 0)
            chunk[last] = atomicAdd(&g_chunk_counter, 1LL);
        __syncthreads();
        const long long chunkID = chunk[last];
        const long long base = chunkID * CS;
        if (base >= outsize)
            break;

        long long sum = 0;
        for (long long i = prevChunkID + tid; i < chunkID; i += TPB) {
            sum += (long long)size_in[i];
        }
        int csize = (int)size_in[chunkID];
        const long long offs = prevOffset + block_sum_reduction(sum, (long long*)&chunk[last + 1]);
        prevChunkID = chunkID;
        prevOffset = offs;

        byte* in = (byte*)&chunk[0 * (CS / sizeof(long long))];
        byte* out = (byte*)&chunk[1 * (CS / sizeof(long long))];
        byte* temp = (byte*)&chunk[2 * (CS / sizeof(long long))];

        const int osize = (int)min((long long)CS, outsize - base);
        bool zero_chunk = false;
        if constexpr (ZeroFast) {
            // size==2 uniquely denotes RZE's all-zero representation for a full chunk.
            // Check the canonical {0x00, 0x40} payload as well, so malformed/noncanonical
            // streams retain the original decode-and-validate behaviour.
            if ((csize == 2) && (osize == CS)) {
                if (tid == 0) {
                    chunk[last + 1] = (data_in[offs] == static_cast<byte>(CS)) &&
                                      (data_in[offs + 1] == static_cast<byte>(CS >> 8));
                }
                __syncthreads();
                zero_chunk = chunk[last + 1] != 0;
            }
        }

        if (zero_chunk) {
            // Produce the decoded magnitudes in the same shared buffer consumed by the
            // common global-store/sign path, without running iRZE_1 then iBIT_2.
            long long* const out_l = (long long*)out;
            for (int i = tid; i < CS / (int)sizeof(long long); i += TPB)
                out_l[i] = 0;
            csize = osize;
            __syncthreads();
        } else {
            g2s(in, &data_in[offs], csize, out);
            byte* tmp = in;
            in = out;
            out = tmp;
            __syncthreads();

            if (csize < osize) {
                byte* tmp2;
                tmp2 = in;
                in = out;
                out = tmp2;
                d_iRZE_1(csize, in, out, temp);
                __syncthreads();
                tmp2 = in;
                in = out;
                out = tmp2;
                d_iBIT_2(csize, in, out, temp);
                __syncthreads();
            }
        }

        if (csize != osize) {
            printf(
                "ERROR: csize %d doesn't match osize %d in chunk %lld\n\n", csize, osize, chunkID);
            __trap();
        }
        long long* const output_l = (long long*)&output[base];
        long long* const out_l = (long long*)out;
        for (int i = tid; i < osize / 8; i += TPB) {
            output_l[i] = out_l[i];
        }
        const int extra = osize % 8;
        if (tid < extra)
            output[base + osize - extra + tid] = out[osize - extra + tid];

        // ---- fused sign unpack for this chunk (decoded u16 mags still in `out`) ----
        // Output is the 1-bit NEGATIVE bitmap.  The legacy mapping writes one uint16 per
        // thread; WarpSign rebuilds each consecutive 32-element word with a ballot.
        __syncthreads(); // inverse stages + output copy done; temp free for reuse
        long long* const s_pbuf = (long long*)temp; // [0..WS) prefix scratch
        long long* const s_aux = s_pbuf + WS;       // [WS]=total, [WS+1]=carry out
        const unsigned short* const s_m = (const unsigned short*)out;
        const int esize = osize >> 1;
        const int EPT = (CS / 2) / TPB; // == 16
        unsigned int nzmask = 0;
        long long excl = 0;
        if (!zero_chunk) {
            long long incl = 0;
            if constexpr (WarpSign) {
                incl = warp_sign_scan(s_m, esize, nzmask, excl, s_pbuf);
            } else {
                const int start = tid * EPT;
                int cnt = 0;
#pragma unroll
                for (int j = 0; j < EPT; ++j) {
                    const int idx = start + j;
                    if ((idx < esize) && (s_m[idx] != 0))
                        ++cnt;
                }
                incl = block_prefix_sum<long long>((long long)cnt, s_pbuf);
                excl = incl - cnt;
            }
            if (tid == TPB - 1)
                s_aux[0] = incl;
        } else {
            if (tid == TPB - 1)
                s_aux[0] = 0;
        }
        __syncthreads();
        propagate_carry(
            (int)(s_aux[0] + 1), chunkID, signcarry, &s_aux[1]); // +1: zero-count sentinel
        __syncthreads();
        if constexpr (WarpSign) {
            const int warp = tid >> 5;
            const int lane = tid & 31;
            const unsigned int lower_lanes = lane == 0 ? 0u : ((1u << lane) - 1u);
            const long long R = zero_chunk ? 0 : ((chunkID == 0) ? 0 : (s_aux[1] - chunkID));
            int group_excl = 0;
#pragma unroll
            for (int j = 0; j < EPT; ++j) {
                const bool nz = ((nzmask >> j) & 1u) != 0;
                const unsigned int group_mask = __ballot_sync(0xFFFFFFFFu, nz);
                bool negative = false;
                if (nz) {
                    const int lane_excl = __popc(group_mask & lower_lanes);
                    const long long rank = R + excl + group_excl + lane_excl;
                    negative = ((packed[rank >> 3] >> (rank & 7)) & 1u) != 0;
                }
                const unsigned int negative_word = __ballot_sync(0xFFFFFFFFu, negative);
                if (lane == 0)
                    sign_bm[chunkID * (CS / 2 / 32) + warp * EPT + j] = negative_word;
                group_excl += __popc(group_mask);
            }
        } else {
            unsigned int bits = 0; // this thread's 16 bitmap bits
            if (!zero_chunk) {
                const int start = tid * EPT;
                const long long R = (chunkID == 0) ? 0 : (s_aux[1] - chunkID);
                long long rank = R + excl;
#pragma unroll
                for (int j = 0; j < EPT; ++j) {
                    const int idx = start + j;
                    if ((idx < esize) && (s_m[idx] != 0)) {
                        if ((packed[rank >> 3] >> (rank & 7)) & 1u)
                            bits |= 1u << j; // negative
                        ++rank;
                    }
                }
            }
            ((unsigned short*)sign_bm)[chunkID * (CS / 2 / 16) + tid] =
                static_cast<unsigned short>(bits);
        }
    } while (true);

    if ((blockIdx.x == 0) && (tid == 0)) {
        *g_outsize = outsize;
    }
}

static void ensure_lc_blocks() {
    if (g_lc_blocks == 0) {
        const cudaDeviceProp& prop = WALTZ::gpu_config().properties;
        const int blocks_per_sm = prop.maxThreadsPerMultiProcessor / TPB;
        g_lc_blocks = prop.multiProcessorCount * blocks_per_sm;
    }
}

uint32_t WALTZ::lossless::sign_compact_pack(const void* d_mag_u16,
                                            const uint32_t* d_sign_bm,
                                            unsigned char* d_packed,
                                            uint32_t n,
                                            cudaStream_t stream) {
    if (n == 0)
        return 0;
    sign_compact_pack_launch(d_mag_u16, d_sign_bm, d_packed, n, stream);
    return sign_compact_pack_finish(stream);
}

void WALTZ::lossless::sign_compact_pack_launch(const void* d_mag_u16,
                                               const uint32_t* d_sign_bm,
                                               unsigned char* d_packed,
                                               uint32_t n,
                                               cudaStream_t stream) {
    if (n == 0)
        return;
    ensure_lc_blocks();
    launch_sign_pack(static_cast<const unsigned short*>(d_mag_u16),
                     d_sign_bm,
                     d_packed,
                     static_cast<long long>(n),
                     stream);
}

uint32_t WALTZ::lossless::sign_compact_pack_finish(cudaStream_t stream) {
    long long nnz = 0;
    cudaMemcpyAsync(&nnz, g_sign_nnz, sizeof(long long), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    return static_cast<uint32_t>(nnz);
}

void WALTZ::lossless::sign_compact_unpack(const void* d_mag_u16,
                                          const unsigned char* d_packed,
                                          uint32_t* d_sign_bm,
                                          uint32_t n,
                                          cudaStream_t stream) {
    if (n == 0)
        return;
    ensure_lc_blocks();
    const long long chunks = ((long long)n + (CS / 2) - 1) / (CS / 2);
    ensure_sign_scratch(chunks);
    cudaMemsetAsync(d_sign_bm, 0, ((size_t)n + 31) / 32 * sizeof(uint32_t), stream);
    d_signbm_count<<<static_cast<int>(chunks), TPB, 0, stream>>>(
        static_cast<const unsigned short*>(d_mag_u16), n, g_sign_fullcarry);
    d_sign_scan<<<1, 1024, 0, stream>>>(g_sign_fullcarry, chunks, g_sign_nnz);
    d_signbm_unpack<false><<<static_cast<int>(chunks), TPB, 0, stream>>>(
        static_cast<const unsigned short*>(d_mag_u16),
        n,
        d_packed,
        nullptr,
        nullptr,
        0,
        d_sign_bm,
        g_sign_fullcarry);
}

void WALTZ::lossless::sign_compact_unpack_blob_tail(const void* d_mag_u16,
                                                    const unsigned char* d_blob,
                                                    const unsigned long long* d_total_bytes,
                                                    const unsigned long long* d_sign_bytes,
                                                    size_t tail_guard,
                                                    uint32_t* d_sign_bm,
                                                    uint32_t n,
                                                    cudaStream_t stream) {
    if (n == 0)
        return;
    ensure_lc_blocks();
    const long long chunks = ((long long)n + (CS / 2) - 1) / (CS / 2);
    ensure_sign_scratch(chunks);
    cudaMemsetAsync(d_sign_bm, 0, ((size_t)n + 31) / 32 * sizeof(uint32_t), stream);
    d_signbm_count<<<static_cast<int>(chunks), TPB, 0, stream>>>(
        static_cast<const unsigned short*>(d_mag_u16), n, g_sign_fullcarry);
    d_sign_scan<<<1, 1024, 0, stream>>>(g_sign_fullcarry, chunks, g_sign_nnz);
    d_signbm_unpack<true><<<static_cast<int>(chunks), TPB, 0, stream>>>(
        static_cast<const unsigned short*>(d_mag_u16),
        n,
        d_blob,
        d_total_bytes,
        d_sign_bytes,
        tail_guard,
        d_sign_bm,
        g_sign_fullcarry);
}

// FUSED magnitude-LC + sign-pack: one kernel, one pass.  (The two-stream concurrent
// version was tried first and lost to contention; true fusion rides the 8KB sign read
// along with each 16KB mag chunk.)  Requires sign_n*2 == mag_insize (u16 mags).
void WALTZ::lossless::lc_compress_bit2_rze1_with_sign_launch(const void* d_mag,
                                                             long long mag_insize,
                                                             const uint32_t* d_sign_bm,
                                                             unsigned char* d_packed,
                                                             uint32_t sign_n,
                                                             cudaStream_t stream) {
    if (mag_insize <= 0)
        return;
    ensure_lc_scratch(mag_insize);
    const long long mchunks = (mag_insize + CS - 1) / CS;
    ensure_sign_scratch(mchunks); // sign carry chain is per MAG chunk (CS/2 elements each)
    const long long packbytes = ((static_cast<long long>(sign_n) + 63) / 64) * 8;
    cudaMemsetAsync(d_packed, 0, packbytes, stream);
    cudaMemsetAsync(g_lc_fullcarry, 0, mchunks * sizeof(long long), stream);
    cudaMemsetAsync(g_sign_fullcarry, 0, mchunks * sizeof(long long), stream);
    d_reset<<<1, 1, 0, stream>>>();
    // Enabled by default; set WALTZ_LC_ZERO_FAST=0 for the byte-identical baseline.
    // The conflict-free warp sign mapping is the default after cross-decoding
    // validation against the legacy mapping.  WALTZ_LC_WARP_SIGN_OFF=1 retains
    // the old path for A/B diagnosis.  Template combinations keep both switches
    // out of the device hot path.
    const bool zero_fast = lc_zero_fast_enabled();
    const bool warp_sign = lc_warp_sign_enabled();
    if (zero_fast) {
        if (warp_sign) {
            d_encode_with_sign<2, true, true>
                <<<g_lc_blocks, TPB, 0, stream>>>(static_cast<const byte*>(d_mag),
                                                  mag_insize,
                                                  g_lc_encoded,
                                                  g_lc_encsize,
                                                  g_lc_fullcarry,
                                                  d_sign_bm,
                                                  d_packed,
                                                  g_sign_fullcarry,
                                                  g_sign_nnz);
        } else {
            d_encode_with_sign<2, true, false>
                <<<g_lc_blocks, TPB, 0, stream>>>(static_cast<const byte*>(d_mag),
                                                  mag_insize,
                                                  g_lc_encoded,
                                                  g_lc_encsize,
                                                  g_lc_fullcarry,
                                                  d_sign_bm,
                                                  d_packed,
                                                  g_sign_fullcarry,
                                                  g_sign_nnz);
        }
    } else {
        if (warp_sign) {
            d_encode_with_sign<2, false, true>
                <<<g_lc_blocks, TPB, 0, stream>>>(static_cast<const byte*>(d_mag),
                                                  mag_insize,
                                                  g_lc_encoded,
                                                  g_lc_encsize,
                                                  g_lc_fullcarry,
                                                  d_sign_bm,
                                                  d_packed,
                                                  g_sign_fullcarry,
                                                  g_sign_nnz);
        } else {
            d_encode_with_sign<2, false, false>
                <<<g_lc_blocks, TPB, 0, stream>>>(static_cast<const byte*>(d_mag),
                                                  mag_insize,
                                                  g_lc_encoded,
                                                  g_lc_encsize,
                                                  g_lc_fullcarry,
                                                  d_sign_bm,
                                                  d_packed,
                                                  g_sign_fullcarry,
                                                  g_sign_nnz);
        }
    }
}

uint32_t WALTZ::lossless::lc_compress_bit2_rze1_with_sign_finish(long long* mag_out,
                                                                 cudaStream_t stream) {
    cudaMemcpyAsync(
        g_lc_host_results, g_lc_encsize, sizeof(long long), cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(
        g_lc_host_results + 1, g_sign_nnz, sizeof(long long), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    *mag_out = g_lc_host_results[0];
    return static_cast<uint32_t>(g_lc_host_results[1]);
}

uint32_t WALTZ::lossless::lc_compress_bit2_rze1_with_sign(const void* d_mag,
                                                          long long mag_insize,
                                                          long long* mag_out,
                                                          const uint32_t* d_sign_bm,
                                                          unsigned char* d_packed,
                                                          uint32_t sign_n,
                                                          cudaStream_t stream) {
    if (mag_insize <= 0) {
        *mag_out = 0;
        return 0;
    }
    lc_compress_bit2_rze1_with_sign_launch(d_mag, mag_insize, d_sign_bm, d_packed, sign_n, stream);
    return lc_compress_bit2_rze1_with_sign_finish(mag_out, stream);
}

// ---------------------------------------------------------------------------
// Decompression entry points.
// ---------------------------------------------------------------------------

static long long* g_dec_outsize = nullptr;
static long long* g_dec_host_outsize = nullptr;

// LC-decode a BIT_2+RZE_1 stream (device->device).  Returns the decoded byte count
// (the original insize, read from the stream header).
long long WALTZ::lossless::lc_decompress(const void* d_in, void* d_out, cudaStream_t stream) {
    ensure_lc_blocks();
    if (g_dec_outsize == nullptr)
        cudaMalloc((void**)&g_dec_outsize, sizeof(long long));
    d_reset<<<1, 1, 0, stream>>>();
    d_decode<<<g_lc_blocks, TPB, 0, stream>>>(
        static_cast<const byte*>(d_in), static_cast<byte*>(d_out), g_dec_outsize);
    long long outsize = 0;
    cudaMemcpyAsync(&outsize, g_dec_outsize, sizeof(long long), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    return outsize;
}

// FUSED decode: LC-decode the u16 magnitudes AND rebuild their 1-bit negative bitmap in
// one kernel (sign unpacked per chunk while the decoded mags are still in shared).
// mag_bytes_expected sizes the sign carry chain (= decoded size, known from the blob header).
// d_sign_bm must hold ceil(n_elems/8192)*1024 bytes (one 1KB region per chunk).
void WALTZ::lossless::lc_decompress_with_sign_launch(const void* d_in,
                                                     void* d_out_mags,
                                                     long long mag_bytes_expected,
                                                     const unsigned char* d_packed,
                                                     uint32_t* d_sign_bm,
                                                     cudaStream_t stream) {
    ensure_lc_blocks();
    if (g_dec_outsize == nullptr)
        cudaMalloc((void**)&g_dec_outsize, sizeof(long long));
    if (g_dec_host_outsize == nullptr)
        cudaMallocHost((void**)&g_dec_host_outsize, sizeof(long long));
    const long long mchunks = (mag_bytes_expected + CS - 1) / CS;
    ensure_sign_scratch(mchunks);
    cudaMemsetAsync(g_sign_fullcarry, 0, mchunks * sizeof(long long), stream);
    d_reset<<<1, 1, 0, stream>>>();
    // Decoder zero shortcut is deliberately an independent opt-in.  The
    // encoder gate above remains enabled by default; unset
    // WALTZ_LC_ZERO_FAST_DECODE keeps the legacy inverse path for diagnosis.
    const bool zero_fast = lc_zero_fast_decode_enabled();
    const bool warp_sign = lc_warp_sign_decode_enabled();
    if (zero_fast) {
        if (warp_sign) {
            d_decode_with_sign<true, true>
                <<<g_lc_blocks, TPB, 0, stream>>>(static_cast<const byte*>(d_in),
                                                  static_cast<byte*>(d_out_mags),
                                                  g_dec_outsize,
                                                  d_packed,
                                                  d_sign_bm,
                                                  g_sign_fullcarry);
        } else {
            d_decode_with_sign<true, false>
                <<<g_lc_blocks, TPB, 0, stream>>>(static_cast<const byte*>(d_in),
                                                  static_cast<byte*>(d_out_mags),
                                                  g_dec_outsize,
                                                  d_packed,
                                                  d_sign_bm,
                                                  g_sign_fullcarry);
        }
    } else {
        if (warp_sign) {
            d_decode_with_sign<false, true>
                <<<g_lc_blocks, TPB, 0, stream>>>(static_cast<const byte*>(d_in),
                                                  static_cast<byte*>(d_out_mags),
                                                  g_dec_outsize,
                                                  d_packed,
                                                  d_sign_bm,
                                                  g_sign_fullcarry);
        } else {
            d_decode_with_sign<false, false>
                <<<g_lc_blocks, TPB, 0, stream>>>(static_cast<const byte*>(d_in),
                                                  static_cast<byte*>(d_out_mags),
                                                  g_dec_outsize,
                                                  d_packed,
                                                  d_sign_bm,
                                                  g_sign_fullcarry);
        }
    }
}

long long WALTZ::lossless::lc_decompress_with_sign_finish(cudaStream_t stream) {
    cudaMemcpyAsync(
        g_dec_host_outsize, g_dec_outsize, sizeof(long long), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    return *g_dec_host_outsize;
}

long long WALTZ::lossless::lc_decompress_with_sign(const void* d_in,
                                                   void* d_out_mags,
                                                   long long mag_bytes_expected,
                                                   const unsigned char* d_packed,
                                                   uint32_t* d_sign_bm,
                                                   cudaStream_t stream) {
    lc_decompress_with_sign_launch(
        d_in, d_out_mags, mag_bytes_expected, d_packed, d_sign_bm, stream);
    return lc_decompress_with_sign_finish(stream);
}
