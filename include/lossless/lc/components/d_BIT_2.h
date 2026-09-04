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

#define swp(x, y, s, m)                                                                            \
    t = ((x) ^ ((y) >> (s))) & (m);                                                                \
    (x) ^= t;                                                                                      \
    (y) ^= t << (s);

// Keep the four 64-bit words owned by a thread in registers throughout BIT_2.
// The original LC implementation repeatedly transformed `a[j]` through a pointer
// into shared memory.  In the fused Waltz kernels that makes the bit shuffle contend
// with RZE for shared-memory bandwidth.  These two involutions are pure per-word
// permutations, so applying them to register values preserves the byte stream.
static __device__ __forceinline__ unsigned long long d_BIT_2_swap_34(unsigned long long v) {
    constexpr unsigned long long m = 0x33333333CCCCCCCCULL;
    const unsigned long long vnm = v & ~m;
    return (v & m) | (vnm >> 34) | (vnm << 34);
}

static __device__ __forceinline__ unsigned long long d_BIT_2_swap_17(unsigned long long v) {
    constexpr unsigned long long m = 0x5555AAAA5555AAAAULL;
    constexpr unsigned long long m1 = 0xFFFF0000FFFF0000ULL;
    const unsigned long long vnm = v & ~m;
    return (v & m) | ((vnm & m1) >> 17) | ((vnm & ~m1) << 17);
}

static __device__ inline bool d_BIT_2(int& csize, byte in[CS], byte out[CS], byte temp[CS]) {
    const int extra = csize % (16 * 16 / 8);
    const int size = (csize - extra) / 2;
    const int tid = threadIdx.x;
    unsigned long long* const in_l = (unsigned long long*)in;
    unsigned short* const out_s = (unsigned short*)out;

    for (int pos = 16 * tid; pos < size; pos += 16 * TPB) {
        const int wi = pos / 4; // process 16 shorts in four 64-bit registers
        unsigned long long a0 = in_l[wi + 0];
        unsigned long long a1 = in_l[wi + 1];
        unsigned long long a2 = in_l[wi + 2];
        unsigned long long a3 = in_l[wi + 3];
        unsigned long long t;

        swp(a0, a2, 8, 0x00FF00FF00FF00FFULL);
        swp(a1, a3, 8, 0x00FF00FF00FF00FFULL);
        swp(a0, a1, 4, 0x0F0F0F0F0F0F0F0FULL);
        swp(a2, a3, 4, 0x0F0F0F0F0F0F0F0FULL);

        a0 = d_BIT_2_swap_17(d_BIT_2_swap_34(a0));
        a1 = d_BIT_2_swap_17(d_BIT_2_swap_34(a1));
        a2 = d_BIT_2_swap_17(d_BIT_2_swap_34(a2));
        a3 = d_BIT_2_swap_17(d_BIT_2_swap_34(a3));

        const int obase = pos / 16;
        const int ostride = size / 16;
        out_s[obase + 0 * ostride] = a0;
        out_s[obase + 1 * ostride] = a0 >> 16;
        out_s[obase + 2 * ostride] = a0 >> 32;
        out_s[obase + 3 * ostride] = a0 >> 48;
        out_s[obase + 4 * ostride] = a1;
        out_s[obase + 5 * ostride] = a1 >> 16;
        out_s[obase + 6 * ostride] = a1 >> 32;
        out_s[obase + 7 * ostride] = a1 >> 48;
        out_s[obase + 8 * ostride] = a2;
        out_s[obase + 9 * ostride] = a2 >> 16;
        out_s[obase + 10 * ostride] = a2 >> 32;
        out_s[obase + 11 * ostride] = a2 >> 48;
        out_s[obase + 12 * ostride] = a3;
        out_s[obase + 13 * ostride] = a3 >> 16;
        out_s[obase + 14 * ostride] = a3 >> 32;
        out_s[obase + 15 * ostride] = a3 >> 48;
    }

    // copy leftover bytes
    if (tid < extra)
        out[csize - extra + tid] = in[csize - extra + tid];
    return true;
}

static __device__ inline void d_iBIT_2(int& csize, byte in[CS], byte out[CS], byte temp[CS]) {
    const int extra = csize % (16 * 16 / 8);
    const int size = (csize - extra) / 2;
    const int tid = threadIdx.x;
    unsigned short* const in_s = (unsigned short*)in;
    unsigned long long* const out_l = (unsigned long long*)out;

    for (int pos = 16 * tid; pos < size; pos += 16 * TPB) {
        const int ibase = pos / 16;
        const int istride = size / 16;
        unsigned long long a0 = in_s[ibase + 0 * istride] |
                                ((unsigned long long)in_s[ibase + 1 * istride] << 16) |
                                ((unsigned long long)in_s[ibase + 2 * istride] << 32) |
                                ((unsigned long long)in_s[ibase + 3 * istride] << 48);
        unsigned long long a1 = in_s[ibase + 4 * istride] |
                                ((unsigned long long)in_s[ibase + 5 * istride] << 16) |
                                ((unsigned long long)in_s[ibase + 6 * istride] << 32) |
                                ((unsigned long long)in_s[ibase + 7 * istride] << 48);
        unsigned long long a2 = in_s[ibase + 8 * istride] |
                                ((unsigned long long)in_s[ibase + 9 * istride] << 16) |
                                ((unsigned long long)in_s[ibase + 10 * istride] << 32) |
                                ((unsigned long long)in_s[ibase + 11 * istride] << 48);
        unsigned long long a3 = in_s[ibase + 12 * istride] |
                                ((unsigned long long)in_s[ibase + 13 * istride] << 16) |
                                ((unsigned long long)in_s[ibase + 14 * istride] << 32) |
                                ((unsigned long long)in_s[ibase + 15 * istride] << 48);
        unsigned long long t;

        swp(a0, a2, 8, 0x00FF00FF00FF00FFULL);
        swp(a1, a3, 8, 0x00FF00FF00FF00FFULL);
        swp(a0, a1, 4, 0x0F0F0F0F0F0F0F0FULL);
        swp(a2, a3, 4, 0x0F0F0F0F0F0F0F0FULL);

        a0 = d_BIT_2_swap_17(d_BIT_2_swap_34(a0));
        a1 = d_BIT_2_swap_17(d_BIT_2_swap_34(a1));
        a2 = d_BIT_2_swap_17(d_BIT_2_swap_34(a2));
        a3 = d_BIT_2_swap_17(d_BIT_2_swap_34(a3));

        const int wi = pos / 4;
        out_l[wi + 0] = a0;
        out_l[wi + 1] = a1;
        out_l[wi + 2] = a2;
        out_l[wi + 3] = a3;
    }

    // copy leftover bytes
    if (tid < extra)
        out[csize - extra + tid] = in[csize - extra + tid];
}
