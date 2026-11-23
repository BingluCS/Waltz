/*
The BIT_2 permutations and RZE_1 hierarchy below are adapted from the LC
framework's h_BIT_2, h_RZE, and GPU component implementations.

BSD 3-Clause License

Copyright (c) 2021-2025, Noushin Azami, Alex Fallin, Brandon Burtchell,
Andrew Rodriguez, Benila Jerald, Yiqian Liu, Anju Mongandampulath Akathoott,
and Martin Burtscher. All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice,
   this list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the distribution.
3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.

URL: https://github.com/burtscher/LC-framework
Sponsor: U.S. Department of Energy, Office of Science, Office of Advanced
Scientific Research (ASCR), contract DE-SC0022223.
*/

#include "wzp_cpu.hpp"

#include <cstring>
#include <limits>
#include <stdexcept>

#if (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__))
#include <immintrin.h>
#define WALTZ_WZP_X86_TARGET 1
#define WALTZ_WZP_AVX2 __attribute__((target("avx2")))
#else
#define WALTZ_WZP_X86_TARGET 0
#endif

namespace waltz_cpu_detail {
namespace {

constexpr std::uint32_t kChunkElements = 8192;
constexpr std::uint32_t kChunkBytes = 16384;
constexpr std::uint32_t kGroupChunks = 16;
constexpr std::uint32_t kSignSlotBytes = 1024;
constexpr std::uint32_t kRawCode = 0x3fff;
constexpr std::uint32_t kBitmapBytes = 2048 + 256 + 32 + 4;

bool has_avx2() {
#if WALTZ_WZP_X86_TARGET
    __builtin_cpu_init();
    // GCC's AVX2 target also enables POPCNT, which it can select for the
    // explicit integer bit-count idiom below. Check both before dispatch.
    return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("popcnt");
#else
    return false;
#endif
}

[[noreturn]] void fail(const char* message) {
    throw std::runtime_error(message);
}

struct Shape {
    std::uint32_t chunks;
    std::uint32_t groups;
    std::size_t header_bytes;
    std::size_t max_bytes;
};

Shape shape(std::uint32_t n) {
    // The production GPU kernels use uint32_t for n * sizeof(uint16_t).
    if (n > std::numeric_limits<std::uint32_t>::max() / 2u)
        fail("WZP element count exceeds the GPU format limit");
    const std::uint32_t chunks = n == 0 ? 0 : 1u + (n - 1u) / kChunkElements;
    const std::uint32_t groups = (chunks + kGroupChunks - 1u) / kGroupChunks;
    const std::uint64_t header = std::uint64_t(groups) * 4u + std::uint64_t(chunks) * 3u;
    const std::uint64_t maximum = header + std::uint64_t(n) * 2u +
                                  (std::uint64_t(n) + 7u) / 8u;
    if (maximum > std::numeric_limits<std::size_t>::max())
        fail("WZP size exceeds the host address space");
    return {chunks, groups, static_cast<std::size_t>(header),
            static_cast<std::size_t>(maximum)};
}

std::uint16_t load16(const std::uint8_t* p) {
    return static_cast<std::uint16_t>(std::uint16_t(p[0]) | (std::uint16_t(p[1]) << 8u));
}

std::uint32_t load24(const std::uint8_t* p) {
    return std::uint32_t(p[0]) | (std::uint32_t(p[1]) << 8u) |
           (std::uint32_t(p[2]) << 16u);
}

std::uint32_t load32(const std::uint8_t* p) {
    return load24(p) | (std::uint32_t(p[3]) << 24u);
}

void store16(std::uint8_t* p, std::uint16_t value) {
    p[0] = static_cast<std::uint8_t>(value);
    p[1] = static_cast<std::uint8_t>(value >> 8u);
}

void store24(std::uint8_t* p, std::uint32_t value) {
    p[0] = static_cast<std::uint8_t>(value);
    p[1] = static_cast<std::uint8_t>(value >> 8u);
    p[2] = static_cast<std::uint8_t>(value >> 16u);
}

void store32(std::uint8_t* p, std::uint32_t value) {
    store24(p, value);
    p[3] = static_cast<std::uint8_t>(value >> 24u);
}

bool overlaps(const void* a, std::size_t a_bytes, const void* b, std::size_t b_bytes) {
    if (a_bytes == 0 || b_bytes == 0)
        return false;
    const std::uintptr_t ap = reinterpret_cast<std::uintptr_t>(a);
    const std::uintptr_t bp = reinterpret_cast<std::uintptr_t>(b);
    const auto maximum = std::numeric_limits<std::uintptr_t>::max();
    if (a_bytes > maximum - ap || b_bytes > maximum - bp)
        fail("WZP buffer address range overflows");
    return ap < bp + b_bytes && bp < ap + a_bytes;
}

std::uint32_t chunk_elements(std::uint32_t n, std::uint32_t chunk) {
    const std::uint32_t remaining = n - chunk * kChunkElements;
    return remaining < kChunkElements ? remaining : kChunkElements;
}

// These exchanges are the exact LC BIT_2 16x16 bit transpose. Work on
// integer values rather than aliased/native-endian byte loads.
void exchange(std::uint64_t& x, std::uint64_t& y, unsigned shift, std::uint64_t mask) {
    const std::uint64_t t = (x ^ (y >> shift)) & mask;
    x ^= t;
    y ^= t << shift;
}

std::uint64_t exchange34(std::uint64_t value) {
    constexpr std::uint64_t mask = 0x33333333CCCCCCCCull;
    const std::uint64_t other = value & ~mask;
    return (value & mask) | (other >> 34u) | (other << 34u);
}

std::uint64_t exchange17(std::uint64_t value) {
    constexpr std::uint64_t mask = 0x5555AAAA5555AAAAull;
    constexpr std::uint64_t upper = 0xFFFF0000FFFF0000ull;
    const std::uint64_t other = value & ~mask;
    return (value & mask) | ((other & upper) >> 17u) | ((other & ~upper) << 17u);
}

void transpose(std::uint64_t (&a)[4]) {
    exchange(a[0], a[2], 8u, 0x00FF00FF00FF00FFull);
    exchange(a[1], a[3], 8u, 0x00FF00FF00FF00FFull);
    exchange(a[0], a[1], 4u, 0x0F0F0F0F0F0F0F0Full);
    exchange(a[2], a[3], 4u, 0x0F0F0F0F0F0F0F0Full);
    for (unsigned j = 0; j != 4; ++j)
        a[j] = exchange17(exchange34(a[j]));
}

void bit_encode_scalar(const std::uint16_t* mag, std::uint32_t elements, std::uint8_t* out,
                       std::uint32_t first_block = 0) {
    const std::uint32_t blocks = elements / 16u;
    for (std::uint32_t block = first_block; block != blocks; ++block) {
        std::uint64_t a[4];
        for (unsigned j = 0; j != 4; ++j) {
            const std::uint16_t* p = mag + block * 16u + j * 4u;
            a[j] = std::uint64_t(p[0]) | (std::uint64_t(p[1]) << 16u) |
                   (std::uint64_t(p[2]) << 32u) | (std::uint64_t(p[3]) << 48u);
        }
        transpose(a);
        for (unsigned plane = 0; plane != 16; ++plane)
            store16(out + (plane * blocks + block) * 2u,
                    static_cast<std::uint16_t>(a[plane / 4u] >> ((plane % 4u) * 16u)));
    }
    for (std::uint32_t i = blocks * 16u; i != elements; ++i)
        store16(out + i * 2u, mag[i]);
}

void bit_decode_scalar(const std::uint8_t* in, std::uint32_t elements, std::uint16_t* mag,
                       std::uint32_t first_block = 0) {
    const std::uint32_t blocks = elements / 16u;
    for (std::uint32_t block = first_block; block != blocks; ++block) {
        std::uint64_t a[4] = {};
        for (unsigned plane = 0; plane != 16; ++plane)
            a[plane / 4u] |= std::uint64_t(load16(in + (plane * blocks + block) * 2u)) <<
                            ((plane % 4u) * 16u);
        transpose(a);
        for (unsigned j = 0; j != 16; ++j)
            mag[block * 16u + j] = static_cast<std::uint16_t>(a[j / 4u] >> ((j % 4u) * 16u));
    }
    for (std::uint32_t i = blocks * 16u; i != elements; ++i)
        mag[i] = load16(in + i * 2u);
}

std::uint32_t popcount32(std::uint32_t v) {
    v -= (v >> 1u) & 0x55555555u;
    v = (v & 0x33333333u) + ((v >> 2u) & 0x33333333u);
    v = (v + (v >> 4u)) & 0x0f0f0f0fu;
    return (v * 0x01010101u) >> 24u;
}

struct SignTables {
    std::uint8_t compact[16][16] = {};
    std::uint8_t expand[16][16] = {};
    std::uint8_t count[16] = {};
};

constexpr SignTables make_sign_tables() {
    SignTables result{};
    for (unsigned mask = 0; mask != 16; ++mask) {
        for (unsigned value = 0; value != 16; ++value) {
            unsigned rank = 0;
            for (unsigned bit = 0; bit != 4; ++bit) {
                if ((mask >> bit) & 1u) {
                    result.compact[mask][value] |= ((value >> bit) & 1u) << rank;
                    result.expand[mask][value] |= ((value >> rank) & 1u) << bit;
                    ++rank;
                }
            }
            result.count[mask] = static_cast<std::uint8_t>(rank);
        }
    }
    return result;
}

constexpr SignTables kSignTables = make_sign_tables();

std::uint32_t compact_sign16(std::uint32_t value, std::uint32_t mask) {
    if (mask == 0xffffu)
        return value;
    std::uint32_t result = 0;
    unsigned rank = 0;
    for (unsigned bit = 0; bit != 16; bit += 4u) {
        const unsigned selected = (mask >> bit) & 15u;
        result |= std::uint32_t(kSignTables.compact[selected][(value >> bit) & 15u]) << rank;
        rank += kSignTables.count[selected];
    }
    return result;
}

std::uint32_t expand_sign16(std::uint32_t value, std::uint32_t mask) {
    if (mask == 0xffffu)
        return value & 0xffffu;
    std::uint32_t result = 0;
    unsigned rank = 0;
    for (unsigned bit = 0; bit != 16; bit += 4u) {
        const unsigned selected = (mask >> bit) & 15u;
        result |= std::uint32_t(kSignTables.expand[selected][(value >> rank) & 15u]) << bit;
        rank += kSignTables.count[selected];
    }
    return result;
}

void compact_signs(const std::uint16_t* masks, const std::uint32_t* signs,
                   std::uint32_t elements, std::uint8_t* out) {
    std::uint64_t pending = 0;
    unsigned bits = 0;
    std::uint32_t written = 0;
    for (std::uint32_t block = 0; block != (elements + 15u) / 16u; ++block) {
        const std::uint32_t mask = masks[block];
        if (mask == 0)
            continue;
        const std::uint32_t value = (signs[block / 2u] >> ((block & 1u) * 16u)) & 0xffffu;
        pending |= std::uint64_t(compact_sign16(value, mask)) << bits;
        bits += popcount32(mask);
        if (bits >= 32u) {
            store32(out + written, static_cast<std::uint32_t>(pending));
            pending >>= 32u;
            bits -= 32u;
            written += 4u;
        }
    }
    while (bits != 0) {
        out[written++] = static_cast<std::uint8_t>(pending);
        pending >>= 8u;
        bits = bits > 8u ? bits - 8u : 0;
    }
}

void expand_signs(const std::uint16_t* masks, const std::uint8_t* packed,
                  std::uint32_t sign_bytes, std::uint32_t elements, std::uint32_t* out) {
    std::uint32_t rank = 0;
    for (std::uint32_t block = 0; block != (elements + 15u) / 16u; ++block) {
        const std::uint32_t mask = masks[block];
        if (mask == 0)
            continue;
        const std::uint32_t byte = rank >> 3u;
        std::uint32_t value = packed[byte];
        if (byte + 3u <= sign_bytes)
            value = load24(packed + byte);
        else for (std::uint32_t i = byte + 1u; i < sign_bytes; ++i)
            value |= std::uint32_t(packed[i]) << ((i - byte) * 8u);
        out[block / 2u] |= expand_sign16(value >> (rank & 7u), mask) << ((block & 1u) * 16u);
        rank += popcount32(mask);
    }
}

#if WALTZ_WZP_X86_TARGET
// Four independent 16x16 bit transposes travel in the four uint64 lanes.
// The surrounding 4x4 lane transpose only changes transport, not LC's bits.
WALTZ_WZP_AVX2 inline void transpose_words(__m256i& a0, __m256i& a1,
                                          __m256i& a2, __m256i& a3) {
    const __m256i lo01 = _mm256_unpacklo_epi64(a0, a1);
    const __m256i hi01 = _mm256_unpackhi_epi64(a0, a1);
    const __m256i lo23 = _mm256_unpacklo_epi64(a2, a3);
    const __m256i hi23 = _mm256_unpackhi_epi64(a2, a3);
    a0 = _mm256_permute2x128_si256(lo01, lo23, 0x20);
    a1 = _mm256_permute2x128_si256(hi01, hi23, 0x20);
    a2 = _mm256_permute2x128_si256(lo01, lo23, 0x31);
    a3 = _mm256_permute2x128_si256(hi01, hi23, 0x31);
}

template<unsigned Shift>
WALTZ_WZP_AVX2 inline void exchange_vec(__m256i& x, __m256i& y, __m256i mask) {
    const __m256i t = _mm256_and_si256(_mm256_xor_si256(x, _mm256_srli_epi64(y, Shift)), mask);
    x = _mm256_xor_si256(x, t);
    y = _mm256_xor_si256(y, _mm256_slli_epi64(t, Shift));
}

WALTZ_WZP_AVX2 inline __m256i exchange_inner(__m256i value) {
    const __m256i mask34 = _mm256_set1_epi64x(0x33333333CCCCCCCCll);
    const __m256i other34 = _mm256_andnot_si256(mask34, value);
    value = _mm256_or_si256(_mm256_and_si256(mask34, value),
        _mm256_or_si256(_mm256_srli_epi64(other34, 34), _mm256_slli_epi64(other34, 34)));
    const __m256i mask17 = _mm256_set1_epi64x(0x5555AAAA5555AAAAll);
    const __m256i upper = _mm256_set1_epi64x(static_cast<long long>(0xFFFF0000FFFF0000ull));
    const __m256i other17 = _mm256_andnot_si256(mask17, value);
    return _mm256_or_si256(_mm256_and_si256(mask17, value),
        _mm256_or_si256(_mm256_srli_epi64(_mm256_and_si256(other17, upper), 17),
                       _mm256_slli_epi64(_mm256_andnot_si256(upper, other17), 17)));
}

WALTZ_WZP_AVX2 inline void transpose_bits(__m256i& a0, __m256i& a1,
                                         __m256i& a2, __m256i& a3) {
    const __m256i mask8 = _mm256_set1_epi64x(0x00FF00FF00FF00FFll);
    const __m256i mask4 = _mm256_set1_epi64x(0x0F0F0F0F0F0F0F0Fll);
    exchange_vec<8>(a0, a2, mask8);
    exchange_vec<8>(a1, a3, mask8);
    exchange_vec<4>(a0, a1, mask4);
    exchange_vec<4>(a2, a3, mask4);
    a0 = exchange_inner(a0);
    a1 = exchange_inner(a1);
    a2 = exchange_inner(a2);
    a3 = exchange_inner(a3);
}

WALTZ_WZP_AVX2 inline void store_planes(std::uint8_t* out, std::uint32_t blocks,
                                       std::uint32_t block, unsigned plane, __m256i a) {
    const __m128i lo = _mm256_castsi256_si128(a);
    const __m128i hi = _mm256_extracti128_si256(a, 1);
    const __m128i even = _mm_unpacklo_epi16(lo, hi);
    const __m128i odd = _mm_unpackhi_epi16(lo, hi);
    const __m128i p01 = _mm_unpacklo_epi16(even, odd);
    const __m128i p23 = _mm_unpackhi_epi16(even, odd);
    _mm_storel_epi64(reinterpret_cast<__m128i*>(out + (plane * blocks + block) * 2u), p01);
    _mm_storel_epi64(reinterpret_cast<__m128i*>(out + ((plane + 1u) * blocks + block) * 2u),
                     _mm_srli_si128(p01, 8));
    _mm_storel_epi64(reinterpret_cast<__m128i*>(out + ((plane + 2u) * blocks + block) * 2u), p23);
    _mm_storel_epi64(reinterpret_cast<__m128i*>(out + ((plane + 3u) * blocks + block) * 2u),
                     _mm_srli_si128(p23, 8));
}

WALTZ_WZP_AVX2 inline __m256i load_planes(const std::uint8_t* in, std::uint32_t blocks,
                                         std::uint32_t block, unsigned plane) {
    const __m128i p0 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(in + (plane * blocks + block) * 2u));
    const __m128i p1 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(in + ((plane + 1u) * blocks + block) * 2u));
    const __m128i p2 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(in + ((plane + 2u) * blocks + block) * 2u));
    const __m128i p3 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(in + ((plane + 3u) * blocks + block) * 2u));
    const __m128i p01 = _mm_unpacklo_epi16(p0, p1);
    const __m128i p23 = _mm_unpacklo_epi16(p2, p3);
    return _mm256_set_m128i(_mm_unpackhi_epi32(p01, p23), _mm_unpacklo_epi32(p01, p23));
}

WALTZ_WZP_AVX2 void bit_encode_avx2(const std::uint16_t* mag, std::uint32_t elements,
                                   std::uint8_t* out) {
    const std::uint32_t blocks = elements / 16u;
    std::uint32_t block = 0;
    for (; block + 4u <= blocks; block += 4u) {
        __m256i a0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(mag + block * 16u));
        __m256i a1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(mag + block * 16u + 16u));
        __m256i a2 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(mag + block * 16u + 32u));
        __m256i a3 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(mag + block * 16u + 48u));
        transpose_words(a0, a1, a2, a3);
        transpose_bits(a0, a1, a2, a3);
        store_planes(out, blocks, block, 0, a0);
        store_planes(out, blocks, block, 4, a1);
        store_planes(out, blocks, block, 8, a2);
        store_planes(out, blocks, block, 12, a3);
    }
    bit_encode_scalar(mag, elements, out, block);
}

WALTZ_WZP_AVX2 void bit_decode_avx2(const std::uint8_t* in, std::uint32_t elements,
                                   std::uint16_t* mag) {
    const std::uint32_t blocks = elements / 16u;
    std::uint32_t block = 0;
    for (; block + 4u <= blocks; block += 4u) {
        __m256i a0 = load_planes(in, blocks, block, 0);
        __m256i a1 = load_planes(in, blocks, block, 4);
        __m256i a2 = load_planes(in, blocks, block, 8);
        __m256i a3 = load_planes(in, blocks, block, 12);
        transpose_bits(a0, a1, a2, a3);
        transpose_words(a0, a1, a2, a3);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(mag + block * 16u), a0);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(mag + block * 16u + 16u), a1);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(mag + block * 16u + 32u), a2);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(mag + block * 16u + 48u), a3);
    }
    bit_decode_scalar(in, elements, mag, block);
}

WALTZ_WZP_AVX2 std::uint32_t count_nonzero_avx2(const std::uint16_t* mag, std::uint32_t elements,
                                               std::uint16_t* masks) {
    const __m256i zero = _mm256_setzero_si256();
    std::uint32_t count = 0, i = 0;
    for (; i + 16u <= elements; i += 16u) {
        const __m256i value = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(mag + i));
        const __m256i equal = _mm256_cmpeq_epi16(value, zero);
        const __m256i packed = _mm256_permute4x64_epi64(_mm256_packs_epi16(equal, equal), 0xd8);
        const std::uint32_t mask = ~static_cast<std::uint32_t>(_mm_movemask_epi8(_mm256_castsi256_si128(packed))) & 0xffffu;
        masks[i / 16u] = static_cast<std::uint16_t>(mask);
        count += popcount32(mask);
    }
    std::uint32_t tail = 0;
    for (std::uint32_t bit = 0; i != elements; ++i, ++bit) {
        tail |= std::uint32_t(mag[i] != 0) << bit;
        count += mag[i] != 0;
    }
    if ((elements & 15u) != 0)
        masks[elements / 16u] = static_cast<std::uint16_t>(tail);
    return count;
}

WALTZ_WZP_AVX2 std::uint32_t rze_size_avx2(const std::uint8_t* in, std::uint32_t bytes,
                                          std::uint8_t* bitmap) {
    std::memset(bitmap, 0, kBitmapBytes);
    const __m256i zero = _mm256_setzero_si256();
    std::uint32_t stored = 0, i = 0;
    for (; i + 32u <= bytes; i += 32u) {
        const __m256i value = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(in + i));
        const std::uint32_t mask = ~static_cast<std::uint32_t>(_mm256_movemask_epi8(_mm256_cmpeq_epi8(value, zero)));
        store32(bitmap + i / 8u, mask);
        stored += popcount32(mask);
    }
    for (; i != bytes; ++i) {
        if (in[i] != 0) {
            bitmap[i >> 3u] |= static_cast<std::uint8_t>(1u << (i & 7u));
            ++stored;
        }
    }
    if (stored == 0)
        return 0;
    std::uint32_t base = 0;
    for (std::uint32_t range = 2048; range != 4; range /= 8u) {
        for (std::uint32_t pos = 0; pos != range; pos += 32u) {
            const __m256i value = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(bitmap + base + pos));
            // Byte zero of every RE level has an implicit previous value zero.
            const __m256i previous = pos == 0
                ? _mm256_alignr_epi8(value, _mm256_permute2x128_si256(value, value, 0x08), 15)
                : _mm256_loadu_si256(reinterpret_cast<const __m256i*>(bitmap + base + pos - 1u));
            const std::uint32_t mask = ~static_cast<std::uint32_t>(_mm256_movemask_epi8(_mm256_cmpeq_epi8(value, previous)));
            store32(bitmap + base + range + pos / 8u, mask);
            stored += popcount32(mask);
        }
        base += range;
    }
    return stored + 4u;
}
#endif

std::uint32_t count_nonzero(const std::uint16_t* mag, std::uint32_t elements, bool avx2,
                             std::uint16_t* masks) {
#if WALTZ_WZP_X86_TARGET
    if (avx2)
        return count_nonzero_avx2(mag, elements, masks);
#else
    (void)avx2;
    (void)masks;
#endif
    std::uint32_t count = 0;
    for (std::uint32_t i = 0; i != elements; ++i)
        count += mag[i] != 0;
    return count;
}

unsigned first_bit(std::uint32_t mask) {
#if defined(__GNUC__) || defined(__clang__)
    return static_cast<unsigned>(__builtin_ctz(mask));
#else
    unsigned bit = 0;
    while ((mask & 1u) == 0) { mask >>= 1u; ++bit; }
    return bit;
#endif
}

void append_sparse(const std::uint8_t* in, std::uint32_t bytes,
                    const std::uint8_t* bitmap, std::uint8_t* out, std::uint32_t& position) {
    for (std::uint32_t i = 0; i < bytes; i += 32u) {
        std::uint32_t mask = load32(bitmap + i / 8u);
        if (mask == 0xffffffffu) {
            std::memcpy(out + position, in + i, 32);
            position += 32u;
        } else while (mask != 0) {
            out[position++] = in[i + first_bit(mask)];
            mask &= mask - 1u;
        }
    }
}

void rze_write_sparse(const std::uint8_t* in, std::uint32_t bytes,
                       const std::uint8_t* bitmap, std::uint8_t* out) {
    std::uint32_t position = 0;
    append_sparse(in, bytes, bitmap, out, position);
    std::uint32_t base = 0;
    for (std::uint32_t range = 2048; range != 4; range /= 8u) {
        append_sparse(bitmap + base, range, bitmap + base + range, out, position);
        base += range;
    }
    std::memcpy(out + position, bitmap + base, 4);
}

// Return the RZE byte count without the two-byte original-size trailer. The
// GPU WZP wrapper elides that trailer because raw chunk length is implicit.
std::uint32_t rze_size(const std::uint8_t* in, std::uint32_t bytes, std::uint8_t* bitmap) {
    std::memset(bitmap, 0, kBitmapBytes);
    std::uint32_t stored = 0;
    for (std::uint32_t i = 0; i != bytes; ++i) {
        if (in[i] != 0) {
            bitmap[i >> 3u] |= static_cast<std::uint8_t>(1u << (i & 7u));
            ++stored;
        }
    }
    if (stored == 0)
        return 0;
    std::uint32_t base = 0;
    for (std::uint32_t range = 2048; range != 4; range /= 8u) {
        std::uint8_t previous = 0;
        for (std::uint32_t i = 0; i != range; ++i) {
            const std::uint8_t value = bitmap[base + i];
            if (value != previous) {
                bitmap[base + range + (i >> 3u)] |=
                    static_cast<std::uint8_t>(1u << (i & 7u));
                ++stored;
            }
            previous = value;
        }
        base += range;
    }
    return stored + 4u;
}

void rze_write(const std::uint8_t* in, std::uint32_t bytes,
               const std::uint8_t* bitmap, std::uint8_t* out) {
    std::uint32_t position = 0;
    for (std::uint32_t i = 0; i != bytes; ++i)
        if (in[i] != 0)
            out[position++] = in[i];
    std::uint32_t base = 0;
    for (std::uint32_t range = 2048; range != 4; range /= 8u) {
        std::uint8_t previous = 0;
        for (std::uint32_t i = 0; i != range; ++i) {
            const std::uint8_t value = bitmap[base + i];
            if (value != previous)
                out[position++] = value;
            previous = value;
        }
        base += range;
    }
    std::memcpy(out + position, bitmap + base, 4);
}

unsigned popcount_byte(std::uint8_t value) {
    unsigned v = value;
    v = v - ((v >> 1u) & 0x55u);
    v = (v & 0x33u) + ((v >> 2u) & 0x33u);
    return (v + (v >> 4u)) & 0x0fu;
}

void rze_decode(const std::uint8_t* in, std::uint32_t stored,
                std::uint32_t raw_bytes, std::uint8_t* expanded,
                std::uint8_t* bitmap, bool sparse) {
    if (stored < 4u)
        fail("WZP truncated RZE bitmap");
    std::uint32_t position = stored - 4u;
    std::memcpy(bitmap + 2048 + 256 + 32, in + position, 4);
    std::uint32_t base = 2048 + 256 + 32;
    for (std::uint32_t range = 32; range <= 2048; range *= 8u) {
        const std::uint8_t* control = bitmap + base;
        std::uint32_t changes = 0;
        for (std::uint32_t i = 0; i != range / 8u; ++i)
            changes += popcount_byte(control[i]);
        if (changes > position)
            fail("WZP RZE transition stream underflows");
        position -= changes;
        std::uint32_t read = position;
        base -= range;
        std::uint8_t value = 0;
        for (std::uint32_t i = 0; i != range; ++i) {
            if ((control[i >> 3u] >> (i & 7u)) & 1u) {
                const std::uint8_t next = in[read++];
                if (next == value)
                    fail("WZP redundant RZE transition");
                value = next;
            }
            bitmap[base + i] = value;
        }
    }
    std::uint32_t nonzero = 0;
    for (std::uint32_t i = 0; i != 2048; ++i)
        nonzero += popcount_byte(bitmap[i]);
    if (nonzero == 0 || nonzero != position)
        fail("WZP RZE value stream length mismatch");
    const std::uint32_t full = raw_bytes / 8u;
    const std::uint32_t partial = raw_bytes & 7u;
    if (partial != 0 && (bitmap[full] >> partial) != 0)
        fail("WZP nonzero partial-chunk RZE padding");
    for (std::uint32_t i = full + (partial != 0); i != 2048; ++i)
        if (bitmap[i] != 0)
            fail("WZP nonzero partial-chunk RZE padding");
    std::uint32_t read = 0;
    if (sparse) {
        std::memset(expanded, 0, raw_bytes);
        for (std::uint32_t i = 0; i < raw_bytes; i += 32u) {
            std::uint32_t mask = load32(bitmap + i / 8u);
            while (mask != 0) {
                const std::uint8_t value = in[read++];
                if (value == 0)
                    fail("WZP zero value in sparse RZE stream");
                expanded[i + first_bit(mask)] = value;
                mask &= mask - 1u;
            }
        }
        return;
    }
    for (std::uint32_t i = 0; i != raw_bytes; ++i) {
        std::uint8_t value = 0;
        if ((bitmap[i >> 3u] >> (i & 7u)) & 1u) {
            value = in[read++];
            if (value == 0)
                fail("WZP zero value in sparse RZE stream");
        }
        expanded[i] = value;
    }
}

struct ChunkSize {
    std::uint32_t magnitude;
    std::uint32_t signs;
    bool raw;
};

ChunkSize inspect_meta(std::uint32_t metadata, std::uint32_t elements) {
    const std::uint32_t code = metadata & kRawCode;
    const std::uint32_t sign_code = metadata >> 14u;
    if (code == 0) {
        if (sign_code != 0)
            fail("WZP zero chunk carries sign metadata");
        return {0, 0, false};
    }
    const std::uint32_t raw_bytes = elements * 2u;
    const bool raw = code == kRawCode;
    if (!raw && (code < 4u || code + 2u >= raw_bytes))
        fail("WZP invalid compressed magnitude length");
    const std::uint32_t sign_bytes = sign_code == 0 ? kSignSlotBytes : sign_code;
    if (sign_bytes > (elements + 7u) / 8u)
        fail("WZP sign stream exceeds the chunk size");
    return {raw ? raw_bytes : code, sign_bytes, raw};
}

std::uint32_t group_length(const std::uint8_t* metadata, std::uint32_t n,
                          std::uint32_t chunks, std::uint32_t group) {
    const std::uint32_t first = group * kGroupChunks;
    const std::uint32_t last = chunks - first < kGroupChunks ? chunks : first + kGroupChunks;
    std::uint32_t length = 0;
    for (std::uint32_t chunk = first; chunk != last; ++chunk) {
        const ChunkSize cs = inspect_meta(load24(metadata + std::size_t(chunk) * 3u),
                                         chunk_elements(n, chunk));
        length += cs.magnitude + cs.signs;
    }
    return length;
}

void swap_range(std::uint32_t* ranges, std::uint32_t a, std::uint32_t b) {
    const std::uint32_t offset = ranges[a * 2u];
    const std::uint32_t length = ranges[a * 2u + 1u];
    ranges[a * 2u] = ranges[b * 2u];
    ranges[a * 2u + 1u] = ranges[b * 2u + 1u];
    ranges[b * 2u] = offset;
    ranges[b * 2u + 1u] = length;
}

void sift_ranges(std::uint32_t* ranges, std::uint32_t root, std::uint32_t count) {
    while (root < count / 2u) {
        std::uint32_t child = root * 2u + 1u;
        if (child + 1u < count && ranges[child * 2u] < ranges[(child + 1u) * 2u])
            ++child;
        if (ranges[root * 2u] >= ranges[child * 2u])
            return;
        swap_range(ranges, root, child);
        root = child;
    }
}

void validate_directory(const std::uint8_t* in, std::size_t payload_bytes,
                        const std::uint8_t* metadata, const Shape& s,
                        std::uint32_t n, std::uint32_t* sign_scratch) {
    if (payload_bytes > std::numeric_limits<std::uint32_t>::max())
        fail("WZP payload exceeds its 32-bit directory");
    if (s.groups == 1) {
        const std::uint32_t length = group_length(metadata, n, s.chunks, 0);
        if (load32(in) != 0 || length != payload_bytes)
            fail("WZP directory does not cover the payload");
        return;
    }
    // For two or more groups, n > 131072, so ceil(n/32) output words are
    // always more than the two words per group required here. Borrow this
    // caller-owned space before constructing the natural sign bitmap.
    for (std::uint32_t group = 0; group != s.groups; ++group) {
        const std::uint32_t offset = load32(in + std::size_t(group) * 4u);
        const std::uint32_t length = group_length(metadata, n, s.chunks, group);
        if (offset > payload_bytes || length > payload_bytes - offset ||
            (length == 0 && offset != 0))
            fail("WZP directory range is outside the payload");
        sign_scratch[group * 2u] = offset;
        sign_scratch[group * 2u + 1u] = length;
    }
    for (std::uint32_t i = s.groups / 2u; i != 0; --i)
        sift_ranges(sign_scratch, i - 1u, s.groups);
    for (std::uint32_t count = s.groups; count > 1u; --count) {
        swap_range(sign_scratch, 0, count - 1u);
        sift_ranges(sign_scratch, 0, count - 1u);
    }
    std::size_t cursor = 0;
    for (std::uint32_t i = 0; i != s.groups; ++i) {
        const std::uint32_t offset = sign_scratch[i * 2u];
        const std::uint32_t length = sign_scratch[i * 2u + 1u];
        if (length == 0)
            continue;
        if (offset != cursor)
            fail("WZP payload groups overlap or leave a gap");
        cursor += length;
    }
    if (cursor != payload_bytes)
        fail("WZP trailing or missing payload bytes");
}

} // namespace

std::size_t wzp_max_bytes(std::uint32_t n) {
    return shape(n).max_bytes;
}

std::size_t wzp_encode(const std::uint16_t* mag, const std::uint32_t* sign,
                       std::uint32_t n, std::uint8_t* out, std::size_t capacity, bool avx2) {
    const Shape s = shape(n);
    if (n == 0)
        return 0;
    if (mag == nullptr || sign == nullptr || out == nullptr)
        fail("WZP null encode buffer");
    if (capacity < s.header_bytes)
        fail("WZP output capacity is too small");
    const std::size_t sign_words = (std::size_t(n) + 31u) / 32u;
    if (overlaps(mag, std::size_t(n) * 2u, sign, sign_words * 4u) ||
        overlaps(mag, std::size_t(n) * 2u, out, capacity) ||
        overlaps(sign, sign_words * 4u, out, capacity))
        fail("WZP encode buffers overlap");
    avx2 = avx2 && has_avx2();
    std::uint8_t transformed[kChunkBytes];
    std::uint8_t bitmap[kBitmapBytes];
    std::uint16_t nonzero_masks[kChunkElements / 16u];
    std::uint8_t* const metadata = out + std::size_t(s.groups) * 4u;
    std::size_t cursor = s.header_bytes;
    for (std::uint32_t group = 0; group != s.groups; ++group) {
        store32(out + std::size_t(group) * 4u, 0);
        bool group_nonzero = false;
        const std::uint32_t first = group * kGroupChunks;
        const std::uint32_t last = s.chunks - first < kGroupChunks ? s.chunks : first + kGroupChunks;
        for (std::uint32_t chunk = first; chunk != last; ++chunk) {
            const std::uint32_t elements = chunk_elements(n, chunk);
            const std::uint32_t base = chunk * kChunkElements;
            const std::uint16_t* const source = mag + base;
            const std::uint32_t nnz = count_nonzero(source, elements, avx2, nonzero_masks);
            if (nnz == 0) {
                store24(metadata + std::size_t(chunk) * 3u, 0);
                continue;
            }
            const std::uint32_t raw_bytes = elements * 2u;
#if WALTZ_WZP_X86_TARGET
            if (avx2)
                bit_encode_avx2(source, elements, transformed);
            else
#endif
                bit_encode_scalar(source, elements, transformed);
            std::uint32_t compressed;
#if WALTZ_WZP_X86_TARGET
            if (avx2)
                compressed = rze_size_avx2(transformed, raw_bytes, bitmap);
            else
#endif
                compressed = rze_size(transformed, raw_bytes, bitmap);
            const bool raw = compressed + 2u >= raw_bytes;
            const std::uint32_t stored = raw ? raw_bytes : compressed;
            const std::uint32_t sign_bytes = (nnz + 7u) / 8u;
            const std::uint32_t chunk_bytes = stored + sign_bytes;
            if (chunk_bytes > capacity - cursor)
                fail("WZP output capacity is too small");
            if (cursor - s.header_bytes >
                std::numeric_limits<std::uint32_t>::max() - chunk_bytes)
                fail("WZP payload exceeds its 32-bit directory");
            if (!group_nonzero) {
                store32(out + std::size_t(group) * 4u,
                        static_cast<std::uint32_t>(cursor - s.header_bytes));
                group_nonzero = true;
            }
            if (raw) {
                for (std::uint32_t i = 0; i != elements; ++i)
                    store16(out + cursor + i * 2u, source[i]);
            } else {
                if (avx2)
                    rze_write_sparse(transformed, raw_bytes, bitmap, out + cursor);
                else
                    rze_write(transformed, raw_bytes, bitmap, out + cursor);
            }
            std::uint8_t* const packed_sign = out + cursor + stored;
            std::memset(packed_sign, 0, sign_bytes);
            std::uint32_t rank = 0;
            if (nnz == elements) {
                // No compaction is needed. Explicit LE stores keep this fast
                // path portable even though sign words are native integers.
                for (std::uint32_t i = 0; i != sign_bytes; ++i)
                    packed_sign[i] = static_cast<std::uint8_t>(sign[(base / 32u) + (i / 4u)] >> ((i & 3u) * 8u));
                if ((nnz & 7u) != 0)
                    packed_sign[sign_bytes - 1u] &= static_cast<std::uint8_t>((1u << (nnz & 7u)) - 1u);
            } else if (avx2) {
                compact_signs(nonzero_masks, sign + base / 32u, elements, packed_sign);
            } else for (std::uint32_t i = 0; i != elements; ++i) {
                if (source[i] == 0)
                    continue;
                const std::uint32_t global = base + i;
                const std::uint32_t negative = (sign[global >> 5u] >> (global & 31u)) & 1u;
                packed_sign[rank >> 3u] |= static_cast<std::uint8_t>(negative << (rank & 7u));
                ++rank;
            }
            const std::uint32_t sign_code = sign_bytes == kSignSlotBytes ? 0u : sign_bytes;
            store24(metadata + std::size_t(chunk) * 3u,
                    (raw ? kRawCode : stored) | (sign_code << 14u));
            cursor += chunk_bytes;
        }
    }
    return cursor;
}

void wzp_decode(const std::uint8_t* in, std::size_t bytes, std::uint32_t n,
                std::uint16_t* mag, std::uint32_t* sign, bool avx2) {
    const Shape s = shape(n);
    if (n == 0) {
        if (bytes != 0)
            fail("WZP nonempty archive for zero elements");
        return;
    }
    if (in == nullptr || mag == nullptr || sign == nullptr)
        fail("WZP null decode buffer");
    if (bytes < s.header_bytes || bytes > s.max_bytes)
        fail("WZP archive size is outside its valid bounds");
    const std::size_t sign_words = (std::size_t(n) + 31u) / 32u;
    if (overlaps(mag, std::size_t(n) * 2u, sign, sign_words * 4u) ||
        overlaps(mag, std::size_t(n) * 2u, in, bytes) ||
        overlaps(sign, sign_words * 4u, in, bytes))
        fail("WZP decode buffers overlap");
    avx2 = avx2 && has_avx2();
    const std::uint8_t* const metadata = in + std::size_t(s.groups) * 4u;
    const std::uint8_t* const payload = in + s.header_bytes;
    validate_directory(in, bytes - s.header_bytes, metadata, s, n, sign);
    std::memset(sign, 0, sign_words * sizeof(std::uint32_t));
    std::uint8_t expanded[kChunkBytes];
    std::uint8_t bitmap[kBitmapBytes];
    std::uint16_t nonzero_masks[kChunkElements / 16u];
    for (std::uint32_t group = 0; group != s.groups; ++group) {
        std::size_t cursor = load32(in + std::size_t(group) * 4u);
        const std::uint32_t first = group * kGroupChunks;
        const std::uint32_t last = s.chunks - first < kGroupChunks ? s.chunks : first + kGroupChunks;
        for (std::uint32_t chunk = first; chunk != last; ++chunk) {
            const std::uint32_t elements = chunk_elements(n, chunk);
            const std::uint32_t base = chunk * kChunkElements;
            const ChunkSize cs = inspect_meta(load24(metadata + std::size_t(chunk) * 3u), elements);
            std::uint16_t* const destination = mag + base;
            if (cs.magnitude == 0) {
                std::memset(destination, 0, std::size_t(elements) * 2u);
                continue;
            }
            const std::uint8_t* const source = payload + cursor;
            if (cs.raw) {
                for (std::uint32_t i = 0; i != elements; ++i)
                    destination[i] = load16(source + i * 2u);
            } else {
                rze_decode(source, cs.magnitude, elements * 2u, expanded, bitmap, avx2);
#if WALTZ_WZP_X86_TARGET
                if (avx2)
                    bit_decode_avx2(expanded, elements, destination);
                else
#endif
                    bit_decode_scalar(expanded, elements, destination);
            }
            const std::uint32_t nnz = count_nonzero(destination, elements, avx2, nonzero_masks);
            if (nnz == 0 || (nnz + 7u) / 8u != cs.signs)
                fail("WZP compact sign length does not match nonzero magnitudes");
            const std::uint8_t* const packed_sign = source + cs.magnitude;
            if ((nnz & 7u) != 0 && (packed_sign[cs.signs - 1u] >> (nnz & 7u)) != 0)
                fail("WZP nonzero compact sign padding");
            std::uint32_t rank = 0;
            if (nnz == elements) {
                const std::uint32_t words = elements / 32u;
                for (std::uint32_t i = 0; i != words; ++i)
                    sign[base / 32u + i] = load32(packed_sign + i * 4u);
                std::uint32_t last = 0;
                for (std::uint32_t i = words * 4u; i != cs.signs; ++i)
                    last |= std::uint32_t(packed_sign[i]) << ((i & 3u) * 8u);
                if ((elements & 31u) != 0)
                    sign[base / 32u + words] = last;
            } else if (avx2) {
                expand_signs(nonzero_masks, packed_sign, cs.signs, elements, sign + base / 32u);
            } else for (std::uint32_t i = 0; i != elements; ++i) {
                if (destination[i] == 0)
                    continue;
                const std::uint32_t negative = (packed_sign[rank >> 3u] >> (rank & 7u)) & 1u;
                const std::uint32_t global = base + i;
                sign[global >> 5u] |= negative << (global & 31u);
                ++rank;
            }
            cursor += cs.magnitude + cs.signs;
        }
    }
}

} // namespace waltz_cpu_detail
