#include "../src/wzp_cpu.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>

// Optional byte-for-byte oracle from the original scalar LC components:
// -DWALTZ_TEST_LC_REFERENCE -I/path/to/LC-framework/components
// No external LC source is needed to build or use the CPU library.
#if defined(WALTZ_TEST_LC_REFERENCE)
#include <cassert>
using byte = unsigned char;
static constexpr int CS = 16384;
#include "h_BIT_2.h"
#include "include/h_RZE.h"
#undef swp
#endif

namespace {

using waltz_cpu_detail::wzp_decode;
using waltz_cpu_detail::wzp_encode;
using waltz_cpu_detail::wzp_max_bytes;

void require(bool condition, const char* message) {
    if (!condition)
        throw std::runtime_error(message);
}

std::uint32_t get24(const std::uint8_t* p) {
    return std::uint32_t(p[0]) | (std::uint32_t(p[1]) << 8u) |
           (std::uint32_t(p[2]) << 16u);
}

std::uint32_t get32(const std::uint8_t* p) {
    return get24(p) | (std::uint32_t(p[3]) << 24u);
}

void put24(std::uint8_t* p, std::uint32_t value) {
    p[0] = static_cast<std::uint8_t>(value);
    p[1] = static_cast<std::uint8_t>(value >> 8u);
    p[2] = static_cast<std::uint8_t>(value >> 16u);
}

void put32(std::uint8_t* p, std::uint32_t value) {
    put24(p, value);
    p[3] = static_cast<std::uint8_t>(value >> 24u);
}

std::uint32_t random_word(std::uint32_t& state) {
    state ^= state << 13u;
    state ^= state >> 17u;
    state ^= state << 5u;
    return state;
}

std::vector<std::uint8_t> encode(const std::vector<std::uint16_t>& mag,
                               const std::vector<std::uint32_t>& signs, bool avx2 = true) {
    std::vector<std::uint8_t> archive(wzp_max_bytes(static_cast<std::uint32_t>(mag.size())), 0xa5);
    const std::size_t bytes = wzp_encode(mag.data(), signs.data(),
                                       static_cast<std::uint32_t>(mag.size()),
                                       archive.data(), archive.size(), avx2);
    archive.resize(bytes);
    return archive;
}

#if defined(WALTZ_TEST_LC_REFERENCE)
std::vector<std::uint8_t> reference_encode(const std::vector<std::uint16_t>& mag,
                                         const std::vector<std::uint32_t>& signs) {
    const std::uint32_t n = static_cast<std::uint32_t>(mag.size());
    const std::uint32_t chunks = (n + 8191u) / 8192u;
    const std::uint32_t groups = (chunks + 15u) / 16u;
    const std::size_t header = std::size_t(groups) * 4u + std::size_t(chunks) * 3u;
    std::vector<std::uint8_t> archive(wzp_max_bytes(n), 0);
    std::size_t cursor = header;
    alignas(16) std::uint8_t raw[CS];
    alignas(16) std::uint8_t bit[CS];
    alignas(16) std::uint8_t rze[CS];
    for (std::uint32_t group = 0; group != groups; ++group) {
        bool nonzero_group = false;
        for (std::uint32_t chunk = group * 16u; chunk != std::min(chunks, group * 16u + 16u); ++chunk) {
            const std::uint32_t base = chunk * 8192u;
            const std::uint32_t count = std::min(8192u, n - base);
            std::uint32_t nnz = 0;
            for (std::uint32_t i = 0; i != count; ++i) {
                raw[i * 2u] = static_cast<std::uint8_t>(mag[base + i]);
                raw[i * 2u + 1u] = static_cast<std::uint8_t>(mag[base + i] >> 8u);
                nnz += mag[base + i] != 0;
            }
            if (nnz == 0)
                continue;
            if (!nonzero_group) {
                put32(archive.data() + group * 4u, static_cast<std::uint32_t>(cursor - header));
                nonzero_group = true;
            }
            int compressed = static_cast<int>(count * 2u);
            h_BIT_2(compressed, raw, bit);
            const bool good = h_RZE<byte>(compressed, bit, rze);
            const bool store_raw = !good || compressed >= static_cast<int>(count * 2u);
            const std::uint32_t stored = store_raw ? count * 2u : static_cast<std::uint32_t>(compressed - 2);
            if (store_raw) {
                for (std::uint32_t i = 0; i != count; ++i) {
                    archive[cursor + i * 2u] = static_cast<std::uint8_t>(mag[base + i]);
                    archive[cursor + i * 2u + 1u] = static_cast<std::uint8_t>(mag[base + i] >> 8u);
                }
            } else {
                std::memcpy(archive.data() + cursor, rze, stored);
            }
            const std::uint32_t sign_bytes = (nnz + 7u) / 8u;
            std::uint32_t rank = 0;
            for (std::uint32_t i = 0; i != count; ++i) {
                if (mag[base + i] == 0)
                    continue;
                const std::uint32_t global = base + i;
                if ((signs[global >> 5u] >> (global & 31u)) & 1u)
                    archive[cursor + stored + (rank >> 3u)] |= static_cast<std::uint8_t>(1u << (rank & 7u));
                ++rank;
            }
            put24(archive.data() + groups * 4u + chunk * 3u,
                  (store_raw ? 0x3fffu : stored) | ((sign_bytes == 1024u ? 0u : sign_bytes) << 14u));
            cursor += stored + sign_bytes;
        }
    }
    archive.resize(cursor);
    return archive;
}
#endif

void check_decode(const std::vector<std::uint8_t>& archive,
                  const std::vector<std::uint16_t>& expected,
                  const std::vector<std::uint32_t>& input_sign, bool avx2 = true) {
    const std::uint32_t n = static_cast<std::uint32_t>(expected.size());
    // Guard magnitudes and the exact minimal ceil(n/32) sign allocation.
    std::vector<std::uint16_t> output(expected.size() + 2u, 0xdead);
    std::vector<std::uint32_t> signs(input_sign.size() + 2u, 0xabcdef01);
    wzp_decode(archive.data(), archive.size(), n, output.data() + 1u, signs.data() + 1u, avx2);
    require(output.front() == 0xdead && output.back() == 0xdead, "magnitude guard overwritten");
    require(signs.front() == 0xabcdef01 && signs.back() == 0xabcdef01, "sign guard overwritten");
    for (std::uint32_t i = 0; i != n; ++i) {
        require(output[i + 1u] == expected[i], "magnitude round trip mismatch");
        const bool actual = ((signs[1u + (i >> 5u)] >> (i & 31u)) & 1u) != 0;
        const bool wanted = expected[i] != 0 && ((input_sign[i >> 5u] >> (i & 31u)) & 1u) != 0;
        require(actual == wanted, "sign round trip mismatch");
    }
    if ((n & 31u) != 0)
        require((signs[signs.size() - 2u] >> (n & 31u)) == 0, "unused output sign bits not zero");
}

void malformed(const std::vector<std::uint8_t>& archive, std::uint32_t n) {
    std::vector<std::uint16_t> mag(std::size_t(n) + 2u, 0xbeef);
    std::vector<std::uint32_t> sign((std::size_t(n) + 31u) / 32u + 2u, 0x12345678);
    for (bool avx2 : {false, true}) {
        bool threw = false;
        try {
            wzp_decode(archive.data(), archive.size(), n, mag.data() + 1u, sign.data() + 1u, avx2);
        } catch (const std::runtime_error&) {
            threw = true;
        }
        require(threw, "malformed archive was silently accepted");
    }
    require(mag.front() == 0xbeef && mag.back() == 0xbeef, "malformed magnitude guard overwritten");
    require(sign.front() == 0x12345678 && sign.back() == 0x12345678,
            "malformed sign guard overwritten");
}

std::vector<std::uint8_t> reverse_groups(const std::vector<std::uint8_t>& archive, std::uint32_t n) {
    const std::uint32_t chunks = (n + 8191u) / 8192u;
    const std::uint32_t groups = (chunks + 15u) / 16u;
    const std::size_t header = groups * 4u + chunks * 3u;
    std::vector<std::uint8_t> output = archive;
    std::size_t cursor = 0;
    for (std::uint32_t g = groups; g != 0; --g) {
        const std::uint32_t group = g - 1u;
        std::uint32_t bytes = 0;
        for (std::uint32_t chunk = group * 16u; chunk != std::min(chunks, group * 16u + 16u); ++chunk) {
            const std::uint32_t m = get24(archive.data() + groups * 4u + chunk * 3u);
            const std::uint32_t code = m & 0x3fffu;
            if (code == 0)
                continue;
            bytes += code == 0x3fffu ? std::min(8192u, n - chunk * 8192u) * 2u : code;
            bytes += (m >> 14u) == 0 ? 1024u : m >> 14u;
        }
        put32(output.data() + group * 4u, bytes == 0 ? 0u : static_cast<std::uint32_t>(cursor));
        if (bytes != 0)
            std::memcpy(output.data() + header + cursor,
                        archive.data() + header + get32(archive.data() + group * 4u), bytes);
        cursor += bytes;
    }
    return output;
}

void round_trips() {
    const std::uint32_t sizes[] = {0, 1, 2, 7, 8, 15, 16, 17, 31, 32, 33, 63, 127,
        255, 256, 511, 8191, 8192, 8193, 16384, 15u * 8192u + 7u,
        16u * 8192u, 16u * 8192u + 1u, 32u * 8192u + 37u};
    std::uint32_t state = 0x728ca19bu;
    unsigned cases = 0;
    for (const std::uint32_t n : sizes) {
        std::vector<std::uint16_t> mag(n);
        std::vector<std::uint32_t> signs((std::size_t(n) + 31u) / 32u);
        for (unsigned pattern = 0; pattern != 8; ++pattern) {
            for (auto& word : signs)
                word = random_word(state);
            for (std::uint32_t i = 0; i != n; ++i) {
                const std::uint32_t r = random_word(state);
                switch (pattern) {
                case 0: mag[i] = 0; break;
                case 1: mag[i] = 1; break;
                case 2: mag[i] = static_cast<std::uint16_t>(r); break;
                case 3: mag[i] = r % 97u == 0 ? static_cast<std::uint16_t>(r >> 16u) : 0; break;
                case 4: mag[i] = static_cast<std::uint16_t>(r & 7u); break;
                case 5: mag[i] = (i / 8192u) % 3u == 0 ? 0 : static_cast<std::uint16_t>(r); break;
                case 6: mag[i] = static_cast<std::uint16_t>((i % 16u == 0) ? 0x8000u : 0); break;
                default: mag[i] = (i / (16u * 8192u)) % 2u == 0 ? 0 : 0xffffu; break;
                }
            }
            const auto archive = encode(mag, signs);
            require(archive == encode(mag, signs, false), "AVX2/scalar encoded archive bytes differ");
            require(archive == encode(mag, signs), "encode is nondeterministic");
            require(archive.size() <= wzp_max_bytes(n), "maximum size bound is wrong");
            check_decode(archive, mag, signs);
            check_decode(archive, mag, signs, false);
#if defined(WALTZ_TEST_LC_REFERENCE)
            require(archive == reference_encode(mag, signs), "encoded bytes differ from scalar LC reference");
#endif
            if (n > 16u * 8192u)
                check_decode(reverse_groups(archive, n), mag, signs);
            if (n != 0) {
                std::vector<std::uint8_t> exact(archive.size());
                require(wzp_encode(mag.data(), signs.data(), n, exact.data(), exact.size()) == exact.size(),
                        "exact-capacity output rejected");
                require(exact == archive, "exact-capacity output differs");
            }
            ++cases;
        }
    }
    std::printf("WZP: %u scalar/AVX2 byte-identical pattern/size round trips passed", cases);
#if defined(WALTZ_TEST_LC_REFERENCE)
    std::printf(" (exact scalar LC byte comparison enabled)");
#endif
    std::printf("\n");
}

void malformed_tests() {
    std::vector<std::uint16_t> mag(8192, 1);
    std::vector<std::uint32_t> sign(256, 0x55555555);
    const auto valid = encode(mag, sign);
    require((get24(valid.data() + 4) & 0x3fffu) != 0x3fffu, "ones did not select compressed path");
    require((get24(valid.data() + 4) >> 14u) == 0, "full sign slot did not use code zero");
    for (std::size_t cut = 0; cut != valid.size(); ++cut)
        malformed(std::vector<std::uint8_t>(valid.begin(), valid.begin() + cut), 8192);
    auto bad = valid;
    bad.push_back(0);
    malformed(bad, 8192);
    bad = valid;
    put32(bad.data(), 1);
    malformed(bad, 8192);
    bad = valid;
    put24(bad.data() + 4, 1);
    malformed(bad, 8192);
    bad = valid;
    put24(bad.data() + 4, 1u << 14u);
    malformed(bad, 8192);
    bad = valid;
    const std::uint32_t stored = get24(bad.data() + 4) & 0x3fffu;
    std::memset(bad.data() + 7u + stored - 4u, 0xff, 4);
    malformed(bad, 8192);
    bad = valid;
    bad[7] = 0;
    malformed(bad, 8192);
    mag.assign(1, 1);
    sign.assign(1, 1);
    const auto small = encode(mag, sign);
    require((get24(small.data() + 4) & 0x3fffu) == 0x3fffu, "tiny chunk did not use raw fallback");
    bad = small;
    bad.back() = 0x80;
    malformed(bad, 1);
    bad = small;
    bad[7] = bad[8] = 0;
    malformed(bad, 1);
    bad = small;
    put24(bad.data() + 4, 0x3fffu);
    malformed(bad, 1);
    std::vector<std::uint8_t> short_output(small.size() + 1u, 0x5a);
    bool capacity_threw = false;
    try {
        (void)wzp_encode(mag.data(), sign.data(), 1, short_output.data() + 1u, small.size() - 1u);
    } catch (const std::runtime_error&) {
        capacity_threw = true;
    }
    require(capacity_threw, "undersized output capacity accepted");
    require(short_output.front() == 0x5a && short_output.back() == 0x5a,
            "undersized encode overwrote a guard byte");
    mag.assign(33u * 8192u, 7);
    sign.assign((mag.size() + 31u) / 32u, 0);
    bad = encode(mag, sign);
    put32(bad.data() + 4, 0);
    malformed(bad, static_cast<std::uint32_t>(mag.size()));
    bad = encode(mag, sign);
    put32(bad.data() + 4, get32(bad.data() + 4) + 1u);
    malformed(bad, static_cast<std::uint32_t>(mag.size()));
    bool threw = false;
    try { (void)wzp_max_bytes(std::numeric_limits<std::uint32_t>::max()); }
    catch (const std::runtime_error&) { threw = true; }
    require(threw, "GPU-incompatible element count accepted");
    require(wzp_encode(nullptr, nullptr, 0, nullptr, 0) == 0, "empty encode failed");
    wzp_decode(nullptr, 0, 0, nullptr, nullptr);
    malformed(std::vector<std::uint8_t>(1, 0), 0);
    std::printf("WZP: malformed archive and padding tests passed\n");
}

void mutation_tests() {
    const std::uint32_t n = 8191;
    std::uint32_t state = 0xe126bd93u;
    std::vector<std::uint16_t> mag(n);
    std::vector<std::uint32_t> signs((n + 31u) / 32u, 0x55aa55aa);
    for (std::uint32_t i = 0; i != n; ++i)
        mag[i] = i % 71u == 0 ? static_cast<std::uint16_t>(i + 1u) : 0;
    const auto original = encode(mag, signs);
    std::vector<std::uint16_t> output(n + 2u, 0xdead);
    std::vector<std::uint32_t> output_sign(signs.size() + 2u, 0xabcdef01);
    std::vector<std::uint16_t> scalar_output(n + 2u, 0xdead);
    std::vector<std::uint32_t> scalar_sign(signs.size() + 2u, 0xabcdef01);
    for (unsigned iteration = 0; iteration != 4000; ++iteration) {
        auto archive = original;
        for (unsigned changes = 1u + random_word(state) % 4u; changes != 0; --changes) {
            // Include metadata, directory and sparse hierarchy corruption.
            const std::size_t position = random_word(state) % archive.size();
            archive[position] ^= static_cast<std::uint8_t>(1u << (random_word(state) % 8u));
        }
        bool accepted[2] = {};
        for (unsigned mode = 0; mode != 2; ++mode) {
            try {
                wzp_decode(archive.data(), archive.size(), n,
                           mode ? output.data() + 1u : scalar_output.data() + 1u,
                           mode ? output_sign.data() + 1u : scalar_sign.data() + 1u, mode != 0);
                accepted[mode] = true;
            } catch (const std::runtime_error&) {
            }
        }
        require(accepted[0] == accepted[1], "scalar/AVX2 mutation acceptance differs");
        if (accepted[0])
            require(output == scalar_output && output_sign == scalar_sign,
                    "scalar/AVX2 mutated archive output differs");
        // A mutation can be a valid alternative archive: WZP has no checksum.
        // Whether accepted or rejected, it must not escape either output span.
        require(output.front() == 0xdead && output.back() == 0xdead,
                "mutated archive overwrote magnitude guard");
        require(output_sign.front() == 0xabcdef01 && output_sign.back() == 0xabcdef01,
                "mutated archive overwrote sign guard");
        require(scalar_output.front() == 0xdead && scalar_output.back() == 0xdead &&
                scalar_sign.front() == 0xabcdef01 && scalar_sign.back() == 0xabcdef01,
                "mutated scalar archive overwrote guard");
    }
    std::printf("WZP: 4000 malformed-stream mutation safety checks passed\n");
}

void benchmark() {
    constexpr std::uint32_t n = 1u << 20u;
    constexpr unsigned repeats = 4;
    std::vector<std::uint16_t> mag(n), output(n);
    std::vector<std::uint32_t> sign(n / 32u), output_sign(n / 32u);
    std::vector<std::uint8_t> scalar(wzp_max_bytes(n)), simd(scalar.size());
    const char* names[] = {"zero", "ones", "random16", "low3", "sparse1pct"};
    std::uint32_t state = 0x294ae5fbu;
    std::printf("WZP benchmark: %u magnitudes, raw-byte GB/s, best of 3 x %u calls\n", n, repeats);
    for (unsigned pattern = 0; pattern != 5; ++pattern) {
        for (auto& value : sign)
            value = random_word(state);
        for (std::uint32_t i = 0; i != n; ++i) {
            const std::uint32_t r = random_word(state);
            mag[i] = pattern == 0 ? 0 : pattern == 1 ? 1 : pattern == 2 ? std::uint16_t(r) :
                     pattern == 3 ? std::uint16_t(r & 7u) : r % 100u == 0 ? std::uint16_t(r) : 0;
        }
        const std::size_t scalar_bytes = wzp_encode(mag.data(), sign.data(), n, scalar.data(), scalar.size(), false);
        const std::size_t simd_bytes = wzp_encode(mag.data(), sign.data(), n, simd.data(), simd.size(), true);
        require(scalar_bytes == simd_bytes && std::memcmp(scalar.data(), simd.data(), scalar_bytes) == 0,
                "benchmark scalar/AVX2 archive differs");
        double best[2][2] = {{1e30, 1e30}, {1e30, 1e30}};
        for (unsigned run = 0; run != 3; ++run) {
            for (unsigned mode = 0; mode != 2; ++mode) {
                auto& archive = mode ? simd : scalar;
                const std::size_t bytes = mode ? simd_bytes : scalar_bytes;
                const auto start = std::chrono::steady_clock::now();
                for (unsigned repeat = 0; repeat != repeats; ++repeat)
                    (void)wzp_encode(mag.data(), sign.data(), n, archive.data(), archive.size(), mode != 0);
                const auto encoded = std::chrono::steady_clock::now();
                for (unsigned repeat = 0; repeat != repeats; ++repeat)
                    wzp_decode(archive.data(), bytes, n, output.data(), output_sign.data(), mode != 0);
                const auto decoded = std::chrono::steady_clock::now();
                best[mode][0] = std::min(best[mode][0], std::chrono::duration<double>(encoded - start).count());
                best[mode][1] = std::min(best[mode][1], std::chrono::duration<double>(decoded - encoded).count());
            }
        }
        const double raw_gigabytes = double(n) * 2.0 * repeats / 1e9;
        std::printf("  %-10s bytes=%zu encode scalar=%.3f AVX2=%.3f (%.2fx) decode scalar=%.3f AVX2=%.3f (%.2fx)\n",
                    names[pattern], scalar_bytes, raw_gigabytes / best[0][0], raw_gigabytes / best[1][0],
                    best[0][0] / best[1][0], raw_gigabytes / best[0][1], raw_gigabytes / best[1][1],
                    best[0][1] / best[1][1]);
    }
}

} // namespace

int main(int argc, char** argv) {
    try {
        if (argc == 2 && std::strcmp(argv[1], "--benchmark") == 0) {
            benchmark();
            return EXIT_SUCCESS;
        }
        round_trips();
        malformed_tests();
        mutation_tests();
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "WZP test failure: %s\n", error.what());
        return EXIT_FAILURE;
    }
}
