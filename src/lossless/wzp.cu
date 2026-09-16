#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cuda/atomic>
#include <stdexcept>
#include <string>
#include <type_traits>

using byte = unsigned char;
static const int CS = 1024 * 16;
static const int TPB = 512;
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
#include "components/d_RZE_1.h"
#include "lossless/wzp.hpp"
#include "utils/Config.hpp"

namespace WALTZ::lossless {
namespace {

constexpr uint32_t ELEMS_PER_CHUNK = CS / sizeof(uint16_t);
constexpr uint32_t GROUP_CHUNKS = 16;
constexpr uint32_t SIGN_SLOT_BYTES = ELEMS_PER_CHUNK / 8;
constexpr uint32_t ITEMS_PER_THREAD = (ELEMS_PER_CHUNK + TPB - 1u) / TPB;
// BIT_2 never touches its temp argument.  Byte-RZE needs 2340 bitmap bytes
// after a 132-byte scan prefix, while the signed path needs only 1536 bytes.
// The old kernels nevertheless reserved a third full 16-KiB chunk.  Keeping a
// compact 3-KiB scratch leaves the byte stream unchanged and permits four
// resident CTAs per SM once register pressure is capped below.
constexpr uint32_t TEMP_BYTES = 3u * 1024u;
constexpr uint32_t STORAGE_BYTES = 2u * CS + TEMP_BYTES;
// Decode-only compatibility with archives produced by the retired prefix encoder.
constexpr uint32_t PREFIX_FORMAT_BIT = 0x80000000u;
constexpr uint32_t PREFIX_CHUNK_BIT = 0x2000u;
constexpr uint32_t PREFIX_SIZE_MASK = 0x1fffu;

void check(cudaError_t error, const char* what) {
    if (error != cudaSuccess)
        throw std::runtime_error(std::string("WZP ") + what + ": " + cudaGetErrorString(error));
}

__device__ unsigned int payload_counter;
__device__ unsigned int nnz_counter;
__device__ unsigned long long chunk_counter;

__global__ void reset_counter_kernel() {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        payload_counter = 0;
        nnz_counter = 0;
        chunk_counter = 0;
    }
}

__device__ __forceinline__ uint32_t load_meta24(const byte* meta, uint32_t chunk) {
    meta += static_cast<size_t>(chunk) * 3u;
    return static_cast<uint32_t>(meta[0]) | (static_cast<uint32_t>(meta[1]) << 8u) |
           (static_cast<uint32_t>(meta[2]) << 16u);
}

__device__ __forceinline__ uint32_t decode_mag_bytes(uint32_t meta,
                                                     uint32_t raw_bytes,
                                                     bool prefix_format = false) {
    const uint32_t code = meta & 0x3fffu;
    if (code == 0x3fffu)
        return raw_bytes;
    return prefix_format && (code & PREFIX_CHUNK_BIT) ? (code & PREFIX_SIZE_MASK) : code;
}

__device__ __forceinline__ bool mag_is_prefixed(uint32_t meta, bool prefix_format) {
    const uint32_t code = meta & 0x3fffu;
    return prefix_format && code != 0x3fffu && (code & PREFIX_CHUNK_BIT) != 0u;
}

__device__ __forceinline__ uint32_t decode_sign_bytes(uint32_t meta) {
    const uint32_t mag_code = meta & 0x3fffu;
    const uint32_t sign_code = meta >> 14u;
    return mag_code == 0 ? 0u : (sign_code == 0 ? SIGN_SLOT_BYTES : sign_code);
}

__device__ __forceinline__ uint32_t zorder_unrank_local(uint32_t rank,
                                                        uint32_t dimx,
                                                        uint32_t dimy,
                                                        uint32_t dimz) {
    const uint32_t ox = dimx, oy = dimy;
    uint32_t x = 0, y = 0, z = 0;
    while (dimx > 1U || dimy > 1U || dimz > 1U) {
        const uint32_t sx0 = dimx - dimx / 2U;
        const uint32_t sy0 = dimy - dimy / 2U;
        const uint32_t sz0 = dimz - dimz / 2U;
        const uint32_t z0_volume = sz0 * dimx * dimy;
        const bool cz = rank >= z0_volume;
        if (cz) {
            rank -= z0_volume;
            z += sz0;
        }
        const uint32_t sz = cz ? dimz / 2U : sz0;
        const uint32_t y0_volume = sz * sy0 * dimx;
        const bool cy = rank >= y0_volume;
        if (cy) {
            rank -= y0_volume;
            y += sy0;
        }
        const uint32_t sy = cy ? dimy / 2U : sy0;
        const uint32_t x0_volume = sz * sy * sx0;
        const bool cx = rank >= x0_volume;
        if (cx) {
            rank -= x0_volume;
            x += sx0;
        }
        dimx = cx ? dimx / 2U : sx0;
        dimy = sy;
        dimz = sz;
    }
    return x + y * ox + z * ox * oy;
}

// In a 16x16x2 brick the recursive octants put the four interleaved XY bits in
// rank bits [0,7] and the single Z bit in bit 8.
__device__ __forceinline__ uint32_t rank_16x16x2(uint32_t natural) {
    const uint32_t x = natural & 15U;
    const uint32_t y = (natural >> 4U) & 15U;
    uint32_t rank = 0;
#pragma unroll
    for (uint32_t bit = 0; bit < 4U; ++bit) {
        rank |= ((x >> bit) & 1U) << (2U * bit);
        rank |= ((y >> bit) & 1U) << (2U * bit + 1U);
    }
    return rank | ((natural >> 8U) << 8U);
}

// Row-major local index -> recursive Morton rank for a complete 16x16xBz
// brick, where Bz is a power of two no greater than 16.  Z participates only
// in the coarsest log2(Bz) octree levels, matching zorder_rank() exactly.
__device__ __forceinline__ uint32_t
rank_16x16_power2_z(uint32_t natural, uint32_t block_z) {
    const uint32_t x = natural & 15U;
    const uint32_t y = (natural >> 4U) & 15U;
    const uint32_t z = natural >> 8U;
    const uint32_t z_bits = 31U - __clz(block_z);
    const uint32_t z_begin = 4U - z_bits;
    uint32_t rank = 0, position = 0;
#pragma unroll
    for (uint32_t bit = 0; bit < 4U; ++bit) {
        rank |= ((x >> bit) & 1U) << position++;
        rank |= ((y >> bit) & 1U) << position++;
        if (bit >= z_begin)
            rank |= ((z >> (bit - z_begin)) & 1U) << position++;
    }
    return rank;
}

struct ReorderRun {
    size_t origin;
    uint32_t rank;
    uint32_t bdx, bdy, bdz;
    bool full;
};

__device__ __forceinline__ ReorderRun
reorder_run(uint32_t zpos, uint32_t nx, uint32_t ny, uint32_t nz, uint32_t block_z) {
    constexpr uint32_t BX = 16U, BY = 16U;
    const size_t nxy = static_cast<size_t>(nx) * ny;
    const uint32_t z0 = static_cast<uint32_t>(zpos / (nxy * block_z)) * block_z;
    const uint32_t bdz = min(block_z, nz - z0);
    uint32_t rem = static_cast<uint32_t>(zpos - static_cast<size_t>(z0) * nxy);
    const uint32_t max_by = (ny - 1U) / BY;
    const uint32_t y0 = min(rem / (nx * bdz * BY), max_by) * BY;
    const uint32_t bdy = min(BY, ny - y0);
    rem -= nx * bdz * y0;
    const uint32_t max_bx = (nx - 1U) / BX;
    const uint32_t x0 = min(rem / (bdy * bdz * BX), max_bx) * BX;
    const uint32_t bdx = min(BX, nx - x0);
    const uint32_t rank = rem - bdy * bdz * x0;
    const size_t origin = static_cast<size_t>(x0) + static_cast<size_t>(y0) * nx +
                          static_cast<size_t>(z0) * nxy;
    return {origin, rank, bdx, bdy, bdz, bdx == BX && bdy == BY && bdz == block_z};
}

__device__ __forceinline__ size_t reordered_natural_index(
    uint32_t zpos,
    uint32_t nx,
    uint32_t ny,
    uint32_t nz,
    uint32_t block_z,
    const uint32_t* __restrict__ offset_table) {
    const ReorderRun run = reorder_run(zpos, nx, ny, nz, block_z);
    if (run.full)
        return run.origin + offset_table[run.rank];
    const uint32_t local = zorder_unrank_local(run.rank, run.bdx, run.bdy, run.bdz);
    const uint32_t lx = local % run.bdx;
    const uint32_t ly = (local / run.bdx) % run.bdy;
    const uint32_t lz = local / (run.bdx * run.bdy);
    return run.origin + lx + static_cast<size_t>(ly) * nx +
           static_cast<size_t>(lz) * nx * ny;
}

__device__ __forceinline__ size_t
full_brick_origin(uint32_t bi, uint32_t nx, uint32_t ny, uint32_t block_z) {
    const uint32_t nbx = nx / 16U;
    const uint32_t nby = ny / 16U;
    const uint32_t bx = bi % nbx;
    const uint32_t byz = bi / nbx;
    const uint32_t by = byz % nby;
    const uint32_t bz = byz / nby;
    return static_cast<size_t>(bx * 16U) + static_cast<size_t>(by * 16U) * nx +
           static_cast<size_t>(bz * block_z) * nx * ny;
}

// Dynamic scheduling is essential because RZE work varies substantially across
// a field. Compact the negative bits while raw magnitudes are already
// in shared memory, so there is no second magnitude read or global sign scan.
__global__
__launch_bounds__(TPB, 2048 / TPB)
void encode_kernel(const byte* __restrict__ input,
    uint32_t input_bytes, const uint32_t* __restrict__ sign_bitmap, byte* __restrict__ slots,
    byte* __restrict__ sign_slots, byte* __restrict__ meta,
    byte* __restrict__ blob, unsigned int* __restrict__ group_done) {
    const uint32_t chunks = input_bytes / CS + (input_bytes % CS != 0U);
    __shared__ __align__(16) long long storage[STORAGE_BYTES / sizeof(long long)];
    constexpr int counter_slot = STORAGE_BYTES / sizeof(long long) - 2 - WS;
    byte* const raw = reinterpret_cast<byte*>(storage);
    byte* const transformed = raw + CS;
    byte* const temp = transformed + CS;

    __shared__ uint32_t block_nnz;
    __shared__ bool pack_ready;
    if (threadIdx.x == 0) block_nnz = 0U;
    while (true) {
        if (threadIdx.x == 0)
            storage[counter_slot] = static_cast<long long>(atomicAdd(&chunk_counter, 1ull));
        __syncthreads();
        const uint32_t chunk = static_cast<uint32_t>(storage[counter_slot]);
        if (chunk >= chunks) {
            if (threadIdx.x == 0 && block_nnz != 0U)
                atomicAdd(&nnz_counter, block_nnz);
            return;
        }

        const uint32_t base = chunk * CS;
        const uint32_t raw_bytes = min(static_cast<uint32_t>(CS), input_bytes - base);
        uint32_t any = 0;
        if ((reinterpret_cast<uintptr_t>(input) & 15U) == 0U) {
            const auto* source128 = reinterpret_cast<const uint4*>(input + base);
            auto* shared128 = reinterpret_cast<uint4*>(raw);
            for (uint32_t i = threadIdx.x; i < raw_bytes / 16u; i += blockDim.x) {
                const uint4 value = source128[i];
                shared128[i] = value;
                any |= value.x | value.y | value.z | value.w;
            }
            const uint32_t extra = raw_bytes & 15u;
            if (threadIdx.x < extra) {
                const byte value = input[base + raw_bytes - extra + threadIdx.x];
                raw[raw_bytes - extra + threadIdx.x] = value;
                any |= value;
            }
        } else {
            for (uint32_t i = threadIdx.x; i < raw_bytes; i += blockDim.x) {
                const byte value = input[base + i];
                raw[i] = value;
                any |= value;
            }
        }
        const bool zero = __syncthreads_or(any != 0) == 0;

        uint32_t chunk_nnz = 0;
        uint32_t local_nnz = 0;
        uint32_t local_negative = 0;
        long long sign_exclusive = 0;
        byte* const chunk_sign_slot = sign_slots + static_cast<size_t>(chunk) * SIGN_SLOT_BYTES;

        if (!zero) {
            auto* sign_stage = reinterpret_cast<uint32_t*>(temp + 512);
            for (uint32_t i = threadIdx.x; i < SIGN_SLOT_BYTES / sizeof(uint32_t);
                 i += blockDim.x)
                sign_stage[i] = 0;
            __syncthreads();

            const auto* mag = reinterpret_cast<const uint16_t*>(raw);
            const uint32_t elements = raw_bytes / sizeof(uint16_t);
            const uint32_t start = threadIdx.x * ITEMS_PER_THREAD;
            #pragma unroll
            for (uint32_t j = 0; j < ITEMS_PER_THREAD; ++j) {
                const uint32_t i = start + j;
                if (i < elements && mag[i] != 0) {
                    const uint32_t global = base / 2u + i;
                    if ((sign_bitmap[global >> 5u] >> (global & 31u)) & 1u)
                        local_negative |= 1u << local_nnz;
                    ++local_nnz;
                }
            }
            const uint32_t inclusive = block_prefix_sum<uint32_t>(local_nnz, temp);
            sign_exclusive = inclusive - local_nnz;
            // The scan's final barrier publishes the last warp's prefix.
            chunk_nnz = reinterpret_cast<const uint32_t*>(temp)[TPB / WS - 1];

            if (local_negative != 0) {
                const uint32_t word = static_cast<uint32_t>(sign_exclusive) >> 5u;
                const uint32_t shift = static_cast<uint32_t>(sign_exclusive) & 31u;
                atomicOr(sign_stage + word, local_negative << shift);
                if (shift + local_nnz > 32u)
                    atomicOr(sign_stage + word + 1u, local_negative >> (32u - shift));
            }
            __syncthreads();
            const uint32_t sign_bytes = (chunk_nnz + 7u) / 8u;
            const byte* sign_stage_bytes = reinterpret_cast<const byte*>(sign_stage);
            for (uint32_t i = threadIdx.x; i < sign_bytes; i += blockDim.x)
                chunk_sign_slot[i] = sign_stage_bytes[i];
            if (threadIdx.x == 0)
                block_nnz += chunk_nnz;
            __syncthreads();
        }

        uint32_t stored_bytes = 0;
        const byte* encoded = raw;
        bool stored_raw = false;
        if (!zero) {
            int compressed_bytes = static_cast<int>(raw_bytes);
            d_BIT_2(compressed_bytes, raw, transformed, temp);
            __syncthreads();
            const bool good = d_RZE_1(compressed_bytes, transformed, raw, temp);
            __syncthreads();
            if (!good || compressed_bytes >= static_cast<int>(raw_bytes)) {
                stored_bytes = raw_bytes;
                encoded = input + base;
                stored_raw = true;
            } else {
                stored_bytes = static_cast<uint32_t>(compressed_bytes - 2);
            }
        }


        byte* const destination = slots + static_cast<size_t>(chunk) * CS;
        for (uint32_t i = threadIdx.x; i < stored_bytes; i += blockDim.x)
            destination[i] = encoded[i];
        if (threadIdx.x == 0) {
            const uint32_t sign_bytes = (chunk_nnz + 7u) / 8u;
            const uint32_t mag_code = stored_raw ? 0x3fffu : stored_bytes;
            const uint32_t sign_code = sign_bytes == SIGN_SLOT_BYTES ? 0u : sign_bytes;
            const uint32_t value = mag_code | (sign_code << 14u);
            byte* m = meta + static_cast<size_t>(chunk) * 3u;
            m[0] = static_cast<byte>(value);
            m[1] = static_cast<byte>(value >> 8u);
            m[2] = static_cast<byte>(value >> 16u);
        }
        if (!zero)
            __syncthreads();
        // Publish the slot/metadata writes. Only the last block in a group
        // acquires the other chunks and packs that group; no block spins.
        if (threadIdx.x == 0) {
            const uint32_t group = chunk / GROUP_CHUNKS;
            const uint32_t count = min(GROUP_CHUNKS, chunks - group * GROUP_CHUNKS);
            cuda::atomic_ref<unsigned int, cuda::thread_scope_device> completed(group_done[group]);
            pack_ready = completed.fetch_add(1U, cuda::memory_order_release) + 1U == count;
            if (pack_ready)
                cuda::atomic_thread_fence(cuda::memory_order_acquire, cuda::thread_scope_device);
        }
        __syncthreads();
        if (pack_ready) {
            __shared__ uint32_t group_prefix[GROUP_CHUNKS + 1];
            __shared__ uint32_t group_offset;
            auto* const group_offsets = reinterpret_cast<uint32_t*>(blob);
            byte* const metadata = meta;
            const size_t metadata_bytes = static_cast<size_t>(chunks) * 3u;
            byte* const payload = metadata + metadata_bytes;
            const uint32_t group = chunk / GROUP_CHUNKS;
            const uint32_t first = group * GROUP_CHUNKS;
            const uint32_t group_count = min(GROUP_CHUNKS, chunks - first);

            // One warp scans the group's lengths to obtain chunk offsets.
            if (threadIdx.x < 32) {
                const uint32_t lane = threadIdx.x;
                uint32_t bytes = 0;
                if (lane < group_count) {
                    const uint32_t m = load_meta24(metadata, first + lane);
                    bytes = decode_mag_bytes(m, min(static_cast<uint32_t>(CS),
                        input_bytes - (first + lane) * CS)) + decode_sign_bytes(m);
                }
                uint32_t inclusive = bytes;
#pragma unroll
                for (uint32_t delta = 1; delta < GROUP_CHUNKS; delta <<= 1u) {
                    const uint32_t prior = __shfl_up_sync(0xffffffffu, inclusive, delta);
                    if (lane >= delta)
                        inclusive += prior;
                }
                if (lane == 0)
                    group_prefix[0] = 0;
                if (lane < group_count)
                    group_prefix[lane + 1u] = inclusive;
                if (lane + 1u == group_count) {
                    group_offset = inclusive ? atomicAdd(&payload_counter, inclusive) : 0U;
                    group_offsets[group] = group_offset;
                }
            }
            __syncthreads();

            // Each warp copies one chunk's magnitudes and signs.
            const uint32_t local = threadIdx.x >> 5u;
            const uint32_t lane = threadIdx.x & 31u;
            if (group_prefix[group_count] != 0U && local < group_count) {
                const uint32_t chunk = first + local;
                const uint32_t m = load_meta24(metadata, chunk);
                const uint32_t mag_bytes = decode_mag_bytes(m, min(static_cast<uint32_t>(CS),
                    input_bytes - chunk * CS));
                const uint32_t sign_bytes = decode_sign_bytes(m);
                byte* destination = payload + group_offset + group_prefix[local];
                const byte* mag_source = slots + static_cast<size_t>(chunk) * CS;
                for (uint32_t i = lane; i < mag_bytes; i += 32u)
                    destination[i] = mag_source[i];
                const byte* sign_source = sign_slots + static_cast<size_t>(chunk) * SIGN_SLOT_BYTES;
                for (uint32_t i = lane; i < sign_bytes; i += 32u)
                    destination[mag_bytes + i] = sign_source[i];
            }
            __syncthreads();
            // All publishers for this group have finished. Reuse the counter
            // on the next encode without another per-call memset/launch.
            if (threadIdx.x == 0) {
                cuda::atomic_ref<unsigned int, cuda::thread_scope_device> completed(group_done[group]);
                completed.store(0U, cuda::memory_order_relaxed);
            }
        }
    }
}

template <bool WithSign, int DirectLayout = 0, typename DirectT = float>
__global__ __launch_bounds__(TPB, 2048 / TPB) void decode_kernel(
    const byte* __restrict__ blob,
    uint32_t chunks,
    uint32_t groups,
    uint32_t output_bytes,
    byte* __restrict__ output,
    uint32_t* __restrict__ sign_bitmap,
    DirectT* __restrict__ direct_coeff = nullptr,
    const uint32_t* __restrict__ direct_offset = nullptr,
    DirectT direct_q = static_cast<DirectT>(0),
    uint32_t direct_nx = 0,
    uint32_t direct_ny = 0,
    uint32_t direct_nz = 0,
    uint32_t direct_block_z = 0) {
    __shared__ long long storage[STORAGE_BYTES / sizeof(long long)];
    byte* const encoded = reinterpret_cast<byte*>(storage);
    byte* const expanded = encoded + CS;
    byte* const temp = expanded + CS;
    const uint32_t chunk = blockIdx.x;
    if (chunk >= chunks)
        return;

    const auto* group_offsets = reinterpret_cast<const uint32_t*>(blob);
    const byte* metadata = reinterpret_cast<const byte*>(group_offsets + groups);
    const size_t metadata_bytes = static_cast<size_t>(chunks) * (WithSign ? 3u : sizeof(uint16_t));
    const byte* payload = metadata + metadata_bytes;
    const uint32_t group = chunk / GROUP_CHUNKS;
    const uint32_t local = chunk % GROUP_CHUNKS;
    const uint32_t base = chunk * CS;
    const uint32_t raw_bytes = min(static_cast<uint32_t>(CS), output_bytes - base);

    const bool prefix_format = (group_offsets[group] & PREFIX_FORMAT_BIT) != 0u;
    uint32_t stored_bytes = 0, sign_bytes = 0;
    bool prefixed = false;
    if constexpr (WithSign) {
        const uint32_t m = load_meta24(metadata, chunk);
        stored_bytes = decode_mag_bytes(m, raw_bytes, prefix_format);
        prefixed = mag_is_prefixed(m, prefix_format);
        sign_bytes = decode_sign_bytes(m);
    } else {
        stored_bytes = reinterpret_cast<const uint16_t*>(metadata)[chunk];
    }

    auto* const shared_begin = reinterpret_cast<uint32_t*>(storage);
    // Decode blocks used to make lane 0 serially walk every preceding chunk in
    // the group.  A warp-wide metadata scan computes the same local byte
    // offset in four shuffle steps.
    if (threadIdx.x < 32) {
        const uint32_t lane = threadIdx.x;
        const uint32_t first = group * GROUP_CHUNKS;
        const uint32_t group_count = min(GROUP_CHUNKS, chunks - first);
        uint32_t bytes = 0;
        if (lane < group_count) {
            if constexpr (WithSign) {
                const uint32_t prior_raw =
                    min(static_cast<uint32_t>(CS), output_bytes - (first + lane) * CS);
                const uint32_t m = load_meta24(metadata, first + lane);
                bytes = decode_mag_bytes(m, prior_raw, prefix_format) + decode_sign_bytes(m);
            } else {
                bytes = reinterpret_cast<const uint16_t*>(metadata)[first + lane];
            }
        }
        uint32_t inclusive = bytes;
#pragma unroll
        for (uint32_t delta = 1; delta < GROUP_CHUNKS; delta <<= 1u) {
            const uint32_t prior = __shfl_up_sync(0xffffffffu, inclusive, delta);
            if (lane >= delta)
                inclusive += prior;
        }
        if (lane == local)
            *shared_begin = (group_offsets[group] & ~PREFIX_FORMAT_BIT) + inclusive - bytes;
    }
    __syncthreads();
    const uint32_t chunk_begin = *shared_begin;
    __syncthreads();

    if (stored_bytes == 0) {
        if constexpr (DirectLayout != 0) {
            const uint32_t elements = raw_bytes / sizeof(uint16_t);
            const uint32_t elem_base = base / sizeof(uint16_t);
            if constexpr (DirectLayout == 2) {
                constexpr uint32_t BN = 16U * 16U * 2U;
                const uint32_t natural = threadIdx.x;
                const uint32_t x = natural & 15U;
                const uint32_t y = (natural >> 4U) & 15U;
                const uint32_t z = natural >> 8U;
                const size_t local = x + static_cast<size_t>(y) * direct_nx +
                                     static_cast<size_t>(z) * direct_nx * direct_ny;
                const uint32_t nbx = direct_nx >> 4U;
                const uint32_t nby = direct_ny >> 4U;
                const size_t nxy = static_cast<size_t>(direct_nx) * direct_ny;
                const uint32_t first_bi = elem_base / BN;
                uint32_t bx = first_bi % nbx;
                const uint32_t byz = first_bi / nbx;
                uint32_t by = byz % nby;
                size_t brick_origin = static_cast<size_t>(bx * 16U) +
                                      static_cast<size_t>(by * 16U) * direct_nx +
                                      static_cast<size_t>(byz / nby * 2U) * nxy;
                for (uint32_t i = threadIdx.x; i < elements; i += blockDim.x) {
                    direct_coeff[brick_origin + local] = static_cast<DirectT>(0);
                    if (++bx < nbx) {
                        brick_origin += 16U;
                    } else {
                        bx = 0U;
                        if (++by < nby) {
                            brick_origin += static_cast<size_t>(16U) * direct_nx -
                                            static_cast<size_t>(nbx - 1U) * 16U;
                        } else {
                            by = 0U;
                            brick_origin += 2U * nxy -
                                            static_cast<size_t>(nby - 1U) * 16U * direct_nx -
                                            static_cast<size_t>(nbx - 1U) * 16U;
                        }
                    }
                }
            } else if constexpr (DirectLayout == 3) {
                const uint32_t block_n = 16U * 16U * direct_block_z;
                const size_t nxy = static_cast<size_t>(direct_nx) * direct_ny;
                for (uint32_t i = threadIdx.x; i < elements; i += blockDim.x) {
                    const uint32_t pos = elem_base + i;
                    const uint32_t bi = pos / block_n;
                    const uint32_t natural = pos - bi * block_n;
                    const uint32_t x = natural & 15U;
                    const uint32_t y = (natural >> 4U) & 15U;
                    const uint32_t z = natural >> 8U;
                    direct_coeff[full_brick_origin(
                                     bi, direct_nx, direct_ny, direct_block_z) +
                                 x + static_cast<size_t>(y) * direct_nx +
                                 static_cast<size_t>(z) * nxy] = static_cast<DirectT>(0);
                }
            } else {
                const bool full_layout = direct_nx % 16U == 0U && direct_ny % 16U == 0U &&
                                         direct_nz % direct_block_z == 0U;
                if (full_layout) {
                    const uint32_t block_n = 16U * 16U * direct_block_z;
                    for (uint32_t i = threadIdx.x; i < elements; i += blockDim.x) {
                        const uint32_t pos = elem_base + i;
                        const uint32_t bi = pos / block_n;
                        const uint32_t rank = pos - bi * block_n;
                        direct_coeff[full_brick_origin(
                            bi, direct_nx, direct_ny, direct_block_z) +
                            direct_offset[rank]] = static_cast<DirectT>(0);
                    }
                } else {
                    for (uint32_t i = threadIdx.x; i < elements; i += blockDim.x)
                        direct_coeff[reordered_natural_index(elem_base + i,
                                                            direct_nx,
                                                            direct_ny,
                                                            direct_nz,
                                                            direct_block_z,
                                                            direct_offset)] = static_cast<DirectT>(0);
                }
            }
        } else {
            auto* destination64 = reinterpret_cast<uint64_t*>(output + base);
            for (uint32_t i = threadIdx.x; i < raw_bytes / 8u; i += blockDim.x)
                destination64[i] = 0;
            for (uint32_t i = (raw_bytes & ~7u) + threadIdx.x; i < raw_bytes; i += blockDim.x)
                output[base + i] = 0;
            if constexpr (WithSign) {
                const uint32_t words = (raw_bytes / 2u + 31u) / 32u;
                const uint32_t word_base = base / 2u / 32u;
                for (uint32_t i = threadIdx.x; i < words; i += blockDim.x)
                    sign_bitmap[word_base + i] = 0;
            }
        }
        return;
    }

    const byte* source = payload + chunk_begin;
    for (uint32_t i = threadIdx.x; i < stored_bytes; i += blockDim.x)
        encoded[i] = source[i];
    __syncthreads();
    byte* decoded = encoded;
    if (prefixed) {
        __shared__ uint32_t prefix_decoded_bytes;
        const uint32_t wanted =
            static_cast<uint32_t>(encoded[0]) | (static_cast<uint32_t>(encoded[1]) << 8u);
        const uint32_t prefix_lanes = encoded[2];
        if (threadIdx.x == 0)
            prefix_decoded_bytes = wanted;
        if (threadIdx.x < prefix_lanes) {
            const uint32_t lane = threadIdx.x;
            const uint32_t span = (wanted + prefix_lanes - 1u) / prefix_lanes;
            const uint32_t begin = min(wanted, lane * span);
            const uint32_t end = min(wanted, begin + span);
            uint32_t lane_offset = 0;
            for (uint32_t i = 0; i < lane; ++i)
                lane_offset += encoded[3u + i];
            const uint32_t lane_bytes = encoded[3u + lane];
            const byte* lane_src = encoded + 3u + prefix_lanes + lane_offset;
            uint64_t bitbuf = 0;
            uint32_t nbits = 0, pos = 0;
            uint32_t i = begin;
            while (i < end) {
                const uint32_t batch_count = min(4u, end - i);
                // Refill once per four symbols instead of re-entering the
                // byte-at-a-time refill loop for every symbol.  Thirty-six
                // bits cover the worst case (four 9-bit literals); at the end
                // of a byte-aligned lane the padding bits are harmless.
                const uint32_t wanted_bits = batch_count * 9u;
                while (nbits < wanted_bits && pos < lane_bytes) {
                    bitbuf |= static_cast<uint64_t>(lane_src[pos++]) << nbits;
                    nbits += 8u;
                }
#pragma unroll
                for (uint32_t batch = 0; batch < 4u; ++batch) {
                    if (batch < batch_count) {
                        if ((bitbuf & 1u) != 0u) {
                            expanded[i] = static_cast<byte>((bitbuf >> 1u) & 0xffu);
                            bitbuf >>= 9u;
                            nbits -= 9u;
                        } else {
                            constexpr byte dict[16] = {0x00,
                                                       0x40,
                                                       0x02,
                                                       0x80,
                                                       0x01,
                                                       0x10,
                                                       0x08,
                                                       0x20,
                                                       0x04,
                                                       0xc0,
                                                       0xff,
                                                       0x03,
                                                       0x30,
                                                       0x0c,
                                                       0x05,
                                                       0xa0};
                            expanded[i] = dict[(bitbuf >> 1u) & 0xfu];
                            bitbuf >>= 5u;
                            nbits -= 5u;
                        }
                        ++i;
                    }
                }
            }
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            expanded[wanted] = static_cast<byte>(raw_bytes);
            expanded[wanted + 1u] = static_cast<byte>(raw_bytes >> 8u);
        }
        __syncthreads();
        int compressed_bytes = static_cast<int>(prefix_decoded_bytes + 2u);
        d_iRZE_1(compressed_bytes, expanded, encoded, temp);
        __syncthreads();
        d_iBIT_2(compressed_bytes, encoded, expanded, temp);
        __syncthreads();
        decoded = expanded;
    } else if (stored_bytes < raw_bytes) {
        if (threadIdx.x == 0) {
            encoded[stored_bytes] = static_cast<byte>(raw_bytes);
            encoded[stored_bytes + 1u] = static_cast<byte>(raw_bytes >> 8u);
        }
        __syncthreads();
        int compressed_bytes = static_cast<int>(stored_bytes + 2u);
        d_iRZE_1(compressed_bytes, encoded, expanded, temp);
        __syncthreads();
        d_iBIT_2(compressed_bytes, expanded, encoded, temp);
        __syncthreads();
    }

    if constexpr (DirectLayout == 0) {
        auto* destination64 = reinterpret_cast<uint64_t*>(output + base);
        const auto* decoded64 = reinterpret_cast<const uint64_t*>(decoded);
        for (uint32_t i = threadIdx.x; i < raw_bytes / 8u; i += blockDim.x)
            destination64[i] = decoded64[i];
        for (uint32_t i = (raw_bytes & ~7u) + threadIdx.x; i < raw_bytes; i += blockDim.x)
            output[base + i] = decoded[i];
    }

    if constexpr (WithSign) {
        __syncthreads();
        // Each thread owns one contiguous 16-element half-word.  Compute one
        // block-wide nonzero prefix, consume the compact sign stream locally,
        // then pair adjacent threads into the natural 32-bit sign bitmap.  The
        // former 512-element tiled scan repeated three CTA barriers 16 times
        // per chunk; this produces the identical ranks with one prefix scan.
        const auto* mag = reinterpret_cast<const uint16_t*>(decoded);
        const uint32_t elements = raw_bytes / sizeof(uint16_t);
        const byte* packed_sign = source + stored_bytes;
        const uint32_t start = threadIdx.x * ITEMS_PER_THREAD;
        uint32_t local_nnz = 0;
#pragma unroll
        for (uint32_t j = 0; j < ITEMS_PER_THREAD; ++j) {
            const uint32_t i = start + j;
            local_nnz += static_cast<uint32_t>(i < elements && mag[i] != 0);
        }
        const uint32_t inclusive =
            block_prefix_sum<uint32_t>(local_nnz, reinterpret_cast<uint32_t*>(temp));
        uint32_t rank = inclusive - local_nnz;
        uint32_t negative_half = 0;
#pragma unroll
        for (uint32_t j = 0; j < ITEMS_PER_THREAD; ++j) {
            const uint32_t i = start + j;
            if (i < elements && mag[i] != 0) {
                const bool negative = ((packed_sign[rank >> 3u] >> (rank & 7u)) & 1u) != 0;
                negative_half |= static_cast<uint32_t>(negative) << j;
                ++rank;
            }
        }
        if constexpr (DirectLayout != 0) {
            const uint32_t elem_base = base / sizeof(uint16_t);
            if constexpr (DirectLayout == 2) {
                // A 16x16x2 brick has exactly one thread per natural element.
                // Invert the access direction so stores are row-major for every
                // aligned volume, independent of its global dimensions.
                auto* const negative_words = reinterpret_cast<uint16_t*>(temp);
                // block_prefix_sum() returns the per-thread value before all
                // warps are necessarily finished reading its shared scratch.
                // Fence the old use before repurposing temp as sign storage.
                __syncthreads();
                negative_words[threadIdx.x] = static_cast<uint16_t>(negative_half);
                __syncthreads();
                constexpr uint32_t BN = 16U * 16U * 2U;
                const uint32_t natural = threadIdx.x;
                const uint32_t source_rank = rank_16x16x2(natural);
                const uint32_t x = natural & 15U;
                const uint32_t y = (natural >> 4U) & 15U;
                const uint32_t z = natural >> 8U;
                const size_t natural_offset = x + static_cast<size_t>(y) * direct_nx +
                                              static_cast<size_t>(z) * direct_nx * direct_ny;
                uint32_t source_i = source_rank;
                const uint32_t nbx = direct_nx >> 4U;
                const uint32_t nby = direct_ny >> 4U;
                const size_t nxy = static_cast<size_t>(direct_nx) * direct_ny;
                const uint32_t first_bi = elem_base / BN;
                uint32_t bx = first_bi % nbx;
                const uint32_t byz = first_bi / nbx;
                uint32_t by = byz % nby;
                size_t brick_origin = static_cast<size_t>(bx * 16U) +
                                      static_cast<size_t>(by * 16U) * direct_nx +
                                      static_cast<size_t>(byz / nby * 2U) * nxy;
                for (uint32_t i = threadIdx.x; i < elements;
                     i += blockDim.x, source_i += BN) {
                    const uint32_t source_word = negative_words[source_i >> 4U];
                    const bool negative = ((source_word >> (source_i & 15U)) & 1U) != 0U;
                    const size_t destination = brick_origin + natural_offset;
                    const DirectT value = static_cast<DirectT>(mag[source_i]) * direct_q;
                    direct_coeff[destination] = negative ? -value : value;
                    if (++bx < nbx) {
                        brick_origin += 16U;
                    } else {
                        bx = 0U;
                        if (++by < nby) {
                            brick_origin += static_cast<size_t>(16U) * direct_nx -
                                            static_cast<size_t>(nbx - 1U) * 16U;
                        } else {
                            by = 0U;
                            brick_origin += 2U * nxy -
                                            static_cast<size_t>(nby - 1U) * 16U * direct_nx -
                                            static_cast<size_t>(nbx - 1U) * 16U;
                        }
                    }
                }
            } else if constexpr (DirectLayout == 3) {
                // Decode/sign extraction naturally owns Morton-contiguous data.
                // Re-index the shared-memory reads instead so each warp writes
                // row-major XYZ coefficients.  This trades a few integer bit
                // operations for fully coalesced output stores and removes the
                // global rank->offset table read on complete deep bricks.
                auto* const negative_words = reinterpret_cast<uint16_t*>(temp);
                __syncthreads();
                negative_words[threadIdx.x] = static_cast<uint16_t>(negative_half);
                __syncthreads();
                const uint32_t block_n = 16U * 16U * direct_block_z;
                const size_t nxy = static_cast<size_t>(direct_nx) * direct_ny;
                for (uint32_t i = threadIdx.x; i < elements; i += blockDim.x) {
                    const uint32_t pos = elem_base + i;
                    const uint32_t bi = pos / block_n;
                    const uint32_t natural = pos - bi * block_n;
                    const uint32_t source_rank =
                        rank_16x16_power2_z(natural, direct_block_z);
                    const uint32_t source_i = bi * block_n + source_rank - elem_base;
                    const uint32_t sign_word = negative_words[source_i >> 4U];
                    const bool negative = ((sign_word >> (source_i & 15U)) & 1U) != 0U;
                    const DirectT value = static_cast<DirectT>(mag[source_i]) * direct_q;
                    const uint32_t x = natural & 15U;
                    const uint32_t y = (natural >> 4U) & 15U;
                    const uint32_t z = natural >> 8U;
                    const size_t destination =
                        full_brick_origin(bi, direct_nx, direct_ny, direct_block_z) + x +
                        static_cast<size_t>(y) * direct_nx + static_cast<size_t>(z) * nxy;
                    direct_coeff[destination] = negative ? -value : value;
                }
            } else {
                const bool full_layout = direct_nx % 16U == 0U && direct_ny % 16U == 0U &&
                                         direct_nz % direct_block_z == 0U;
                if (full_layout) {
                    // The decoder assigns each thread 16 consecutive Morton ranks so
                    // it can unpack signs locally.  Writing rank `start + j` directly
                    // makes a warp access 0,16,32,... at each unrolled `j`, which badly
                    // scatters both the offset-table reads and the output stores for
                    // deep bricks.  Publish the per-thread sign words, then remap the
                    // store phase so every warp handles consecutive ranks, matching
                    // the standalone inverse-reorder kernel while retaining the
                    // decode/dequant fusion.
                    auto* const negative_words = reinterpret_cast<uint16_t*>(temp);
                    __syncthreads();
                    negative_words[threadIdx.x] = static_cast<uint16_t>(negative_half);
                    __syncthreads();
                    const uint32_t block_n = 16U * 16U * direct_block_z;
                    for (uint32_t i = threadIdx.x; i < elements; i += blockDim.x) {
                        const uint32_t pos = elem_base + i;
                        const uint32_t bi = pos / block_n;
                        const uint32_t rank = pos - bi * block_n;
                        const uint32_t sign_word = negative_words[i >> 4U];
                        const bool negative = ((sign_word >> (i & 15U)) & 1U) != 0U;
                        const DirectT value = static_cast<DirectT>(mag[i]) * direct_q;
                        const size_t origin =
                            full_brick_origin(bi, direct_nx, direct_ny, direct_block_z);
                        direct_coeff[origin + direct_offset[rank]] = negative ? -value : value;
                    }
                } else {
                    const uint32_t valid =
                        start < elements ? min(ITEMS_PER_THREAD, elements - start) : 0U;
                    ReorderRun run{};
                    bool one_brick = false;
                    if (valid != 0U) {
                        run = reorder_run(
                            elem_base + start, direct_nx, direct_ny, direct_nz, direct_block_z);
                        one_brick = run.rank + valid <= run.bdx * run.bdy * run.bdz;
                    }
#pragma unroll
                    for (uint32_t j = 0; j < ITEMS_PER_THREAD; ++j) {
                        const uint32_t i = start + j;
                        if (i < elements) {
                            const DirectT value = static_cast<DirectT>(mag[i]) * direct_q;
                            size_t destination;
                            if (one_brick) {
                                if (run.full) {
                                    destination = run.origin + direct_offset[run.rank + j];
                                } else {
                                    const uint32_t local = zorder_unrank_local(
                                        run.rank + j, run.bdx, run.bdy, run.bdz);
                                    const uint32_t lx = local % run.bdx;
                                    const uint32_t ly = (local / run.bdx) % run.bdy;
                                    const uint32_t lz = local / (run.bdx * run.bdy);
                                    destination =
                                        run.origin + lx + static_cast<size_t>(ly) * direct_nx +
                                        static_cast<size_t>(lz) * direct_nx * direct_ny;
                                }
                            } else {
                                destination = reordered_natural_index(elem_base + i,
                                                                      direct_nx,
                                                                      direct_ny,
                                                                      direct_nz,
                                                                      direct_block_z,
                                                                      direct_offset);
                            }
                            direct_coeff[destination] =
                                ((negative_half >> j) & 1U) ? -value : value;
                        }
                    }
                }
            }
        } else {
            const uint32_t upper = __shfl_down_sync(0xffffffffu, negative_half, 1);
            if ((threadIdx.x & 1u) == 0u && start < elements)
                sign_bitmap[(base / 2u + start) >> 5u] =
                    negative_half | (upper << ITEMS_PER_THREAD);
        }
        (void)sign_bytes;
    }
}

void release() {
    if (WZPConfig::device >= 0) {
        int previous = 0;
        cudaGetDevice(&previous);
        cudaSetDevice(WZPConfig::device);
        cudaFree(WZPConfig::slots);
        cudaFree(WZPConfig::sign_slots);
        cudaFree(WZPConfig::blob);
        cudaFree(WZPConfig::group_done);
        if (WZPConfig::timing_begin)
            cudaEventDestroy(WZPConfig::timing_begin);
        if (WZPConfig::timing_end)
            cudaEventDestroy(WZPConfig::timing_end);
        cudaSetDevice(previous);
    }
    if (WZPConfig::host_results)
        cudaFreeHost(WZPConfig::host_results);
    WZPConfig::device = -1;
    WZPConfig::capacity_n = 0;
    WZPConfig::slots = WZPConfig::sign_slots = WZPConfig::blob = nullptr;
    WZPConfig::blob_capacity = 0;
    WZPConfig::group_done = nullptr;
    WZPConfig::host_results = nullptr;
    WZPConfig::timing_begin = WZPConfig::timing_end = nullptr;
    WZPConfig::signed_payload_offset = 0;
}

void reserve(uint32_t n, cudaStream_t stream) {
    const WALTZ::GPUConfig& gpu = WALTZ::gpu_config();
    const int device = gpu.device;
    const uint32_t chunks = (n + ELEMS_PER_CHUNK - 1u) / ELEMS_PER_CHUNK;
    if (WZPConfig::device == device && WZPConfig::capacity_n >= n)
        return;
    release();
    WZPConfig::device = device;
    WZPConfig::capacity_n = n;
    const uint32_t groups = (chunks + GROUP_CHUNKS - 1u) / GROUP_CHUNKS;
    const size_t input_bytes = static_cast<size_t>(n) * sizeof(uint16_t);
    cudaMalloc(&WZPConfig::slots, static_cast<size_t>(chunks) * CS);
    cudaMalloc(&WZPConfig::sign_slots, static_cast<size_t>(chunks) * SIGN_SLOT_BYTES);
    WZPConfig::blob_capacity = static_cast<size_t>(groups) * sizeof(uint32_t) +
                      static_cast<size_t>(chunks) * 3u + input_bytes +
                      (static_cast<size_t>(n) + 7u) / 8u;
    cudaMalloc(&WZPConfig::blob, WZPConfig::blob_capacity);
    cudaMalloc(&WZPConfig::group_done, static_cast<size_t>(groups) * sizeof(uint32_t));
    cudaMallocHost(&WZPConfig::host_results, 2 * sizeof(uint32_t));
    cudaEventCreate(&WZPConfig::timing_begin);
    cudaEventCreate(&WZPConfig::timing_end);

    cudaMemsetAsync(WZPConfig::group_done, 0, static_cast<size_t>(groups) * sizeof(uint32_t), stream);
    reset_counter_kernel<<<1, 1, 0, stream>>>();
    cudaStreamSynchronize(stream);
}

template <bool WithSign>
void decode_impl(const void* blob,
                 uint32_t n,
                 uint16_t* output,
                 uint32_t* sign_bitmap,
                 double* ms,
                 cudaStream_t stream) {
    reserve(n, stream);
    const uint32_t output_bytes = n * sizeof(uint16_t);
    const uint32_t chunks = (output_bytes + CS - 1u) / CS;
    const uint32_t groups = (chunks + GROUP_CHUNKS - 1u) / GROUP_CHUNKS;
    check(cudaEventRecord(WZPConfig::timing_begin, stream), "record decode begin");
    decode_kernel<WithSign><<<chunks, TPB, 0, stream>>>(static_cast<const byte*>(blob),
                                                        chunks,
                                                        groups,
                                                        output_bytes,
                                                        reinterpret_cast<byte*>(output),
                                                        sign_bitmap);
    check(cudaEventRecord(WZPConfig::timing_end, stream), "record decode end");
    check(cudaEventSynchronize(WZPConfig::timing_end), "wait decode");
    float elapsed = 0;
    check(cudaEventElapsedTime(&elapsed, WZPConfig::timing_begin, WZPConfig::timing_end), "measure decode");
    if (ms)
        *ms = elapsed;
    check(cudaGetLastError(), "decode kernel");
}

template <typename DirectT>
void decode_reordered_impl(const void* blob,
                           uint32_t n,
                           DirectT* output,
                           const uint32_t* offset_table,
                           DirectT q,
                           dim3 dims,
                           uint32_t block_z,
                           double* ms,
                           cudaStream_t stream) {
    reserve(n, stream);
    const uint32_t output_bytes = n * sizeof(uint16_t);
    const uint32_t chunks = (output_bytes + CS - 1u) / CS;
    const uint32_t groups = (chunks + GROUP_CHUNKS - 1u) / GROUP_CHUNKS;
    check(cudaEventRecord(WZPConfig::timing_begin, stream), "record fused reordered decode begin");
    const bool aligned_bz2 = block_z == 2U && dims.x % 16U == 0U && dims.y % 16U == 0U &&
                             dims.z % 2U == 0U;
    const bool full_layout = dims.x % 16U == 0U && dims.y % 16U == 0U &&
                             dims.z % block_z == 0U;
    const bool natural_store = full_layout && block_z == 16U;
    if (aligned_bz2)
        decode_kernel<true, 2, DirectT><<<chunks, TPB, 0, stream>>>(static_cast<const byte*>(blob),
                                                           chunks,
                                                           groups,
                                                           output_bytes,
                                                           nullptr,
                                                           nullptr,
                                                           output,
                                                           offset_table,
                                                           q,
                                                           dims.x,
                                                           dims.y,
                                                           dims.z,
                                                           block_z);
    else {
        bool launched = false;
        if constexpr (std::is_same_v<DirectT, float>) {
            if (natural_store) {
                decode_kernel<true, 3, DirectT><<<chunks, TPB, 0, stream>>>(
                    static_cast<const byte*>(blob), chunks, groups, output_bytes, nullptr, nullptr,
                    output, offset_table, q, dims.x, dims.y, dims.z, block_z);
                launched = true;
            }
        }
        if (!launched)
            decode_kernel<true, 1, DirectT><<<chunks, TPB, 0, stream>>>(
                static_cast<const byte*>(blob), chunks, groups, output_bytes, nullptr, nullptr,
                output, offset_table, q, dims.x, dims.y, dims.z, block_z);
    }
    check(cudaEventRecord(WZPConfig::timing_end, stream), "record fused reordered decode end");
    check(cudaEventSynchronize(WZPConfig::timing_end), "wait fused reordered decode");
    float elapsed = 0;
    check(cudaEventElapsedTime(&elapsed, WZPConfig::timing_begin, WZPConfig::timing_end),
          "measure fused reordered decode");
    if (ms)
        *ms = elapsed;
    check(cudaGetLastError(), "fused reordered decode kernel");
}

} // namespace

size_t wzp_maxsize(uint32_t n) {
    const uint32_t chunks = (n + ELEMS_PER_CHUNK - 1u) / ELEMS_PER_CHUNK;
    const uint32_t groups = (chunks + GROUP_CHUNKS - 1u) / GROUP_CHUNKS;
    return static_cast<size_t>(groups) * sizeof(uint32_t) + static_cast<size_t>(chunks) * 3u +
           static_cast<size_t>(n) * sizeof(uint16_t) + (static_cast<size_t>(n) + 7u) / 8u;
}

void wzp_prealloc(uint32_t n, cudaStream_t stream) {
    reserve(n, stream);
}

size_t wzp_encode_with_sign(const uint16_t* input,
                            const uint32_t* sign_bitmap,
                            uint32_t n,
                            uint32_t* nnz,
                            double* ms,
                            cudaStream_t stream) {
    wzp_encode_with_sign_launch(input, sign_bitmap, n, stream);
    return wzp_encode_with_sign_finish(nnz, ms);
}

void wzp_encode_with_sign_launch(const uint16_t* input, const uint32_t* sign_bitmap,
                                 uint32_t n, cudaStream_t stream) {
    reserve(n, stream);
    const uint32_t input_bytes = n * sizeof(uint16_t);
    const uint32_t chunks = (input_bytes + CS - 1u) / CS;
    const uint32_t groups = (chunks + GROUP_CHUNKS - 1u) / GROUP_CHUNKS;
    const size_t metadata_offset = static_cast<size_t>(groups) * sizeof(uint32_t);
    WZPConfig::signed_payload_offset = metadata_offset + static_cast<size_t>(chunks) * 3u;
    const byte* const raw = reinterpret_cast<const byte*>(input);
    byte* const metadata = WZPConfig::blob + metadata_offset;
    cudaEventRecord(WZPConfig::timing_begin, stream);
    encode_kernel<<<512, TPB, 0, stream>>>(raw, input_bytes, sign_bitmap,
        WZPConfig::slots, WZPConfig::sign_slots, metadata, WZPConfig::blob, WZPConfig::group_done);

    cudaMemcpyFromSymbolAsync(WZPConfig::host_results, payload_counter, sizeof(uint32_t), 0, cudaMemcpyDeviceToHost, stream);
    cudaMemcpyFromSymbolAsync(WZPConfig::host_results + 1, nnz_counter, sizeof(uint32_t), 0, cudaMemcpyDeviceToHost, stream);
    cudaEventRecord(WZPConfig::timing_end, stream);
    reset_counter_kernel<<<1, 1, 0, stream>>>();
    cudaGetLastError();
}

size_t wzp_encode_with_sign_finish(uint32_t* nnz, double* ms) {
    check(cudaEventSynchronize(WZPConfig::timing_end), "wait signed encode");
    float elapsed = 0;
    check(cudaEventElapsedTime(&elapsed, WZPConfig::timing_begin, WZPConfig::timing_end), "measure signed encode");
    if (ms)
        *ms = elapsed;
    if (nnz)
        *nnz = WZPConfig::host_results[1];
    return WZPConfig::signed_payload_offset + static_cast<size_t>(WZPConfig::host_results[0]);
}

const void* wzp_encoded_buf() {
    return WZPConfig::blob;
}

void wzp_decode(const void* blob, uint32_t n, uint16_t* output, double* ms, cudaStream_t stream) {
    decode_impl<false>(blob, n, output, nullptr, ms, stream);
}

void wzp_decode_with_sign(const void* blob,
                          uint32_t n,
                          uint16_t* output,
                          uint32_t* sign_bitmap,
                          double* ms,
                          cudaStream_t stream) {
    decode_impl<true>(blob, n, output, sign_bitmap, ms, stream);
}

void wzp_decode_reordered_float(const void* blob,
                                uint32_t n,
                                float* output,
                                const uint32_t* offset_table,
                                float q,
                                dim3 dims,
                                uint32_t block_z,
                                double* ms,
                                cudaStream_t stream) {
    decode_reordered_impl(blob, n, output, offset_table, q, dims, block_z, ms, stream);
}

void wzp_decode_reordered_double(const void* blob,
                                 uint32_t n,
                                 double* output,
                                 const uint32_t* offset_table,
                                 double q,
                                 dim3 dims,
                                 uint32_t block_z,
                                 double* ms,
                                 cudaStream_t stream) {
    decode_reordered_impl(blob, n, output, offset_table, q, dims, block_z, ms, stream);
}

} // namespace WALTZ::lossless
