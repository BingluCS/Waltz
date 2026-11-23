#include "canonicalize.h"
#include "utils/Blob.hpp"
#include "waltz.hpp"

#include <cub/cub.cuh>
#include <cfenv>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <limits>

namespace {
using byte = unsigned char;
constexpr uint32_t chunk_elements = 8192;
constexpr uint32_t group_chunks = 16;
thread_local char last_error[256] = {};

struct Failure { const char* text; };
void require(bool good, const char* text) { if (!good) throw Failure{text}; }
void cuda_check(cudaError_t status) {
    if (status != cudaSuccess) {
        std::snprintf(last_error, sizeof(last_error), "CUDA: %s", cudaGetErrorString(status));
        throw Failure{last_error};
    }
}
// Per-host-thread, per-device reusable scratch. The public API synchronizes
// on both success and failure. Within a call every operation uses one stream,
// so released inner-sort scratch may be reused in that same stream safely.
// No C++ containers, no unbounded number of cached slots.
struct ScratchCache {
    struct Slot {void* data=nullptr;size_t capacity=0;int device=-1;bool used=false;};
    Slot slots[16];
    int acquire(size_t bytes) {
        int device;cuda_check(cudaGetDevice(&device));
        int fit=-1,spare=-1,empty=-1;
        for(int i=0;i<16;++i)if(!slots[i].used){
            auto& s=slots[i];
            if(!s.data){if(empty<0)empty=i;continue;}
            if(s.device!=device)continue;
            if(s.capacity>=bytes&&(fit<0||s.capacity<slots[fit].capacity))fit=i;
            if(spare<0||s.capacity>slots[spare].capacity)spare=i;
        }
        int i=fit>=0?fit:(spare>=0?spare:empty);
        require(i>=0,"portable GPU scratch cache exhausted");
        auto& s=slots[i];
        if(fit<0){
            if(s.data){cuda_check(cudaFree(s.data));s.data=nullptr;s.capacity=0;}
            cuda_check(cudaMalloc(&s.data,bytes));s.capacity=bytes;s.device=device;
        }
        s.used=true;return i;
    }
    void clear() noexcept {
        int original=-1;cudaGetDevice(&original);
        for(auto& s:slots)if(s.data&&!s.used){
            if(cudaSetDevice(s.device)==cudaSuccess)cudaFree(s.data);
            s.data=nullptr;s.capacity=0;s.device=-1;
        }
        if(original>=0)cudaSetDevice(original);
    }
    ~ScratchCache(){clear();}
};
thread_local ScratchCache scratch_cache;
struct Allocation {
    void* data = nullptr;
    int slot=-1;
    explicit Allocation(size_t bytes) {if(bytes){slot=scratch_cache.acquire(bytes);data=scratch_cache.slots[slot].data;}}
    ~Allocation() {if(slot>=0)scratch_cache.slots[slot].used=false;}
    Allocation(const Allocation&) = delete;
    Allocation& operator=(const Allocation&) = delete;
};
size_t aligned(size_t n) {
    require(n <= std::numeric_limits<size_t>::max() - 7, "section alignment overflows");
    return (n + 7) & ~size_t(7);
}
size_t section_end(size_t at, size_t bytes, size_t limit) {
    require(at <= limit && bytes <= limit - at, "truncated Waltz section");
    const size_t end = aligned(at + bytes);
    require(end <= limit, "truncated Waltz section padding");
    return end;
}
void disjoint(const void* a, size_t as, const void* b, size_t bs) {
    const uintptr_t x = reinterpret_cast<uintptr_t>(a), y = reinterpret_cast<uintptr_t>(b);
    require(as <= UINTPTR_MAX - x && bs <= UINTPTR_MAX - y, "device pointer range overflows");
    require(x >= y + bs || y >= x + as, "device input/output ranges overlap");
}

__device__ void bad(int* status, int code) { atomicCAS(status, 0, code); }
template <typename T>
__global__ void finite_input(const T* input, uint32_t n, int* status) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n && !isfinite(double(input[i]))) bad(status, 8);
}
template <typename T>
__global__ void strict_bound(const T* input, const T* decoded, uint32_t n, double bound, int* status) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const double a = double(decoded[i]), b = double(input[i]);
    if (!isfinite(a) || !isfinite(b)) { bad(status, 9); return; }
    const double high = __dsub_rn(a, b), magnitude = fabs(high);
    if (magnitude > bound || !isfinite(high)) { bad(status, 10); return; }
    if (magnitude == bound) {
        // Error-free TwoDiff resolves the only comparison ambiguity: the
        // rounded difference equals the represented bound but the exact
        // subtraction lies just outside it. Widely separated FP32 exponents
        // can also need this tail despite promotion to FP64. Explicit
        // intrinsics retain all rounding points.
        const double bv = __dsub_rn(a, high);
        const double av = __dadd_rn(high, bv);
        const double br = __dsub_rn(bv, b);
        const double ar = __dsub_rn(a, av);
        const double low = __dadd_rn(ar, br);
        if ((high > 0 && low > 0) || (high < 0 && low < 0)) bad(status, 11);
    }
}
__device__ uint32_t meta24(const byte* p, uint32_t chunk) {
    p += size_t(chunk) * 3;
    return uint32_t(p[0]) | uint32_t(p[1]) << 8 | uint32_t(p[2]) << 16;
}

__global__ void group_sizes(const byte* metadata, uint32_t n, uint32_t chunks,
                             uint32_t groups, uint64_t* lengths, int* status) {
    const uint32_t group = blockIdx.x * blockDim.x + threadIdx.x;
    if (group >= groups) return;
    uint64_t bytes = 0;
    const uint32_t begin = group * group_chunks;
    const uint32_t end = min(begin + group_chunks, chunks);
    for (uint32_t c = begin; c < end; ++c) {
        const uint32_t valid = min(chunk_elements, n - c * chunk_elements);
        const uint32_t m = meta24(metadata, c);
        const uint32_t mag_code = m & 0x3fffU;
        const uint32_t sign_code = m >> 14;
        if (!mag_code) {
            if (sign_code) bad(status, 1);
            continue;
        }
        const uint32_t mag = mag_code == 0x3fffU ? valid * 2 : mag_code;
        const uint32_t sign = sign_code ? sign_code : 1024;
        if ((mag_code != 0x3fffU && (mag < 2 || mag >= valid * 2)) || sign > (valid + 7) / 8)
            bad(status, 2);
        bytes += mag + sign;
    }
    lengths[group] = bytes;
}

__global__ void repack_groups(const byte* src_payload, const uint32_t* src_offsets,
                               byte* dst_payload, uint32_t* dst_offsets,
                               const uint64_t* lengths, const uint64_t* offsets,
                               uint32_t groups, size_t payload_bytes, int* status) {
    const uint32_t group = blockIdx.x;
    if (group >= groups) return;
    const uint64_t bytes = lengths[group], dst = offsets[group];
    const uint64_t src = src_offsets[group];
    if (dst > UINT32_MAX || dst > payload_bytes || bytes > payload_bytes - dst ||
        (bytes && (src > payload_bytes || bytes > payload_bytes - src))) {
        if (!threadIdx.x) bad(status, 3);
        return;
    }
    if (group + 1 == groups && dst + bytes != payload_bytes) {
        if (!threadIdx.x) bad(status, 4);
        return;
    }
    if (!threadIdx.x) dst_offsets[group] = bytes ? uint32_t(dst) : 0;
    for (uint64_t i = threadIdx.x; i < bytes; i += blockDim.x)
        dst_payload[dst + i] = src_payload[src + i];
}

template <typename Value>
__global__ void validate_indices(const uint32_t* indices, const Value* values,
                                  uint32_t count, uint32_t n, bool quant, int* status) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    if (indices[i] >= n || (i && indices[i - 1] == indices[i])) bad(status, 5);
    if (quant) {
        if (!(values[i] > Value(65535))) bad(status, 6);
    } else if (!isfinite(double(values[i]))) bad(status, 7);
}

template <typename Value>
void sort_pairs(const byte* input, byte* output, size_t index_at, size_t value_at,
                uint32_t count, uint32_t n, bool quant, int* status, cudaStream_t stream) {
    if (!count) return;
    const auto* ikeys = reinterpret_cast<const uint32_t*>(input + index_at);
    const auto* ivals = reinterpret_cast<const Value*>(input + value_at);
    auto* okeys = reinterpret_cast<uint32_t*>(output + index_at);
    auto* ovals = reinterpret_cast<Value*>(output + value_at);
    size_t temp_bytes = 0;
    cuda_check(cub::DeviceRadixSort::SortPairs(nullptr, temp_bytes, ikeys, okeys, ivals, ovals,
                                              count, 0, 32, stream));
    Allocation temp(temp_bytes);
    cuda_check(cub::DeviceRadixSort::SortPairs(temp.data, temp_bytes, ikeys, okeys, ivals, ovals,
                                              count, 0, 32, stream));
    validate_indices<<<(count + 255) / 256, 256, 0, stream>>>(okeys, ovals, count, n, quant, status);
    cuda_check(cudaGetLastError());
    // Return the lease; subsequent operations on this stream remain ordered.
}

void canonicalize_impl(const void* device_input, size_t bytes, uint32_t n,
                        void* device_output, size_t capacity, size_t* written, cudaStream_t stream) {
    require(device_input && device_output && written && n, "null argument or zero element count");
    require(n <= UINT32_MAX / 2, "element count exceeds GPU WZP byte-count range");
    require(device_input != device_output, "canonicalization requires separate device buffers");
    require(bytes >= sizeof(WALTZ::WaltzBlobHeader) && capacity >= bytes, "insufficient archive capacity");
    disjoint(device_input, bytes, device_output, bytes);
    WALTZ::WaltzBlobHeader h{};
    cuda_check(cudaMemcpyAsync(&h, device_input, sizeof(h), cudaMemcpyDeviceToHost, stream));
    cuda_check(cudaStreamSynchronize(stream));
    require(WALTZ::WaltzBlobHeader::valid_flags(h.flags), "invalid Waltz archive flags");
    require(h.pwe_encoding() == WALTZ::PweEncoding::Native, "portable profile requires native PWE");
    require(h.synthesis() == WALTZ::Synthesis::HaloAllLevels, "portable profile requires all-level native halo synthesis");
    require(h.levels_xy <= 5 && h.levels_z <= 5 &&
            (!h.use_dyadic() || h.levels_xy == h.levels_z) && h.reorder_bz >= 1 && h.reorder_bz <= 32,
            "invalid portable transform metadata");
    require(h.q > 0 && std::isfinite(h.q), "invalid quantization step");
    require(h.n_pwe <= n && h.n_qo <= n && h.n_pwe <= INT32_MAX && h.n_qo <= INT32_MAX,
            "invalid side-channel counts");
    require(h.pwe_bytes == uint64_t(h.n_pwe) * h.element_bytes(), "invalid native PWE size");
    require(h.mag_comp_bytes <= UINT32_MAX, "WZP magnitude length exceeds portable offset range");
    const uint32_t chunks = uint32_t((uint64_t(n) + chunk_elements - 1) / chunk_elements);
    const uint32_t groups = (chunks + group_chunks - 1) / group_chunks;
    const size_t prefix = size_t(groups) * 4 + size_t(chunks) * 3;
    require(h.mag_comp_bytes >= prefix, "truncated WZP directory/metadata");
    const size_t lengths[] = {size_t(h.mag_comp_bytes), size_t(h.n_pwe) * 4, h.pwe_bytes,
                               size_t(h.n_qo) * 4, size_t(h.n_qo) * 4};
    size_t offsets[5], at = aligned(sizeof(h));
    for (unsigned s = 0; s < 5; ++s) { offsets[s] = at; at = section_end(at, lengths[s], bytes); }
    require(at == bytes, "archive length does not match Waltz sections");
    const auto src = static_cast<const byte*>(device_input);
    auto dst = static_cast<byte*>(device_output);
    Allocation status_alloc(sizeof(int));
    auto* status = static_cast<int*>(status_alloc.data);
    cuda_check(cudaMemsetAsync(status, 0, sizeof(int), stream));
    cuda_check(cudaMemsetAsync(dst, 0, bytes, stream));
    cuda_check(cudaMemcpyAsync(dst, src, sizeof(h), cudaMemcpyDeviceToDevice, stream));
    cuda_check(cudaMemcpyAsync(dst + offsets[0] + size_t(groups) * 4,
                               src + offsets[0] + size_t(groups) * 4,
                               size_t(chunks) * 3, cudaMemcpyDeviceToDevice, stream));
    Allocation length_alloc(size_t(groups) * sizeof(uint64_t)), offset_alloc(size_t(groups) * sizeof(uint64_t));
    auto* group_lengths = static_cast<uint64_t*>(length_alloc.data);
    auto* group_offsets = static_cast<uint64_t*>(offset_alloc.data);
    group_sizes<<<(groups + 255) / 256, 256, 0, stream>>>(src + offsets[0] + size_t(groups) * 4,
                                                       n, chunks, groups, group_lengths, status);
    cuda_check(cudaGetLastError());
    size_t scan_bytes = 0;
    cuda_check(cub::DeviceScan::ExclusiveSum(nullptr, scan_bytes, group_lengths, group_offsets, groups, stream));
    Allocation scan(scan_bytes);
    cuda_check(cub::DeviceScan::ExclusiveSum(scan.data, scan_bytes, group_lengths, group_offsets, groups, stream));
    repack_groups<<<groups, 256, 0, stream>>>(src + offsets[0] + prefix,
        reinterpret_cast<const uint32_t*>(src + offsets[0]), dst + offsets[0] + prefix,
        reinterpret_cast<uint32_t*>(dst + offsets[0]), group_lengths, group_offsets, groups,
        size_t(h.mag_comp_bytes) - prefix, status);
    cuda_check(cudaGetLastError());
    if (h.element_bytes() == 4)
        sort_pairs<float>(src, dst, offsets[1], offsets[2], h.n_pwe, n, false, status, stream);
    else sort_pairs<double>(src, dst, offsets[1], offsets[2], h.n_pwe, n, false, status, stream);
    sort_pairs<int32_t>(src, dst, offsets[3], offsets[4], h.n_qo, n, true, status, stream);
    int device_status = 0;
    cuda_check(cudaMemcpyAsync(&device_status, status, sizeof(device_status), cudaMemcpyDeviceToHost, stream));
    cuda_check(cudaStreamSynchronize(stream));
    require(device_status == 0, "invalid WZP metadata, group bounds, indices or side-channel values");
    *written = bytes;
}

void check_environment() {
    const char* halo = std::getenv("WALTZ_HALO3D");
    const char* small = std::getenv("WALTZ_SMALL_PWE_SCALE");
    require(!halo || !std::strcmp(halo, "all") || !std::strcmp(halo, "auto") || !std::strcmp(halo, "both"),
            "portable encoder requires WALTZ_HALO3D=all/default");
    require(small && !std::strcmp(small, "0"), "portable encoder requires WALTZ_SMALL_PWE_SCALE=0");
    require(std::fegetround() == FE_TONEAREST, "portable GPU encoder requires nearest-even host rounding");
}

template <typename T>
void compress_impl(const T* input, uint32_t nx, uint32_t ny, uint32_t nz, double eb,
                   void* output, size_t capacity, size_t* written, cudaStream_t stream) {
    require(input && output && written && nx && ny && nz, "null argument or zero dimension");
    const uint64_t xy = uint64_t(nx) * ny;
    require(xy <= UINT32_MAX / 2 && nz <= (UINT32_MAX / 2) / xy,
            "shape exceeds GPU WZP byte-count range");
    const uint32_t n = uint32_t(xy * nz);
    require(eb > 0 && std::isfinite(eb) && std::isfinite(1.5 * eb), "invalid absolute error bound");
    const T quantum = T(1.5 * eb), reciprocal = T(1.0 / (1.5 * eb));
    require(quantum > T(0) && std::isfinite(quantum) && std::isfinite(reciprocal),
            "quantum/reciprocal not representable in native type");
    check_environment();
    int device = 0, shared = 0;
    cuda_check(cudaGetDevice(&device));
    cuda_check(cudaDeviceGetAttribute(&shared, cudaDevAttrMaxSharedMemoryPerBlockOptin, device));
    require(shared >= 65536, "portable all-level encoder requires 64 KiB opt-in shared memory");
    const size_t gpu_capacity = WALTZ::compressed_buffer_capacity<T>(n);
    require(capacity >= gpu_capacity, "portable GPU output requires complete GPU capacity bound");
    disjoint(input, size_t(n) * sizeof(T), output, capacity);
    Allocation status_alloc(sizeof(int));
    auto* status = static_cast<int*>(status_alloc.data);
    cuda_check(cudaMemsetAsync(status, 0, sizeof(int), stream));
    finite_input<<<(n + 255) / 256, 256, 0, stream>>>(input, n, status);
    cuda_check(cudaGetLastError());
    int input_status = 0;
    cuda_check(cudaMemcpyAsync(&input_status, status, sizeof(int), cudaMemcpyDeviceToHost, stream));
    cuda_check(cudaStreamSynchronize(stream));
    require(input_status == 0, "nonfinite input unsupported in portable profile");
    Allocation encoded(gpu_capacity);
    WALTZ::Config config;
    config.ndims = 3; config.dimx = nx; config.dimy = ny; config.dimz = nz; config.num = n;
    config.errorBoundMode = WALTZ::EB_ABS; config.absErrorBound = eb;
    const size_t bytes = WALTZ::d_compress<T>(config, const_cast<T*>(input), static_cast<char*>(encoded.data), nullptr,
                                              reinterpret_cast<void*>(stream));
    WALTZ::WaltzBlobHeader header{};
    cuda_check(cudaMemcpyAsync(&header, encoded.data, sizeof(header), cudaMemcpyDeviceToHost, stream));
    cuda_check(cudaStreamSynchronize(stream));
    // The legacy portable path has no independent overflow flag for these
    // fixed-cap channels. Reaching a cap below N is conservatively ambiguous.
    // A cap of N is not ambiguous: each element can emit at most one entry.
    const uint32_t pwe_cap = n < (1U << 22) ? n : (1U << 22);
    const uint32_t qo_cap = n < (1U << 20) ? n : (1U << 20);
    require((n <= pwe_cap || header.n_pwe < pwe_cap) && (n <= qo_cap || header.n_qo < qo_cap),
            "legacy GPU side-channel count reached its fixed cap; portable encoder fails closed");
    canonicalize_impl(encoded.data, bytes, n, output, capacity, written, stream);
    Allocation reconstructed(size_t(n) * sizeof(T));
    const size_t decoded_bytes = WALTZ::d_decompress<T>(config, static_cast<char*>(output),
                                                        static_cast<T*>(reconstructed.data), nullptr,
                                                        reinterpret_cast<void*>(stream));
    require(decoded_bytes == size_t(n) * sizeof(T), "verification decode has wrong size");
    strict_bound<<<(n + 255) / 256, 256, 0, stream>>>(input, static_cast<const T*>(reconstructed.data), n, eb, status);
    cuda_check(cudaGetLastError());
    int bound_status = 0;
    cuda_check(cudaMemcpyAsync(&bound_status, status, sizeof(int), cudaMemcpyDeviceToHost, stream));
    cuda_check(cudaStreamSynchronize(stream));
    require(bound_status == 0, "legacy GPU reconstruction cannot meet strict pointwise bound; no valid archive emitted");
}

template <typename Fn>
int protect(Fn&& fn, size_t* written, cudaStream_t stream) {
    if (written) *written = 0;
    last_error[0] = 0;
    try { fn(); cuda_check(cudaStreamSynchronize(stream)); return 0; }
    catch (const Failure& e) {
        if (e.text != last_error) std::snprintf(last_error, sizeof(last_error), "%s", e.text);
    } catch (const std::exception& e) {
        std::snprintf(last_error, sizeof(last_error), "%s", e.what());
    } catch (...) { std::snprintf(last_error, sizeof(last_error), "unknown GPU portable codec failure"); }
    if (written) *written = 0;
    // Ensure no queued operation retains the caller's storage on error.
    cudaStreamSynchronize(stream);
    return -1;
}
} // namespace

extern "C" int waltz_gpu_canonicalize(const void* input, size_t bytes, uint32_t n,
                                       void* output, size_t capacity, size_t* written, cudaStream_t stream) {
    return protect([&] { canonicalize_impl(input, bytes, n, output, capacity, written, stream); }, written, stream);
}
extern "C" int waltz_gpu_portable_compress_f32(const float* input, uint32_t nx, uint32_t ny, uint32_t nz,
                                               double eb, void* output, size_t capacity, size_t* written,
                                               cudaStream_t stream) {
    return protect([&] { compress_impl(input, nx, ny, nz, eb, output, capacity, written, stream); }, written, stream);
}
extern "C" int waltz_gpu_portable_compress_f64(const double* input, uint32_t nx, uint32_t ny, uint32_t nz,
                                               double eb, void* output, size_t capacity, size_t* written,
                                               cudaStream_t stream) {
    return protect([&] { compress_impl(input, nx, ny, nz, eb, output, capacity, written, stream); }, written, stream);
}
extern "C" const char* waltz_gpu_portable_last_error(void) { return last_error; }
extern "C" void waltz_gpu_portable_release_workspace(void) {scratch_cache.clear();}
