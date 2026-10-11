#include "lossless/reorder.hpp"
#include "lossless/wzp.hpp"
#include "transformers/CDF97.hpp"
#include "transformers/ICDF97.hpp"
#include "waltz.hpp"
#include "utils/Blob.hpp"
#include "utils/Timer.hpp"
#include "utils/Statistics.hpp"
#include "utils/def.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <type_traits>

namespace WALTZ {

// Reuse only temporary storage, never coefficients or results. Calls already
// synchronize before returning; the next call overwrites all required contents.
class CompressionWorkspace {
    char* data = nullptr;
    size_t capacity = 0;
    int device = -1;

    void release() {
        if (data == nullptr)
            return;
        int previous = 0;
        if (cudaGetDevice(&previous) == cudaSuccess) {
            cudaSetDevice(device);
            cudaFree(data);
            cudaSetDevice(previous);
        }
        data = nullptr;
        capacity = 0;
    }

  public:
    ~CompressionWorkspace() { release(); }

    char* reserve(size_t bytes) {
        int current = 0;
        cudaGetDevice(&current);
        if (device != current || capacity < bytes) {
            release();
            device = current;
            if (cudaMalloc(reinterpret_cast<void**>(&data), bytes) != cudaSuccess)
                throw std::runtime_error("Waltz: cannot allocate compression workspace");
            capacity = bytes;
        }
        return data;
    }
};

static thread_local CompressionWorkspace compression_workspace;

static inline size_t blob_align(size_t v) {
    return (v + 7) & ~static_cast<size_t>(7);
}

struct BlobSection {
    const void* data;
    size_t bytes;
};

template <typename T>
static size_t Waltz_gather(void* output, bool use_dyadic,
                     uint32_t reorder_bz, uint8_t levels_xy, uint8_t levels_z,
                     double quantity, size_t mag_bytes,
                     uint32_t pwe_count, const uint32_t* pwe_index, const T* pwe_error,
                     uint32_t qout_count, const uint32_t* quant_index,
                     const int32_t* quant_value, cudaStream_t stream) {
    WaltzBlobHeader hdr{};
    hdr.use_dyadic = static_cast<uint8_t>(use_dyadic);
    hdr.reorder_bz = static_cast<uint8_t>(reorder_bz);
    hdr.levels_xy = levels_xy;
    hdr.levels_z = levels_z;
    hdr.q = quantity;
    hdr.mag_comp_bytes = static_cast<uint64_t>(mag_bytes);
    hdr.n_pwe = pwe_count;
    hdr.n_qo = qout_count;
    // Keep section order and 8-byte alignment identical to Waltz_scatter.
    const BlobSection sections[] = {
        {lossless::wzp_encoded_buf(), mag_bytes},
        {pwe_index, pwe_count * sizeof(uint32_t)},
        {pwe_error, pwe_count * sizeof(T)},
        {quant_index, qout_count * sizeof(uint32_t)},
        {quant_value, qout_count * sizeof(int32_t)},
    };
    auto* cmpdata = static_cast<unsigned char*>(output);
    cudaMemcpyAsync(cmpdata, &hdr, sizeof(hdr), cudaMemcpyHostToDevice, stream);
    size_t offset = blob_align(sizeof(hdr));
    for (const auto& section : sections) {
        if (section.bytes != 0U)
            cudaMemcpyAsync(cmpdata + offset, section.data, section.bytes,
                            cudaMemcpyDeviceToDevice, stream);
        offset = blob_align(offset + section.bytes);
    }
    // Complete the header copy before its host storage can go out of scope.
    cudaStreamSynchronize(stream);
    return offset;
}

struct BlobView {
    const unsigned char* magnitude;
    const unsigned char* pwe_index;
    const unsigned char* pwe_error;
    const unsigned char* quant_index;
    const unsigned char* quant_value;
    size_t bytes;
};

// Split the blob into device views, without allocating or copying payloads.
template <typename T>
static BlobView Waltz_scatter(const void* input, const WaltzBlobHeader& hdr) {
    const auto* cmpdata = static_cast<const unsigned char*>(input);
    const size_t sizes[] = {
        static_cast<size_t>(hdr.mag_comp_bytes),
        hdr.n_pwe * sizeof(uint32_t),
        static_cast<size_t>(hdr.n_pwe) * sizeof(T),
        hdr.n_qo * sizeof(uint32_t),
        hdr.n_qo * sizeof(int32_t),
    };
    BlobView view{};
    const unsigned char** sections[] = {
        &view.magnitude, &view.pwe_index, &view.pwe_error,
        &view.quant_index, &view.quant_value,
    };
    size_t offset = blob_align(sizeof(hdr));
    for (size_t i = 0; i < sizeof(sizes) / sizeof(sizes[0]); ++i) {
        *sections[i] = cmpdata + offset;
        offset = blob_align(offset + sizes[i]);
    }
    view.bytes = offset;
    return view;
}

// Decompression: apply the exact, original-precision PWE error.
template <typename T>
__global__ void apply_pwe_outliers(T* x, const uint32_t* oidx, const T* oerr, uint32_t n) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        x[oidx[i]] -= oerr[i];
}

// decompression: patch the q-outliers (mags clamped to 65535 in the u16 stream) with the
// exact magnitude -- the sign is taken from the already-dequantized (clamped) value.
template <typename T>
__global__ void
apply_q_outliers(T* iq, const uint32_t* qidx, const int32_t* qval, uint32_t n, double q) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        const T v = static_cast<T>(qval[i]) * static_cast<T>(q);
        iq[qidx[i]] = (iq[qidx[i]] < T(0)) ? -v : v;
    }
}

// Tiny compression-result copies use this reusable pinned array so they can be
// queued together and completed by one synchronization.
static unsigned int* g_host_outlier_counts = nullptr;
static WaltzBlobHeader* g_pipeline_host_header = nullptr;

static uint8_t calculate_dwt_levels_1d(size_t len) {
    uint8_t levels = 0;
    while (len > 8 && levels < 5) {
        len = (len + 1) / 2;
        ++levels;
    }
    return levels;
}

void calculate_dwt_levels(dim3 data_dims, uint8_t& levels_xy, uint8_t& levels_z) {
    levels_xy = calculate_dwt_levels_1d(std::min(data_dims.x, data_dims.y));
    levels_z = calculate_dwt_levels_1d(data_dims.z);
}

// Conservatively map forward flat-column tiles to inverse row tiles. Only the
// high band is proven here: the low band remains marked possibly nonzero.
static __global__ void map_z_high_flags(const uint32_t* producer, uint32_t* inverse,
    uint32_t nx, uint32_t ny, uint32_t producer_columns, uint32_t inverse_columns) {
    const uint32_t nx_tiles = (nx + inverse_columns - 1U) / inverse_columns;
    const uint32_t t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= nx_tiles * ny)
        return;
    const uint32_t y = t / nx_tiles;
    const uint32_t x0 = (t - y * nx_tiles) * inverse_columns;
    const uint32_t columns = min(inverse_columns, nx - x0);
    const size_t begin = static_cast<size_t>(y) * nx + x0;
    const uint32_t first = static_cast<uint32_t>(begin / producer_columns);
    const uint32_t last = static_cast<uint32_t>((begin + columns - 1U) / producer_columns);
    uint32_t high_nonzero = 0U;
    for (uint32_t tile = first; tile <= last; ++tile)
        high_nonzero |= producer[tile];
    inverse[t] = high_nonzero ? 3U : 1U;
}

template <typename T, int NDIMS>
size_t d_WALTZ_compress(T* d_oridata, char* d_cmpData, const WALTZ::Config& config, const char* cmpPath, void* stream) {
    dim3 data_dims(config.dimx, config.dimy, config.dimz);
    uint32_t datasize = config.num;
    if constexpr (NDIMS != 3) {
        throw std::runtime_error("CPU CDF97 path currently only supports 3D");
    }
    double real_error = config.absErrorBound;
    double quantity = 1.5 * real_error;


    const cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    if (g_host_outlier_counts == nullptr)
        cudaMallocHost(reinterpret_cast<void**>(&g_host_outlier_counts), 2U * sizeof(unsigned int));

    uint8_t levels_xy = 0, levels_z = 0;
    calculate_dwt_levels(data_dims, levels_xy, levels_z);
    // SPERR's dimension-only rule: equal levels use dyadic, otherwise XY+Z.
    // Both level counts are capped at 5, covering SPERR's >=5 exception too.
    const bool use_dyadic = levels_xy == levels_z;
    const uint8_t dyadic_level = std::min(levels_xy, levels_z);

    if (!use_dyadic && levels_z > 0 &&
        data_dims.z > CDF97::DWT_Z_STATIC_BYTES / (2U * sizeof(T)))
        throw std::invalid_argument("Waltz: plane Z column exceeds the 45 KiB static shared-memory capacity");

    const uint32_t reorder_bz = WALTZ::lossless::select_block_zband(data_dims, levels_z, use_dyadic != 0U);
    const uint32_t rank_shapes = (data_dims.x % 16U || data_dims.y % 16U || data_dims.z % reorder_bz) ? 8U : 1U;

    CDF97::dwt3d_prealloc<T>();
    if (CDF97::idwt3d_prealloc<T>(data_dims, use_dyadic != 0U, false, cuda_stream) != cudaSuccess)
        throw std::runtime_error("Waltz: cannot configure IDWT workspace");

    uint32_t* forward_high_flags = nullptr;
    uint32_t* inverse_high_flags = nullptr;
    uint32_t forward_columns = 0U, inverse_columns = 0U, inverse_tiles = 0U;
    if constexpr (std::is_same_v<T, double>) {
        // The exact low-only synthesis path requires aligned double2 columns.
        // Keep both the forward and inverse shared/global dispatch unchanged.
        if (use_dyadic && levels_z > 0 && (data_dims.x & 1U) == 0U &&
            static_cast<size_t>(data_dims.z) * 16U <= WALTZ::IDWT_SHARED_BYTES / sizeof(T)) {
            const uint32_t columns = data_dims.x * data_dims.y;
            forward_columns = std::min(static_cast<uint32_t>(32768U / sizeof(T) / data_dims.z), columns);
            if (forward_columns >= 2U)
                forward_columns &= ~1U;
            const uint32_t forward_tiles = (columns + forward_columns - 1U) / forward_columns;
            const uint32_t sm_count = WALTZ::gpu_config().properties.multiProcessorCount;
            if (forward_tiles > 2U * sm_count) {
                inverse_columns = (CDF97::idwt_z_shared_bytes<T>() / sizeof(T) / data_dims.z) & ~1U;
                inverse_tiles = ((data_dims.x + inverse_columns - 1U) / inverse_columns) * data_dims.y;
                static thread_local CompressionWorkspace high_flag_storage;
                forward_high_flags = reinterpret_cast<uint32_t*>(high_flag_storage.reserve(
                    (static_cast<size_t>(forward_tiles) + inverse_tiles) * sizeof(uint32_t)));
                inverse_high_flags = forward_high_flags + forward_tiles;
            }
        }
    }

    T* d_data = nullptr;
    T* tmp_data = nullptr;
    T* d_q = nullptr;
    uint16_t* d_q_abs = nullptr;
    uint32_t* d_q_sign = nullptr;
    uint32_t* d_qoi = nullptr;
    int32_t* d_qov = nullptr;
    uint32_t* d_poi = nullptr;
    T* d_poe = nullptr;
    uint16_t* d_tb = nullptr;
    unsigned int* d_cnt = nullptr;
    unsigned int* d_qoc = nullptr;
    const size_t bytes = static_cast<size_t>(datasize) * sizeof(T);
    const uint32_t qout_cap = std::min(datasize, 1U << 20U);
    const uint32_t pout_cap = std::min(datasize, 1U << 22U);
    const size_t mag_bytes = static_cast<size_t>(datasize) * sizeof(uint16_t);
    const size_t sb_bytes = ((static_cast<size_t>(datasize) + 8191) / 8192) * 1024;
    static thread_local GPUTimerCache<2> timer_cache;
    GPUTimer* const timers = timer_cache.get();
    GPUTimer& block_rank_timer = timers[0];

    // All scratch buffers have the same lifetime. Allocate once, preserving
    // cudaMalloc's 256-byte alignment for every typed/vectorized view.
    const size_t scratch_sizes[] = {
        bytes, bytes, use_dyadic ? 0U : bytes, mag_bytes, sb_bytes,
        static_cast<size_t>(qout_cap) * (sizeof(uint32_t) + sizeof(int32_t)),
        static_cast<size_t>(pout_cap) * (sizeof(T) + sizeof(uint32_t)),
        rank_shapes * 16U * 16U * reorder_bz * sizeof(uint16_t),
        2U * sizeof(unsigned int),
    };
    size_t scratch_bytes = 0;
    for (size_t size : scratch_sizes)
        scratch_bytes += (size + 255U) & ~size_t(255U);
    char* scratch = compression_workspace.reserve(scratch_bytes);
    void* buffers[9];
    size_t offset = 0;
    for (size_t i = 0; i < 9; ++i) {
        buffers[i] = scratch + offset;
        offset += (scratch_sizes[i] + 255U) & ~size_t(255U);
    }
    d_data = static_cast<T*>(buffers[0]);
    tmp_data = static_cast<T*>(buffers[1]);
    d_q = use_dyadic ? d_data : static_cast<T*>(buffers[2]);
    d_q_abs = static_cast<uint16_t*>(buffers[3]);
    d_q_sign = static_cast<uint32_t*>(buffers[4]);
    d_qoi = static_cast<uint32_t*>(buffers[5]);
    d_qov = reinterpret_cast<int32_t*>(d_qoi + qout_cap);
    d_poe = static_cast<T*>(buffers[6]);
    d_poi = reinterpret_cast<uint32_t*>(d_poe + pout_cap);
    d_tb = static_cast<uint16_t*>(buffers[7]);
    d_cnt = static_cast<unsigned int*>(buffers[8]);
    d_qoc = d_cnt + 1;

    block_rank_timer.start(stream);
    WALTZ::lossless::d_build_block_rank<<<dim3(reorder_bz, rank_shapes), 256, 0, cuda_stream>>>
        (d_tb, 16U, 16U, reorder_bz, data_dims);
    block_rank_timer.record_stop(stream);

    // The sparse emitter skips zero magnitudes; reconstructed coefficients are
    // still written at every finalized position. Sign bits use atomic OR.
    cudaMemsetAsync(d_q_abs, 0, mag_bytes, cuda_stream);
    cudaMemsetAsync(d_q_sign, 0, sb_bytes, cuda_stream);
    cudaMemsetAsync(d_qoc, 0, sizeof(unsigned int), cuda_stream);

    CDF97::set_quant_params<T>(d_q_abs, d_q_sign, reinterpret_cast<T*>(d_q), d_tb,
        data_dims, reorder_bz, quantity, d_qoi, d_qov, d_qoc, qout_cap, cuda_stream);
    double gpu_predict_ms;
    size_t blob_bytes = 0; // assembled compressed-cmpdata size (returned)
    GPUTimer& timer = timers[1];
    timer.start(stream);
    CDF97::d_reset<<<1, 1, 0, cuda_stream>>>();
    if (use_dyadic) {
        CDF97::dwt3d_xy_halo_z<T>(d_data, tmp_data, data_dims, dyadic_level, reorder_bz,
            cuda_stream, d_oridata, forward_high_flags);
    } else {
        CDF97::dwt3d_plane_xy_halo_z<T>(
            d_data, tmp_data, data_dims, levels_xy, levels_z, d_oridata, reorder_bz, cuda_stream);
    }
    if (forward_high_flags != nullptr)
        map_z_high_flags<<<(inverse_tiles + 255U) / 256U, 256, 0, cuda_stream>>>(
            forward_high_flags, inverse_high_flags, data_dims.x, data_dims.y,
            forward_columns, inverse_columns);
    timer.record_stop(stream);

    // Error-bounded compression: lossless magnitude/sign runs on the main
    // stream while reconstruction runs on an independent stream.
    const double t_quant = 0.0;
    cudaStream_t cuda_stream1;
    cudaEvent_t overlap_done, overlap_begin;
    long long magc;
    double t_mag;

    WALTZ::lossless::wzp_encode(d_q_abs, d_q_sign, datasize, cuda_stream);

    cudaStreamCreateWithFlags(&cuda_stream1, cudaStreamNonBlocking);
    cudaStreamWaitEvent(cuda_stream1, timer.end, 0);
    cudaEventCreate(&overlap_begin);
    cudaEventCreate(&overlap_done);
    cudaMemsetAsync(d_cnt, 0, sizeof(unsigned int), cuda_stream1);
    cudaEventRecord(overlap_begin, cuda_stream1);
    if (use_dyadic) {
        CDF97::idwt3d_xy_halo_z<T>(d_q, tmp_data, data_dims, dyadic_level,
            d_oridata, real_error, d_cnt, d_poi, d_poe, pout_cap, cuda_stream1, inverse_high_flags);
    } else {
        // Supported plane inputs have levels_z >= 1; PWE is fused into Z synthesis.
        CDF97::idwt3d_plane_xy_halo_z<T>(
            d_q, tmp_data, data_dims, levels_xy, levels_z, d_oridata,
            real_error, d_cnt, d_poi, d_poe, pout_cap, false, cuda_stream1);
    }
    cudaEventRecord(overlap_done, cuda_stream1);

    // WZP packs magnitude and sign together in independently decodable groups.
    magc = static_cast<long long>(WALTZ::lossless::wzp_encode_with_sign_finish(nullptr, &t_mag));

    cudaEventSynchronize(overlap_done);
    float overlap_ms = 0.0f;
    cudaEventElapsedTime(&overlap_ms, overlap_begin, overlap_done);
    const double t_idwt = overlap_ms;
    cudaMemcpyAsync(g_host_outlier_counts, d_cnt, sizeof(unsigned int), cudaMemcpyDeviceToHost, cuda_stream);
    cudaMemcpyAsync(g_host_outlier_counts + 1, d_qoc, sizeof(unsigned int), cudaMemcpyDeviceToHost, cuda_stream);
    cudaStreamSynchronize(cuda_stream);
    const unsigned int no = g_host_outlier_counts[0];
    const unsigned int nqo = g_host_outlier_counts[1];
    // Both result-count copies above are synchronization points.
    // Querying the already-complete events here preserves the
    // per-stage kernel timings without inserting bubbles between
    // dependent stages in the actual cE2E path.
    gpu_predict_ms = timer.elapsed_ready();
    cudaEventDestroy(overlap_begin);
    cudaEventDestroy(overlap_done);
    cudaStreamDestroy(cuda_stream1);
    const uint32_t pwe_count = min(no, pout_cap);
    const uint32_t qout_count = min(nqo, qout_cap);
    // ---- assemble the compressed blob into d_cmpData (header + sections) ----
    blob_bytes = Waltz_gather<T>(d_cmpData, use_dyadic, reorder_bz, levels_xy, levels_z, quantity,
        static_cast<size_t>(magc), pwe_count, d_poi, d_poe, qout_count, d_qoi, d_qov, cuda_stream);

    Waltz_print_compression<T>(bytes,
        {block_rank_timer.elapsed_ready(), gpu_predict_ms, t_quant, t_mag, t_idwt, 0.0},
        static_cast<size_t>(magc), pwe_count, qout_count, blob_bytes);

    return blob_bytes;
}


template <typename T, int NDIMS>
size_t d_WALTZ_decompress(
    char* d_cmpData, T* d_decData, const WALTZ::Config& config, const char* decPath, void* stream) {
    if constexpr (NDIMS != 3) {
        throw std::runtime_error("decompress currently only supports 3D");
    }
    const cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    WaltzBlobHeader hdr{};
    if (g_pipeline_host_header == nullptr)
        cudaMallocHost(reinterpret_cast<void**>(&g_pipeline_host_header),
                       sizeof(WaltzBlobHeader));
    cudaMemcpyAsync(g_pipeline_host_header, d_cmpData, sizeof(hdr),
                    cudaMemcpyDeviceToHost, cuda_stream);
    cudaStreamSynchronize(cuda_stream);
    hdr = *g_pipeline_host_header;
    const uint32_t N = config.num;
    dim3 dims(config.dimx, config.dimy, config.dimz);
    const bool dec_use_dyadic = hdr.use_dyadic != 0U;
    const uint32_t dec_quant_bz = hdr.reorder_bz;
    if (dec_quant_bz < 1U || dec_quant_bz > 32U)
        throw std::runtime_error("invalid Waltz reorder z-band");
    const BlobView sections = Waltz_scatter<T>(d_cmpData, hdr);

    const bool full_reorder_bricks = dims.x % 16U == 0U && dims.y % 16U == 0U &&
                                      dims.z % dec_quant_bz == 0U;
    const bool complete_bz2_layout = dec_quant_bz == 2U && full_reorder_bricks;
    const bool needs_offsets = !complete_bz2_layout &&
        !(std::is_same_v<T, float> && full_reorder_bricks && dec_quant_bz == 16U);

    // WZP always writes dequantized row-major coefficients directly, including edges.
    uint32_t* d_tb = nullptr;
    const uint32_t entries = 16U * 16U * dec_quant_bz;
    const uint32_t offset_shapes = full_reorder_bricks ? 1U : 8U;
    struct OffsetCache {
        CompressionWorkspace storage;
        dim3 dims{0U, 0U, 0U};
        uint32_t bz = 0;
        int device = -1;
    };
    static thread_local OffsetCache offsets;
    bool build_offsets = false;
    int current_device = -1;
    if (needs_offsets) {
        cudaGetDevice(&current_device);
        d_tb = reinterpret_cast<uint32_t*>(offsets.storage.reserve(
            static_cast<size_t>(offset_shapes) * entries * sizeof(uint32_t)));
        build_offsets = offsets.device != current_device || offsets.bz != dec_quant_bz ||
            offsets.dims.x != dims.x || offsets.dims.y != dims.y || offsets.dims.z != dims.z;
    }

    if (CDF97::idwt3d_prealloc<T>(dims, dec_use_dyadic, true, cuda_stream) != cudaSuccess)
        throw std::runtime_error("Waltz: cannot configure IDWT workspace");
    static thread_local GPUTimerCache<4> timer_cache;
    GPUTimer* const timers = timer_cache.get();
    GPUTimer& block_offset_timer = timers[0];
    GPUTimer& unquant_timer = timers[1];
    GPUTimer& idwt_timer = timers[2];
    GPUTimer& pwe_timer = timers[3];
    T* const d_coeff = d_decData;

    // A few bytes per XY brick summarize only this decode's nonzero chunks.
    // WZP's metadata prepass overwrites every flag inside the measured API.
    static thread_local CompressionWorkspace zero_column_storage;
    uint32_t* xy_nonzero = nullptr;
    uint32_t zero_columns = 0U;
    if (dec_use_dyadic && static_cast<size_t>(dims.z) * 16U <=
            WALTZ::IDWT_SHARED_BYTES / sizeof(T)) {
        zero_columns = CDF97::idwt_z_shared_bytes<T>() / sizeof(T) / dims.z;
        constexpr uint32_t align_mask = std::is_same_v<T, float> ? 3U : 1U;
        if ((dims.x & align_mask) == 0U && zero_columns > align_mask)
            zero_columns &= ~align_mask;
        const size_t tiles = static_cast<size_t>((dims.x + zero_columns - 1U) / zero_columns) * dims.y;
        // A workload-size heuristic avoids a metadata launch for very short
        // passes. Two SM waves is not a proven break-even threshold.
        const size_t sm_count = WALTZ::gpu_config().properties.multiProcessorCount;
        if (tiles > 2U * sm_count)
            xy_nonzero = reinterpret_cast<uint32_t*>(zero_column_storage.reserve(tiles * sizeof(uint32_t)));
    }

    if (build_offsets) {
        block_offset_timer.start(stream);
        WALTZ::lossless::d_build_block_ioffset<<<dim3(dec_quant_bz, offset_shapes), 256, 0, cuda_stream>>>(
            d_tb, 16U, 16U, dec_quant_bz, dims.x, static_cast<size_t>(dims.x) * dims.y, dims);
        block_offset_timer.record_stop(stream);
        offsets.dims = dims;
        offsets.bz = dec_quant_bz;
        offsets.device = current_device;
    }
    if constexpr (std::is_same_v<T, float>) {
        WALTZ::lossless::wzp_decode_reordered_float(sections.magnitude,
            N, d_coeff, d_tb, static_cast<float>(hdr.q), dims,
            dec_quant_bz, nullptr, cuda_stream, xy_nonzero, zero_columns);
    } else if constexpr (std::is_same_v<T, double>) {
        WALTZ::lossless::wzp_decode_reordered_double(
            sections.magnitude, N, d_coeff, d_tb,
            hdr.q, dims, dec_quant_bz, nullptr, cuda_stream, xy_nonzero, zero_columns);
    }
    if (hdr.n_qo) {
        unquant_timer.start(stream); // Only quantization-outlier patches remain separate.
        const int tb2 = 256, gb2 = static_cast<int>((hdr.n_qo + tb2 - 1) / tb2);
        apply_q_outliers<T><<<gb2, tb2, 0, cuda_stream>>>(d_coeff,
            reinterpret_cast<const uint32_t*>(sections.quant_index),
            reinterpret_cast<const int32_t*>(sections.quant_value), hdr.n_qo, hdr.q);
        unquant_timer.record_stop(stream);
    }
    idwt_timer.start(stream); // 5) IDWT (in place)
    if (dec_use_dyadic) {
        const int level = std::min(hdr.levels_xy, hdr.levels_z);
        CDF97::idwt3d_xy_halo_z<T>(
            d_decData, static_cast<T*>(WALTZ::IDWTConfig<T>::tmp), dims, level,
            nullptr, 0.0, nullptr, nullptr, nullptr, 0, cuda_stream, xy_nonzero);
    } else {
        CDF97::idwt3d_plane_xy_halo_z<T>(
            d_decData, nullptr, dims, hdr.levels_xy, hdr.levels_z,
            nullptr, 0.0, nullptr, nullptr, nullptr, 0, false, cuda_stream);
    }
    idwt_timer.record_stop(stream);
    pwe_timer.start(stream); // 6) PWE outlier corrections
    if (hdr.n_pwe) {
        const int tb2 = 256;
        const int gb2 = static_cast<int>((hdr.n_pwe + tb2 - 1) / tb2);
        apply_pwe_outliers<T><<<gb2, tb2, 0, cuda_stream>>>(d_decData,
                                           reinterpret_cast<const uint32_t*>(sections.pwe_index),
                                           reinterpret_cast<const T*>(sections.pwe_error),
                                           hdr.n_pwe);
    }
    pwe_timer.record_stop(stream);
    cudaStreamSynchronize(cuda_stream);
    // The final stream synchronization above is required by the public decode
    // contract.  Read all interval events only now, so timing instrumentation
    // adds no submission bubbles between decode stages.
    const double t_offset = build_offsets ? block_offset_timer.elapsed_ready() : 0.0;
    const double t_wzp = WALTZ::lossless::wzp_decode_elapsed();
    const double t_uq = hdr.n_qo ? unquant_timer.elapsed_ready() : 0.0;
    const double t_idwt = idwt_timer.elapsed_ready();
    const double t_pw = pwe_timer.elapsed_ready();
    const size_t orig = static_cast<size_t>(N) * sizeof(T);
    Waltz_print_decompression(orig, sections.bytes, {t_offset, t_wzp, t_uq, t_idwt, t_pw});

    if (decPath != nullptr) { // optional host-side dump of the reconstruction
        T* h = new T[N];
        cudaMemcpy(h, d_decData, orig, cudaMemcpyDeviceToHost);
        FILE* f = std::fopen(decPath, "wb");
        if (f != nullptr) {
            std::fwrite(h, sizeof(T), N, f);
            std::fclose(f);
        }
        delete[] h;
    }
    return static_cast<size_t>(N) * sizeof(T);
}

// GPU extrema scan used by callers to resolve REL before entering d_compress.
template <typename T>
void d_extrema(T* d_data, uint64_t n, double& out_min, double& out_max, void* stream) {
    T ext[2] = {T(0), T(0)};
    waltz::extrema_scan<T>(d_data, n, ext, reinterpret_cast<cudaStream_t>(stream));
    out_min = static_cast<double>(ext[0]);
    out_max = static_cast<double>(ext[1]);
}
template void d_extrema<float>(float*, uint64_t, double&, double&, void*);
template void d_extrema<double>(double*, uint64_t, double&, double&, void*);

template size_t d_WALTZ_compress<float, 3>(float*, char*, const WALTZ::Config&, const char*, void*);
template size_t
d_WALTZ_compress<double, 3>(double*, char*, const WALTZ::Config&, const char*, void*);
template size_t
d_WALTZ_decompress<float, 3>(char*, float*, const WALTZ::Config&, const char*, void*);
template size_t
d_WALTZ_decompress<double, 3>(char*, double*, const WALTZ::Config&, const char*, void*);

} // namespace WALTZ
