#include "lossless/reorder.hpp"
#include "lossless/wzp.hpp"
#include "transformers/CDF97.hpp"
#include "transformers/ICDF97.hpp"
#include "transformers/all_halo.cuh"
#include "waltz.hpp"
#include "waltz_cuda.h"
#include "utils/Blob.hpp"
#include "utils/Timer.hpp"
#include "utils/Statistics.hpp"
#include "utils/def.hpp"
#include "small_pwe.cuh"
#include "native_repairs.cuh"
#include <cfenv>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <string>
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

struct BlobSection {
    const void* data;
    size_t bytes;
};

struct GatherPlan {
    BlobSection sections[5];
    size_t offsets[5];
    uint32_t block_ends[5];
    SmallPweHeader small_header{};
    bool prepared_small=false;
};

// One launch assembles the header and all archive sections. Each CTA copies
// one 4 KiB segment, with native aligned accesses and a bounded byte tail.
// Section payloads remain unchanged; this replaces up to six API copy calls.
static __global__ void gather_sections(unsigned char* output,WaltzBlobHeader header,GatherPlan plan) {
    if(blockIdx.x==0 && threadIdx.x==0)
        *reinterpret_cast<WaltzBlobHeader*>(output)=header;
    unsigned section=0;
    while(section<5 && blockIdx.x>=plan.block_ends[section]) ++section;
    if(section==5) return; // Empty archive payload still writes its header.
    const uint32_t begin=section?plan.block_ends[section-1]:0;
    const size_t pos=size_t(blockIdx.x-begin)*4096U+threadIdx.x*16U;
    const size_t bytes=plan.sections[section].bytes;
    const auto* src=static_cast<const unsigned char*>(plan.sections[section].data);
    auto* dst=output+plan.offsets[section];
    // Define the bytes between aligned sections too. These are copied when an
    // archive is downloaded, even though the decoder never consumes them.
    const unsigned padding=unsigned((8U-(bytes&7U))&7U);
    if(blockIdx.x+1==plan.block_ends[section] && threadIdx.x<padding)
        dst[bytes+threadIdx.x]=0;
    if (!src) return; // Sorted directly into this span; only its padding was needed.
    if(section==2 && plan.prepared_small && pos<sizeof(SmallPweHeader)) {
        // The first two threads synthesize the compact header while assembling
        // their disjoint 16-byte chunks. Never read its uninitialized workspace.
        const auto* header_bytes=reinterpret_cast<const unsigned char*>(&plan.small_header);
        for(size_t k=pos;k<bytes && k<pos+16U;++k)
            dst[k]=k<sizeof(SmallPweHeader)?header_bytes[k]:src[k];
        return;
    }
    if(pos+16U<=bytes) {
        if(((reinterpret_cast<uintptr_t>(src)|reinterpret_cast<uintptr_t>(dst))&7U)==0) {
            reinterpret_cast<uint2*>(dst+pos)[0]=reinterpret_cast<const uint2*>(src+pos)[0];
            reinterpret_cast<uint2*>(dst+pos)[1]=reinterpret_cast<const uint2*>(src+pos)[1];
        } else {
            #pragma unroll
            for(unsigned k=0;k<4;++k)
                reinterpret_cast<uint32_t*>(dst+pos)[k]=reinterpret_cast<const uint32_t*>(src+pos)[k];
        }
    } else if(pos<bytes) {
        for(size_t k=pos;k<bytes && k<pos+16U;++k) dst[k]=src[k];
    }
}

static void check_pwe_cuda(cudaError_t status) {
    if (status != cudaSuccess)
        throw std::runtime_error(std::string("Waltz PWE: ") + cudaGetErrorString(status));
}

// The API joins this stream before returning. Reusing its handles does not
// cache reconstruction or coefficients, and avoids per-call create/destroy.
struct CompressionOverlap {
    cudaStream_t stream=nullptr;
    cudaEvent_t begin=nullptr, reconstructed=nullptr, done=nullptr;
    int device=-1;
    void release() {
        if(device<0) return;
        int previous=0;
        if(cudaGetDevice(&previous)==cudaSuccess) {
            cudaSetDevice(device);
            if(begin) cudaEventDestroy(begin);
            if(reconstructed) cudaEventDestroy(reconstructed);
            if(done) cudaEventDestroy(done);
            if(stream) cudaStreamDestroy(stream);
            cudaSetDevice(previous);
        }
        stream=nullptr;begin=nullptr;reconstructed=nullptr;done=nullptr;device=-1;
    }
    ~CompressionOverlap(){release();}
    void ensure() {
        int current=-1;check_pwe_cuda(cudaGetDevice(&current));
        if(current==device && stream && begin && reconstructed && done) return;
        release();device=current;
        check_pwe_cuda(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
        check_pwe_cuda(cudaEventCreate(&begin));
        check_pwe_cuda(cudaEventCreate(&reconstructed));
        check_pwe_cuda(cudaEventCreate(&done));
    }
};
static thread_local CompressionOverlap compression_overlap;

// Keep the tested precision presets explicit. Arbitrary scales could consume
// more than the requested error budget; malformed settings must not silently
// disable correction or select an unvalidated precision.
static double small_pwe_scale() {
    const char* value = std::getenv("WALTZ_SMALL_PWE_SCALE");
    if (value == nullptr)
        return 0.0; // Same native residuals as CPU and HIP by default.
    char* end = nullptr;
    const double scale = std::strtod(value, &end);
    if (end == value || *end != '\0' ||
        (scale != 0.0 && scale != 16.0 && scale != 64.0))
        throw std::invalid_argument("Waltz: WALTZ_SMALL_PWE_SCALE must be 0, 16 or 64");
    return scale;
}

// Compression-only selection. Synthesis is always taken from the archive on
// decode, so an environment change cannot silently change repair arithmetic.
static unsigned halo3d_mode() {
    const char* value=std::getenv("WALTZ_HALO3D");
    // Default uses all-level lifting. Compact CUDA presets can use short plane-Z.
    // Explicit old modes remain available only for diagnostic ablations.
    if(value==nullptr || std::strcmp(value,"auto")==0 || std::strcmp(value,"all")==0 || std::strcmp(value,"both")==0)
        return 7U;
    if(std::strcmp(value,"0")==0) return 0U;
    if(std::strcmp(value,"forward")==0) return 1U;
    if(std::strcmp(value,"inverse")==0) return 2U;
    throw std::invalid_argument("Waltz: invalid WALTZ_HALO3D mode");
}

template<class T>
static bool fused_plane_z_eligible(dim3 dims, int levels_xy, int levels_z, bool dyadic, bool all_levels) {
    if constexpr(std::is_same_v<T,float>) {
        return all_levels && !dyadic && levels_xy>0 && levels_z>=2 &&
            dims.z>16 && dims.z<=128 && (dims.x&3U)==0 && uint64_t(dims.x)*dims.y>=64 &&
            CDF97::dwt_wide_z_supported();
    }
    return false;
}

template <typename T>
static size_t Waltz_gather(void* output, bool use_dyadic,
                     uint32_t reorder_bz, uint8_t levels_xy, uint8_t levels_z,
                     double quantity, size_t mag_bytes,
                     uint32_t pwe_count, const uint32_t* pwe_index, const T* pwe_error,
                     uint32_t qout_count, const uint32_t* quant_index,
                     const int32_t* quant_value, double repair_step, cudaStream_t stream,
                     bool halo_inverse, const unsigned char* prepared_small,
                     uint32_t prepared_bytes, bool all_levels, bool fused_plane_z, bool ordered) {
    WaltzBlobHeader hdr{};
    const auto synthesis = all_levels ? Synthesis::HaloAllLevels :
        halo_inverse ? Synthesis::HaloLifting3D : Synthesis::Convolution;
    hdr.reorder_bz = static_cast<uint8_t>(reorder_bz);
    hdr.levels_xy = levels_xy;
    hdr.levels_z = levels_z;
    hdr.q = quantity;
    hdr.mag_comp_bytes = static_cast<uint64_t>(mag_bytes);
    hdr.n_pwe = pwe_count;
    hdr.n_qo = qout_count;
    const void* error_data = pwe_error;
    size_t error_bytes = size_t(pwe_count) * sizeof(T);
    SmallPweHeader small_header{};
    bool use_prepared=false;
    if (prepared_small && pwe_count) {
        small_header.groups=(pwe_count+31U)/32U;
        small_header.step=repair_step;
        small_header.payload_bytes=prepared_bytes;
        const size_t packed_bytes=sizeof(SmallPweHeader)+size_t(small_header.groups)*sizeof(SmallPweGroup)+prepared_bytes;
        if(packed_bytes<error_bytes) {
            error_data=prepared_small;error_bytes=packed_bytes;
            use_prepared=true;
        }
    }
    hdr.flags = WaltzBlobHeader::pack_flags(use_dyadic, sizeof(T),
        use_prepared ? PweEncoding::Small : PweEncoding::Native, synthesis);
    if (fused_plane_z) hdr.flags |= WaltzBlobHeader::CudaFusedPlaneZ;
    hdr.pwe_bytes = static_cast<uint32_t>(error_bytes);
    const BlobSection sections[] = {
        {lossless::wzp_encoded_buf(), mag_bytes},
        {pwe_index, pwe_count * sizeof(uint32_t)},
        {error_data, error_bytes},
        {quant_index, qout_count * sizeof(uint32_t)},
        {quant_value, qout_count * sizeof(int32_t)},
    };
    auto* cmpdata = static_cast<unsigned char*>(output);
    const auto layout=BlobLayout::checked(hdr);
    GatherPlan plan{{}, {BlobSize,layout.pi,layout.pv,layout.qi,layout.qv}};
    plan.small_header=small_header;plan.prepared_small=use_prepared;
    uint32_t blocks=0;
    for (unsigned i=0;i<5;++i) {
        plan.sections[i]=sections[i];
        const bool sorted = ordered && ((i==1 || i==2) ? pwe_count>1 : i>=3 && qout_count>1);
        if (sorted) plan.sections[i].data=nullptr;
        blocks+=sorted ? unsigned((sections[i].bytes&7U)!=0)
                       : static_cast<uint32_t>((sections[i].bytes+4095U)/4096U);
        plan.block_ends[i]=blocks;
    }
    static thread_local CompressionWorkspace sort_scratch;
    if (ordered && pwe_count>1)
        check_pwe_cuda(native_repairs::write_pairs(pwe_index,pwe_error,pwe_count,
            cmpdata+plan.offsets[1],cmpdata+plan.offsets[2],sort_scratch,stream));
    if (ordered && qout_count>1)
        check_pwe_cuda(native_repairs::write_pairs(quant_index,quant_value,qout_count,
            cmpdata+plan.offsets[3],cmpdata+plan.offsets[4],sort_scratch,stream));
    gather_sections<<<std::max(blocks,1U),256,0,stream>>>(cmpdata,hdr,plan);
    check_pwe_cuda(cudaGetLastError());
    check_pwe_cuda(cudaStreamSynchronize(stream));
    return layout.end;
}

struct BlobView {
    const unsigned char* magnitude;
    const unsigned char* pwe_index;
    const unsigned char* pwe_error;
    const unsigned char* quant_index;
    const unsigned char* quant_value;
    size_t bytes;
};

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

void calculate_dwt_levels(dim3 data_dims, uint8_t& levels_xy, uint8_t& levels_z) {
    levels_xy = dwt_levels(std::min(data_dims.x, data_dims.y));
    levels_z = dwt_levels(data_dims.z);
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
    const double repair_scale = small_pwe_scale();
    const unsigned requested_halo = halo3d_mode();
    const bool all_levels=(requested_halo&4U)!=0;
    const bool ordered=all_levels && repair_scale==0.0;
    if (!config.dimx || !config.dimy || !config.dimz || config.dimx>UINT32_MAX/2U ||
        config.dimy>(UINT32_MAX/2U)/config.dimx ||
        config.dimz>(UINT32_MAX/2U)/(config.dimx*config.dimy) ||
        config.num!=config.dimx*config.dimy*config.dimz || !d_oridata || !d_cmpData)
        throw std::invalid_argument("Waltz: invalid buffers or shape for WZP");
    const uintptr_t src_address=reinterpret_cast<uintptr_t>(d_oridata);
    const uintptr_t dst_address=reinterpret_cast<uintptr_t>(d_cmpData);
    if (src_address<=dst_address ? dst_address-src_address<config.num*sizeof(T)
                                 : src_address-dst_address<compressed_buffer_capacity<T>(config.num))
        throw std::invalid_argument("Waltz: input and output buffers overlap");
    if (!(real_error>0) || !std::isfinite(quantity) || !(T(quantity)>0) ||
        !std::isfinite(T(quantity)) || !std::isfinite(T(1.0/quantity)) ||
        (ordered && std::fegetround()!=FE_TONEAREST))
        throw std::invalid_argument("Waltz: invalid bound or rounding mode");

    const cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    if (g_host_outlier_counts == nullptr)
        check_pwe_cuda(cudaMallocHost(reinterpret_cast<void**>(&g_host_outlier_counts), 4U * sizeof(unsigned int)));

    uint8_t levels_xy = 0, levels_z = 0;
    calculate_dwt_levels(data_dims, levels_xy, levels_z);
    // SPERR's dimension-only rule: equal levels use dyadic, otherwise XY+Z.
    // Both level counts are capped at 5, covering SPERR's >=5 exception too.
    const bool use_dyadic = levels_xy == levels_z;
    // The canonical CPU/CUDA/HIP encoder profile requires native residuals
    // (scale 0) and bit-identical all-level lifting, not CUDA-only convolution.
    const bool fused_plane_z=repair_scale>0.0 &&
        fused_plane_z_eligible<T>(data_dims,levels_xy,levels_z,use_dyadic,all_levels);
    const uint8_t dyadic_level = std::min(levels_xy, levels_z);
    const bool halo_eligible = !all_levels && use_dyadic && dyadic_level>0 &&
        data_dims.x>=2U && data_dims.y>=2U && data_dims.z>=2U &&
        WALTZ::gpu_config().properties.sharedMemPerBlockOptin>=65536U;
    const bool halo_forward = halo_eligible && (requested_halo & 1U);
    const bool halo_inverse = halo_eligible && (requested_halo & 2U);
    if(halo_inverse) check_pwe_cuda(CDF97::configure_halo3d_inverse<T>());
    if(all_levels && use_dyadic && dyadic_level && WALTZ::gpu_config().properties.sharedMemPerBlockOptin<65536U)
        throw std::runtime_error("Waltz: all-level halo requires 64 KiB opt-in shared memory");

    if (!all_levels && !use_dyadic && levels_z > 0 &&
        data_dims.z > CDF97::DWT_Z_STATIC_BYTES / (2U * sizeof(T)))
        throw std::invalid_argument("Waltz: plane Z column exceeds the 45 KiB static shared-memory capacity");

    const uint32_t reorder_bz = WALTZ::lossless::select_block_zband(data_dims, levels_z, use_dyadic != 0U);
    const uint32_t rank_shapes = (data_dims.x % 16U || data_dims.y % 16U || data_dims.z % reorder_bz) ? 8U : 1U;

    if(!all_levels || fused_plane_z) CDF97::dwt3d_prealloc<T>();
    if ((!all_levels || fused_plane_z) && CDF97::idwt3d_prealloc<T>(data_dims, use_dyadic != 0U, false, cuda_stream) != cudaSuccess)
        throw std::runtime_error("Waltz: cannot configure IDWT workspace");

    uint32_t* forward_high_flags = nullptr;
    uint32_t* inverse_high_flags = nullptr;
    uint32_t forward_columns = 0U, inverse_columns = 0U, inverse_tiles = 0U;
    if constexpr (std::is_same_v<T, double>) {
        // The exact low-only synthesis path requires aligned double2 columns.
        // Keep both the forward and inverse shared/global dispatch unchanged.
        if (!all_levels && !halo_forward && !halo_inverse && use_dyadic && levels_z > 0 && (data_dims.x & 1U) == 0U &&
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
        bytes, bytes, use_dyadic && !all_levels ? 0U : bytes, mag_bytes, sb_bytes,
        static_cast<size_t>(qout_cap) * (sizeof(uint32_t) + sizeof(int32_t)),
        static_cast<size_t>(pout_cap) * (sizeof(T) + sizeof(uint32_t)),
        0U, // Geometry rank lookup now has its own cache.
        4U * sizeof(unsigned int), // PWE count, Q-outlier count, repair status/size, WZP payload
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
    d_q = use_dyadic && !all_levels ? d_data : static_cast<T*>(buffers[2]);
    d_q_abs = static_cast<uint16_t*>(buffers[3]);
    d_q_sign = static_cast<uint32_t*>(buffers[4]);
    d_qoi = static_cast<uint32_t*>(buffers[5]);
    d_qov = reinterpret_cast<int32_t*>(d_qoi + qout_cap);
    d_poe = static_cast<T*>(buffers[6]);
    d_poi = reinterpret_cast<uint32_t*>(d_poe + pout_cap);
    d_tb = static_cast<uint16_t*>(buffers[7]);
    d_cnt = static_cast<unsigned int*>(buffers[8]);
    d_qoc = d_cnt + 1;
    auto* packed_counter=d_cnt+2;
    const double repair_step=repair_scale>0.0?real_error/repair_scale:0.0;
    const bool async_small=repair_step>0.0 && std::isfinite(repair_step);
    unsigned char* packed_storage=nullptr;
    if(async_small) {
        static thread_local CompressionWorkspace packed_workspace;
        const size_t max_bytes=sizeof(SmallPweHeader)+size_t((pout_cap+31U)/32U)*sizeof(SmallPweGroup)+
            size_t(pout_cap)*sizeof(T)+8U;
        packed_storage=reinterpret_cast<unsigned char*>(packed_workspace.reserve(max_bytes));
    }

    // Geometry-only lookup cache. Content-dependent coefficients/results are
    // recomputed on every call. API synchronization makes cross-stream reuse safe.
    struct RankCache {
        CompressionWorkspace storage;
        dim3 dims{0,0,0};
        uint32_t bz=0;
        int device=-1;
    };
    static thread_local RankCache rank_cache;
    int rank_device=-1;check_pwe_cuda(cudaGetDevice(&rank_device));
    d_tb=reinterpret_cast<uint16_t*>(rank_cache.storage.reserve(
        size_t(rank_shapes)*16U*16U*reorder_bz*sizeof(uint16_t)));
    const bool build_rank=rank_cache.device!=rank_device || rank_cache.bz!=reorder_bz ||
        rank_cache.dims.x!=data_dims.x || rank_cache.dims.y!=data_dims.y || rank_cache.dims.z!=data_dims.z;
    if(build_rank) {
    block_rank_timer.start(stream);
    WALTZ::lossless::d_build_block_rank<<<dim3(reorder_bz, rank_shapes), 256, 0, cuda_stream>>>
        (d_tb, 16U, 16U, reorder_bz, data_dims);
    block_rank_timer.record_stop(stream);
    rank_cache.device=rank_device;rank_cache.bz=reorder_bz;rank_cache.dims=data_dims;
    }

    // The sparse emitter skips zero magnitudes; reconstructed coefficients are
    // still written at every finalized position. Sign bits use atomic OR.
    cudaMemsetAsync(d_q_abs, 0, mag_bytes, cuda_stream);
    cudaMemsetAsync(d_q_sign, 0, sb_bytes, cuda_stream);
    cudaMemsetAsync(d_qoc, 0, sizeof(unsigned int), cuda_stream);
    cudaMemsetAsync(packed_counter,0,sizeof(unsigned int),cuda_stream);

    CDF97::set_quant_params<T>(d_q_abs, d_q_sign, reinterpret_cast<T*>(d_q), d_tb,
        data_dims, reorder_bz, quantity, d_qoi, d_qov, d_qoc, qout_cap, cuda_stream);
    double gpu_predict_ms;
    size_t blob_bytes = 0; // assembled compressed-cmpdata size (returned)
    GPUTimer& timer = timers[1];
    timer.start(stream);
    if(all_levels) {
        // The regular Morton emitter replaces Z division/modulo with bit
        // operations. Full bricks alone are insufficient: BZ must be a power
        // of two (e.g. 192^3 selects BZ=6 despite having no edge bricks).
        const bool regular=reorder_bz && (reorder_bz&(reorder_bz-1U))==0 &&
            (data_dims.x&15U)==0 && (data_dims.y&15U)==0 && data_dims.z%reorder_bz==0;
        if(regular) allhalo::forward<T,true>(d_oridata,d_data,tmp_data,d_q,data_dims,levels_xy,levels_z,use_dyadic,cuda_stream,fused_plane_z);
        else allhalo::forward<T,false>(d_oridata,d_data,tmp_data,d_q,data_dims,levels_xy,levels_z,use_dyadic,cuda_stream,fused_plane_z);
        check_pwe_cuda(cudaGetLastError());
    } else if (use_dyadic) {
        CDF97::d_reset<<<1, 1, 0, cuda_stream>>>();
        CDF97::dwt3d_xy_halo_z<T>(d_data, tmp_data, data_dims, dyadic_level, reorder_bz,
            cuda_stream, d_oridata, forward_high_flags, halo_forward);
    } else {
        CDF97::d_reset<<<1, 1, 0, cuda_stream>>>();
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
    compression_overlap.ensure();
    const cudaStream_t cuda_stream1=compression_overlap.stream;
    const cudaEvent_t overlap_done=compression_overlap.done, overlap_begin=compression_overlap.begin;
    const size_t wzp_metadata = WALTZ::lossless::wzp_enqueue(d_q_abs,d_q_sign,datasize,d_cnt+3,cuda_stream);

    cudaStreamWaitEvent(cuda_stream1, timer.end, 0);
    cudaMemsetAsync(d_cnt, 0, sizeof(unsigned int), cuda_stream1);
    cudaEventRecord(overlap_begin, cuda_stream1);
    if(ordered) {
        // Native repairs do not use the compact-payload counter. Check the
        // reconstructed value here; never decompress the completed archive.
        allhalo::inverse(d_q,d_data,tmp_data,data_dims,levels_xy,levels_z,use_dyadic,
            native_repairs::Emit<T>{d_oridata,real_error,T(real_error),d_cnt,d_poi,d_poe,pout_cap,packed_counter},
            cuda_stream1);
        check_pwe_cuda(cudaGetLastError());
    } else if(all_levels) {
        allhalo::inverse(
            d_q,d_data,tmp_data,data_dims,levels_xy,levels_z,use_dyadic,
            CDF97::Halo3DSynthEmit<T,true>{nullptr,d_oridata,real_error,std::nextafter(float(real_error),0.0f),d_cnt,d_poi,d_poe,pout_cap},cuda_stream1,fused_plane_z);
        check_pwe_cuda(cudaGetLastError());
    } else if (use_dyadic) {
        check_pwe_cuda(CDF97::idwt3d_xy_halo_z<T>(d_q, tmp_data, data_dims, dyadic_level,
            d_oridata, real_error, d_cnt, d_poi, d_poe, pout_cap, cuda_stream1, inverse_high_flags, halo_inverse));
    } else {
        // Supported plane inputs have levels_z >= 1; PWE is fused into Z synthesis.
        CDF97::idwt3d_plane_xy_halo_z<T>(
            d_q, tmp_data, data_dims, levels_xy, levels_z, d_oridata,
            real_error, d_cnt, d_poi, d_poe, pout_cap, false, cuda_stream1);
    }
    cudaEventRecord(compression_overlap.reconstructed,cuda_stream1);
    if(async_small) {
        const unsigned blocks=std::min((pout_cap+255U)/256U,
            2U*unsigned(WALTZ::gpu_config().properties.multiProcessorCount));
        pack_small_pwe_async<T><<<blocks,256,0,cuda_stream1>>>(
            d_oridata,d_poi,d_poe,d_cnt,pout_cap,repair_step,packed_storage,packed_counter);
        check_pwe_cuda(cudaGetLastError());
    }
    cudaEventRecord(overlap_done, cuda_stream1);

    check_pwe_cuda(cudaStreamWaitEvent(cuda_stream,overlap_done,0));
    check_pwe_cuda(cudaMemcpyAsync(g_host_outlier_counts,d_cnt,4U*sizeof(unsigned int),cudaMemcpyDeviceToHost,cuda_stream));
    check_pwe_cuda(cudaStreamSynchronize(cuda_stream));
    const size_t magc = wzp_metadata + g_host_outlier_counts[3];
    const double t_mag = WALTZ::lossless::wzp_elapsed();
    float overlap_ms = 0.0f;
    cudaEventElapsedTime(&overlap_ms, overlap_begin, compression_overlap.reconstructed);
    const double t_idwt = overlap_ms;
    float pack_ms=0.0f;
    cudaEventElapsedTime(&pack_ms,compression_overlap.reconstructed,overlap_done);
    const unsigned int no = g_host_outlier_counts[0];
    const unsigned int nqo = g_host_outlier_counts[1];
    if (ordered && (g_host_outlier_counts[2] || no>pout_cap || nqo>qout_cap))
        throw std::runtime_error("Waltz: nonfinite input, pointwise bound failure or repair capacity exceeded");
    // The contiguous result-count copy above is a synchronization point.
    // Querying the already-complete events here preserves the
    // per-stage kernel timings without inserting bubbles between
    // dependent stages in the actual cE2E path.
    gpu_predict_ms = timer.elapsed_ready();
    const uint32_t pwe_count = min(no, pout_cap);
    const uint32_t qout_count = min(nqo, qout_cap);
    // ---- assemble the compressed blob into d_cmpData (header + sections) ----
    blob_bytes = Waltz_gather<T>(d_cmpData, use_dyadic, reorder_bz, levels_xy, levels_z, quantity,
        static_cast<size_t>(magc), pwe_count, d_poi, d_poe, qout_count, d_qoi, d_qov,
        repair_step,cuda_stream,halo_inverse,packed_storage,g_host_outlier_counts[2],all_levels,fused_plane_z,ordered);

    Waltz_print_compression<T>(bytes,
        {build_rank ? block_rank_timer.elapsed_ready() : 0.0, gpu_predict_ms, t_quant, t_mag, t_idwt, double(pack_ms)},
        static_cast<size_t>(magc), pwe_count, qout_count, blob_bytes);

    return blob_bytes;
}


template <typename T, int NDIMS>
static size_t decode_impl(
    char* d_cmpData, T* d_decData, const WALTZ::Config& config, const char* decPath,
    void* stream, size_t canonical_bytes) {
    if constexpr (NDIMS != 3) {
        throw std::runtime_error("decompress currently only supports 3D");
    }
    const cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    WaltzBlobHeader hdr{};
    if (g_pipeline_host_header == nullptr)
        check_pwe_cuda(cudaMallocHost(reinterpret_cast<void**>(&g_pipeline_host_header),
                                     sizeof(WaltzBlobHeader)));
    check_pwe_cuda(cudaMemcpyAsync(g_pipeline_host_header, d_cmpData, sizeof(hdr),
                                   cudaMemcpyDeviceToHost, cuda_stream));
    check_pwe_cuda(cudaStreamSynchronize(cuda_stream));
    hdr = *g_pipeline_host_header;
    const auto layout=BlobLayout::checked(hdr);
    const uint32_t N = config.num;
    dim3 dims(config.dimx, config.dimy, config.dimz);
    if (canonical_bytes) {
        uint8_t xy, z;
        calculate_dwt_levels(dims, xy, z);
        const bool dyadic = xy == z;
        const size_t chunks = (size_t(N) + 8191) / 8192;
        if (hdr.flags != WaltzBlobHeader::pack_flags(dyadic, sizeof(T), PweEncoding::Native,
                                                     Synthesis::HaloAllLevels) ||
            hdr.levels_xy != xy || hdr.levels_z != z ||
            hdr.reorder_bz != lossless::select_block_zband(dims, z, dyadic) ||
            !(T(hdr.q) > 0) || !std::isfinite(T(hdr.q)) ||
            hdr.mag_comp_bytes < 3*chunks + 4*((chunks+15)/16) ||
            hdr.mag_comp_bytes > canonical_bytes - BlobSize ||
            hdr.n_pwe > N || hdr.n_qo > N || hdr.pwe_bytes != size_t(hdr.n_pwe)*sizeof(T) ||
            canonical_bytes != layout.end)
            throw std::runtime_error("invalid canonical Waltz chunk shape, header or length");
    }
    const bool fused_plane_z=(hdr.flags&WaltzBlobHeader::CudaFusedPlaneZ)!=0;
    hdr.flags &= ~WaltzBlobHeader::CudaFusedPlaneZ;
    if (!WaltzBlobHeader::valid_flags(hdr.flags) || hdr.element_bytes() != sizeof(T) ||
        hdr.n_pwe > N || hdr.n_qo > N)
        throw std::runtime_error("invalid Waltz archive header");
    if (hdr.pwe_encoding() == PweEncoding::Small) {
        const size_t prefix = sizeof(SmallPweHeader) +
            ((size_t(hdr.n_pwe) + 31U) / 32U) * sizeof(SmallPweGroup);
        if (!hdr.n_pwe || hdr.pwe_bytes < prefix ||
            hdr.pwe_bytes >= size_t(hdr.n_pwe) * sizeof(T))
            throw std::runtime_error("invalid Waltz compact PWE size");
    } else if (hdr.pwe_bytes != size_t(hdr.n_pwe) * sizeof(T))
        throw std::runtime_error("invalid Waltz native PWE section");
    const bool dec_use_dyadic = hdr.use_dyadic();
    const bool all_levels=hdr.synthesis()==Synthesis::HaloAllLevels;
    const bool halo_inverse = hdr.synthesis() == Synthesis::HaloLifting3D;
    if(all_levels && (hdr.levels_xy>CDF97_MAX_LEVELS || hdr.levels_z>CDF97_MAX_LEVELS ||
        (dec_use_dyadic && hdr.levels_xy!=hdr.levels_z)))
        throw std::runtime_error("Waltz: invalid all-level halo metadata");
    if(fused_plane_z) {
        uint8_t expected_xy,expected_z;
        calculate_dwt_levels(dims,expected_xy,expected_z);
        if(hdr.levels_xy!=expected_xy || hdr.levels_z!=expected_z || expected_xy==expected_z ||
           !fused_plane_z_eligible<T>(dims,expected_xy,expected_z,dec_use_dyadic,all_levels))
            throw std::runtime_error("Waltz: invalid short plane-Z metadata or unsupported GPU");
    }
    if(halo_inverse && (!dec_use_dyadic ||
        !hdr.levels_xy || !hdr.levels_z || dims.x<2U || dims.y<2U || dims.z<2U))
        throw std::runtime_error("Waltz: invalid halo synthesis metadata");
    if(halo_inverse) check_pwe_cuda(CDF97::configure_halo3d_inverse<T>());
    const uint32_t dec_quant_bz = hdr.reorder_bz;
    if (dec_quant_bz < 1U || dec_quant_bz > 32U)
        throw std::runtime_error("invalid Waltz reorder z-band");
    const auto* src=reinterpret_cast<const unsigned char*>(d_cmpData);
    const BlobView sections{src+BlobSize,src+layout.pi,src+layout.pv,src+layout.qi,src+layout.qv,layout.end};

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

    if ((!all_levels || fused_plane_z) &&
        CDF97::idwt3d_prealloc<T>(dims, dec_use_dyadic, !all_levels, cuda_stream) != cudaSuccess)
        throw std::runtime_error("Waltz: cannot configure IDWT workspace");
    static thread_local GPUTimerCache<4> timer_cache;
    GPUTimer* const timers = timer_cache.get();
    GPUTimer& block_offset_timer = timers[0];
    GPUTimer& unquant_timer = timers[1];
    GPUTimer& idwt_timer = timers[2];
    GPUTimer& pwe_timer = timers[3];
    // All-level synthesis keeps coefficients and coarse work separate from
    // the caller output. The explicit finest-only diagnostic retains its
    // previous IDWT workspace. Neither path permits in-place halo races.
    static thread_local CompressionWorkspace all_decode_storage;
    const size_t native_bytes=size_t(N)*sizeof(T);
    T* all_scratch=nullptr;
    if(all_levels)all_scratch=reinterpret_cast<T*>(all_decode_storage.reserve(2*native_bytes));
    T* const d_coeff=all_levels ? all_scratch : halo_inverse ? static_cast<T*>(WALTZ::IDWTConfig<T>::tmp) : d_decData;

    // A few bytes per XY brick summarize only this decode's nonzero chunks.
    // WZP's metadata prepass overwrites every flag inside the measured API.
    static thread_local CompressionWorkspace zero_column_storage;
    uint32_t* xy_nonzero = nullptr;
    uint32_t zero_columns = 0U;
    if (!all_levels && !halo_inverse && dec_use_dyadic && static_cast<size_t>(dims.z) * 16U <=
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
    if(all_levels) {
        allhalo::inverse(
            d_coeff,all_scratch+N,d_decData,dims,hdr.levels_xy,hdr.levels_z,dec_use_dyadic,
            allhalo::Store<T>{d_decData},cuda_stream,fused_plane_z);
        check_pwe_cuda(cudaGetLastError());
    } else if (dec_use_dyadic) {
        const int level = std::min(hdr.levels_xy, hdr.levels_z);
        check_pwe_cuda(CDF97::idwt3d_xy_halo_z<T>(
            d_coeff, halo_inverse ? d_decData : static_cast<T*>(WALTZ::IDWTConfig<T>::tmp), dims, level,
            nullptr, 0.0, nullptr, nullptr, nullptr, 0, cuda_stream, xy_nonzero, halo_inverse));
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
        if (hdr.pwe_encoding() == PweEncoding::Small) {
            apply_small_pwe<T><<<gb2, tb2, 0, cuda_stream>>>(d_decData,
                reinterpret_cast<const uint32_t*>(sections.pwe_index),
                sections.pwe_error,hdr.n_pwe,hdr.pwe_bytes);
        } else apply_pwe_outliers<T><<<gb2, tb2, 0, cuda_stream>>>(d_decData,
                                           reinterpret_cast<const uint32_t*>(sections.pwe_index),
                                           reinterpret_cast<const T*>(sections.pwe_error),
                                           hdr.n_pwe);
    }
    pwe_timer.record_stop(stream);
    check_pwe_cuda(cudaStreamSynchronize(cuda_stream));
    // The final stream synchronization above is required by the public decode
    // contract.  Read all interval events only now, so timing instrumentation
    // adds no submission bubbles between decode stages.
    const double t_offset = build_offsets ? block_offset_timer.elapsed_ready() : 0.0;
    const double t_wzp = WALTZ::lossless::wzp_decode_elapsed();
    const double t_uq = hdr.n_qo ? unquant_timer.elapsed_ready() : 0.0;
    const double t_idwt = idwt_timer.elapsed_ready();
    const double t_pw = pwe_timer.elapsed_ready();
    const size_t orig = static_cast<size_t>(N) * sizeof(T);
    if (!canonical_bytes)
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

template <typename T, int NDIMS>
size_t d_WALTZ_decompress(char* archive, T* output, const Config& cfg,
                         const char* path, void* stream) {
    return decode_impl<T,NDIMS>(archive, output, cfg, path, stream, 0);
}

extern "C" const char* waltz_cuda_decode_chunk(uint32_t nx, uint32_t ny, uint32_t nz, int width,
        const void* archive, size_t bytes, void* output, size_t out_bytes, void* stream) {
    static thread_local char error[256];
    static std::mutex lock;
    try {
        std::lock_guard<std::mutex> guard(lock);
        if ((width != 4 && width != 8) || !nx || !ny || !nz || !archive || !output ||
            bytes < BlobSize || uint64_t(nx)*ny > UINT32_MAX / unsigned(width) / nz ||
            out_bytes != uint64_t(nx)*ny*nz*width)
            throw std::invalid_argument("invalid chunk dimensions or buffer sizes");
        int device;
        check_pwe_cuda(cudaGetDevice(&device));
        if (device != gpu_config().device)
            throw std::runtime_error("use one CUDA device per Waltz training process");
        // The variadic constructor collapses singleton axes. Keep chunk geometry.
        Config cfg;
        cfg.ndims = 3; cfg.dimx = nx; cfg.dimy = ny; cfg.dimz = nz;
        cfg.num = uint64_t(nx)*ny*nz;
        auto* src = const_cast<char*>(static_cast<const char*>(archive));
        if (width == 4) decode_impl<float,3>(src, static_cast<float*>(output), cfg, nullptr, stream, bytes);
        else decode_impl<double,3>(src, static_cast<double*>(output), cfg, nullptr, stream, bytes);
        return nullptr;
    } catch (const std::exception& e) {
        std::snprintf(error, sizeof(error), "%s", e.what());
    } catch (...) {
        std::snprintf(error, sizeof(error), "unknown CUDA decode error");
    }
    (void)cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(stream));
    return error;
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

#include "batch_decode.cuh"
