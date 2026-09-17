#include "lossless/reorder.hpp"
#include "lossless/wzp.hpp"
#include "transformers/CDF97.hpp"
#include "transformers/ICDF97.hpp"
#include "waltz.hpp"
#include "utils/Blob.hpp"
#include "utils/Timer.hpp"
#include "utils/Statistics.hpp"
#include "utils/def.hpp"

#include <cmath>
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

// Fast autotune statistics over the whole volume at compile-time stride STEP.
// Adaptive sampling starts at 8 and refines uncertain decisions at stride 4. Sampled points keep
// their +-1 ORIGINAL- resolution neighbours (subsample-then-diff would measure lag-4 statistics).
// acc[0..5] = sum gx2, gy2, gz2, gxgy, gxgz, gygz   (gradient structure tensor)
// nzc       = #{ max(|gx|,|gy|,|gz|) < ctau }       (near-zero-gradient fraction:
//             "background that quantizes to zero at this bound" -- needs NO field
//             min/range, only the known absolute bound; ctau = 2*tau)
template <typename T, uint32_t STEP>
__global__ void d_isotropy_stats(const T* __restrict__ vol, uint32_t nx, uint32_t ny,
    uint32_t nz, double ctau, double* acc, unsigned int* nzc) {
    static_assert(STEP == 4 || STEP == 8, "autotune sampling stride must be 4 or 8");
    const uint32_t isx = (nx - 2 + STEP - 1) / STEP;
    const uint32_t isy = (ny - 2 + STEP - 1) / STEP;
    const uint32_t isz = (nz - 2 + STEP - 1) / STEP;
    const uint64_t total = static_cast<uint64_t>(isx) * isy * isz;
    const uint32_t stride = static_cast<uint32_t>(gridDim.x) * blockDim.x;
    const uint64_t sy = nx, sz = static_cast<uint64_t>(nx) * ny;
    double l[6] = {0, 0, 0, 0, 0, 0};
    unsigned int lnz = 0;

    const uint64_t isxy = static_cast<uint64_t>(isx) * isy;
    for (uint64_t g = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x; g < total;
         g += stride) {
        const uint32_t lx = 1u + STEP * (g % isx);
        const uint32_t ly = 1u + STEP * ((g / isx) % isy);
        const uint32_t lzc = 1u + STEP * (g / isxy);

        const uint64_t base = lx + ly * sy + lzc * sz;

        const T vxp = vol[base + 1];
        const T vyp = vol[base + sy];
        const T vzp = vol[base + sz];

        const T gx = 0.5 * (vxp - vol[base - 1]);
        const T gy = 0.5 * (vyp - vol[base - sy]);
        const T gz = 0.5 * (vzp - vol[base - sz]);

        l[0] += gx * gx;
        l[1] += gy * gy;
        l[2] += gz * gz;
        l[3] += gx * gy;
        l[4] += gx * gz;
        l[5] += gy * gz;
        const double ga = fabs(gx) > fabs(gy) ? fabs(gx) : fabs(gy);
        if ((ga > fabs(gz) ? ga : fabs(gz)) < ctau)
            ++lnz;
    }
    // warp-shuffle tree reduction (same-address shared atomics serialize 256-way on
    // dense fields: measured 3.6x slowdown vs sparse on identical dims)
    const unsigned int FULL = 0xFFFFFFFFu;
    #pragma unroll
    for (int i = 0; i < 6; ++i)
        for (int o = 16; o > 0; o >>= 1)
            l[i] += __shfl_down_sync(FULL, l[i], o);
    for (int o = 16; o > 0; o >>= 1)
        lnz += __shfl_down_sync(FULL, lnz, o);
    __shared__ double swarp[8][6];
    __shared__ unsigned int snz[8];
    const int wid = threadIdx.x >> 5, lane = threadIdx.x & 31;
    if (lane == 0) {
        for (int i = 0; i < 6; ++i)
            swarp[wid][i] = l[i];
        snz[wid] = lnz;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        double t[6] = {0, 0, 0, 0, 0, 0};
        unsigned int z = 0;
        const int nw = blockDim.x >> 5;
        for (int w = 0; w < nw; ++w) {
            for (int i = 0; i < 6; ++i)
                t[i] += swarp[w][i];
            z += snz[w];
        }
        for (int i = 0; i < 6; ++i)
            if (t[i] != 0.0)
                atomicAdd(&acc[i], t[i]);
        if (z)
            atomicAdd(nzc, z);
    }
}

// Tiny compression-result copies use this reusable pinned array so they can be
// queued together and completed by one synchronization.
static unsigned int* g_host_outlier_counts = nullptr;
static WaltzBlobHeader* g_pipeline_host_header = nullptr;
static double* g_tune_acc = nullptr;
static unsigned int* g_tune_nzc = nullptr;
static double* g_tune_host_acc = nullptr;
static unsigned int* g_tune_host_nzc = nullptr;
static cudaEvent_t g_tune_begin = nullptr;
static cudaEvent_t g_tune_end = nullptr;
constexpr size_t TUNE_RESULT_BYTES = 6U * sizeof(double) + sizeof(unsigned int);
void calculate_dwt_levels(dim3 data_dims, uint8_t& levels_xy, uint8_t& levels_z);

static void reserve_tuner() {
    if (g_tune_acc == nullptr) {
        cudaMalloc(reinterpret_cast<void**>(&g_tune_acc), TUNE_RESULT_BYTES);
        g_tune_nzc = reinterpret_cast<unsigned int*>(g_tune_acc + 6);
    }
    if (g_tune_host_acc == nullptr) {
        cudaMallocHost(reinterpret_cast<void**>(&g_tune_host_acc), TUNE_RESULT_BYTES);
        g_tune_host_nzc = reinterpret_cast<unsigned int*>(g_tune_host_acc + 6);
    }
    if (g_tune_begin == nullptr) {
        cudaEventCreate(&g_tune_begin);
        cudaEventCreate(&g_tune_end);
    }
}

template <typename T, uint32_t STEP>
static void sample_isotropy(const T* d_data, dim3 dims, double ctau,
    cudaStream_t st, double& cs, double& nz0, double& time_ms) {
    static_assert(STEP == 4u || STEP == 8u, "autotune sampling stride must be 4 or 8");

    const uint32_t isx = (dims.x - 2u + STEP - 1u) / STEP;
    const uint32_t isy = (dims.y - 2u + STEP - 1u) / STEP;
    const uint32_t isz = (dims.z - 2u + STEP - 1u) / STEP;
    const size_t points = static_cast<size_t>(isx) * isy * isz;
    const int tb = 256;
    const int gb = std::min(static_cast<int>((points + tb - 1) / tb), 4096);

    reserve_tuner();

    cudaEventRecord(g_tune_begin, st);
    cudaMemsetAsync(g_tune_acc, 0, TUNE_RESULT_BYTES, st);
    d_isotropy_stats<T, STEP><<<gb, tb, 0, st>>>(
        d_data, dims.x, dims.y, dims.z, ctau, g_tune_acc, g_tune_nzc);
    cudaMemcpyAsync(g_tune_host_acc, g_tune_acc, TUNE_RESULT_BYTES,
                    cudaMemcpyDeviceToHost, st);
    cudaEventRecord(g_tune_end, st);
    // This sample is only a few microseconds long. cudaEventSynchronize can
    // spend another 0.1--0.2 ms putting the host thread to sleep and waking it
    // again, which used to dominate the tuner. Poll this one short event so the
    // topology decision is available as soon as the D2H result completes.
    while (cudaEventQuery(g_tune_end) == cudaErrorNotReady) {
    }
    float elapsed_ms = 0.0f;
    cudaEventElapsedTime(&elapsed_ms, g_tune_begin, g_tune_end);
    time_ms = elapsed_ms;

    cs = 0.0;
    const double pts = static_cast<double>(points);
    const double Jxx = g_tune_host_acc[0] / pts, Jyy = g_tune_host_acc[1] / pts;
    const double Jzz = g_tune_host_acc[2] / pts, Jxy = g_tune_host_acc[3] / pts;
    const double Jxz = g_tune_host_acc[4] / pts, Jyz = g_tune_host_acc[5] / pts;
    {
        const double q = (Jxx + Jyy + Jzz) / 3.0;
        const double p1 = Jxy * Jxy + Jxz * Jxz + Jyz * Jyz;
        const double p2 =
            (Jxx - q) * (Jxx - q) + (Jyy - q) * (Jyy - q) + (Jzz - q) * (Jzz - q) + 2.0 * p1;
        const double p = std::sqrt(std::max(p2 / 6.0, 0.0));
        if (p > 0.0) {
            const double bxx = (Jxx - q) / p, byy = (Jyy - q) / p, bzz = (Jzz - q) / p;
            const double bxy = Jxy / p, bxz = Jxz / p, byz = Jyz / p;
            const double detB = bxx * (byy * bzz - byz * byz) - bxy * (bxy * bzz - byz * bxz) +
                                bxz * (bxy * byz - byy * bxz);
            const double rr = std::min(1.0, std::max(-1.0, detB / 2.0));
            const double phi = std::acos(rr) / 3.0;
            const double l1 = q + 2.0 * p * std::cos(phi);
            const double l3 = q + 2.0 * p * std::cos(phi + 2.0943951023931953); // +2pi/3
            if (l1 > 0.0)
                cs = std::max(0.0, l3 / l1);
        } else if (q > 0.0) {
            cs = 1.0; // isotropic: all eigenvalues equal
        }
    }
    nz0 = *g_tune_host_nzc / pts;
}

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

// CR-first topology decision using statistics already produced by d_isotropy_stats.
// The REL boundaries are half-decades, so arbitrary bounds map to the nearest tested
// decade without log10/pow.  No field names or exact dimensions are encoded here;
// very_flat is a geometry-only distinction.
static int pick_cr_first_mode(dim3 dims, double rel, double cs, double nz0) {
    // ABS callers do not provide a normalized bound.  Preserve the conservative
    // sphericity fallback rather than interpreting rel=0 as an extremely tight REL.
    if (rel <= 0.0)
        return cs >= 0.21 ? 0 : 1;

    constexpr double R1_R2 = 3.162277660168379e-2;
    constexpr double R2_R3 = 3.162277660168379e-3;
    constexpr double R3_R4 = 3.162277660168379e-4;
    constexpr double R4_R5 = 3.162277660168379e-5;
    const bool very_flat = static_cast<uint64_t>(dims.z) * 8u <= std::min(dims.x, dims.y);

    if (very_flat) {
        if (rel >= R1_R2)
            return cs >= 0.10 && cs <= 0.13 ? 0 : 1;
        if (rel >= R3_R4)
            return 1;
        if (rel >= R4_R5)
            return cs >= 0.10 && cs <= 0.121 && nz0 > 0.80 ? 0 : 1;
        return cs >= 0.02 && cs <= 0.121 && nz0 > 0.60 ? 0 : 1;
    }

    if (rel >= R1_R2) {
        const bool sparse_low = cs >= 0.03 && cs <= 0.05 && nz0 < 0.995;
        return sparse_low || (cs >= 0.21 && cs <= 0.265) || cs >= 0.45 ? 0 : 1;
    }
    if (rel >= R2_R3) {
        const bool sparse_low = cs >= 0.03 && cs <= 0.05 && nz0 > 0.98;
        return sparse_low || (cs >= 0.21 && cs <= 0.30) || cs >= 0.45 ? 0 : 1;
    }
    if (rel >= R3_R4) {
        const bool sparse_low = cs >= 0.06 && cs <= 0.09 && nz0 > 0.95;
        return sparse_low || (cs >= 0.21 && cs <= 0.30) || cs >= 0.45 ? 0 : 1;
    }
    const bool sparse_low = cs >= 0.06 && cs <= 0.09 && nz0 > 0.90;
    return sparse_low || (cs >= 0.21 && cs <= 0.30) ? 0 : 1;
}

// ---------------------------------------------------------------------------
// FAST transform auto-tuner. Sample at stride 8 first and refine uncertain decisions at stride 4.
//
// CR-first decision based on decomposition geometry, sampled field statistics, and
// the current REL bound.
//
//   cs  = lambda3/lambda1 of J = mean(grad f * grad f^T)      [sphericity]
//   nz0 = #{ max(|gx|,|gy|,|gz|) < 2*tau } / points           [zero-quantizing frac]
// rel selects a half-decade rule; nz0 also changes with the current error bound.  Thus
// the choice adapts to error without another kernel, trial transform, or device read.
// ---------------------------------------------------------------------------
template <typename T>
static int autotune_pick_mode_fast(const T* d_data, dim3 dims, double tau,
    double rel_hint, cudaStream_t st, double* tune_ms) {
    const double ctau = 2.0 * tau;
    double cs = 0.0, nz0 = 0.0, sample_ms = 0.0;
    sample_isotropy<T, 8>(d_data, dims, ctau, st, cs, nz0, sample_ms);
    double accumulated_ms = sample_ms;

    const bool uncertain_cs = cs >= 0.10 && cs <= 0.32;
    const bool uncertain_nz = rel_hint <= 1.5e-4 && nz0 >= 0.45 && nz0 <= 0.55;
    if (uncertain_cs || uncertain_nz) {
        sample_isotropy<T, 4>(d_data, dims, ctau, st, cs, nz0, sample_ms);
        accumulated_ms += sample_ms;
    }
    const int mode = pick_cr_first_mode(dims, rel_hint, cs, nz0);
    *tune_ms = accumulated_ms;
    return mode;
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
    uint8_t use_dyadic = 0;
    uint8_t dyadic_level = std::min(levels_xy, levels_z);

    double tune_ms = 0.0;
    const int mode = autotune_pick_mode_fast<T>(d_oridata, data_dims, real_error, config.relErrorBound,
        cuda_stream, &tune_ms);
    use_dyadic = mode == 0 ? 1 : 0;

    if (!use_dyadic && levels_z > 0 &&
        data_dims.z > CDF97::DWT_Z_STATIC_BYTES / (2U * sizeof(T)))
        throw std::invalid_argument("Waltz: plane Z column exceeds the 45 KiB static shared-memory capacity");

    const uint32_t reorder_bz = WALTZ::lossless::select_block_zband(data_dims, levels_z, use_dyadic != 0U);
    const uint32_t rank_shapes = (data_dims.x % 16U || data_dims.y % 16U || data_dims.z % reorder_bz) ? 8U : 1U;

    CDF97::dwt3d_prealloc<T>();
    if (CDF97::idwt3d_prealloc<T>(data_dims, use_dyadic != 0U, false, cuda_stream) != cudaSuccess)
        throw std::runtime_error("Waltz: cannot configure IDWT workspace");

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
    const bool plane_z_direct = !use_dyadic;
    GPUTimer block_rank_timer;

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

    if (!plane_z_direct)
        cudaMemcpyAsync(d_data, d_oridata, bytes, cudaMemcpyDeviceToDevice, cuda_stream);

    block_rank_timer.start(stream);
    WALTZ::lossless::d_build_block_rank<<<dim3(reorder_bz, rank_shapes), 256, 0, cuda_stream>>>
        (d_tb, 16U, 16U, reorder_bz, data_dims);
    block_rank_timer.record_stop(stream);

    if (!use_dyadic)
        cudaMemsetAsync(d_q, 0, bytes, cuda_stream);
    cudaMemsetAsync(d_q_abs, 0, mag_bytes, cuda_stream);
    cudaMemsetAsync(d_q_sign, 0, sb_bytes, cuda_stream);
    cudaMemsetAsync(d_qoc, 0, sizeof(unsigned int), cuda_stream);

    CDF97::set_quant_params<T>(d_q_abs, d_q_sign, reinterpret_cast<T*>(d_q), d_tb,
        data_dims, reorder_bz, quantity, d_qoi, d_qov, d_qoc, qout_cap, cuda_stream);
    cudaStreamSynchronize(cuda_stream);

    double gpu_predict_ms;
    size_t blob_bytes = 0; // assembled compressed-cmpdata size (returned)
    GPUTimer timer;
    timer.start(stream);
    CDF97::d_reset<<<1, 1, 0, cuda_stream>>>();
    if (use_dyadic) {
        CDF97::dwt3d_xy_halo_z<T>(d_data, tmp_data, data_dims, dyadic_level, reorder_bz, cuda_stream);
    } else {
        CDF97::dwt3d_plane_xy_halo_z<T>(
            d_data, tmp_data, data_dims, levels_xy, levels_z, d_oridata, reorder_bz, cuda_stream);
    }
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
            d_oridata, real_error, d_cnt, d_poi, d_poe, pout_cap, cuda_stream1);
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
        {tune_ms, block_rank_timer.elapsed_ready(), gpu_predict_ms, t_quant, t_mag, t_idwt, 0.0},
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

    // WZP always writes dequantized row-major coefficients directly, including edges.
    uint32_t* d_tb = nullptr;
    const uint32_t entries = 16U * 16U * dec_quant_bz;
    const uint32_t offset_shapes = full_reorder_bricks ? 1U : 8U;
    if (!complete_bz2_layout)
        cudaMalloc(reinterpret_cast<void**>(&d_tb), offset_shapes * entries * sizeof(uint32_t));

    if (CDF97::idwt3d_prealloc<T>(dims, dec_use_dyadic, true, cuda_stream) != cudaSuccess)
        throw std::runtime_error("Waltz: cannot configure IDWT workspace");
    GPUTimer block_offset_timer, unquant_timer, idwt_timer, pwe_timer;
    double wzp_decode_kernel_ms = 0.0;
    T* const d_coeff = d_decData;

    if (!complete_bz2_layout) {
        block_offset_timer.start(stream);
        WALTZ::lossless::d_build_block_ioffset<<<dim3(dec_quant_bz, offset_shapes), 256, 0, cuda_stream>>>(
            d_tb, 16U, 16U, dec_quant_bz, dims.x, static_cast<size_t>(dims.x) * dims.y, dims);
        block_offset_timer.record_stop(stream);
    }
    if constexpr (std::is_same_v<T, float>) {
        WALTZ::lossless::wzp_decode_reordered_float(sections.magnitude,
            N, d_coeff, d_tb, static_cast<float>(hdr.q), dims,
            dec_quant_bz, &wzp_decode_kernel_ms, cuda_stream);
    } else if constexpr (std::is_same_v<T, double>) {
        WALTZ::lossless::wzp_decode_reordered_double(
            sections.magnitude, N, d_coeff, d_tb,
            hdr.q, dims, dec_quant_bz, &wzp_decode_kernel_ms, cuda_stream);
    }
    unquant_timer.start(stream); // Only quantization-outlier patches remain separate.
    if (hdr.n_qo) {
        const int tb2 = 256, gb2 = static_cast<int>((hdr.n_qo + tb2 - 1) / tb2);
        apply_q_outliers<T><<<gb2, tb2, 0, cuda_stream>>>(d_coeff,
            reinterpret_cast<const uint32_t*>(sections.quant_index),
            reinterpret_cast<const int32_t*>(sections.quant_value), hdr.n_qo, hdr.q);
    }
    unquant_timer.record_stop(stream);
    idwt_timer.start(stream); // 5) IDWT (in place)
    if (dec_use_dyadic) {
        const int level = std::min(hdr.levels_xy, hdr.levels_z);
        CDF97::idwt3d_xy_halo_z<T>(
            d_decData, static_cast<T*>(WALTZ::IDWTConfig<T>::tmp), dims, level,
            nullptr, 0.0, nullptr, nullptr, nullptr, 0, cuda_stream);
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
    const double t_offset = complete_bz2_layout ? 0.0 : block_offset_timer.elapsed_ready();
    const double t_wzp = wzp_decode_kernel_ms;
    const double t_uq = unquant_timer.elapsed_ready();
    const double t_idwt = idwt_timer.elapsed_ready();
    const double t_pw = pwe_timer.elapsed_ready();
    if (d_tb)
        cudaFree(d_tb);
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
