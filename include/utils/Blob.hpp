#pragma once

#include <cstdint>
#include <cstddef>

namespace WALTZ {

#if defined(__CUDACC__) || defined(__HIPCC__)
#define WALTZ_BLOB_HD __host__ __device__
#else
#define WALTZ_BLOB_HD
#endif

enum class PweEncoding : uint8_t { Native = 0, Small = 1 };
enum class Synthesis : uint8_t {
    Convolution = 0, HaloLifting3D = 1, HaloAllLevels = 2,
    // Values 3–6 were experimental layouts and are not accepted.
};

// Little-endian archive header; 8-byte-aligned sections follow:
// WZP magnitude/sign stream, PWE indices/errors, quantization outlier indices/values.
// Dimensions and element count come from Config; element size is checked on decode.
// No magic or version field. Bits 0–2 are dyadic, compact PWE and FP64;
// bits 3–4 encode synthesis as 1–3. Zero is not a synthesis mode.
// Bit 5 selects the CUDA-only short plane-Z convolution variant of HaloAllLevels.
// Portable decoders must reject bit 5, rather than decoding it with lifting.
struct WaltzBlobHeader {
    static constexpr uint8_t CudaFusedPlaneZ = 0x20;
    uint8_t flags, levels_xy, levels_z, reorder_bz;
    uint32_t pwe_bytes;
    double q;
    uint64_t mag_comp_bytes;
    uint32_t n_pwe, n_qo;

    WALTZ_BLOB_HD static constexpr uint8_t pack_flags(bool dyadic, unsigned width, PweEncoding pwe, Synthesis synth) {
        return uint8_t(unsigned(dyadic) | (unsigned(pwe) << 1U) |
            (width == 8 ? 4U : 0U) | ((unsigned(synth) + 1U) << 3U));
    }
    // Validate the base format. The CUDA decoder separately checks its Z mode.
    WALTZ_BLOB_HD static constexpr bool valid_flags(uint8_t f) { return !(f & 0xe0U) && (f & 0x18U); }
    WALTZ_BLOB_HD static constexpr unsigned width(uint8_t f) { return (f & 4U) ? 8U : 4U; }
    WALTZ_BLOB_HD static constexpr Synthesis synthesis_of(uint8_t f) { return Synthesis(int((f >> 3U) & 3U) - 1); }
    WALTZ_BLOB_HD static constexpr PweEncoding repair_of(uint8_t f) { return PweEncoding((f >> 1U) & 1U); }
    WALTZ_BLOB_HD constexpr unsigned element_bytes() const { return width(flags); }
    WALTZ_BLOB_HD constexpr bool use_dyadic() const { return flags & 1U; }
    WALTZ_BLOB_HD constexpr Synthesis synthesis() const { return synthesis_of(flags); }
    WALTZ_BLOB_HD constexpr PweEncoding pwe_encoding() const { return repair_of(flags); }
};
inline constexpr size_t BlobSize = sizeof(WaltzBlobHeader);
inline constexpr size_t BlobPweBytes = offsetof(WaltzBlobHeader, pwe_bytes);
inline constexpr size_t BlobQuantum = offsetof(WaltzBlobHeader, q);
inline constexpr size_t BlobWzpBytes = offsetof(WaltzBlobHeader, mag_comp_bytes);
inline constexpr size_t BlobPweCount = offsetof(WaltzBlobHeader, n_pwe);
inline constexpr size_t BlobQoutCount = offsetof(WaltzBlobHeader, n_qo);
static_assert(BlobSize == 32 && BlobPweBytes == 4 && BlobQuantum == 8 &&
              BlobWzpBytes == 16 && BlobPweCount == 24 && BlobQoutCount == 28);
#undef WALTZ_BLOB_HD

inline constexpr uint32_t SMALL_PWE_MAGIC = 0x31575057U; // WPW1
struct SmallPweHeader {
    uint32_t magic = SMALL_PWE_MAGIC;
    uint32_t groups = 0;
    uint64_t payload_bytes = 0;
    double step = 0;
};
struct SmallPweGroup { uint32_t offset, native_mask; };
static_assert(sizeof(SmallPweHeader) == 24);
static_assert(sizeof(SmallPweGroup) == 8);

} // namespace WALTZ
