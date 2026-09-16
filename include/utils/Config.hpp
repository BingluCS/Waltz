//
// Created by Bing LU on 25/5/26.
//

#pragma once

#include "def.hpp"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>

#define WALTZ_UINT8 1
#define WALTZ_INT8 11
#define WALTZ_UINT16 2
#define WALTZ_INT16 12
#define WALTZ_UINT32 4
#define WALTZ_INT32 14
#define WALTZ_UINT64 8
#define WALTZ_INT64 18

#define WALTZ_FLOAT 24
#define WALTZ_DOUBLE 28

namespace WALTZ {

inline constexpr uint32_t IDWT_SHARED_BYTES = 32U * 1024U;

enum EB { EB_ABS, EB_REL, EB_PSNR, EB_L2NORM, EB_ABS_AND_REL, EB_ABS_OR_REL };

enum CMP { HOST, DEVICE };

enum class QuantMode : uint8_t {
    None,
    High,
    All,
};

// T records PWE outliers without writing dst; F writes the reconstruction.
enum class PWE : uint8_t {
    T,
    F,
};

struct GPUConfig {
    int device = 0;
    cudaDeviceProp properties{};

    int sm_count() const { return properties.multiProcessorCount; }
};

inline const GPUConfig& gpu_config() {
    static const GPUConfig config = [] {
        GPUConfig value;
        check_cuda(cudaGetDevice(&value.device), "query current CUDA device");
        check_cuda(cudaGetDeviceProperties(&value.properties, value.device),
                   "query CUDA device properties");
        return value;
    }();
    return config;
}

template <typename T> struct DWTConfig {
    static inline int x_blocks = 0;
    static inline int y_blocks = 0;
    static inline int z_blocks = 0;
};

// WZP workspace; device selection comes from GPUConfig.
struct WZPConfig {
    static inline int device = -1;
    static inline uint32_t capacity_n = 0;
    static inline unsigned char* slots = nullptr;
    static inline unsigned char* sign_slots = nullptr;
    static inline unsigned char* blob = nullptr;
    static inline uint32_t* group_done = nullptr;
    static inline size_t blob_capacity = 0;
    static inline uint32_t* host_results = nullptr;
    // Shared by serialized WZP operations; finish signed encode before reuse.
    static inline cudaEvent_t timing_begin = nullptr;
    static inline cudaEvent_t timing_end = nullptr;
    static inline size_t signed_payload_offset = 0;
};

template <typename T> struct IDWTConfig {
    // Persistent global-memory scratch; capacities are in bytes.
    static inline void* tmp = nullptr;
    static inline size_t cap = 0;
    static inline size_t mapped_cap = 0;
    // Single-level dyadic launch grids. z_global_blocks is the one-time init guard.
    static inline int z_global_blocks = 0;
    static inline int z_static_blocks = 0;
    static inline int yx_blocks = 0;
    static inline int yx_halo_blocks = 0;
    // All-level plane launch grids, keyed by configured dimensions below.
    static inline int plane_yx_blocks = 0;
    static inline int plane_z_blocks = 0;
    static inline size_t z_smem_bytes = 0;
    static inline uint32_t z_capacity = 0;
    static inline uint32_t configured_nx = 0;
    static inline uint32_t configured_nz = 0;
};

const std::unordered_map<std::string, EB> EB_MAP = {
    {"ABS", EB_ABS}, {"REL", EB_REL}, {"PSNR", EB_PSNR},
    // {"NORM", EB_L2NORM},
    // {"ABS_AND_REL", EB_ABS_AND_REL},
    // {"ABS_OR_REL", EB_ABS_OR_REL},
};

ALWAYS_INLINE std::string to_lower(const std::string& s) {
    std::string out = s;
    std::transform(out.begin(), out.end(), out.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return out;
}

template <typename EnumType>
ALWAYS_INLINE void match_enum(const std::string& input,
                              const std::unordered_map<std::string, EnumType>& table,
                              uint8_t& out) {
    std::string input_lc = to_lower(input);
    for (const auto& [key, val] : table) {
        if (to_lower(key) == input_lc) {
            out = static_cast<uint8_t>(val);
        }
    }
}

class Config {
  public:
    Config() : Config(1) {}

    template <class... Dims> Config(Dims... args) {
        auto dims_ = {static_cast<uint64_t>(args)...};
        setDims(dims_.begin(), dims_.end());
    }

    template <class IterDim> uint64_t setDims(IterDim begin, IterDim end) {
        dimx = dimy = dimz = dimw = 1;
        ndims = 0;
        num = 1;

        for (auto it = begin; it != end; ++it) {
            const uint64_t dim = static_cast<uint64_t>(*it);
            if (dim <= 1) {
                continue;
            }
            if (ndims >= 4) {
                throw std::invalid_argument("Config supports at most 4 dimensions");
            }
            setDim(ndims, dim);
            ++ndims;
            if (dim > std::numeric_limits<uint64_t>::max() / num)
                throw std::overflow_error("Config element count exceeds uint64 capacity");
            num *= dim;
        }

        if (ndims == 0) {
            ndims = 1;
            dimx = 1;
            num = 1;
        }

        return num;
    }

    void setDim(uint8_t i, uint64_t value) {
        switch (i) {
        case 0:
            dimx = value;
            break;
        case 1:
            dimy = value;
            break;
        case 2:
            dimz = value;
            break;
        case 3:
            dimw = value;
            break;
        default:
            throw std::invalid_argument("dimension index out of range");
        }
    }

    uint8_t ndims;
    uint64_t dimx, dimy, dimz, dimw;
    uint64_t chunkx{256}, chunky{256}, chunkz{256}, chunkw{256};
    uint64_t num;
    uint64_t blockSize = 0;
    uint8_t errorBoundMode = EB_ABS;
    double absErrorBound = 0;
    double relErrorBound = 0;
    double psnrErrorBound = 0;
    double l2normErrorBound = 0;
};

} // namespace WALTZ
