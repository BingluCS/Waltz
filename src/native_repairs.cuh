#pragma once

#include <cub/device/device_radix_sort.cuh>
#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>

namespace WALTZ::native_repairs {

// Slow only for errors on the rounding boundary. Compare the exact difference
// of the represented values, including the low part of a rounded tie.
__device__ inline bool within(double a, double b, double bound) {
    const double hi = __dsub_rn(a, b);
    if (!isfinite(hi) || fabs(hi) > bound) return false;
    if (fabs(hi) < bound) return true;
    const double bv = __dsub_rn(a, hi);
    const double lo = __dadd_rn(__dsub_rn(a, __dadd_rn(hi, bv)), __dsub_rn(bv, b));
    return !((hi > 0 && lo > 0) || (hi < 0 && lo < 0));
}

// Final lifting output: record exactly the CPU's native residual. Check the
// value the decoder will return here, while the sample is still in registers.
template<class T> struct Emit {
    const T* original;
    double bound;
    T native_bound;
    unsigned* count;
    uint32_t* indices;
    T* errors;
    unsigned capacity;
    unsigned* invalid;
    __device__ void operator()(T value, size_t i) const {
        const T ref = original[i], err = value - ref;
        const T mag = fabs(err);
        if (mag < native_bound) return;
        const bool patch = mag > native_bound;
        const T corrected = patch ? value - err : value;
        if (!isfinite(err) || !within(double(corrected), double(ref), bound)) {
            atomicOr(invalid, 1U);
            return;
        }
        if (patch) {
            const unsigned slot = atomicAdd(count, 1U);
            if (slot < capacity) { indices[slot] = uint32_t(i); errors[slot] = err; }
        }
    }
};

// Sort only the atomic side records, directly into their final archive spans.
// WZP already writes in logical order by group lookback.
template<class T, class Workspace> cudaError_t write_pairs(
    const uint32_t* keys, const T* vals, uint32_t count, unsigned char* key_dst,
    unsigned char* val_dst, Workspace& workspace, cudaStream_t stream) {
    auto* out_keys = reinterpret_cast<uint32_t*>(key_dst);
    auto* out_vals = reinterpret_cast<T*>(val_dst);
    size_t bytes = 0;
    auto run = [&](void* scratch) {
        return cub::DeviceRadixSort::SortPairs(scratch, bytes, keys, out_keys,
                                               vals, out_vals, count, 0, 32, stream);
    };
    auto e = run(nullptr);
    return e == cudaSuccess ? run(workspace.reserve(bytes)) : e;
}
} // namespace WALTZ::native_repairs
