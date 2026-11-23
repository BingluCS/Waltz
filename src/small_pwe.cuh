#pragma once
#include <cuda_runtime.h>
#include <math_constants.h>
#include <cstdint>
#include <cmath>
#include <type_traits>
#include "utils/Blob.hpp"

namespace WALTZ {

template<class T> __device__ __forceinline__ T repair_sub(T a, T b) {
    if constexpr (std::is_same_v<T, float>) return __fsub_rn(a,b);
    else return __dsub_rn(a,b);
}

// e is the rounded native subtraction (encoder synthesis - original).
// Adjacent representable errors enclose the true subtraction. Directed native
// additions/subtractions conservatively enclose synthesis - new_correction.
// This relies on identical encoder/decoder synthesis, just like native PWE.
// Direct-error synthesis is disabled in the production path used here.
template<class T> __device__ __forceinline__ bool safe_small_pwe(
    T original, T e, T correction, double budget) {
    if (!isfinite(e) || !isfinite(correction) || !isfinite(original)) return false;
    double err_lo, err_hi;
    if constexpr (std::is_same_v<T, float>) {
        float lo = __fadd_rd(original, nextafterf(e, -CUDART_INF_F));
        float hi = __fadd_ru(original, nextafterf(e, CUDART_INF_F));
        lo = __fsub_rd(lo, correction);
        hi = __fsub_ru(hi, correction);
        err_lo = double(lo) - double(original);
        err_hi = double(hi) - double(original);
    } else {
        double lo = __dadd_rd(original, nextafter(e, -CUDART_INF));
        double hi = __dadd_ru(original, nextafter(e, CUDART_INF));
        lo = __dsub_rd(lo, correction);
        hi = __dsub_ru(hi, correction);
        err_lo = __dsub_rd(lo, original);
        err_hi = __dsub_ru(hi, original);
    }
    return isfinite(err_lo) && isfinite(err_hi) &&
           err_lo >= -budget && err_hi <= budget;
}

// Each warp owns 32 PWE records. Codes occupy one byte; a bitmap indexes exact
// native fallback errors using popcount. A whole group can instead stay native.
// Payload reservations are independent; the directory preserves logical order.
template<class T> __device__ __forceinline__ void pack_small_pwe_group(
    const T* original, const uint32_t* indices, const T* errors, uint32_t n,
    double step, double budget, SmallPweGroup* directory,
    unsigned char* payload, uint32_t* payload_size, uint32_t i) {
    const uint32_t lane = threadIdx.x & 31U, group = i >> 5U;
    const uint32_t start = group * 32U;
    if (start >= n) return; // warp-uniform
    const uint32_t count = min(32U, n-start);
    const bool valid = i < n;
    T e = valid ? errors[i] : T(0);
    int code = 0;
    bool compact = false;
    if (valid) {
        double rounded = nearbyint(double(e) / step);
        if (isfinite(rounded) && rounded >= -127.0 && rounded <= 127.0) {
            code = int(rounded);
            const T correction = T(double(code)*step);
            compact = safe_small_pwe(original[indices[i]], e, correction, budget);
        }
    }
    const uint32_t mask = __ballot_sync(0xffffffffU, valid && !compact);
    const uint32_t raw_count = __popc(mask);
    const uint32_t raw_start = (count + sizeof(T)-1U) & ~(sizeof(T)-1U);
    const uint32_t small_size = raw_start + raw_count * sizeof(T);
    const uint32_t native_size = count * sizeof(T);
    const bool use_small = small_size < native_size;
    const uint32_t bytes = ((use_small ? small_size : native_size)+7U)&~7U;
    uint32_t offset = 0;
    if (lane == 0) {
        offset = atomicAdd(payload_size, bytes);
        directory[group] = {offset | (use_small ? 0U : 0x80000000U), mask};
    }
    offset = __shfl_sync(0xffffffffU, offset, 0);
    unsigned char* out = payload + offset;
    if (valid) {
        if (!use_small) reinterpret_cast<T*>(out)[lane] = e;
        else {
            reinterpret_cast<int8_t*>(out)[lane] = compact ? int8_t(code) : int8_t(0);
            if (!compact) {
                const uint32_t prefix = __popc(mask & ((1U << lane)-1U));
                reinterpret_cast<T*>(out+raw_start)[prefix] = e;
            }
        }
    }
    // Define alignment bytes too (no uninitialized archive padding).
    const uint32_t used = use_small ? small_size : native_size;
    if (lane < bytes-used) out[used+lane] = 0;
    if (use_small && lane < raw_start-count) out[count+lane] = 0;
}

// Queued after synthesis on the same stream: consume the GPU count directly.
// A bounded grid avoids launching capacity-sized grids when repairs are rare.
// The actual count (not capacity) determines the unchanged WPW1 layout.
template<class T> __global__ void pack_small_pwe_async(
    const T* original, const uint32_t* indices, const T* errors,
    const uint32_t* device_count, uint32_t capacity, double step,
    unsigned char* storage, uint32_t* payload_size) {
    const uint32_t n=min(*device_count,capacity), groups=(n+31U)/32U;
    auto* directory=reinterpret_cast<SmallPweGroup*>(storage+sizeof(SmallPweHeader));
    auto* payload=reinterpret_cast<unsigned char*>(directory+groups);
    const uint32_t stride=gridDim.x*blockDim.x;
    for(uint32_t i=blockIdx.x*blockDim.x+threadIdx.x;(i&~31U)<n;i+=stride)
        pack_small_pwe_group(original,indices,errors,n,step,step,directory,payload,payload_size,i);
    // Gather writes the header using the one host-visible result-count copy.
}

template<class T> __global__ void apply_small_pwe(
    T* output, const uint32_t* indices, const unsigned char* section, uint32_t n,
    uint32_t section_bytes) {
    const auto* header = reinterpret_cast<const SmallPweHeader*>(section);
    const size_t prefix = sizeof(SmallPweHeader) +
        ((size_t(n)+31U)/32U) * sizeof(SmallPweGroup);
    // Fail closed on an unsupported or inconsistent WPW header. CUDA reports
    // the trap at the API's checked synchronization; never silently misdecode.
    if (header->magic != SMALL_PWE_MAGIC || header->groups != (size_t(n)+31U)/32U ||
        !(header->step > 0.0) || !isfinite(header->step) || prefix > section_bytes ||
        header->payload_bytes != section_bytes-prefix) {
        asm volatile("trap;");
        return;
    }
    const auto* directory = reinterpret_cast<const SmallPweGroup*>(header+1);
    const unsigned char* payload = reinterpret_cast<const unsigned char*>(directory+header->groups);
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const uint32_t lane = i&31U, group = i>>5U;
    const SmallPweGroup desc = directory[group];
    const uint32_t count = min(32U,n-group*32U);
    const uint32_t raw_start = (count+sizeof(T)-1U)&~(sizeof(T)-1U);
    const size_t offset = desc.offset & 0x7fffffffU;
    const size_t group_bytes = (desc.offset & 0x80000000U) ? count*sizeof(T) :
        raw_start + __popc(desc.native_mask)*sizeof(T);
    if ((offset & 7U) || offset > header->payload_bytes ||
        group_bytes > header->payload_bytes-offset) {
        asm volatile("trap;");
        return;
    }
    const unsigned char* data = payload + (desc.offset & 0x7fffffffU);
    T correction;
    if (desc.offset & 0x80000000U) correction = reinterpret_cast<const T*>(data)[lane];
    else if ((desc.native_mask >> lane) & 1U) {
        const uint32_t prefix = __popc(desc.native_mask & ((1U<<lane)-1U));
        correction = reinterpret_cast<const T*>(data+raw_start)[prefix];
    } else correction = T(double(reinterpret_cast<const int8_t*>(data)[lane])*header->step);
    const uint32_t index = indices[i];
    output[index] = repair_sub(output[index],correction);
}
} // namespace WALTZ
