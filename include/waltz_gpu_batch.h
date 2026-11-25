#pragma once
#include <stdint.h>
#include <stddef.h>
#ifdef __cplusplus
extern "C" {
#endif
typedef struct {
    uint32_t x,y,z,fx,fy,fz,samples,sides,width;
} waltz_batch_shape;
/* Trusted canonical archives. CPU and device copies contain identical packed
 * bytes, followed by aligned uint64 (offset,size) pairs in sample/side/ZYX
 * order. Size zero means fill. Scratch/output must be distinct device
 * allocations. Output is contiguous (side,sample,z,y,x). One batch per call
 * on the supplied stream; no global workspace. Returns NULL on success. */
const char* waltz_gpu_decode_batch(const waltz_batch_shape* shape,
    const void* host,const void* packed,size_t bytes,void* scratch,size_t scratch_bytes,
    void* output,size_t output_bytes,double fill,void* stream);
#ifdef __cplusplus
}
#endif
