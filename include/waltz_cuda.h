#pragma once
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
/* Decode a trusted canonical HDF5 chunk on the calling thread's current CUDA
 * device and stream. Both pointers are device pointers. One device per process;
 * do not run other Waltz CUDA APIs concurrently. Returns NULL on success, or
 * a thread-local diagnostic. Completion is synchronous. */
const char* waltz_cuda_decode_chunk(uint32_t nx, uint32_t ny, uint32_t nz, int width,
                                    const void* archive, size_t archive_bytes,
                                    void* output, size_t output_bytes, void* stream);
#ifdef __cplusplus
}
#endif
