#ifndef WALTZ_GPU_CANONICALIZE_H
#define WALTZ_GPU_CANONICALIZE_H

#include <stddef.h>
#include <stdint.h>
#include <cuda_runtime_api.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Canonical, native-PWE WLT2 portable profile. Input/output are separate device
 * buffers. WZP groups and side channels are reordered without CPU payload
 * staging. Returns 0 on success, -1 on error; written is zero on failure.
 * This API synchronizes stream before returning, including on failure.
 * Device scratch is cached per host thread/device and grows on demand. Include
 * growth and synchronization in complete-call timing, not only the kernels.
 * Call waltz_gpu_portable_release_workspace() to release this thread's cache.
 * n is the shape's exact element count. Compact PWE/optimized schedules are
 * intentionally rejected rather than mislabeled byte-portable.
 */
int waltz_gpu_canonicalize(const void* device_input, size_t archive_bytes,
                           uint32_t n, void* device_output, size_t capacity,
                           size_t* written, cudaStream_t stream);

/* Complete portable GPU encoder wrappers around the matching CUDA library.
 * Environment must select WALTZ_SMALL_PWE_SCALE=0, WALTZ_HALO3D=all (or the
 * equivalent default/auto/both). Link against a build of the current codec.
 * They do not mutate global environment. Device input is not modified.
 * Caller output needs WALTZ::compressed_buffer_capacity<T>(nx*ny*nz) bytes.
 * Workspace growth, original encode, canonicalization, verification decode,
 * strict pointwise validation and synchronization are all part of the wall
 * cost. Verification is required because the legacy GPU repair predicate uses
 * a native-rounded subtraction and is not by itself a strict real-value test.
 */
int waltz_gpu_portable_compress_f32(const float* device_input,
                                   uint32_t nx, uint32_t ny, uint32_t nz,
                                   double absolute_error, void* device_output,
                                   size_t capacity, size_t* written,
                                   cudaStream_t stream);
int waltz_gpu_portable_compress_f64(const double* device_input,
                                   uint32_t nx, uint32_t ny, uint32_t nz,
                                   double absolute_error, void* device_output,
                                   size_t capacity, size_t* written,
                                   cudaStream_t stream);
const char* waltz_gpu_portable_last_error(void);

/* Release this host thread's cached portable scratch on all used devices.
 * Public calls complete their stream before returning, including failure.
 * Does not release the original GPU codec's separate persistent workspace or
 * make the underlying codec safe for concurrent calls. Thread exit also releases
 * this cache. Retained capacity is reused across repeated calls. */
void waltz_gpu_portable_release_workspace(void);

#ifdef __cplusplus
}
#endif
#endif
