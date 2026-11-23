#ifndef WALTZ_CPU_H
#define WALTZ_CPU_H
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
typedef enum { WALTZ_CPU_F32=4, WALTZ_CPU_F64=8 } waltz_cpu_type;
typedef enum { WALTZ_CPU_AUTO=0, WALTZ_CPU_SCALAR=1, WALTZ_CPU_AVX2=2 } waltz_cpu_backend;
typedef struct {
    uint32_t nx, ny, nz;
    waltz_cpu_type type;
    double absolute_error;
    waltz_cpu_backend backend;
} waltz_cpu_config;
typedef struct waltz_cpu_workspace waltz_cpu_workspace;
/* Caller owns input/output/archive. Workspace is reusable but not concurrently
 * shareable. Input samples require natural type alignment. WLT2 does not store
 * dimensions: provide the same shape on decode. Codec functions return 0 on
 * success and -1 on failure; last_error contains the diagnostic. Compression
 * sets written=0 on failure but may have modified archive memory. Output of a
 * failed decode is unchanged. Encoders use the canonical portable profile;
 * arbitrary existing optimized GPU profiles need not produce identical bytes. */
waltz_cpu_workspace* waltz_cpu_create(void);
void waltz_cpu_destroy(waltz_cpu_workspace*);
size_t waltz_cpu_max_compressed_bytes(const waltz_cpu_config*);
int waltz_cpu_compress(waltz_cpu_workspace*, const waltz_cpu_config*,
                       const void* input, void* archive, size_t capacity, size_t* written);
int waltz_cpu_decompress(waltz_cpu_workspace*, const waltz_cpu_config*,
                         const void* archive, size_t archive_bytes, void* output, size_t output_bytes);
const char* waltz_cpu_last_error(const waltz_cpu_workspace*);
int waltz_cpu_has_avx2(void);
#ifdef __cplusplus
}
#endif
#endif
