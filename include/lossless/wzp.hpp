#pragma once

#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>

namespace WALTZ::lossless {

// Waltz Zero-aware Parallel magnitude codec.  The archive is chunk independent:
// zero chunks have no payload, while nonzero chunks use the fast bit-plane sparse
// hierarchy.  All metadata and payload remain device resident.
void wzp_prealloc(uint32_t n, cudaStream_t stream = 0);
size_t wzp_maxsize(uint32_t n);
size_t wzp_encode(const uint16_t* input, uint32_t n, double* ms, cudaStream_t stream = 0);
size_t wzp_encode_with_sign(const uint16_t* input,
                            const uint32_t* sign_bitmap,
                            uint32_t n,
                            uint32_t* nnz,
                            double* ms,
                            cudaStream_t stream = 0);
void wzp_encode_with_sign_launch(const uint16_t* input,
                                 const uint32_t* sign_bitmap,
                                 uint32_t n,
                                 cudaStream_t stream = 0);
size_t wzp_encode_with_sign_finish(uint32_t* nnz, double* ms, cudaStream_t stream = 0);
const void* wzp_encoded_buf();
void wzp_decode(
    const void* blob, uint32_t n, uint16_t* output, double* ms, cudaStream_t stream = 0);
void wzp_decode_with_sign(const void* blob,
                          uint32_t n,
                          uint16_t* output,
                          uint32_t* sign_bitmap,
                          double* ms,
                          cudaStream_t stream = 0);
void wzp_decode_reordered_float(const void* blob,
                                uint32_t n,
                                float* output,
                                const uint32_t* offset_table,
                                float q,
                                dim3 dims,
                                uint32_t block_z,
                                double* ms,
                                cudaStream_t stream = 0);
void wzp_decode_reordered_double(const void* blob,
                                 uint32_t n,
                                 double* output,
                                 const uint32_t* offset_table,
                                 double q,
                                 dim3 dims,
                                 uint32_t block_z,
                                 double* ms,
                                 cudaStream_t stream = 0);

} // namespace WALTZ::lossless
