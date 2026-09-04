#pragma once

// GPU lossless back-end: wraps the LC-framework pipeline "BIT_4 RZE_1" so Waltz can
// compress the quantized + z-ordered coefficients on-device and report a ratio.
//
// Mimics SPERR's encoder-call style (hand the coder the device data, get back the
// encoded length).  The implementation lives in src/lossless/lc.cu because the
// LC-framework headers define their own CS / TPB constants that collide with
// CDF97.hpp's CS / TPB macros -- so they must stay in an isolated translation unit.
// This header only declares the interface and is safe to include alongside CDF97.hpp.

#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>

namespace WALTZ {
namespace lossless {

// Compress `insize` bytes located at the 8-byte-aligned DEVICE pointer `d_in` using
// the GPU LC-framework pipeline BIT_4 (4-byte bit-plane shuffle) -> RZE_1 (run-of-
// zeros elimination).  Returns the compressed size in bytes (0 on empty input).
// Device-resident: no host round-trip of the payload.
long long lc_compress_bit4_rze1(const void* d_in, long long insize, cudaStream_t stream = 0);

// Same pipeline but BIT_2 (2-byte bit-plane shuffle) -- matches uint16 payloads, where
// BIT_4 would interleave two values per word and degrade the RZE zero-run patterns.
long long lc_compress_bit2_rze1(const void* d_in, long long insize, cudaStream_t stream = 0);

// Same, but RZE_1 only (no bit-plane shuffle) -- better suited to a sparse
// sign bit-mask where the work is the zero-run elimination.
long long lc_compress_rze1(const void* d_in, long long insize, cudaStream_t stream = 0);

// Sign compaction + bit-packing from (u16 magnitudes + 1-bit negative bitmap): for each
// nonzero magnitude (rank order over the z-ordered array), its negative bit is packed
// into a dense bit-stream written to d_packed (8-byte-aligned device buffer of
// >= ceil(n/8) bytes, zeroed here).  Returns nnz (number of packed bits).
uint32_t sign_compact_pack(const void* d_mag_u16,
                           const uint32_t* d_sign_bm,
                           unsigned char* d_packed,
                           uint32_t n,
                           cudaStream_t stream = 0);
void sign_compact_pack_launch(const void* d_mag_u16,
                              const uint32_t* d_sign_bm,
                              unsigned char* d_packed,
                              uint32_t n,
                              cudaStream_t stream = 0);
uint32_t sign_compact_pack_finish(cudaStream_t stream = 0);
void sign_compact_unpack(const void* d_mag_u16,
                         const unsigned char* d_packed,
                         uint32_t* d_sign_bm,
                         uint32_t n,
                         cudaStream_t stream = 0);
void sign_compact_unpack_blob_tail(const void* d_mag_u16,
                                   const unsigned char* d_blob,
                                   const unsigned long long* d_total_bytes,
                                   const unsigned long long* d_sign_bytes,
                                   size_t tail_guard,
                                   uint32_t* d_sign_bm,
                                   uint32_t n,
                                   cudaStream_t stream = 0);

// FUSED: BIT_2+RZE_1 compress the (uint16) magnitudes AND pack the negative-sign bits of
// each chunk's nonzeros in the same kernel (nonzero from the in-shared mags, negative
// from the 1-bit z-order bitmap).  Writes the magnitude compressed size to *mag_out and
// returns nnz.
uint32_t lc_compress_bit2_rze1_with_sign(const void* d_mag,
                                         long long mag_insize,
                                         long long* mag_out,
                                         const uint32_t* d_sign_bm,
                                         unsigned char* d_packed,
                                         uint32_t sign_n,
                                         cudaStream_t stream = 0);

// Split-phase form of the fused encoder.  `launch` only enqueues work; `finish`
// retrieves the two small size scalars and synchronizes its stream.  This lets
// callers enqueue independent reconstruction work after LC has been admitted
// to the GPU, avoiding launch-order starvation in multi-stream pipelines.
void lc_compress_bit2_rze1_with_sign_launch(const void* d_mag,
                                            long long mag_insize,
                                            const uint32_t* d_sign_bm,
                                            unsigned char* d_packed,
                                            uint32_t sign_n,
                                            cudaStream_t stream = 0);
uint32_t lc_compress_bit2_rze1_with_sign_finish(long long* mag_out, cudaStream_t stream = 0);

// Device pointer to the most recent LC-encoded stream (internal scratch; valid until the
// next compress call).  Used to assemble the compressed blob without an extra copy API.
const void* lc_encoded_buf();

// Decompression: LC-decode a BIT_2+RZE_1 stream (device->device); returns the decoded
// byte count (the original insize, from the stream header).  d_out must hold that many.
long long lc_decompress(const void* d_in, void* d_out, cudaStream_t stream = 0);

// FUSED decode: LC-decode the u16 magnitudes AND rebuild their 1-bit negative bitmap in
// one kernel (the sign of each chunk is unpacked while its decoded mags are still in
// shared memory).  d_sign_bm must hold ceil(n_elems/8192)*1024 bytes.
long long lc_decompress_with_sign(const void* d_in,
                                  void* d_out_mags,
                                  long long mag_bytes_expected,
                                  const unsigned char* d_packed,
                                  uint32_t* d_sign_bm,
                                  cudaStream_t stream = 0);

// Split-phase form used by the timed Waltz pipeline.  `launch` enqueues only
// device work; `finish` retrieves the decoded-size scalar.  This keeps the
// scalar D2H synchronization outside the decoder's kernel interval.
void lc_decompress_with_sign_launch(const void* d_in,
                                    void* d_out_mags,
                                    long long mag_bytes_expected,
                                    const unsigned char* d_packed,
                                    uint32_t* d_sign_bm,
                                    cudaStream_t stream = 0);
long long lc_decompress_with_sign_finish(cudaStream_t stream = 0);

// Convenience: compression ratio = original / compressed.
inline double lc_ratio(long long insize, long long compressed) {
    return (compressed > 0) ? static_cast<double>(insize) / static_cast<double>(compressed) : 0.0;
}

} // namespace lossless
} // namespace WALTZ
