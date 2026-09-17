# Waltz

Waltz is a CUDA research implementation of error-bounded lossy compression for
3D scientific float32 and float64 fields. It combines CDF 9/7 wavelet transforms,
fused quantization and block reordering, WZP magnitude/sign coding, and
reconstruction-error correction.

## Current implementation

The source synchronized on 2026-09-17 includes:

- A data-driven tuner selecting dyadic or plane (XY + Z) transforms.
- Quantization and forward reordering fused into DWT coefficient writes.
- Plane float/double IDWT sharing `idwt_yx_plane<T>` and
  `idwt_z_all_dynamic<T>`. The latter calls `idwt_z_columns_all<T>`:
  original detail bands stay fixed while reconstructed low bands use two
  scratch regions, avoiding per-level copies of untouched high-frequency data.
  Float retains its float4 path. Inverse plane Z uses dynamic shared memory.
- Signed WZP encoding and group packing in one kernel with a fixed grid of
  512 blocks and dynamic chunk scheduling. WZP is the only lossless backend;
  experimental LC selection and old-format compatibility paths are removed.
- WZP decoding fused with reordering and dequantization. Quantization-outlier
  restoration remains a separate step when required.
- Regular/edge lookup tables, including a timed decode `block-offset` stage.
- Reusable compression workspace and the `dwt3d_prealloc`,
  `idwt3d_prealloc`, and `wzp_prealloc` resource setup entries.
- Shared file I/O in `utils/io.hpp` and reporting in `utils/Statistics.hpp`.

P4/dual-transform and auxiliary entropy-coder experiments are not enabled.
Benchmark archives, local datasets and prebuilt binaries are not synchronized.

## Build

Requires Linux, CMake 3.21+, a CUDA-supported C++17 compiler, and the NVIDIA
CUDA Toolkit. The current update is tested with CUDA 13.1 and H100 (sm_90).
Select the architecture for your target GPU; the latest change has not been
rebenchmarked on every model.

```bash
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.1/bin/nvcc \
  -DCMAKE_CUDA_ARCHITECTURES=90
cmake --build build -j4
./build/waltz -h
```

For example, use architecture `89` for RTX 4090 or `120` for RTX PRO 6000
Blackwell with a toolkit supporting that target.

The default build creates `build/libwaltz.so` and `build/waltz`.
Use `-DBUILD_SHARED_LIBS=OFF` for a static library.

Optional benchmark and regression targets:

```bash
cmake -S . -B build -DWALTZ_BUILD_TOOLS=ON -DWALTZ_BUILD_TESTS=ON
cmake --build build -j4
CUDA_VISIBLE_DEVICES=0 ctest --test-dir build --output-on-failure
```

Install the library, headers, CLI and CMake package:

```bash
cmake --install build --prefix /path/to/waltz-install
```

Downstream projects can use `find_package(Waltz CONFIG REQUIRED)` and link
`Waltz::waltz`.

## Usage

Run `./build/waltz -h` to print help and exit successfully. No dataset or GPU
initialization is needed for help.

```text
Compress:
  waltz [-f|-d] -i input -z compressed -3 nx ny nz -M REL|ABS bound [-o output]

Decompress:
  waltz [-f|-d] -z compressed -o output -3 nx ny nz
  waltz [-f|-d] -i compressed -o output -3 nx ny nz
```

| Option | Meaning |
| --- | --- |
| `-h` | Print help and exit. |
| `-f` | float32 input/output (default). |
| `-d` | float64 input/output; this is a type flag, not a decompression flag. |
| `-i input` | Raw input for compression, or compressed input for standalone decode without `-M`. |
| `-z compressed` | Compressed output for compression; compressed input for standalone decode. |
| `-o output` | Write reconstruction. With compression, also decode and verify against the original input. |
| `-3 nx ny nz` | Original dimensions, required for both operations. |
| `-M REL bound` | Compress using absolute tolerance `bound * (max(input) - min(input))`. |
| `-M ABS bound` | Compress using the supplied absolute tolerance. |

Input is a headerless, native-endian array with X varying fastest and exactly
`nx * ny * nz` elements. Compressed files store neither dimensions nor data
type. Decode with the same `-3` dimensions and `-f`/`-d` type used to compress.

### Compression only

```bash
./build/waltz -f -i field.f32 -z field.waltz \
  -3 512 512 512 -M REL 1e-3
```

Without `-o`, the command writes the compressed file but does not run the
standalone decompression or host-side quality check.

### Compression, reconstruction and verification

```bash
./build/waltz -d -i density.d64 -z density.waltz -o density.out.d64 \
  -3 384 384 256 -M REL 1e-3
```

The driver reopens the compressed file, reconstructs it, then prints
`Max_E`, `Max_RE`, `PSNR` and `NRMSE` against the original.
Compression and decompression each print `Compression Ratio` to five decimal
places. Use `-M ABS 1e-6` instead for an absolute tolerance.

### Standalone decompression

```bash
./build/waltz -f -z field.waltz -o field.out.f32 -3 512 512 512
```

Omit `-M`. Standalone decode writes the output and prints stage timings and CR;
it does not print PSNR/error statistics because no original reference is supplied.

## Warm-up and timing

```bash
CUDA_VISIBLE_DEVICES=0 WALTZ_WARMUP=1 WALTZ_REPORT_CE2E=1 \
  ./build/waltz -f -i field.f32 -z field.waltz -o field.out.f32 \
  -3 512 512 512 -M REL 1e-3
```

- `WALTZ_WARMUP=1`: run each requested API once without normal reporting,
  then run it again for the reported result. If both operations are requested,
  compression and decompression each receive their own warm-up call.
  Warm-up is disabled when unset or zero.
- `WALTZ_REPORT_CE2E=1`: print API wall time around the measured
  `WALTZ::d_compress` / `WALTZ::d_decompress` call, including the final stream
  synchronization. Internally this switch is enabled by the variable's presence;
  unset it to disable reporting.
- REL range calculation occurs once before compression warm-up and timing.
  File I/O, external input/output transfers, caller-owned buffer allocations,
  and host-side verification are outside these API timers.
  Internal API allocations/work, synchronization and reporting remain inside.
- Stage timings and API wall time are different measurements. Compression's
  printed stage total uses `max(WZP, IDWT)` for the overlapping stages, not their
  sum; it is not a complete GPU timeline or process wall time.
- Throughput labels currently say GB/s but use GiB/s (bytes divided by
  `1024^3`, then divided by seconds).
- The obsolete `WALTZ_ASYNC_ALLOC` warm-up switch is no longer used.

## Library API

The APIs consume CUDA device pointers and take a stream as an opaque `void*`.
The caller must resolve REL to an absolute bound before calling
`d_compress`; the CLI already does this outside its API timer.

```cpp
#include <waltz.hpp>

WALTZ::Config conf(nx, ny, nz);
conf.relErrorBound = 1e-3;  // Also retained as the autotuner's relative hint.
conf.errorBoundMode = WALTZ::EB_ABS;

cudaStream_t stream;
cudaStreamCreate(&stream);
float *d_input = nullptr, *d_output = nullptr;
char* d_compressed = nullptr;
const size_t bytes = conf.num * sizeof(float);
cudaMalloc(&d_input, bytes);
cudaMalloc(&d_output, bytes);
cudaMalloc(&d_compressed, WALTZ::compressed_buffer_capacity<float>(conf.num));
cudaMemcpyAsync(d_input, h_input, bytes, cudaMemcpyHostToDevice, stream);

// h_input is the caller's populated host array.
double minimum = 0.0, maximum = 0.0;
WALTZ::d_extrema<float>(d_input, conf.num, minimum, maximum, stream);
conf.absErrorBound = conf.relErrorBound * (maximum - minimum);
// For an ABS bound, set conf.absErrorBound directly and omit d_extrema.

const size_t compressed_bytes =
    WALTZ::d_compress<float>(conf, d_input, d_compressed, nullptr, stream);
const size_t output_bytes =
    WALTZ::d_decompress<float>(conf, d_compressed, d_output, nullptr, stream);
cudaStreamSynchronize(stream);

// Consume/save the output before releasing caller-owned buffers.
cudaFree(d_compressed);
cudaFree(d_output);
cudaFree(d_input);
cudaStreamDestroy(stream);
```

The compressed-output pointer is required; its capacity must be at least
`compressed_buffer_capacity<T>(conf.num)`. The compression path argument
does not enable/disable output and does not write a file. File I/O belongs to
the caller. Supply the original shape and matching template type on decode.

## Licensing

No project-wide license has been declared. Vendored LC-framework components
used inside WZP retain their BSD 3-Clause notices; see
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). Removing the LC backend does
not remove these component notices.
