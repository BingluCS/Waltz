# Waltz

Waltz is a CUDA research implementation of error-bounded lossy compression for
three-dimensional scientific floating-point fields. It combines a GPU CDF 9/7
wavelet transform, fused coefficient quantization and block reordering, a GPU
lossless backend, inverse transformation, and exact correction of the small
number of samples that would otherwise exceed the requested point-wise error
bound.

The current production path uses multi-kernel transforms with quantization
fused into the final coefficient writes. WZP is the default magnitude/sign
backend. A lightweight data-driven tuner selects between dyadic and plane
transform topologies. Forward Z/axis kernels use `DWT_TPB=512`, inverse kernels
use `IDWT_TPB=384`, and fused forward XY kernels launch with 256 threads.
One-time prewarming is disabled by default.

The source update dated 2026-09-16 includes:

- Dyadic DWT: fused XY followed by a single-level Z kernel, with static shared
  memory or direct global loads selected for each level. Float and double use
  the common `dwt_3d_z_single_static<T>` / `dwt_3d_z_single_global<T>` entries.
- Plane DWT: all Z levels in one launch, followed by fused XY/quantization
  launches per level. Plane Z-all now uses a fixed 45 KiB static shared-memory
  allocation; float and double share the `dwt3d_plane_xy_halo_z<T>` launcher.
- `QuantMode::None`, `High`, and `All` name the quantization modes. Z tile
  alignment and quantization-block regularity are handled independently.
- Single-level IDWT Z uses a runtime vectorization flag for float4/aligned
  double2 loads. Launch grids and scratch buffers live in `IDWTConfig<T>`;
  active dimensions are computed once before the inverse level loop.
- Dyadic float IDWT uses whole-row Y-then-X fusion with a 128x16 halo fallback.
  `PWE::T` / `PWE::F` select error recording versus reconstruction writes at
  compile time. The double path retains its type-specific implementation.
- Plane float/double IDWT share `idwt_yx_plane<T>` with 256-thread launches and
  `idwt_z_all_dynamic<T>`. XY uses static shared memory (32 KiB float, 48 KiB
  double); inverse Z-all retains dynamically sized shared memory. Intermediate
  XY launches preserve the outer coefficients needed by the next level.
- Signed WZP encoding and group packing run in one kernel with a fixed grid of
  512 blocks and dynamic chunk scheduling. Workspace allocation and timing
  events are reused through `WZPConfig`. The unsigned-only encoder and the
  experimental prefix-encoding switch were removed; legacy decoding remains.
- `Waltz_gather` / `Waltz_scatter` centralize compressed-blob assembly and
  section views. The existing section order and alignment are preserved.
- Removal of cooperative launch paths, unused Y-chain code, and disabled
  shared-memory search code. Compression overlaps lossless coding with IDWT;
  supported WZP decode layouts fuse reordering and dequantization.

Unadopted P4/dual-transform, reconstruction-shift, adaptive WZP and auxiliary
side-stream codec experiments are not enabled in this release. Benchmark
snapshots, generated binaries and local datasets are not part of the repository.

## Status

This repository is a research prototype. The current interface supports 3D
`float32` and `float64` fields. The compressed blob layout is not yet a stable
or portable file-format specification, so archives should be decoded by a
compatible Waltz revision.

The repository does not yet declare a project-wide license. Vendored
LC-framework files retain their BSD 3-Clause notices; see
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). Choose a license for Waltz
before publishing or redistributing the repository.

## Requirements

- Linux
- CMake 3.21 or newer
- A C++17 compiler supported by the installed CUDA toolkit
- NVIDIA CUDA Toolkit (this snapshot is validated with 13.1; older toolkits
  have not been revalidated)
- An NVIDIA GPU with compute capability 7.0 or newer

Earlier revisions were validated on NVIDIA H100, H200, and RTX 4090 GPUs.
The updated wide float IDWT implementation was compared on H100, RTX 4090
and RTX PRO 6000 Blackwell, including kernel and API end-to-end timings.
It is not claimed to be the fastest implementation on every GPU or shape;
the complete synchronized repository is checked separately below.
Set the CUDA architecture explicitly:

| GPU | CMake architecture |
| --- | --- |
| H100 / H200 | `90` |
| RTX 4090 | `89` |
| RTX PRO 6000 Blackwell | `120` (tested with CUDA 13.1) |
| A100 | `80` |
| RTX 30 series | `86` |

## Build

```bash
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=90
cmake --build build -j
```

If multiple CUDA toolkits are installed, select one explicitly, for example:

```bash
cmake -S . -B build \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
  -DCMAKE_CUDA_ARCHITECTURES=90
```

The default build produces:

- `build/libwaltz.so` (or a static library with `-DBUILD_SHARED_LIBS=OFF`)
- `build/waltz`, the command-line compression and verification driver

Install the library, headers, CLI, and CMake package metadata with:

```bash
cmake --install build --prefix /path/to/waltz-install
```

A downstream CMake project can then use:

```cmake
find_package(Waltz CONFIG REQUIRED)
target_link_libraries(my_target PRIVATE Waltz::waltz)
```

Optional targets:

```bash
# WZP lossless-backend benchmark
cmake -S . -B build-tools -DWALTZ_BUILD_TOOLS=ON \
  -DCMAKE_CUDA_ARCHITECTURES=90
cmake --build build-tools -j
```

The optional WZP regression checks zero/sparse/raw chunks, partial chunks,
unaligned inputs, workspace growth and repeated split/convenience API calls:

```bash
cmake -S . -B build-tests -DWALTZ_BUILD_TESTS=ON \
  -DCMAKE_CUDA_ARCHITECTURES=90
cmake --build build-tests -j
CUDA_VISIBLE_DEVICES=0 ctest --test-dir build-tests --output-on-failure
```

## Command-line use

The input is a headerless, native-endian binary array in X-fastest order. The
element count must equal `NX * NY * NZ`.

Float32 with a relative error bound:

```bash
./build/waltz -f \
  -i /path/to/field.f32 \
  -z field.waltz \
  -o field.reconstructed.f32 \
  -3 512 512 512 \
  -M REL 1e-3
```

Float64 with an absolute error bound:

```bash
./build/waltz -d \
  -i /path/to/field.f64 \
  -z field.waltz \
  -o field.reconstructed.f64 \
  -3 384 384 256 \
  -M ABS 1e-6
```

`-o` is optional. When supplied during compression, the driver reopens the
compressed file, decompresses it, writes the reconstruction, and reports
Max_E, Max_RE, PSNR and NRMSE against the original input. Host verification is
outside kernel and API e2e timing. Compression and decompression each report
their compression ratio and stage timings; decode includes `block-offset`.

An existing compressed file can also be decompressed independently:

```bash
./build/waltz -f -z field.waltz -o field.reconstructed.f32 -3 512 512 512
```

The current compressed header stores neither dimensions nor data type. Supply
the original `-3` dimensions and `-f`/`-d` type on decode. Standalone decode
without an original reference does not report PSNR or error statistics.
Old development formats are not supported; regenerate old compressed files.

Use `CUDA_VISIBLE_DEVICES` to select a GPU:

```bash
CUDA_VISIBLE_DEVICES=0 ./build/waltz ...
```

## Library API

The public entry points operate on CUDA device memory and accept a CUDA stream
as an opaque `void*`:

```cpp
#include <waltz.hpp>

WALTZ::Config config(nx, ny, nz);
config.errorBoundMode = WALTZ::EB_REL;
config.relErrorBound = 1e-3;

float* d_input = nullptr;
float* d_reconstructed = nullptr;
char* d_compressed = nullptr;
cudaStream_t stream = nullptr;

const size_t elements = config.num;
const size_t capacity = WALTZ::compressed_buffer_capacity<float>(elements);
cudaMalloc(&d_input, elements * sizeof(float));
cudaMalloc(&d_reconstructed, elements * sizeof(float));
cudaMalloc(&d_compressed, capacity);

const size_t compressed_bytes = WALTZ::d_compress<float>(
    config, d_input, d_compressed, "enabled", stream);
const size_t reconstructed_bytes = WALTZ::d_decompress<float>(
    config, d_compressed, d_reconstructed, nullptr, stream);
```

The caller owns all three device buffers. `d_compressed` must have at least the
capacity returned by `compressed_buffer_capacity<T>()`. In the current API the
non-null compression path argument enables production of a blob; the library
does not itself write that path.

## Runtime switches

Normal operation requires no environment variables. The most useful diagnostic
switches are:

| Variable | Effect |
| --- | --- |
| `WALTZ_REPORT_CE2E=1` | report wall time for the timed compression/decompression region; see scope below |
| `WALTZ_ASYNC_ALLOC=1` | enable explicit one-time pipeline preallocation/warmup |

The default benchmark path is cold-start compatible: no hidden warmup pass is
performed unless `WALTZ_ASYNC_ALLOC=1` is explicitly set.

`WALTZ_REPORT_CE2E` starts the compression timer after optional prewarming and
the REL range scan. It excludes input loading, input H2D transfer, and archive
file output. The decompression timer includes reconstruction file output when
a non-null output path is passed. To compare device-resident API wall time,
time the API externally and pass a null reconstruction path. The stage table's
`total` is an aggregate of GPU intervals, not a full process E2E measurement.
Throughput labels currently say GB/s, but the calculations use GiB/s.

## Validation snapshot

The synchronized 2026-09-16 repository was freshly built and installed with
CUDA 13.1 for H100 (`sm_90`), including the CLI, tuner, benchmark and optional
regression target. An external CMake consumer using
`find_package(Waltz CONFIG REQUIRED)` also compiled and ran successfully.

On H100 CUDA1, regression against a fresh build of the current production
source passed 8 configurations, each run in both builds with 3 complete
compression/decompression calls (48 round trips in total). These cover the
four fields below with WZP, plus synthetic 65x67x33 float and double fields
with both WZP and LC. Transform-mode sequences matched; final compressed size,
CR, PSNR, Max_RE and decoded-data hashes matched exactly for every
configuration. Normal SCALE and Hurricane retained plane mode. This is a
correctness/synchronization check, not a fresh cross-GPU performance benchmark.

The signed-WZP regression also passed 54 cases with 9 encode/decode calls each,
covering counter reuse, allocation growth, zero/sparse/raw chunks and unaligned
inputs. Detailed synchronization notes are in
[docs/SYNC_20260916.md](docs/SYNC_20260916.md).

The following fields retain the same compressed sizes and reconstruction
quality as the production source:

| Field | Type and dimensions | REL | CR | PSNR (dB) | Max_RE |
| --- | --- | ---: | ---: | ---: | ---: |
| NYX temperature | f32, 512 x 512 x 512 | 1e-3 | 344.03x | 80.9177 | 0.000999992 |
| Hurricane Uf48 | f32, 500 x 500 x 100 | 1e-3 | 36.38x | 72.1091 | 0.000999990 |
| SCALE U | f32, 1200 x 1200 x 98 | 1e-3 | 124.50x | 75.8876 | 0.000999997 |
| Miranda density | f64, 384 x 384 x 256 | 1e-3 | 250.90x | 78.5292 | 0.000999977 |

These values characterize the listed fields and error bound; they are not a
general compression-ratio guarantee.

## Repository layout

```text
include/                 public API and CUDA transform/codec headers
src/waltz.cu             compression and decompression pipeline
src/lossless/            WZP GPU backend
examples/waltz.cu        command-line driver
tools/wzp_gpu_bench.cu   optional WZP round-trip benchmark
tools/wzp_single_kernel_test.cu  optional signed-WZP regression
```

## Known limitations

- Only 3D float32 and float64 compression/decompression is supported.
- When the tuner selects plane mode with a Z transform, the static Z-all
  buffer requires `NZ <= 5760` for float32 or `NZ <= 2880` for float64.
  Larger plane columns currently raise an error; there is no automatic
  dynamic-shared-memory fallback for this stage.
- The command-line tool does not yet implement standalone reopening of a saved
  `.waltz` archive.
- The blob header currently follows the native host ABI and endianness.
- Cached workspaces and timing events are shared mutable state. The API is
  not a general concurrent multi-stream or multi-device interface; complete a
  split signed encode with `wzp_encode_with_sign_finish` before reusing WZP.
- Internal workspaces and a few bounded outlier side channels use fixed
  capacities; production deployments should expose capacity planning and error
  reporting at the API boundary.
- The API and on-disk format may change while the implementation is under active
  development.
