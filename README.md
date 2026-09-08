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

The source update dated 2026-09-08 includes:

- Dyadic DWT: fused XY followed by a single-level Z kernel, with static shared
  memory or direct global loads selected for each level. Float and double use
  the common `dwt_3d_z_single_static<T>` / `dwt_3d_z_single_global<T>` entries.
- Plane DWT: all Z levels in one launch, followed by fused XY/quantization
  launches per level. Plane Z-all retains its shared-memory configuration.
- `QuantMode::None`, `High`, and `All` name the quantization modes. Z tile
  alignment and quantization-block regularity are handled independently.
- Updated single-level IDWT Z kernels, including the bounded 32-bit reflection
  calculation in the float4 boundary path that resolved the observed RTX PRO
  6000 timeout in repeated tests.
- Removal of cooperative launch paths, unused Y-chain code, and disabled
  shared-memory search code. Compression overlaps lossless coding with IDWT;
  supported WZP decode layouts fuse reordering and dequantization.

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
- NVIDIA CUDA Toolkit 11.8 or newer
- An NVIDIA GPU with compute capability 7.0 or newer

Earlier revisions were validated on NVIDIA H100, H200, and RTX 4090 GPUs.
The updated IDWT reflection path was validated on H100 and RTX PRO 6000
Blackwell; the complete synchronized repository is checked separately below.
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
- `build/tune_demo`, the standalone topology-tuner harness

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

`-o` is optional. When supplied, the driver immediately decodes the in-memory
blob, writes the reconstructed field, and reports Max_E, Max_RE, PSNR, NRMSE,
compressed size, compression ratio, and stage timings. The driver writes the
compressed blob named by `-z`; it does not yet provide a separate command that
reopens an existing blob in a later process.

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
| `WALTZ_WZP=0` | select the legacy LC magnitude backend instead of default WZP |
| `WALTZ_WZP_DEQUANT_FUSE_OFF=1` | disable fused WZP decode/reorder/dequantization for comparison |

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

The synchronized 2026-09-08 repository is validated on H100 with the default
WZP backend and prewarming disabled. The following fields retain the same
compressed sizes and reconstruction quality as the production source:

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
src/lossless/            LC and WZP GPU backends
examples/waltz.cu        command-line driver
examples/tune_demo.cu    transform-topology tuner harness
tools/wzp_gpu_bench.cu   optional WZP round-trip benchmark
```

## Known limitations

- Only 3D float32 and float64 compression/decompression is supported.
- The command-line tool does not yet implement standalone reopening of a saved
  `.waltz` archive.
- The blob header currently follows the native host ABI and endianness.
- Internal workspaces and a few bounded outlier side channels use fixed
  capacities; production deployments should expose capacity planning and error
  reporting at the API boundary.
- The API and on-disk format may change while the implementation is under active
  development.
