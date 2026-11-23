# Waltz CPU and canonical CUDA profile

The CPU C API is [waltz_cpu.h](include/waltz_cpu.h). A scalar path is the
reference, and runtime dispatch enables AVX2/FMA where available. Build and
test without CUDA:

```bash
cmake -S cpu -B cpu/build -DCMAKE_BUILD_TYPE=Release
cmake --build cpu/build -j4
ctest --test-dir cpu/build --output-on-failure
```

The CPU CLI accepts explicit absolute error bounds:

```bash
cpu/build/waltz_cpu_cli c f32 72 70 25 0.001 input.f32 field.waltz auto
cpu/build/waltz_cpu_cli d f32 72 70 25 0.001 field.waltz output.f32 auto
```

For CUDA interchange, use [canonicalize.h](gpu/canonicalize.h) and build
the canonical wrapper against the CUDA library built from this checkout:

```bash
cmake -S cpu -B cpu/build-interop -DCMAKE_BUILD_TYPE=Release \
  -DWALTZ_CPU_BUILD_GPU_INTEROP=ON -DWALTZ_GPU_LIBRARY="$PWD/build/libwaltz.so" \
  -DCMAKE_CUDA_ARCHITECTURES=90
cmake --build cpu/build-interop -j4
CUDA_VISIBLE_DEVICES=0 cpu/build-interop/waltz_cpu_gpu_interop --full
```

The wrapper requires `WALTZ_HALO3D=all` (or unset) and
`WALTZ_SMALL_PWE_SCALE=0`. It canonicalizes WZP groups and repair records
and checks the strict pointwise bound before returning the archive.
Setting the environment alone on the regular CUDA CLI does not
canonicalize its output. Use the canonical wrapper for byte comparisons.
CUDA-only short-Z convolution and compact repair output are not part of
the canonical profile.
