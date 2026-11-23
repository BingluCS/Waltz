# Native AMD/DCU backend

The standalone [C API](waltz_hip.h) uses native HIP kernels. The default
target is `gfx936` on the DCU SDK; load the SDK with HIP and rocPRIM
before building:

```bash
bash platform/hip/build.sh gfx936
```

The optional second argument is an output directory. The build generates
HIP kernels from the checked-out CUDA wavelet, Morton and WZP source and
produces `libwaltz_hip_native.so` in that directory. It does not require
the CUDA library or write to the shared CUDA sources.

This backend uses the canonical native-repair, all-level lifting profile.
It does not read the ordinary CUDA CLI's compact repairs or CUDA-only fused
short-Z convolution. For byte-portable CUDA compression use the
[canonical wrapper](../../cpu/README.md).
