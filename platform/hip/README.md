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
Use the ordinary CUDA `WALTZ::d_compress` with defaults for byte-identical
output. The HIP decoder does not support the optional CUDA-only compact
repairs or fused short-Z convolution. See the [interop tests](../../cpu/README.md).

On wave64 devices, the XY lifting kernels use a full wave for sufficiently
wide FP32/FP64 forward strips and FP32 inverse strips. On one-plane inputs
the waves own adjacent Y strips. A plane inverse reads the even/even band
from the previously reconstructed low-pass plane when present; other bands
still come from the immutable coefficients. Smaller levels retain their
narrower or generic kernels. The encoder marks pointwise
repairs and quantizer overflow in two temporary bitmaps. It scans the tile
counts and emits only marked records in increasing index order; the bitmaps
are not part of the archive. The [edge-case test](tests/interop_edges.cpp)
checks encoded bytes and decoded values against the CPU, including partial
waves, partial bitmap tiles, zero fields, and repeated workspace use.

For large FP32 dyadic volumes, the finest inverse level still runs Z and
then XY in separate kernels. When X is divisible by four, each Z thread
can process two adjacent columns with the same indices without crossing
the low/high X split. XY pairs the final stores (decoder), or the input
loads and repair-mask updates (encoder). If the external FP32 buffer is
not 8-byte aligned, XY uses scalar I/O. This changes neither the lifting
order nor the repair decision. Plane, FP64 and coarse levels use their
original paths.

On a Hygon BW150 at REL `1e-3`, the paired-column inverse raised pooled
compression from 55.09 to 56.45 GB/s and decompression from 83.21 to
86.81 GB/s across six representative fields. Only NYX and JHTDB used the
new path.

On complete 16×16 Morton bricks with a power-of-two Z extent, the encoder
quantizes final detail bands as the transform emits them (Z for dyadic,
XY for the plane schedule). The low-pass band stays unquantized while a
coarser level still needs it. The fused path clears its 16-bit transport
buffer once and scatters only nonzero codes: sign and magnitude share a
word, with a separate 16-bit scratch entry for magnitudes of 32767 or
larger. Irregular bricks keep the independent quantization pass. One
64-lane wave per WZP chunk writes the payload to its original offset.
These are temporary or scheduling changes; the archive is unchanged.
The [packed-code test](tests/packed_codes.cpp) compares both producer
representations across ZERO, compressed and RAW blocks. Build it like
the bitmap test below, substituting `packed_codes.cpp` for `rze_maps.cpp`.

The HIP WZP decoder expands full 16 KiB magnitude blocks by reading the
four-byte repetition map and computing prefix counts at each level; partial
compressed blocks still use the LC decoder. Packed signs are expanded in
groups of 16 consecutive coefficients. Neither change modifies the wire
format. The [bitmap test](tests/rze_maps.cpp) compares the device expansion
with 320 independently constructed streams. To run it after building on
the BW150, from the repository root with the DCU SDK loaded:

```bash
hipcc --gcc-install-dir="$(dirname "$(g++ -print-libgcc-file-name)")" --offload-arch=gfx936 -O2 -std=c++17 \
  -Iplatform/hip/build-gfx936/generated -Iinclude -Iinclude/lossless/lc \
  -x hip platform/hip/tests/rze_maps.cpp -o /tmp/waltz-rze-maps
/tmp/waltz-rze-maps
```
