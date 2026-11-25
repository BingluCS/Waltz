# Waltz for NumPy, HDF5 and PyTorch

The NumPy bindings, HDF5 filter and CPU DataLoader use the
[canonical CPU codec](../cpu/README.md), with scalar or AVX2/FMA dispatch.
They require no GPU. The optional [GPU DataLoader](#decode-batches-on-the-gpu)
reads the same files with a native CUDA or HIP decoder. Neither path changes
the archive format.

## Install

From the repository root, choose one of the following. A source build needs
CMake 3.21+ and a C/C++17 toolchain. The HDF5 plugin additionally needs HDF5
development headers (for example, `libhdf5-dev` on Ubuntu). Tested on Linux.

```bash
python -m pip install .                                          # NumPy only
CMAKE_ARGS="-DWALTZ_BUILD_HDF5_FILTER=ON" python -m pip install '.[hdf5]'
CMAKE_ARGS="-DWALTZ_BUILD_HDF5_FILTER=ON" python -m pip install '.[loader]'  # includes h5py + torch
```

## NumPy: one field, one archive

```python
import numpy as np
import pywaltz

x = np.random.default_rng(7).normal(size=(17, 48, 64)).astype("float32")
blob = pywaltz.compress(x, abs_error=1e-3)
y = pywaltz.decompress(blob, shape=x.shape, dtype=x.dtype)
print("CR:", x.nbytes / len(blob))
assert np.max(np.abs(x.astype("float64") - y)) <= 1e-3
```

Supported inputs are finite native FP32/FP64 arrays of shape `(y,x)` or
`(z,y,x)`, X fastest on the wire; the CPU CLI takes `nx ny nz` in the **reverse**
order. Non-contiguous arrays are accepted, and compression never overwrites the
input. Keep the original shape and dtype alongside standalone bytes; decoding
does not need the bound. For a field-wide range-relative tolerance, compute
the **positive absolute bound once** from the original field, e.g.
`rel * (float(x.max()) - float(x.min()))` for nonconstant data. Constant
fields require an explicit positive absolute bound.

## HDF5: one chunk, one archive

```python
import h5py
import numpy as np
from pywaltz.hdf5 import create_dataset, enable_filter

# Four volumes, indexed as (sample, z, y, x). "x" mode will not overwrite a file.
a = np.random.default_rng(8).normal(size=(4, 17, 48, 64)).astype("float32")
with h5py.File("volumes.h5", "x") as f:
    d = create_dataset(f, "fields", shape=a.shape, chunks=(1, 17, 24, 32),
                       dtype=a.dtype, abs_error=1e-3)
    d[...] = a

# In a fresh Python process importing pywaltz, call enable_filter() before reading.
enable_filter()
with h5py.File("volumes.h5", "r") as f:
    tile = f["fields"][2, :, :24, :32]  # only the intersecting chunk is needed
    assert np.max(np.abs(tile.astype("float64") - a[2, :, :24, :32])) <= 1e-3
```

The dataset remains a numeric dataset, not an opaque byte array. The filter is
mandatory: HDF5 must not silently skip Waltz. All chunks use the same supplied
absolute bound. For `(n,z,y,x)` stacks, use a leading chunk size of 1. With
2-D frames `(n,y,x)`, use chunks `(1,cy,cx)` to keep samples independent.
Choose spatial chunks to match the slices you actually read. Each chunk is an
independent field (edge chunks may be padded), so CR can differ from compressing
the whole volume. Prefer writing complete chunks once: a partial overwrite of
an existing lossy chunk may compress already reconstructed values again.

For an ordinary h5py, C/C++ HDF5, or HDF5 command-line reader **without**
Python registration, set this in its environment **before launching it**:

```bash
export HDF5_PLUGIN_PATH="$(python -c 'from pywaltz.hdf5 import plugin_path; print(plugin_path())')${HDF5_PLUGIN_PATH:+:$HDF5_PLUGIN_PATH}"
```

On a different machine, install the plugin for **that** machine; the file
does not embed the plugin. Filter ID `61197` is private/unregistered; do not
load an unrelated filter using that ID. Currently the filter accepts only
little-endian IEEE FP32/FP64 chunks. Use `create_dataset` to supply the actual
chunk geometry and type, not just the numeric ID. The plugin is a normal H5Z
module: for non-Python builds it can also be built with
`cmake -S cpu -B cpu/build-h5 -DWALTZ_BUILD_HDF5_FILTER=ON` and
`cmake --build cpu/build-h5 --target h5waltz`; point `HDF5_PLUGIN_PATH` at
`cpu/build-h5` in that case.

## PyTorch: read samples or time pairs

```python
from torch.utils.data import DataLoader
from pywaltz.dataloader import WaltzDataset

if __name__ == "__main__":
    # For ordered time frames: item i is (fields[i], fields[i+1]).
    ds = WaltzDataset("volumes.h5", key="fields", lead=1)
    for x, target in DataLoader(ds, batch_size=2, num_workers=2):
        print(x.shape, target.shape)  # (batch,z,y,x); tensors are on the CPU
```

Use the default `lead=0` for unrelated fields: each item is one tensor, not
a pair. Each worker opens its own read-only HDF5 handle on first use. A raw
3-D field without a sample axis is interpreted by the loader as a stack of 2-D
slices; for one 3-D sample store it as `(1,z,y,x)`. If training on a GPU,
transfer batches in the training process, not in a DataLoader worker.

For real input, [pack_volumes.py](examples/pack_volumes.py) maps headerless
raw volumes from disk, and [read_batches.py](examples/read_batches.py)
iterates the resulting dataset. For example, two time steps of the same field:

```bash
python python/examples/pack_volumes.py train.h5 step0.f32 step1.f32 \
    --shape 128 256 512 --dtype float32 --chunks 32 64 64 --abs-error 1e-3
python python/examples/read_batches.py train.h5 --lead 1 --workers 2
```

### Decode batches on the GPU

Use `WaltzGPUDataLoader` **instead of** wrapping `WaltzDataset` in a PyTorch
DataLoader. Its workers read raw HDF5 chunks without invoking the CPU HDF5
filter. They pack the compressed archives and an offset table into one byte
tensor per batch. The native backend validates the headers from that tensor's
host copy, then runs WZP and each inverse lifting stage across all chunks on
the device. The final stage places valid samples directly in the output tensor,
including partial edge tiles. No decoded field is copied back to the host;
when repairs are present, one four-byte status is returned per batch.

Install pywaltz, h5py and a **device-matched** PyTorch build. Use NVIDIA's
CUDA build for NVIDIA and the vendor's HIP build for DCU; both expose
`torch.device("cuda:0")`. Build and select **one** matching native library,
from the same source revision:

```bash
# NVIDIA H100 example (change 90 for another GPU)
cmake -S . -B build -DCMAKE_CUDA_ARCHITECTURES=90 -DBUILD_SHARED_LIBS=ON
cmake --build build --target waltz_shared -j4
export WALTZ_CUDA_LIBRARY="$PWD/build/libwaltz.so"

# OR: DCU BW150 with the DTK environment loaded
bash platform/hip/build.sh gfx936
export WALTZ_HIP_LIBRARY="$PWD/platform/hip/build-gfx936/libwaltz_hip_native.so"
```

```python
from pywaltz.gpu_dataloader import WaltzGPUDataLoader

if __name__ == "__main__":
    batches = WaltzGPUDataLoader("train.h5", key="fields", device="cuda:0",
                                 batch_size=4, num_workers=2, lead=1)
    for x, target in batches:
        assert x.is_cuda and target.is_cuda  # already on device; no .to() needed
```

Use `lead=0` for individual samples. The CLI equivalent is
`python python/examples/read_batches.py train.h5 --workers 2 --device-decode cuda:0`.
An explicit `library="/absolute/path/to/lib...so"` overrides the environment.
Only the writer and the CPU reader need the HDF5 filter plugin; raw GPU
reads need h5py but do not need to load that plugin.

- Input: **trusted** canonical Waltz files with Waltz as the only, mandatory
  filter. Use `create_dataset` above; default CUDA-only encoding is not
  supported here. There is no silent CPU fallback.
- Layout: fixed `(n,y,x)` or `(n,z,y,x)`, with a leading chunk size of 1.
  Spatial tiling and partial edge tiles work; unwritten chunks use the HDF5
  fill value. Unsupported filters or layouts raise an error.
- Execution: one training process per accelerator. On both CUDA and HIP the
  batch uses PyTorch's current stream and waits for completion once per batch.
  The DataLoader default, `decode_mode="auto"`, uses the batch backend when
  available and within its launch and memory limits (256 MiB of scratch);
  otherwise it uses the previous per-chunk **GPU** path. Use `"batch"` to
  require batching, or `"serial"` to explicitly select the previous decoder.
  The file format and reconstruction rules are identical.
- To compare the new and previous GPU paths on your own file, run
  `python python/examples/bench_dataloader.py train.h5 --baseline serial`.
  Omit `--baseline serial` for CPU decode **plus** host-to-device copy.
  It warms both paths, alternates their order and times complete epochs until
  the tensors are ready on device. This is loader throughput, not kernel or
  model-training throughput.

Run the integration tests after installing the loader-enabled build:

```bash
python -m unittest discover -s python/tests -v
```

Set `WALTZ_TEST_GPU_LOADER=1` as well as the library variable above to run
the real-device test (including tiles, fill chunks, a non-default stream and
repeated epochs).
