"""Pack same-shape raw fields into a Waltz HDF5 sample stack, in argument order."""

import argparse
import math
from pathlib import Path

import h5py
import numpy as np

from pywaltz.hdf5 import create_dataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("inputs", type=Path, nargs="+")
    parser.add_argument("--shape", nargs=3, type=int, required=True, metavar=("Z", "Y", "X"))
    parser.add_argument("--chunks", nargs=3, type=int, metavar=("Z", "Y", "X"))
    parser.add_argument("--dtype", choices=["float32", "float64"], required=True)
    parser.add_argument("--abs-error", type=float, required=True)
    args = parser.parse_args()

    shape = tuple(args.shape)
    dtype = np.dtype(args.dtype)
    chunks = tuple(args.chunks) if args.chunks else tuple(min(v, 64) for v in shape)
    if min(shape) <= 0 or any(not 0 < c <= s for c, s in zip(chunks, shape)):
        parser.error("expected positive dimensions and chunks no larger than the field")
    if not math.isfinite(args.abs_error) or args.abs_error <= 0:
        parser.error("--abs-error must be positive and finite")
    for path in args.inputs:
        if path.stat().st_size != math.prod(shape) * dtype.itemsize:
            parser.error(f"{path}: file size does not match --shape and --dtype")
    with h5py.File(args.output, "x") as f:  # never overwrite an existing result
        data = create_dataset(f, "fields", shape=(len(args.inputs), *shape),
                              chunks=(1, *chunks), dtype=dtype, abs_error=args.abs_error)
        for i, path in enumerate(args.inputs):
            data[i] = np.memmap(path, dtype=dtype, mode="r", shape=shape)
        f.flush()
        print(f"{data.name}  shape={data.shape}  chunks={data.chunks}  "
              f"chunk payload={data.id.get_storage_size()} bytes")


if __name__ == "__main__":
    main()
