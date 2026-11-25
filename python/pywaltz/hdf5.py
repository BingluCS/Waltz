"""Create a regular chunked HDF5 dataset with the mandatory Waltz filter."""

import operator
import os
from contextlib import closing
from pathlib import Path
import struct
import sys

import h5py

from .codec import _bound, _geometry


FILTER_ID = 61197  # Private HDF5 filter ID. See python/README.md.


def plugin_path():
    """Directory of the wheel's HDF5 plugin, if built."""
    return Path(__file__).with_name("plugins")


def enable_filter():
    """Enable the bundled plugin or use the one on HDF5_PLUGIN_PATH."""
    path = os.fsencode(plugin_path())
    pl = h5py.h5pl
    if os.path.isdir(path) and path not in (pl.get(i) for i in range(pl.size())):
        pl.prepend(path)
    if not h5py.h5z.filter_avail(FILTER_ID):
        raise RuntimeError("Waltz HDF5 filter not found; build with "
                           "WALTZ_BUILD_HDF5_FILTER=ON or set HDF5_PLUGIN_PATH")


def create_dataset(group, name, *, shape, chunks, dtype, abs_error):
    """Create an empty fixed-size dataset. Assign data using normal h5py slices.

    2-D / 3-D chunks use (y,x) / (z,y,x). A 4-D stack (n,z,y,x)
    must have one sample per chunk along n. Edge chunks use HDF5 zero fill.
    """
    shape = tuple(map(operator.index, shape))
    chunks = tuple(map(operator.index, chunks))
    if len(shape) not in (2, 3, 4) or len(chunks) != len(shape):
        raise ValueError("shape and chunks must have the same rank (2, 3 or 4)")
    if any(c <= 0 or c > s for c, s in zip(chunks, shape)):
        raise ValueError("require 0 < chunk extent <= dataset extent on each axis")
    if len(chunks) == 4 and chunks[0] != 1:
        raise ValueError("4-D data must use a leading chunk extent of 1")
    if sys.byteorder != "little":
        raise ValueError("the Waltz HDF5 filter currently requires a little-endian host")
    dims, dtype = _geometry(chunks[-3:], dtype)
    lo, hi = struct.unpack("<II", struct.pack("<d", _bound(abs_error)))
    enable_filter()
    # h5py's high-level compression=ID marks external filters OPTIONAL.
    # Use a mandatory filter so a codec error cannot silently store raw chunks.
    with closing(h5py.h5p.create(h5py.h5p.DATASET_CREATE)) as dcpl:
        dcpl.set_chunk(chunks)
        dcpl.set_filter(FILTER_ID, h5py.h5z.FLAG_MANDATORY, (*dims, dtype.itemsize, lo, hi))
        with closing(h5py.h5s.create_simple(shape)) as space, closing(h5py.h5t.py_create(dtype)) as tid:
            did = h5py.h5d.create(group.id, name.encode("utf-8"), tid, space, dcpl=dcpl)
    return h5py.Dataset(did)
