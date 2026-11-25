"""Thin bindings to waltz_cpu.h; archives have no Python-specific header."""

import ctypes as C
import math
import operator
import sys
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path

import numpy as np


class _Config(C.Structure):
    _fields_ = [("nx", C.c_uint32), ("ny", C.c_uint32), ("nz", C.c_uint32),
                ("type", C.c_int), ("absolute_error", C.c_double), ("backend", C.c_int)]


def _geometry(shape, dtype):
    shape = tuple(operator.index(v) for v in shape)
    dtype = np.dtype(dtype)
    if dtype not in (np.dtype("float32"), np.dtype("float64")):
        raise TypeError("Waltz requires native float32 or float64")
    if len(shape) not in (2, 3) or min(shape) <= 0:
        raise ValueError("expected positive (y,x) or (z,y,x) dimensions")
    if math.prod(shape) * dtype.itemsize > 0xffffffff:
        raise ValueError("field exceeds the native Waltz archive limit")
    return (*reversed(shape), 1)[:3], dtype


def _bound(value):
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError("abs_error must be finite and positive")
    return value


@lru_cache(maxsize=1)
def _library():
    suffix = ".dylib" if sys.platform == "darwin" else ".so"
    lib = C.CDLL(str(Path(__file__).with_name("libwaltz_cpu" + suffix)))
    ptr, size, cfg = C.c_void_p, C.c_size_t, C.POINTER(_Config)
    for name, result, args in [
        ("create", ptr, []), ("destroy", None, [ptr]), ("last_error", C.c_char_p, [ptr]),
        ("max_compressed_bytes", size, [cfg]),
        ("compress", C.c_int, [ptr, cfg, ptr, ptr, size, C.POINTER(size)]),
        ("decompress", C.c_int, [ptr, cfg, ptr, size, ptr, size]),
    ]:
        fn = getattr(lib, "waltz_cpu_" + name)
        fn.restype, fn.argtypes = result, args
    return lib


@contextmanager
def _workspace(lib):
    ws = lib.waltz_cpu_create()
    if not ws:
        raise MemoryError("cannot allocate Waltz workspace")
    try:
        yield ws
    finally:
        lib.waltz_cpu_destroy(ws)


def compress(array, *, abs_error):
    """Return canonical archive bytes. The input is never modified."""
    a = np.asarray(array)
    dims, dtype = _geometry(a.shape, a.dtype)
    cfg = _Config(*dims, dtype.itemsize, _bound(abs_error), 0)
    a = np.require(a, requirements=["C", "A"])
    lib = _library()
    capacity = lib.waltz_cpu_max_compressed_bytes(C.byref(cfg))
    if not capacity:
        raise ValueError("unsupported Waltz configuration")
    out = np.empty(capacity, dtype=np.uint8)
    written = C.c_size_t()
    with _workspace(lib) as ws:
        if lib.waltz_cpu_compress(ws, C.byref(cfg), a.ctypes.data, out.ctypes.data,
                                  capacity, C.byref(written)):
            raise ValueError(lib.waltz_cpu_last_error(ws).decode())
    return out[:written.value].tobytes()


def decompress(archive, *, shape, dtype):
    """Decode bytes using the original NumPy shape and dtype (not stored together in the archive)."""
    shape = tuple(shape)
    dims, dtype = _geometry(shape, dtype)
    cfg = _Config(*dims, dtype.itemsize, 0.0, 0)
    src = np.frombuffer(archive, dtype=np.uint8)
    if src.size < 32:
        raise ValueError("truncated Waltz header")
    out = np.empty(shape, dtype=dtype)
    lib = _library()
    with _workspace(lib) as ws:
        if lib.waltz_cpu_decompress(ws, C.byref(cfg), src.ctypes.data, src.size,
                                    out.ctypes.data, out.nbytes):
            raise ValueError(lib.waltz_cpu_last_error(ws).decode())
    return out
