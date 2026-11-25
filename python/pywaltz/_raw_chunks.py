"""Worker-side raw HDF5 I/O. No accelerator initialization or CPU decoding."""
from dataclasses import dataclass
from itertools import product
import math
import operator
import os
import struct

import h5py
import torch
from torch.utils.data import Dataset, get_worker_info

from .hdf5 import FILTER_ID


class RawFrames(Dataset):
    def __init__(self, path, key, lead):
        self.path, self.key, self.lead = os.fspath(path), key, operator.index(lead)
        with h5py.File(path, "r") as f:
            d = f[key]
            if d.ndim not in (3, 4) or not d.chunks or d.chunks[0] != 1:
                raise ValueError("expected (n,y,x) or (n,z,y,x) with one sample per chunk along n")
            dcpl = d.id.get_create_plist()
            if dcpl.get_nfilters() != 1 or dcpl.get_filter(0)[0] != FILTER_ID:
                raise ValueError("expected Waltz as the only HDF5 filter")
            self.shape, self.tile = d.shape[1:], d.chunks[1:]
            # Immutable training files: for a fully allocated dataset, the
            # chunk index need not be consulted twice per direct read. Don't
            # enumerate a huge index just to enable this shortcut.
            slots = d.shape[0]*math.prod((s+c-1)//c for s,c in zip(self.shape,self.tile))
            self.dense = slots<=65536 and d.id.get_num_chunks()==slots
            self.xyz = (*reversed(self.tile), 1)[:3]
            self.width = d.dtype.itemsize
            if d.dtype.str not in ("<f4", "<f8"):
                raise TypeError("GPU decoding requires little-endian FP32/FP64")
            if math.prod(self.tile)*self.width > 0x7fffffff:
                raise ValueError("chunk exceeds the portable GPU size limit")
            _, flags, cd, _ = dcpl.get_filter(0)
            if flags & h5py.h5z.FLAG_OPTIONAL or len(cd) != 6 or cd[:4] != (*self.xyz, self.width):
                raise ValueError("Waltz filter metadata does not match chunk geometry or dtype")
            self.bound, = struct.unpack("<d", struct.pack("<II", *cd[4:]))
            if not (math.isfinite(self.bound) and self.bound > 0):
                raise ValueError("invalid error bound in the Waltz filter metadata")
            if not 0 <= self.lead < d.shape[0]:
                raise ValueError("lead must be in [0, number of samples)")
            self.length = d.shape[0] - self.lead
            self.fill = float(d.fillvalue)
        self._file = self._data = None

    def __len__(self):
        return self.length

    def origins(self):
        return product(*(range(0, n, c) for n, c in zip(self.shape, self.tile)))

    def _frame(self, d, i):
        records = []
        for origin in self.origins():
            offset = (i, *origin)
            if not self.dense and d.id.get_chunk_info_by_coord(offset).size == 0:
                records.append(None)  # HDF5 fill value
                continue
            mask, blob = d.id.read_direct_chunk(offset)
            if mask or len(blob) < 32:
                raise ValueError(f"skipped filter or short Waltz chunk at {offset}")
            records.append(blob)
        return records

    def _sample(self, d, i):
        return [self._frame(d, j) for j in ((i, i+self.lead) if self.lead else (i,))]

    def __getitem__(self, i):
        if not 0 <= i < self.length:
            raise IndexError(i)
        if get_worker_info() is None:
            with h5py.File(self.path, "r") as f:
                return self._sample(f[self.key], i)
        if self._file is None:
            self._file = h5py.File(self.path, "r")
            self._data = self._file[self.key]
        return self._sample(self._data, i)

    def __getitems__(self, indices):
        # Torch dispatches one read request per batch. With zero workers,
        # keep a handle for this call only, not once for every sample.
        if get_worker_info() is None:
            for i in indices:
                if not 0 <= i < self.length:
                    raise IndexError(i)
            with h5py.File(self.path,"r") as f:
                d=f[self.key]
                return [self._sample(d,i) for i in indices]
        return [self[i] for i in indices]


@dataclass
class WireBatch:
    payload: torch.Tensor
    spans: list
    count: int
    indexed: bool = False

    def pin_memory(self):
        return WireBatch(self.payload.pin_memory(), self.spans, self.count, self.indexed)


def pack(samples, *, indexed=False):
    # One tensor for worker IPC, optional pinning and one H2D per batch.
    data, spans = bytearray(), []
    for sample in samples:
        for frame in sample:
            for blob in frame:
                if blob is None:
                    spans.append((0, 0))
                else:
                    data.extend(b"\0" * (-len(data) % 16))
                    spans.append((len(data), len(blob)))
                    data.extend(blob)
    if indexed:
        data.extend(b"\0" * (-len(data) % 16))
        entry = struct.Struct("<QQ")
        for offset, size in spans:
            data.extend(entry.pack(offset, size))
    payload = torch.frombuffer(data, dtype=torch.uint8) if data else torch.empty(0, dtype=torch.uint8)
    return WireBatch(payload, spans, len(samples), indexed)
