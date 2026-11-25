"""PyTorch map-style dataset for Waltz-filtered HDF5 samples."""

import operator
import os

import h5py
import torch
from torch.utils.data import Dataset, get_worker_info

from .hdf5 import enable_filter


class WaltzDataset(Dataset):
    """Read fields[i], or (fields[i], fields[i+lead]) for a positive lead.

    The first axis is samples/time. Only indexing triggers decompression.
    """

    def __init__(self, path, key="fields", *, lead=0):
        self.path, self.key = os.fspath(path), key
        self.lead = operator.index(lead)
        enable_filter()
        with h5py.File(self.path, "r") as f:
            d = f[key]
            if d.ndim not in (3, 4):
                raise ValueError("expected a stack (n,y,x) or (n,z,y,x)")
            if not 0 <= self.lead < d.shape[0]:
                raise ValueError("lead must be in [0, number of samples)")
            self.length = d.shape[0] - self.lead
        self._file = self._dataset = None

    def __len__(self):
        return self.length

    def __getstate__(self):
        return dict(self.__dict__, _file=None, _dataset=None)

    def _sample(self, data, i):
        x = torch.from_numpy(data[i])
        return (x, torch.from_numpy(data[i + self.lead])) if self.lead else x

    def __getitem__(self, index):
        i = operator.index(index)
        if not 0 <= i < self.length:
            raise IndexError(i)
        if get_worker_info() is None:
            # Never leave a parent handle that might be inherited on fork.
            with h5py.File(self.path, "r") as f:
                return self._sample(f[self.key], i)
        if self._file is None:
            enable_filter()  # spawn starts with a new HDF5 plugin registry
            self._file = h5py.File(self.path, "r")
            self._dataset = self._file[self.key]
        return self._sample(self._dataset, i)

    def close(self):
        if self._file is not None:
            self._file.close()
            self._file = self._dataset = None
