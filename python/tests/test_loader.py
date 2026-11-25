import multiprocessing as mp
from pathlib import Path
import tempfile
import unittest

import h5py
import numpy as np
from torch.utils.data import DataLoader
from pywaltz.dataloader import WaltzDataset
from pywaltz.hdf5 import create_dataset


class LoaderTests(unittest.TestCase):
    def test_worker_reads_and_pair_order(self):
        a = np.random.default_rng(13).normal(size=(5, 17, 20, 23)).astype("f")
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "train.h5"
            with h5py.File(path, "x") as f:
                d = create_dataset(f, "fields", shape=a.shape, chunks=(1, 17, 12, 20),
                                   dtype=a.dtype, abs_error=.001)
                d[...] = a
            with h5py.File(path, "r") as f:
                expected = f["fields"][:]
            np.testing.assert_array_equal(WaltzDataset(path)[0].numpy(), expected[0])
            ds = WaltzDataset(path, lead=1)
            self.assertEqual(len(ds), 4)
            contexts = [None] + [c for c in ("spawn", "fork") if c in mp.get_all_start_methods()]
            for ctx in contexts:
                with self.subTest(context=ctx):
                    batches = list(DataLoader(ds, batch_size=2, num_workers=2 if ctx else 0,
                                              multiprocessing_context=ctx))
                    np.testing.assert_array_equal(np.concatenate([p[0].numpy() for p in batches]), expected[:-1])
                    np.testing.assert_array_equal(np.concatenate([p[1].numpy() for p in batches]), expected[1:])
