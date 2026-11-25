import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np
import pywaltz
from pywaltz.hdf5 import FILTER_ID, create_dataset, plugin_path


class FilterTests(unittest.TestCase):
    def test_chunk_archives_and_edge_reads(self):
        for shape, chunks, dtype in (
            ((33, 43), (17, 22), "float32"),
            ((18, 37, 43), (16, 21, 27), "float64"),
            ((3, 18, 37, 43), (1, 16, 21, 27), "float32"),
        ):
            with self.subTest(shape=shape), tempfile.TemporaryDirectory() as temp:
                path = Path(temp) / "vol.h5"
                a = np.random.default_rng(7).uniform(-4, 4, shape).astype(dtype)
                with h5py.File(path, "x") as f:
                    d = create_dataset(f, "x", shape=shape, chunks=chunks, dtype=dtype, abs_error=1e-3)
                    d[...] = a
                with h5py.File(path, "r") as f:
                    d = f["x"]
                    fid, flags, *_ = d.id.get_create_plist().get_filter(0)
                    self.assertEqual(fid, FILTER_ID)
                    self.assertFalse(flags & h5py.h5z.FLAG_OPTIONAL)
                    for sel in d.iter_chunks():
                        offset = tuple(s.start for s in sel)
                        mask, blob = d.id.read_direct_chunk(offset)
                        self.assertEqual(mask, 0)
                        tile = np.zeros(chunks, dtype=dtype)
                        tile[tuple(slice(0, s.stop - s.start) for s in sel)] = a[sel]
                        if tile.ndim == 4:
                            tile = tile[0]
                        self.assertEqual(blob, pywaltz.compress(tile, abs_error=1e-3))
                    self.assertLessEqual(np.max(abs(d[:].astype("d") - a)), 1e-3)
                # Fresh, ordinary h5py process: no import or manual registration of pywaltz.
                env = dict(os.environ, HDF5_PLUGIN_PATH=str(plugin_path()))
                subprocess.run([sys.executable, "-c",
                                "import sys,h5py; f=h5py.File(sys.argv[1]); "
                                "print(f['x'][...].shape); f.close()", str(path)],
                               check=True, env=env, stdout=subprocess.DEVNULL)

    def test_invalid_dataset(self):
        with tempfile.TemporaryDirectory() as temp, h5py.File(Path(temp) / "x.h5", "x") as f:
            for chunks, dt, exc in (((2, 2, 2, 2), "f", ValueError),
                                    ((1, 2, 2, 2), "i", TypeError),
                                    ((1, 2, 2, 2), ">f4", TypeError)):
                with self.assertRaises(exc):
                    create_dataset(f, "x", shape=(2, 2, 2, 2), chunks=chunks, dtype=dt, abs_error=.1)
