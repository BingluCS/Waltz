"""Set WALTZ_TEST_GPU_LOADER=1 and WALTZ_{CUDA,HIP}_LIBRARY for device tests."""
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import h5py
import numpy as np
import torch

from pywaltz._raw_chunks import RawFrames, pack
from pywaltz.gpu_dataloader import WaltzGPUDataLoader, _Decoder
from pywaltz.hdf5 import create_dataset


def fixture(path):
    with h5py.File(path, "x") as f:
        for key,shape,chunk,dt in [
            ("tiles", (5,19,23,29), (1,12,15,17), "float64"),
            ("frames", (5,25,31), (1,25,31), "float32"),
        ]:
            d = create_dataset(f, key, shape=shape, chunks=chunk, dtype=dt, abs_error=.001)
            a = np.random.default_rng(17).normal(size=shape).astype(dt)
            d[:3] = a[:3]
            # Samples 3 and 4 are unallocated. Test an all-fill batch as well.
    # The writer's raw chunk cache can still contain the original values.
    # Reopen to compare against the CPU *reconstruction* on disk.
    with h5py.File(path, "r") as f:
        return {key: f[key][:] for key in ("tiles", "frames")}


class RawChunkTests(unittest.TestCase):
    def test_wire_only_no_h5py_decode(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"train.h5"
            fixture(path)
            with h5py.File(path) as f:
                mask, blob = f["frames"].id.read_direct_chunk((0,0,0))
                self.assertEqual(mask, 0)
            with patch.object(h5py.Dataset, "__getitem__", side_effect=AssertionError("CPU decode")):
                ds = RawFrames(path, "frames", 0)
                wire = pack([ds[0], ds[3]])
                self.assertEqual(wire.spans, [(0,len(blob)), (0,0)])
                self.assertEqual(wire.payload.numpy().tobytes(), blob)

    def test_incompatible_layout(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"plain.h5"
            with h5py.File(path, "x") as f:
                f.create_dataset("x", shape=(4,9,13), chunks=(1,9,13), dtype="f", compression="gzip")
            with self.assertRaisesRegex(ValueError, "only HDF5 filter"):
                WaltzGPUDataLoader(path,"x")


@unittest.skipUnless(os.getenv("WALTZ_TEST_GPU_LOADER")=="1", "opt-in native device test")
class DeviceTests(unittest.TestCase):
    @unittest.skipUnless(torch.version.hip, "HIP serial-session lifecycle")
    def test_lazy_hip_session(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"lazy.h5"
            ref = fixture(path)["frames"][:1].tobytes()
            ds = RawFrames(path, "frames", 0)
            wire = pack([ds[0]], indexed=True)
            device = torch.device("cuda", torch.cuda.current_device())
            for mode in ("batch", "auto", "serial"):
                decoder = _Decoder(ds, device, None, mode)
                try:
                    self.assertIsNone(decoder.session)
                    self.assertEqual(decoder.batch(ds, wire).cpu().numpy().tobytes(), ref)
                    if mode != "serial":
                        self.assertIsNone(decoder.session)
                    if mode == "batch":
                        continue
                    decoder.multi = None  # Exercise auto fallback after a successful batch.
                    self.assertEqual(decoder.batch(ds, wire).cpu().numpy().tobytes(), ref)
                    self.assertIsNotNone(decoder.session)
                finally:
                    decoder.close()
                self.assertIsNone(decoder.session)

    def test_gpu_matches_h5py(self):
        self.assertTrue(torch.cuda.is_available())
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"train.h5"
            expected = fixture(path)
            stream = torch.cuda.Stream()
            for key,lead,workers in [("tiles",1,2), ("frames",0,0)]:
                with self.subTest(key=key), torch.cuda.stream(stream):
                    loader = WaltzGPUDataLoader(path,key,lead=lead,num_workers=workers,
                                                batch_size=2,persistent_workers=bool(workers),
                                                decode_mode="batch")
                    for epoch in range(2):
                        i = 0
                        for result in loader:
                            batch = result if lead else (result,)
                            for j,x in enumerate(batch):
                                self.assertTrue(x.is_cuda)
                                ref = expected[key][i+j:i+j+len(x)]
                                self.assertEqual(tuple(x.shape), ref.shape)
                                self.assertEqual(x.cpu().numpy().tobytes(), ref.tobytes())
                                self.assertEqual((2*x+1).cpu().numpy().tobytes(), (2*ref+1).tobytes())
                            i += len(batch[0])
                        self.assertEqual(i, 5-lead)

    def test_batch_schedules(self):
        # Exercise coarse XYZ, both plane phases, Z only, zero levels, edge
        # tiles, outliers, and nontrivial values in every sample position.
        shapes = [(64,64,64), (35,19,21), (11,45,48), (37,5,7), (4,6,7)]
        for dt in ("float32", "float64"):
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp)/"schedules.h5"
                with h5py.File(path,"x") as f:
                    for i,tile in enumerate(shapes):
                        shape=(3,tile[0]+1,tile[1],tile[2])
                        ds=create_dataset(f,str(i),shape=shape,chunks=(1,*tile),dtype=dt,abs_error=.001)
                        a=np.random.default_rng(i).normal(size=shape).astype(dt)
                        a[:,1,2,3]=100.
                        ds[:]=a
                with h5py.File(path,"r") as f:
                    for i in range(len(shapes)):
                        with self.subTest(dtype=dt, tile=shapes[i]):
                            x=next(iter(WaltzGPUDataLoader(path,str(i),batch_size=3,decode_mode="batch")))
                            np.testing.assert_array_equal(x.cpu().numpy().view("u"+str(x.element_size())),
                                f[str(i)][:].view("u"+str(x.element_size())))
