import unittest

import numpy as np
import pywaltz


class CodecTests(unittest.TestCase):
    def test_round_trip(self):
        for dt in ("float32", "float64"):
            for shape in ((31, 47), (17, 32, 35), (32, 32, 32)):
                with self.subTest(dtype=dt, shape=shape):
                    a = np.random.default_rng(9).normal(size=shape).astype(dt)[..., ::-1]
                    before = a.copy()
                    blob = pywaltz.compress(a, abs_error=1e-3)
                    self.assertEqual(blob, pywaltz.compress(before, abs_error=1e-3))
                    b = pywaltz.decompress(blob, shape=a.shape, dtype=a.dtype)
                    np.testing.assert_array_equal(a, before)
                    self.assertEqual(b.dtype, a.dtype)
                    self.assertEqual(b.shape, a.shape)
                    self.assertLessEqual(np.max(abs(a.astype("d") - b)), 1e-3)

    def test_rejections(self):
        a = np.ones((4, 8), dtype="f")
        for bound in (0, -1, np.inf, np.nan):
            with self.assertRaises(ValueError):
                pywaltz.compress(a, abs_error=bound)
        with self.assertRaises(TypeError):
            pywaltz.compress(a.astype("i"), abs_error=1)
        a[0, 0] = np.nan
        with self.assertRaises(ValueError):
            pywaltz.compress(a, abs_error=1)
        with self.assertRaises(ValueError):
            pywaltz.decompress(b"", shape=(4, 8), dtype="f")
        with self.assertRaises(ValueError):
            pywaltz.decompress(bytes(32), shape=(2**32, 2), dtype="d")
