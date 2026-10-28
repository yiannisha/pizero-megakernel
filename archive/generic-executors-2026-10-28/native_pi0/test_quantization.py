import unittest

import numpy as np

from native_pi0.quantization import E2M1, UE4M3, _nearest_even, interleave_gate_up, pack_nvfp4


class PackingTests(unittest.TestCase):
    def test_flashrt_reciprocal_midpoint_semantics(self):
        values = np.zeros((1, 16), dtype=np.float32)
        values[0, :2] = [5.625, 1.171875]
        # SF=.9375. x/SF is exactly 1.25, but x*float32(1/SF) lies above
        # that midpoint. FlashRT's hardware converter therefore chooses 1.5.
        packed = pack_nvfp4(values)
        self.assertEqual(packed.scales[0, 0], 55)
        self.assertEqual(packed.data[0, 0], 0x37)

    def test_formats_and_ties(self):
        np.testing.assert_array_equal(_nearest_even(E2M1, E2M1), np.arange(8))
        midpoints = (E2M1[:-1] + E2M1[1:]) / 2
        np.testing.assert_array_equal(_nearest_even(midpoints, E2M1), [0, 2, 2, 4, 4, 6, 6])
        self.assertEqual(UE4M3[-1], 448)
        self.assertEqual(UE4M3[1], 2 ** -9)

    def test_exact_codes_and_nibble_order(self):
        x = np.concatenate((E2M1, -E2M1))[None, :]
        packed = pack_nvfp4(x)
        np.testing.assert_array_equal(packed.scales, [[56]])  # scale=1
        np.testing.assert_array_equal(packed.data, [[0x10, 0x32, 0x54, 0x76, 0x98, 0xBA, 0xDC, 0xFE]])
        np.testing.assert_array_equal(packed.dequantize(), x)

    def test_zero_tiny_and_padding(self):
        x = np.full((3, 17), 1e-12, dtype=np.float32)
        x[1] *= -1
        p = pack_nvfp4(x, row_alignment=128, column_alignment=128)
        self.assertFalse(p.scales.any())
        self.assertFalse(p.data.any())
        self.assertEqual(p.data.shape, (128, 64))
        self.assertEqual(p.scales.shape, (128, 8))
        self.assertEqual(p.dequantize().shape, x.shape)
        ceil = pack_nvfp4(x, scale_policy="ceil")
        self.assertTrue((ceil.scales[:, :1] == 1).all())
        self.assertTrue(np.isfinite(ceil.dequantize()).all())

    def test_global_scale_and_representative_shapes(self):
        rng = np.random.default_rng(7)
        for rows, cols in ((51, 1024), (256, 1152), (768, 2048), (3, 4304)):
            x = rng.normal(size=(rows, cols)).astype(np.float32)
            p = pack_nvfp4(x, global_scale=.125)
            error = np.linalg.norm(p.dequantize() - x) / np.linalg.norm(x)
            self.assertLess(error, .13)
            self.assertTrue(np.isfinite(p.dequantize()).all())

    def test_gate_up_and_invalid_inputs(self):
        gate = np.arange(12).reshape(3, 4)
        result = interleave_gate_up(gate, gate + 100)
        np.testing.assert_array_equal(result[::2], gate)
        np.testing.assert_array_equal(result[1::2], gate + 100)
        for x in (np.array([[np.nan]]), np.array([[np.inf]]), np.empty((0, 4)), np.ones(4)):
            with self.assertRaises(ValueError):
                pack_nvfp4(x)
        for scale in (0, -1, float("nan")):
            with self.assertRaises(ValueError):
                pack_nvfp4([[1]], global_scale=scale)


if __name__ == "__main__":
    unittest.main()
