import tempfile
import unittest
from pathlib import Path

import numpy as np

from native_pi0.calibration import ActivationCalibration, cache_key, checkpoint_digest, fp32_descale


class CalibrationTests(unittest.TestCase):
    def test_aggregation_and_step_coverage(self):
        c = ActivationCalibration()
        for sample in ("camera_a", "camera_b"):
            for step in range(10):
                c.observe("expert.0.qkv", [0, -(step + 1) * 448], sample=sample, step=step)
        record = c.finalize(percentile=100, required_steps=range(10))["expert.0.qkv"]
        self.assertEqual(record["fp8_scale"], 10)
        self.assertEqual(record["per_step_max"]["9"], 4480)
        self.assertEqual(record["observations"], 2)
        self.assertEqual(record["amax_records_above_range"], 0)
        self.assertEqual(c.finalize(percentile=0)["expert.0.qkv"]["fp8_scale"], 1)
        with self.assertRaises(ValueError):
            c.finalize(required_steps=range(11))
        with self.assertRaises(ValueError):
            c.observe("expert.0.qkv", [1], sample="camera_a", step=0)

    def test_checkpoint_and_configuration_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory) / "model.safetensors"
            p.write_bytes(b"a" * 70000)
            before = checkpoint_digest([p])
            with p.open("r+b") as stream:
                stream.seek(69999)
                stream.write(b"b")
            after = checkpoint_digest([p])
            self.assertNotEqual(before, after)
            key = cache_key(after, {"steps": 10, "views": 3}, "observations-v1")
            self.assertEqual(key, cache_key(after, {"views": 3, "steps": 10}, "observations-v1"))
            self.assertNotEqual(key, cache_key(after, {"steps": 5, "views": 3}, "observations-v1"))
            self.assertNotEqual(key, cache_key(after, {"steps": 10, "views": 3}, "observations-v2"))

    def test_zero_and_nonfinite(self):
        c = ActivationCalibration()
        c.observe("prefix.qkv", [0], sample="zero")
        self.assertGreater(c.finalize()["prefix.qkv"]["fp8_scale"], 0)
        for value in (float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                c.observe("bad", [value], sample="bad")
        self.assertEqual(fp32_descale(.0123, .0456), np.float32(.0123) * np.float32(.0456))
        with self.assertRaises(ValueError):
            fp32_descale(1e30, 1e30)


if __name__ == "__main__":
    unittest.main()
