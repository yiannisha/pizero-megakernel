"""Pinned shard coverage and corruption guards, without downloading weight data."""
import copy
import json
import unittest

from cosmos3_edge.spec import SOURCES
from cosmos3_edge.weights import expected_shapes, validate_headers


class HeaderValidationTests(unittest.TestCase):
    def setUp(self):
        self.headers = {p.name.removesuffix('.header.json'): json.loads(p.read_text())
                        for p in SOURCES.glob('*.safetensors.header.json')}

    def test_checkpoint_coverage(self):
        self.assertEqual(validate_headers(self.headers),
                         dict(tensors=549, shards=2, total_bytes=6739314048))
        shapes = expected_shapes()
        self.assertEqual(shapes['layers.27.mlp_moe_gen.up_proj.weight'], (9216, 2048))
        self.assertEqual(shapes['layers.27.self_attn.add_k_proj.weight'], (1024, 2048))

    def test_corrupt_headers_rejected(self):
        for field, value in [('dtype', 'F16'), ('shape', [1]), ('data_offsets', [0, 1])]:
            headers = copy.deepcopy(self.headers)
            shard = next(iter(headers))
            name = next(k for k in headers[shard] if k != '__metadata__')
            headers[shard][name][field] = value
            with self.assertRaises(ValueError):
                validate_headers(headers)

    def test_missing_tensor_rejected(self):
        shard = next(iter(self.headers))
        name = next(k for k in self.headers[shard] if k != '__metadata__')
        del self.headers[shard][name]
        with self.assertRaises(ValueError):
            validate_headers(self.headers)


if __name__ == '__main__':
    unittest.main()
