"""Checks dependency discovery without needing CUDA or FlashRT imports."""

import tempfile
import unittest
from pathlib import Path

from audit_flashrt import local_dependencies


class DependencyDiscovery(unittest.TestCase):
    def test_nested_cpp_and_relative_python_dependencies(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            contents = {
                "flash_rt/models/pi0/pipeline.py": "from ...core import quant\n",
                "flash_rt/core/__init__.py": "",
                "flash_rt/core/quant.py": "from . import scales\n",
                "flash_rt/core/scales.py": "import flash_rt.core.quant\n",
                "csrc/kernels/pi05/thor/op.cu": '#include "kernels/pi05/thor/op.cuh"\n',
                "csrc/kernels/pi05/thor/op.cuh": '#include "local.cuh"\n#include "external.h"\n',
                "csrc/kernels/pi05/thor/local.cuh": '#include "op.cuh"\n',
            }
            for name, content in contents.items():
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(content)
            files, edges, missing = local_dependencies(root, [
                root / "flash_rt/models/pi0/pipeline.py",
                root / "csrc/kernels/pi05/thor/op.cu",
            ])
            self.assertEqual({str(p.relative_to(root)) for p in files}, set(contents))
            self.assertTrue(any(e["target"] == "flash_rt/core/scales.py" for e in edges))
            self.assertEqual(missing, [{"source": "csrc/kernels/pi05/thor/op.cuh",
                                        "line": 2, "include": "external.h"}])


if __name__ == "__main__":
    unittest.main()
