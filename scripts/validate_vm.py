"""Exercise original Megakernels VM resources on SM100; not model validation."""
import ctypes
import subprocess
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main():
    with tempfile.TemporaryDirectory() as directory:
        library = Path(directory) / "probe.so"
        subprocess.run([
            "nvcc", "-O3", "-lineinfo", "-std=c++20", "-arch=sm_100a", "-DKITTENS_HOPPER",
            "-DKITTENS_BLACKWELL", "--expt-relaxed-constexpr", "--expt-extended-lambda",
            "-I" + str(ROOT / "third_party/Megakernels/include"),
            "-I" + str(ROOT / "third_party/Megakernels/ThunderKittens/include"),
            "-shared", "-Xcompiler", "-fPIC", str(ROOT / "native_pi0/vm_probe.cu"),
            "-lcuda", "-o", str(library),
        ], check=True)
        probe = ctypes.CDLL(str(library)).run_vm_probe
        probe.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
        probe.restype = ctypes.c_int
        for workers, instructions in ((1, 1), (1, 257), (16, 17), (148, 65)):
            for replay in range(3):
                output = np.zeros((workers, instructions, 16), dtype=np.int32)
                err = probe(workers, instructions, output.ctypes.data)
                if err:
                    raise RuntimeError(f"VM probe CUDA error {err} at {workers}/{instructions}/{replay}")
                if not np.all(output > 0) or not np.all(output == output[:, :1, :1]):
                    raise AssertionError(f"Consumer participation or tensor allocation mismatch: {output.tolist()[:2]}")
            print(f"VM resource lifecycle passed: {workers} CTAs, {instructions} instructions, three launches", flush=True)


if __name__ == "__main__":
    main()
