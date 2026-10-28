"""Compare offline rounding with B200 CUDA conversions, including every tie."""

import ctypes
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from native_pi0.quantization import E2M1, UE4M3, _nearest_even, scale_offsets, swizzle_scales, unswizzle_scales


def main():
    mid4 = (E2M1[:-1] + E2M1[1:]) / 2
    mid8 = (UE4M3[:-1] + UE4M3[1:]) / 2
    points = np.concatenate((E2M1, UE4M3, mid4, mid8,
                             np.nextafter(mid4, np.float32(0)),
                             np.nextafter(mid4, np.float32(np.inf)),
                             np.nextafter(mid8, np.float32(0)),
                             np.nextafter(mid8, np.float32(np.inf)),
                             np.random.default_rng(0).uniform(0, 500, 10000))).astype(np.float32)
    points = np.concatenate((points, -points))
    with tempfile.TemporaryDirectory() as directory:
        library = Path(directory) / "oracle.so"
        subprocess.run(["nvcc", "-O3", "-arch=sm_100a", "-shared", "-Xcompiler", "-fPIC",
                        "-I" + str(ROOT / "third_party/FlashRT/csrc"),
                        "-I" + str(ROOT / "third_party/cutlass/include"),
                        str(ROOT / "third_party/FlashRT/csrc/quantize/nvfp4_sf_reshape_sm120.cu"),
                        str(ROOT / "native_pi0/validate_quantization.cu"), "-o", str(library)], check=True)
        loaded = ctypes.CDLL(str(library))
        oracle = loaded.quantization_oracle
        oracle.argtypes = [ctypes.c_void_p] * 3 + [ctypes.c_int]
        oracle.restype = ctypes.c_int
        fp4, fp8 = np.empty(points.size, np.uint8), np.empty(points.size, np.uint8)
        error = oracle(points.ctypes.data, fp4.ctypes.data, fp8.ctypes.data, points.size)
        if error:
            raise RuntimeError(f"CUDA oracle failed: {error}")
        layout = loaded.scale_layout_oracle
        layout.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
        layout.restype = ctypes.c_int
        sm100 = loaded.sm100_layout_oracle
        sm100.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int]
        sm100.restype = ctypes.c_int64
        for rows, blocks in ((1, 1), (51, 64), (127, 3), (129, 5), (256, 72), (768, 128), (4304, 72)):
            linear = np.random.default_rng(rows).integers(0, 127, (rows, blocks), dtype=np.uint8)
            expected = swizzle_scales(linear)
            for operand in (0, 1):
                offsets = np.empty((rows, blocks), dtype=np.int64)
                capacity = sm100(offsets.ctypes.data, rows, blocks, operand)
                np.testing.assert_array_equal(offsets, scale_offsets(rows, blocks))
                if capacity != expected.size:
                    raise AssertionError(f"CUTLASS scale capacity {capacity} != {expected.size}")
            actual = np.empty_like(expected)
            error = layout(linear.ctypes.data, actual.ctypes.data, rows, blocks)
            if error:
                raise RuntimeError(f"Scale layout oracle failed: {error}")
            np.testing.assert_array_equal(actual, expected)
            np.testing.assert_array_equal(unswizzle_scales(actual, rows, blocks), linear)
    sign = np.signbit(points).astype(np.uint8)
    expected4 = _nearest_even(np.abs(points), E2M1).astype(np.uint8) | (sign << 3)
    expected8 = _nearest_even(np.abs(points), UE4M3).astype(np.uint8) | (sign << 7)
    np.testing.assert_array_equal(fp4, expected4)
    np.testing.assert_array_equal(fp8, expected8)
    print(f"Hardware E2M1 and E4M3 conversions match for {points.size} values, including midpoint neighbors")
    print("Scale layouts match pinned FlashRT CUDA reshape for seven shapes, including padded regions")
    print("SFA/SFB offsets and capacities match CUTLASS Sm1xxBlockScaledConfig<16> directly")


if __name__ == "__main__":
    main()
