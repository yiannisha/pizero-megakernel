"""Offline NVFP4 packing oracle for native operator development.

Matches FlashRT nvfp4_skinny_common.cuh's activation quant_block16 and
ue4m3_ceil weight scale policy. This is setup/reference code, never inference.
Scales here are linear [row, K/16]; SM100 MMA swizzling is a separate layout
transform and must not be inferred from this representation.
"""

from dataclasses import dataclass

import numpy as np


E2M1 = np.array([0, .5, 1, 1.5, 2, 3, 4, 6], dtype=np.float32)
UE4M3 = np.array([
    (code & 7) * 2.0 ** -9 if code < 8
    else (1 + (code & 7) / 8) * 2.0 ** ((code >> 3) - 7)
    for code in range(127)
], dtype=np.float32)


def _nearest_even(values, table):
    """Nearest code, with even-code midpoint ties and finite saturation."""
    hi = np.searchsorted(table, values).clip(0, len(table) - 1)
    lo = (hi - 1).clip(0)
    dl, dh = values - table[lo], table[hi] - values
    return np.where((dl < dh) | ((dl == dh) & ((lo & 1) == 0)), lo, hi)


@dataclass(frozen=True)
class PackedNVFP4:
    data: np.ndarray
    scales: np.ndarray
    global_scale: np.float32
    logical_shape: tuple[int, int]

    def dequantize(self):
        codes = np.empty((self.data.shape[0], self.data.shape[1] * 2), dtype=np.uint8)
        codes[:, 0::2] = self.data & 15
        codes[:, 1::2] = self.data >> 4
        values = E2M1[codes & 7] * np.where(codes & 8, -1, 1).astype(np.float32)
        values *= np.repeat(UE4M3[self.scales], 16, axis=1)
        values *= self.global_scale
        rows, cols = self.logical_shape
        return values[:rows, :cols]


def pack_nvfp4(matrix, *, global_scale=1.0, scale_policy="nearest", row_alignment=1,
               column_alignment=16):
    """Pack an [N,K] weight or [M,K] activation matrix, low nibble first.

    `nearest` is FlashRT's activation policy; `ceil` is its conservative weight
    policy (its optional squared-error weight search is not implemented here).
    Padding is deterministic zero in both codes and scales. Tiny blocks whose
    nearest scale rounds to zero produce zero, matching quant_block16.
    """
    x = np.asarray(matrix, dtype=np.float32)
    gs = np.float32(global_scale)
    if x.ndim != 2 or not all(x.shape) or not np.isfinite(x).all():
        raise ValueError("Expected a nonempty finite matrix")
    if not np.isfinite(gs) or gs <= 0:
        raise ValueError("global_scale must be finite and positive")
    if scale_policy not in ("nearest", "ceil"):
        raise ValueError("scale_policy must be nearest or ceil")
    if (not isinstance(row_alignment, int) or row_alignment < 1
            or not isinstance(column_alignment, int) or column_alignment < 16
            or column_alignment % 16):
        raise ValueError("Invalid row alignment or column alignment (multiple of 16 required)")
    rows, cols = x.shape
    nr = (rows + row_alignment - 1) // row_alignment * row_alignment
    nk = (cols + column_alignment - 1) // column_alignment * column_alignment
    padded = np.zeros((nr, nk), dtype=np.float32)
    with np.errstate(over="ignore"):
        padded[:rows, :cols] = x * np.float32(np.float32(1) / gs)
    if not np.isfinite(padded).all():
        raise ValueError("global_scale overflows normalized values")
    blocks = padded.reshape(nr, nk // 16, 16)
    desired = np.max(np.abs(blocks), axis=-1) * np.float32(1 / 6)
    if scale_policy == "ceil":
        codes = np.searchsorted(UE4M3, desired).clip(0, 126)
    else:
        codes = _nearest_even(desired, UE4M3)
    scales = UE4M3[codes]
    # FlashRT quant_block16 multiplies by a rounded FP32 reciprocal. Direct
    # division differs at E2M1 midpoint ties for non-power-of-two block scales.
    reciprocal = np.divide(np.float32(1), scales, out=np.zeros_like(scales), where=scales > 0)
    normalized = blocks * reciprocal[..., None]
    normalized[scales == 0] = np.float32(0)
    q = _nearest_even(np.abs(normalized), E2M1).astype(np.uint8)
    q |= (np.signbit(normalized).astype(np.uint8) << 3)
    q = q.reshape(nr, nk)
    return PackedNVFP4((q[:, 0::2] | (q[:, 1::2] << 4)).copy(),
                       codes.astype(np.uint8), gs, (rows, cols))


def interleave_gate_up(gate, up):
    """[N,K] matrices -> adjacent gate/up rows for compact GeGLU epilogues."""
    gate, up = np.asarray(gate), np.asarray(up)
    if gate.ndim != 2 or gate.shape != up.shape:
        raise ValueError("gate and up must be equal-shaped matrices")
    return np.stack((gate, up), axis=1).reshape(2 * gate.shape[0], gate.shape[1])


def scale_offsets(rows, blocks):
    """Sm1xx per-16 scale byte offsets from FlashRT's SF reshape kernel.

    A 128-row by 4-block atom occupies 512 bytes. Each row's four bytes
    are interleaved across the four 32-row bands. Both operand sides use
    this representation in FlashRT; MMA operand/TMEM layout is separate.
    """
    if not isinstance(rows, int) or not isinstance(blocks, int) or min(rows, blocks) < 1:
        raise ValueError("rows and blocks must be positive integers")
    r = np.arange(rows, dtype=np.int64)[:, None]
    c = np.arange(blocks, dtype=np.int64)[None, :]
    return ((r // 128) * ((blocks + 3) // 4) + c // 4) * 512 + (r % 32) * 16 + (r % 128 // 32) * 4 + c % 4


def swizzle_scales(linear):
    """Pack scale bytes with deterministic zeroes in all padded positions."""
    linear = np.asarray(linear)
    if linear.ndim != 2 or linear.dtype != np.uint8 or not all(linear.shape):
        raise ValueError("Expected a nonempty uint8 scale matrix")
    rows, blocks = linear.shape
    output = np.zeros(((rows + 127) // 128) * ((blocks + 3) // 4) * 512, dtype=np.uint8)
    output[scale_offsets(rows, blocks)] = linear
    return output


def unswizzle_scales(swizzled, rows, blocks):
    offsets = scale_offsets(rows, blocks)
    swizzled = np.asarray(swizzled)
    size = ((rows + 127) // 128) * ((blocks + 3) // 4) * 512
    if swizzled.ndim != 1 or swizzled.dtype != np.uint8 or swizzled.size != size:
        raise ValueError("Incorrect swizzled scale buffer size or dtype")
    return swizzled[offsets].copy()
