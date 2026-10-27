"""Offline activation calibration; no host decisions belong in inference.

Uses FlashRT core/calibration.py's percentile-of-amax approach. A record is
one named GEMM input for one observation and denoising step. The caller must
collect these at the actual unquantized producer boundaries on real inputs.
"""

import hashlib
import json
from pathlib import Path

import numpy as np


def checkpoint_digest(files):
    """Hash complete checkpoint contents, including every shard supplied."""
    files = sorted(Path(p) for p in files)
    if not files or len({p.name for p in files}) != len(files):
        raise ValueError("Supply nonempty checkpoint files with unique names")
    digest = hashlib.sha256(b"native-pi0-checkpoint-v1\0")
    for path in files:
        content = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                content.update(block)
        digest.update(path.name.encode() + b"\0" + content.digest())
    return digest.hexdigest()


def cache_key(checkpoint, config, dataset_digest):
    if not checkpoint or not dataset_digest:
        raise ValueError("Checkpoint and observation dataset identities are required")
    payload = {"version": 1, "checkpoint": checkpoint, "config": config,
               "dataset": dataset_digest}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode()).hexdigest()


class ActivationCalibration:
    def __init__(self):
        self._records = {}

    def observe(self, operation, values, *, sample, step=None):
        """Record host-visible unquantized values outside timed inference."""
        values = np.asarray(values, dtype=np.float32)
        if not isinstance(operation, str) or not operation:
            raise ValueError("Operation name is required")
        if not values.size or not np.isfinite(values).all():
            raise ValueError("Calibration activations must be nonempty and finite")
        if not isinstance(sample, str) or not sample:
            raise ValueError("Observation identity is required")
        if step is not None and (not isinstance(step, int) or step < 0):
            raise ValueError("Denoising step must be nonnegative")
        identity = (sample, step)
        records = self._records.setdefault(operation, {})
        if identity in records:
            raise ValueError("Duplicate operation/observation/step record")
        records[identity] = float(np.abs(values).max())

    def finalize(self, *, percentile=99.9, required_steps=None):
        if not 0 <= percentile <= 100 or not self._records:
            raise ValueError("Nonempty calibration and percentile in [0,100] required")
        required = None if required_steps is None else set(required_steps)
        result = {}
        for operation, records in sorted(self._records.items()):
            steps = {step for _, step in records if step is not None}
            if steps and required is not None:
                for sample, _ in records:
                    actual = {step for s, step in records if s == sample}
                    if actual != required:
                        raise ValueError(f"Incomplete denoising coverage for {operation}/{sample}")
            amax = np.array(list(records.values()), dtype=np.float64)
            selected = np.float32(np.percentile(amax, percentile))
            scale = np.float32(max(float(selected) / 448, 1e-12))
            result[operation] = {
                "fp8_scale": float(scale), "percentile": percentile,
                "selected_amax": float(selected), "max_amax": float(amax.max()),
                "records": len(records), "observations": len({s for s, _ in records}),
                "amax_records_above_range": int(np.count_nonzero(amax > float(scale) * 448)),
                "per_step_max": {str(step): max(value for (_, s), value in records.items() if s == step)
                                 for step in sorted(steps)},
            }
        return result


def fp32_descale(activation_scale, weight_scale):
    a, w = np.float32(activation_scale), np.float32(weight_scale)
    with np.errstate(over="ignore", invalid="ignore"):
        result = np.float32(a * w)
    if not np.isfinite(result) or min(a, w, result) <= 0:
        raise ValueError("Descale factors must be finite and positive")
    return result
