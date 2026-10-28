"""Native ReLU² FFN with direct FP4 producer-consumer storage.

Checkpoint packing and tensor allocation are setup operations. forward() only
launches the original persistent Megakernels VM, including both projections.
"""
from contextlib import ExitStack

import numpy as np
import torch

from native_pi0.dag_matmul import ResidentDagMatmul
from native_pi0.primitives import ResidentNVFP4
from native_pi0.quantization import PackedNVFP4, pack_nvfp4


class NativeFFN:
    """Packed activation → up → ReLU²/FP4 → down → BF16 (+ residual).

    This is a reusable FFN operation, not a complete Cosmos inference API.
    Input packing happens before forward. Instances own reusable buffers and
    require ordered calls; independent instances can use independent streams.
    """
    def __init__(self, library, activation: PackedNVFP4, up: PackedNVFP4,
                 down: PackedNVFP4, *, hidden_scale=1., residual=None,
                 workers=148, device="cuda", node_order=None):
        m, width = activation.logical_shape
        intermediate, up_width = up.logical_shape
        output_width, down_width = down.logical_shape
        if (width != up_width or down_width != intermediate or output_width != width
                or width % 128 or intermediate % 128):
            raise ValueError("FFN requires up=(intermediate,hidden), down=(hidden,intermediate) with widths divisible by 128")
        if activation.data.shape[1]*2 != width or up.data.shape != (intermediate, width//2) or down.data.shape != (width, intermediate//2):
            raise ValueError("FFN weight and column padding must match the logical model widths")
        self.rows, self.width = m, width
        self._stack = ExitStack()
        try:
            self.up = self._stack.enter_context(ResidentNVFP4(
                library, activation, up, workers=workers, device=device,
                output_dtype=torch.uint8, output_relu2=True,
                output_global_scale=hidden_scale))
            placeholder = pack_nvfp4(np.zeros((m, intermediate), np.float32),
                                    global_scale=hidden_scale, row_alignment=128,
                                    column_alignment=64)
            padded_residual = None
            if residual is not None:
                residual = torch.as_tensor(residual)
                if tuple(residual.shape) != (m, width):
                    raise ValueError("Residual must match logical input dimensions")
                padded_residual = torch.zeros((activation.data.shape[0], width), dtype=torch.bfloat16, device=device)
                padded_residual[:m].copy_(residual)
            self.down = self._stack.enter_context(ResidentNVFP4(
                library, placeholder, down, workers=workers, device=device,
                output_dtype=torch.bfloat16, residual=padded_residual))
            self.graph = self._stack.enter_context(ResidentDagMatmul(
                library, [self.up, self.down], workers=workers, node_order=node_order))
            self.output = self.down.output[:m]
        except BaseException:
            self._stack.close()
            raise

    @classmethod
    def from_checkpoint(cls, library, activation, weights, layer, *, stream="gen",
                        up_scale=1., down_scale=1., **kwargs):
        """Offline weight packing; scales must later be calibrated for the site."""
        from .spec import EdgeSpec
        if not isinstance(layer, int) or not 0 <= layer < EdgeSpec.from_checkpoint().layers:
            raise ValueError("Layer index outside the pinned transformer")
        if stream not in ("gen", "und"):
            raise ValueError("Stream must be gen or und")
        prefix = f"layers.{layer}.mlp" + ("_moe_gen" if stream == "gen" else "")
        up = pack_nvfp4(weights.load(prefix+".up_proj.weight").float().numpy(),
                        global_scale=up_scale, scale_policy="ceil")
        down = pack_nvfp4(weights.load(prefix+".down_proj.weight").float().numpy(),
                          global_scale=down_scale, scale_policy="ceil")
        return cls(library, activation, up, down, **kwargs)

    def forward(self):
        self.graph.forward()
        return self.output

    def close(self):
        self._stack.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


class NativeDenseFFN:
    """One-launch BF16 FFN reference; optionally use prepared FP8 for up.

    The up epilogue computes ReLU² on its FP32 accumulator and rounds once to
    BF16. This fusion differs from framework BF16 intermediate rounding and
    must be measured against the model reference before selecting a policy.
    """
    def __init__(self, library, activation, up_weight, down_weight, *,
                 up_precision='bf16', activation_scale=1., weight_scale=1.,
                 residual=None, workers=148, device='cuda', node_order=None):
        from .gemm import NativeProjection
        from .schedule import DenseSchedule
        if activation.ndim != 2 or up_weight.ndim != 2 or down_weight.ndim != 2:
            raise ValueError('FFN tensors must be matrices')
        m, h = activation.shape
        intermediate, width = up_weight.shape
        if width != h or down_weight.shape != (h, intermediate) or intermediate % 128 or h % 128:
            raise ValueError('Incompatible FFN dimensions')
        self._stack = ExitStack()
        try:
            self.up = self._stack.enter_context(NativeProjection(
                library, activation, up_weight, precision=up_precision,
                activation_scale=activation_scale, weight_scale=weight_scale,
                relu2=True, workers=workers, device=device))
            self.down = self._stack.enter_context(NativeProjection(
                library, torch.zeros((m, intermediate), dtype=torch.bfloat16), down_weight,
                residual=residual, workers=workers, device=device))
            self.graph = self._stack.enter_context(DenseSchedule(
                library, [self.up, self.down], workers=workers, node_order=node_order))
            self.output = self.down.output
        except BaseException:
            self._stack.close()
            raise

    def forward(self):
        self.graph.forward()
        return self.output

    def close(self):
        self._stack.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
