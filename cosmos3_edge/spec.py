"""Checkpoint-derived configuration; no inference framework dependency."""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

SOURCES = Path(__file__).parent / "sources"


def verify_sources() -> dict:
    manifest = json.loads((SOURCES / "manifest.json").read_text())
    for name, entry in manifest["files"].items():
        data = (SOURCES / entry["local"]).read_bytes()
        if hashlib.sha256(data).hexdigest() != entry["sha256"]:
            raise ValueError(f"pinned checkpoint metadata checksum mismatch: {name}")
    return manifest


@dataclass(frozen=True)
class EdgeSpec:
    layers: int
    hidden: int
    ffn: int
    query_heads: int
    kv_heads: int
    head_dim: int
    rms_eps: float
    action_dim: int
    domains: int
    latent_channels: int
    patch_size: int
    rope_axes: tuple[int, ...]
    vocab: int

    @classmethod
    def from_checkpoint(cls) -> "EdgeSpec":
        verify_sources()
        cfg = json.loads((SOURCES / "transformer__config.json").read_text())
        if cfg["hidden_act"] != "relu2" or not cfg["use_und_k_norm_for_gen"]:
            raise ValueError("unsupported generator mathematics")
        result = cls(*(cfg[k] for k in (
            "num_hidden_layers", "hidden_size", "intermediate_size",
            "num_attention_heads", "num_key_value_heads", "head_dim",
            "rms_norm_eps", "action_dim", "num_embodiment_domains",
            "latent_channel", "latent_patch_size")),
            tuple(cfg["rope_axes_dim"]), cfg["vocab_size"])
        if result.hidden != result.query_heads * result.head_dim:
            raise ValueError("query projection does not span hidden width")
        if result.query_heads % result.kv_heads:
            raise ValueError("unsupported GQA ratio")
        if sum(result.rope_axes) > result.head_dim:
            raise ValueError("RoPE exceeds head dimension")
        return result

    @property
    def projections(self) -> dict[str, tuple[int, int]]:
        """Logical (output, input) weight dimensions for either stream."""
        return dict(q=(self.hidden, self.hidden),
                    k=(self.kv_heads * self.head_dim, self.hidden),
                    v=(self.kv_heads * self.head_dim, self.hidden),
                    o=(self.hidden, self.hidden),
                    up=(self.ffn, self.hidden), down=(self.hidden, self.ffn))
