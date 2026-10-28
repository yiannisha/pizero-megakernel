"""Explicit stream mapping and lazy checkpoint access for offline preparation.

No FlashRT imports or runtime dispatch. Loading tensors is setup-only.
"""
from pathlib import Path
import json

from .spec import EdgeSpec, SOURCES, verify_sources


def expected_shapes(spec: EdgeSpec | None = None) -> dict[str, tuple[int, ...]]:
    s = spec or EdgeSpec.from_checkpoint()
    h, f, d = s.hidden, s.ffn, s.head_dim
    shapes = {
        "action_modality_embed": (h,),
        "action_proj_in.bias.weight": (s.domains, h),
        "action_proj_in.fc.weight": (s.domains, s.action_dim * h),
        "action_proj_out.bias.weight": (s.domains, s.action_dim),
        "action_proj_out.fc.weight": (s.domains, s.action_dim * h),
        "embed_tokens.weight": (s.vocab, h), "lm_head.weight": (s.vocab, h),
        "norm.weight": (h,), "norm_moe_gen.weight": (h,),
        "proj_in.bias": (h,),
        "proj_in.weight": (h, s.latent_channels * s.patch_size**2),
        "proj_out.bias": (s.latent_channels * s.patch_size**2,),
        "proj_out.weight": (s.latent_channels * s.patch_size**2, h),
        "time_embedder.linear_1.bias": (h,),
        "time_embedder.linear_1.weight": (h, 256),
        "time_embedder.linear_2.bias": (h,),
        "time_embedder.linear_2.weight": (h, h),
    }
    layer = {}
    for stream in ("", "_moe_gen"):
        layer[f"input_layernorm{stream}.weight"] = (h,)
        layer[f"post_attention_layernorm{stream}.weight"] = (h,)
        layer[f"mlp{stream}.up_proj.weight"] = (f, h)
        layer[f"mlp{stream}.down_proj.weight"] = (h, f)
    for name, op in (("to_q", "q"), ("to_k", "k"), ("to_v", "v"),
                     ("to_out", "o"), ("add_q_proj", "q"),
                     ("add_k_proj", "k"), ("add_v_proj", "v"), ("to_add_out", "o")):
        layer[f"self_attn.{name}.weight"] = s.projections[op]
    for name in ("norm_added_q", "norm_added_k", "k_norm_und_for_gen"):
        layer[f"self_attn.{name}.weight"] = (d,)
    for i in range(s.layers):
        shapes.update({f"layers.{i}.{name}": shape for name, shape in layer.items()})
    return shapes


def validate_headers(headers: dict[str, dict]) -> dict:
    """Validate every indexed tensor, shape, dtype, and byte range before loading."""
    verify_sources()
    index = json.loads((SOURCES / "transformer__diffusion_pytorch_model.safetensors.index.json").read_text())
    mapping = index["weight_map"]
    expected = expected_shapes()
    if set(mapping) != set(expected):
        raise ValueError(f"checkpoint key mismatch: missing={set(expected)-set(mapping)}, extra={set(mapping)-set(expected)}")
    if set(headers) != set(mapping.values()):
        raise ValueError("shard header set differs from index")
    found = {}
    sizes = {"BF16": 2, "F32": 4, "F16": 2}
    total = 0
    for shard, header in headers.items():
        ranges = []
        for name, entry in header.items():
            if name == "__metadata__":
                continue
            if name in found or mapping.get(name) != shard:
                raise ValueError(f"unexpected or duplicate tensor: {name}")
            shape = tuple(entry["shape"])
            if shape != expected[name] or entry["dtype"] != "BF16":
                raise ValueError(f"{name}: expected BF16 {expected[name]}, got {entry}")
            start, end = entry["data_offsets"]
            count = 1
            for dim in shape:
                count *= dim
            if start < 0 or end-start != count*sizes[entry["dtype"]]:
                raise ValueError(f"invalid tensor byte range: {name}")
            ranges.append((start, end)); found[name] = entry; total += end-start
        cursor = 0
        for start, end in sorted(ranges):
            if start != cursor:
                raise ValueError(f"noncontiguous or overlapping shard data: {shard}")
            cursor = end
    if set(found) != set(mapping) or total != index["metadata"]["total_size"]:
        raise ValueError("checkpoint header coverage or size mismatch")
    return {"tensors": len(found), "shards": len(headers), "total_bytes": total}


class CheckpointWeights:
    """Read only the requested safetensor; never execute model operators."""
    def __init__(self, checkpoint: str | Path, *, verify_integrity: bool = True):
        self.root = Path(checkpoint) / "transformer"
        pinned = (SOURCES / "transformer__diffusion_pytorch_model.safetensors.index.json").read_bytes()
        if json.loads((self.root / "diffusion_pytorch_model.safetensors.index.json").read_bytes()) != json.loads(pinned):
            raise ValueError("local weight index differs from pinned checkpoint")
        self.mapping = json.loads(pinned)["weight_map"]
        import struct
        headers = {}
        for shard in set(self.mapping.values()):
            with (self.root / shard).open("rb") as f:
                length, = struct.unpack("<Q", f.read(8))
                if length > 16*1024*1024:
                    raise ValueError("unreasonable safetensor header size")
                headers[shard] = json.loads(f.read(length))
                data_bytes = max(v["data_offsets"][1] for k, v in headers[shard].items() if k != "__metadata__")
                if (self.root / shard).stat().st_size != 8+length+data_bytes:
                    raise ValueError(f"truncated or extra data in {shard}")
        self.validation = validate_headers(headers)
        if verify_integrity:
            from .download_checkpoint import digest
            files = json.loads((SOURCES / "weight_files.json").read_text())
            for shard in headers:
                expected = files["transformer/"+shard]
                if digest(self.root / shard) != expected["sha256"]:
                    raise ValueError(f"checkpoint content hash mismatch: {shard}")
        self.validation["weight_content_sha256_verified"] = verify_integrity

    def load(self, name: str, *, device="cpu"):
        from safetensors import safe_open
        if name not in self.mapping:
            raise KeyError(name)
        with safe_open(self.root / self.mapping[name], framework="pt", device=str(device)) as f:
            return f.get_tensor(name)
