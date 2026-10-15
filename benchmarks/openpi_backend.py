"""OpenPI inference with a capture-safe, fixed-count Euler loop."""

import math
from types import MethodType, SimpleNamespace

import torch
from openpi.models_pytorch.pi0_pytorch import (
    PI0Pytorch,
    create_sinusoidal_pos_embedding,
    make_att_2d_masks,
)


def capture_prefix(self, images, masks, tokens, token_mask):
    """OpenPI embed_prefix without a Python-list-to-CUDA tensor transfer."""
    embeddings, padding = [], []
    for image, mask in zip(images, masks, strict=True):
        embedded = self.paligemma_with_expert.embed_image(image)
        embeddings.append(embedded)
        padding.append(mask[:, None].expand(embedded.shape[:2]))
    language = self.paligemma_with_expert.embed_language_tokens(tokens)
    embeddings.append(language * math.sqrt(language.shape[-1]))
    padding.append(token_mask)
    embedded = torch.cat(embeddings, dim=1)
    pad = torch.cat(padding, dim=1)
    return embedded, pad, torch.zeros_like(pad)


def capture_suffix(self, state, actions, timestep):
    """OpenPI pi0 embed_suffix with attention constants constructed on-device."""
    if self.state_proj.weight.dtype == torch.float32:
        state = state.float()
    state_embedding = self.state_proj(state)[:, None, :]
    time_embedding = create_sinusoidal_pos_embedding(
        timestep,
        self.action_in_proj.out_features,
        min_period=4e-3,
        max_period=4.0,
        device=timestep.device,
    ).to(timestep.dtype)
    action_embedding = self.action_in_proj(actions)
    time_embedding = time_embedding[:, None, :].expand_as(action_embedding)
    fused = torch.cat([action_embedding, time_embedding], dim=2)
    fused = self.action_time_mlp_out(
        torch.nn.functional.silu(self.action_time_mlp_in(fused))
    )
    embedded = torch.cat([state_embedding, fused], dim=1)
    pad = torch.ones(embedded.shape[:2], device=embedded.device, dtype=torch.bool)
    att = torch.cat(
        [
            torch.ones(2, device=embedded.device, dtype=embedded.dtype),
            torch.zeros(
                self.config.action_horizon - 1,
                device=embedded.device,
                dtype=embedded.dtype,
            ),
        ]
    )[None, :].expand(embedded.shape[0], -1)
    return embedded, pad, att, None


@torch.no_grad()
def fixed_sample(model, images, masks, tokens, token_mask, state, noise, steps=10):
    # Inputs are already normalized, resized, and on the GPU. No policy transforms.
    prefix, pad, att = model.embed_prefix(images, masks, tokens, token_mask)
    mask = model._prepare_attention_masks_4d(make_att_2d_masks(pad, att))
    positions = torch.cumsum(pad, dim=1) - 1
    _, cache = model.paligemma_with_expert.forward(
        attention_mask=mask,
        position_ids=positions,
        inputs_embeds=[prefix, None],
        use_cache=True,
    )
    dt = torch.full((), -1.0 / steps, device=noise.device, dtype=torch.float32)
    time = torch.ones((), device=noise.device, dtype=torch.float32)
    x = noise
    for _ in range(steps):
        v = model.denoise_step(state, pad, cache, x, time.expand(state.shape[0]))
        x = x + dt * v
        time = time + dt
    return x


def build(args, images, state, noise, tokens):
    config = SimpleNamespace(
        pi05=False,
        paligemma_variant="gemma_2b",
        action_expert_variant="gemma_300m",
        dtype="bfloat16",
        action_dim=32,
        action_horizon=args.chunk_size,
        pytorch_compile_mode=None,
    )
    # Initialization is untimed; allocate directly on GPU to avoid a large CPU copy.
    with torch.device("cuda"):
        model = PI0Pytorch(config).eval()
    if args.openpi_checkpoint:
        from safetensors.torch import load_model

        load_model(model, args.openpi_checkpoint, strict=True)
    model.upstream_embed_prefix = model.embed_prefix
    model.upstream_embed_suffix = model.embed_suffix
    model.embed_prefix = MethodType(capture_prefix, model)
    model.embed_suffix = MethodType(capture_suffix, model)
    # OpenPI intentionally retains FP32 norms, patch embedding, and action heads.
    model.paligemma_with_expert.paligemma.language_model.config._attn_implementation = (
        "eager"
    )
    model.paligemma_with_expert.gemma_expert.model.config._attn_implementation = "eager"
    imgs = [im.unsqueeze(0).permute(0, 3, 1, 2).float() for im in images]
    masks = [torch.ones(1, device="cuda", dtype=torch.bool) for _ in imgs]
    token_mask = torch.ones_like(tokens, dtype=torch.bool)
    inputs = (imgs, masks, tokens, token_mask, state[None].float(), noise[None].float())

    def eager():
        return fixed_sample(model, *inputs)

    return model, eager, inputs
