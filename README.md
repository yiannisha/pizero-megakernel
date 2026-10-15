# π0 megakernel workspace

BF16 latency baselines for the full π0 model (SigLIP vision tower, Gemma 2B
prefix, Gemma 300M action expert, ten Euler denoising steps):

- OpenPI eager
- OpenPI compiled with Inductor (`max-autotune-no-cudagraphs`)
- OpenPI eager captured in an explicit CUDA graph
- realtime-vla Triton, using its upstream CUDA graph

## Setup

Requires Linux, Python 3.11+, and a CUDA GPU supporting BF16. The tested
runtime is PyTorch 2.8.0 + CUDA 12.8 / Triton 3.4.0 on NVIDIA B200.

```bash
uv venv .venv
uv pip install --python .venv/bin/python -r requirements.lock
.venv/bin/python scripts/setup.py
.venv/bin/python benchmarks/run.py --backend all --output results/bf16.json
```

`setup.py` clones the two repositories at the commits in `upstreams.json` and
installs OpenPI's required transformers 4.53.2 overlay into the venv. These
sources are fetched dependencies, not vendored project code. Their licenses
remain in `third_party/*/LICENSE`.

For a short smoke run:

```bash
.venv/bin/python benchmarks/run.py --backend all --warmup 2 --iterations 3 --output results/smoke.json
```

Select a backend with `--backend openpi-eager`, `openpi-compiled`,
`openpi-cudagraph`, or `realtime-triton`. Shape controls are `--views {1,2,3}`,
`--prompt-len N`, and `--chunk-size N`. Defaults: batch 1, three 224×224
views, zero language tokens, chunk 50, action dimension 32, ten flow steps.
`--chunk-size 63` matches the shape highlighted in realtime-vla's performance
reports; it is not OpenPI's usual chunk length. Empty prompt means literally
zero language tokens, not a tokenizer's encoding of an empty string.

## Measurement contract

Inputs are normalized GPU-resident images, state, language token IDs, and
fixed supplied diffusion noise. The timed operation includes vision encoding,
prefix KV construction, and every denoising step. It excludes checkpoint
loading, compilation, graph capture, resizing, tokenization, host/device
transfers, and robot policy transforms. Triton's public forward includes its
three device-to-device input copies and uses cached language embeddings;
OpenPI performs its language embedding lookup. No sleeps or CPU work are
inserted into the timed model call.

Every backend runs in a fresh subprocess, with warmup followed by synchronized
individual inference measurements. JSON includes raw samples, median/p10/p90,
CUDA-event and host wall times, setup time, peak allocated memory, shapes,
precision policy, seeds, software versions, GPU, and upstream revisions.
First-use compilation can take several minutes.
Wall measurements include event submission and completion overhead; they are
not a CPU-only dispatch measurement. CUDA events can include GPU idle gaps
caused by eager Python submission. These are single-request latency results,
not throughput measurements. Run on an otherwise idle GPU.

OpenPI's BF16 policy intentionally keeps patch embedding, normalization
parameters, and action heads in FP32. Triton uses BF16 weights and activations
with FP32 accumulators and its upstream fused operations. These are the
upstream BF16 inference policies, not an assertion that every operation is BF16.
OpenPI's attention implementation remains eager as selected by upstream
`sample_actions`; compiling it still permits Inductor optimization.

The OpenPI adapter replaces the tensor-controlled `while` with exactly ten
Euler steps and constructs constant attention masks and timestep scalars on the
GPU so CUDA capture is possible. Its inference-only embedding adapters preserve
the upstream operations while avoiding Python-list-to-CUDA tensor copies. It uses
the upstream vision/language modules, KV-cache, and denoising code. Each run
checks this adapter against upstream `sample_actions` with the same prepared inputs and noise, and checks the
compiled/captured result against eager. The upstream comparison restores both
original embedding methods as well as the original sampling loop. Capture uses
eager kernels; the compiled baseline explicitly disables implicit Inductor CUDA
graphs.

## Weights and correctness

By default, these are **synthetic latency benchmarks**: all weights are
initialized, but OpenPI and Triton use independently initialized weights.
They do not establish pretrained policy accuracy or cross-backend numerical
parity. Finite outputs and expected output shapes are mandatory. In particular,
the adapter does not reproduce realtime-vla's upstream benchmark's use of
uninitialized `torch.empty` weights.

For trained runs, pass OpenPI's `model.safetensors` with
`--openpi-checkpoint /path/model.safetensors`. Pass a complete Triton weight
dictionary serialized with `torch.save` using
`--triton-checkpoint /path/triton.pt`. The latter must use upstream Pi0Inference's
fused layout and contain `language_embeds` of shape `(prompt_len, 2048)`.
The upstream `convert_from_jax.py` documents and implements that layout; its
pickle output must be converted to a `torch.save` tensor dictionary in a trusted
environment. Both loaders reject missing or mismatched weights. Token IDs in
this harness are synthetic; pretrained Triton language embeddings must match
those IDs to claim a common language input. Using checkpoint paths alone does
not establish cross-backend equivalence.

Upstream references: [OpenPI](https://github.com/Physical-Intelligence/openpi)
and [realtime-vla](https://github.com/dexmal/realtime-vla). The future megakernel
should use the same workload and timing boundary and add a separate parity gate
with shared trained weights and real tokenized observations.
