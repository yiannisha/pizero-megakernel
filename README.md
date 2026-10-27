# π0 megakernel workspace

Native implementation work is in progress under `native_pi0/`; it does not yet
provide complete π0 inference. The source audit and remaining requirements are
tracked in `native_pi0/FLASHRT_PORTING_MATRIX.md` and `PLAN.md`.

On B200, the development NVFP4 projection can be validated and measured with:

```bash
python scripts/validate_nvfp4.py
python benchmarks/native_primitives.py --workload expert --workers 20 64 148
compute-sanitizer --tool memcheck --error-exitcode 1 \
  python benchmarks/native_primitives.py --validation-only --output /tmp/native-fp4-validation.json
```

The primitive benchmark packs synthetic operands before timing and invokes one
original Megakernels VM kernel per call. It records projection latency and
checks CUDA launch count, dequantized-operand parity, and stream compatibility.
It is not an end-to-end model or pretrained policy benchmark. Its CUDA-event
interval includes host submission gaps from the Python/ctypes wrapper;
synchronized wall samples are reported separately. `--workload vision` and
`--workload prefix` select additional projection shapes. Use a distinct
`--output` path when preserving different worker sweeps. Validation-only mode
skips timing and CUPTI profiling, allowing use with compute-sanitizer.
`--epilogue bf16`, `bias`, or `bias-residual` measures direct BF16 output and
optional native fusion; `python scripts/validate_epilogues.py --profile`
checks its numerical behavior and one-launch contract.
`python scripts/validate_fp4_output.py --profile` checks direct FP4 output,
including fused GELU and bias/residual terms against pinned CUDA references.
Packed outputs use `ResidentNVFP4(output_dtype=torch.uint8)` and expose
`output_scales` separately. `output_geglu=True` selects compact GeGLU for
interleaved gate/up weights; `python scripts/validate_geglu.py --profile`
checks full expert/prefix shapes using driver launch callbacks. `--k-tile
128` or `256` selects larger native FP4 staging tiles in the primitive tools.
`python scripts/validate_norm.py --profile` checks native RMSNorm/LayerNorm
with optional residual addition, affine/AWQ scaling, and direct FP4/FP8 output.
`--row-tile 1`, `4`, or `8` groups rows per VM instruction. The producer avoids
a global normalized intermediate; complete model execution remains under development.
Setup applies the recorded VM startup barrier patch in `native_pi0/patches`.
Run `compute-sanitizer --tool synccheck --error-exitcode 1 python scripts/validate_vm.py`
to check the patched VM lifecycle.

`python scripts/validate_norm_projection.py --profile` checks a native
normalization → NVFP4 projection chain in one VM launch. Its resident wrapper
is `native_pi0.norm_projection.ResidentNormProjection`, with FP32 or BF16
projection output. All CTAs can help produce FP4 rows; projection loaders wait
on release/acquire readiness tags before TMA reads. `--producer-rows 1`, `4`,
or `8` selects ownership granularity. Add `--benchmark-output PATH` to record
matched one-launch and validation-only two-launch timings. Inputs and reused
outputs stay on the GPU during forward. Each instance requires stream ordering;
independent instances can run concurrently. Graph capture is unsupported by
this development wrapper's host-generated epochs. This chain is not the full model.

`python scripts/validate_dag.py --profile --report results/native_dag_scheduler.json`
checks an opt-in VM instruction-fetch callback, device ready queue, and terminal
instruction retirement. The scalar probe covers repeated epochs, fan-in/out,
permuted graphs, and concurrent instances. Queue initialization occurs inside
the native invocation. Setup also applies `megakernels-dynamic-instructions.patch`;
static VM programs retain their original fetch path. Connecting native model
operators to this scheduler remains in progress.

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
The pinned realtime-vla revision selects its specialized optimized path only on
SM120 with two/three views, chunk 63, and no prompt. B200 uses its general
Triton path; JSON records the upstream optimization flag.

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

Inductor's default fusion can remove intermediate BF16 rounding, so the compiled
comparison reports normalized RMSE, maximum absolute error divided by reference
RMS, cosine similarity, and the original `rtol=0.02, atol=0.02` pointwise result.
The compiled gate requires normalized RMSE ≤ 2 BF16 epsilons (0.015625), normalized
maximum error ≤ 8 epsilons (0.0625), and cosine ≥ 0.999. These are explicit
numerical smoke-test limits, not a pretrained policy accuracy guarantee. CUDA
graph and original-OpenPI comparisons use `rtol=atol=1e-5`. The compiler's precision
settings are not changed to imitate eager's intermediate rounding.

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
