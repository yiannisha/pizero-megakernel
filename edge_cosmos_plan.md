# Codex Implementation Specification
## Cosmos 3 Edge — Fully Native SOTA Megakernel

### Mission

Implement a **single-launch, fully native, aggressively optimized Cosmos 3 Edge inference engine** on NVIDIA B200 (SM100a), using the Megakernels persistent GPU instruction VM.

The implementation must integrate every technically transferable optimization from FlashRT's Cosmos 3 Edge, Cosmos 3 Reasoner, and relevant related model kernels **inside the native VM itself**.

FlashRT must serve as an implementation and optimization source—not as an alternate backend or external runtime dependency.

The objective is to achieve state-of-the-art inference performance while preserving numerical fidelity and useful downstream model behavior.

**Target repositories**

- Native runtime: https://github.com/yiannisha/Megakernels
- FlashRT: https://github.com/flashrt-project/FlashRT
- Official model: https://github.com/NVIDIA/cosmos
- Checkpoint: https://huggingface.co/nvidia/Cosmos3-Edge

Implement within the `Megakernels` repository, in a dedicated `cosmos3_edge/` module, unless the existing project architecture makes a separate closely integrated repository preferable.

Pin source revisions and the exact checkpoint revision before beginning implementation.

---

# 1. Architectural requirements

## 1.1 One unified native execution engine

Reuse and extend the genuine Megakernels instruction VM:

- `include/megakernel.cuh`
- `include/controller/`
- `include/config.cuh`
- `gemma/native.cu`
- `gemma/fused.cuh`
- `gemma/attention.cuh`
- `gemma/model.py`
- `demos/low-latency-llama/matvec_pipeline.cuh`

Preserve the architecture:

- Persistent resident CTAs.
- Device-side instruction streams.
- Controller/loader/consumer/storer specialization.
- TMA-based asynchronous memory operations.
- Native tensor-core execution.
- Shared-memory page management.
- Producer-consumer counters and semaphores.
- Tile-level dependency scheduling.
- Single CUDA launch per complete fixed-shape model execution.

Extend these mechanisms to support large multi-token transformers, diffusion loops, and multimodal attention.

Do not implement a different runtime as a shortcut.

## 1.2 Forbidden execution paths

The completed native model must not call:

- PyTorch forward operations.
- Triton kernels.
- Host-dispatched FlashRT kernels.
- cuBLAS or cuBLASLt.
- Externally launched CUTLASS GEMMs.
- FlashAttention library launches.
- CUDA Graphs wrapping multiple model kernels.

CUTLASS/CuTe algorithms may be ported into native VM operators, but the existing `GemmUniversalAdapter` host-launch path is not acceptable.

Python may perform checkpoint loading, offline calibration, tensor allocation, configuration selection and model input preparation.

## 1.3 Cosmos 3 Edge execution targets

Implement two model capabilities using the same native VM and reusable operator library.

**A. Generator / diffusion tower**

Primary milestone: AV inverse-dynamics generation, with full 30-step native denoising.

Extend the same engine to supported image/video generation workflows, including dynamic vision latents.

**B. Reasoner / autoregressive tower**

Implement:

- Native SigLIP2 vision encoding.
- Multimodal projector.
- Parallel prompt prefill.
- Native autoregressive decoding.
- Quantized KV caching.
- Native sampling.

These are native execution specializations of one architecture, not separate FlashRT versus Megakernels strategies.

The completed AV denoising request must execute all 30 model evaluations and scheduler updates in **one launch**.

For Reasoner, initially implement one native decode-token launch, then extend to a persistent device-side decoding loop that can generate multiple tokens per launch with an explicit maximum length and stopping policy.

---

# 2. Exact model specification

Audit the checkpoint and official model code before hardcoding any constants.

The currently identified Cosmos 3 Edge transformer configuration is:

| Parameter | Value |
|---|---|
| Transformer layers | 28 |
| Hidden size | 2048 |
| FFN size | 9216 |
| Query heads | 16 |
| KV heads | 8 |
| Head dimension | 128 |
| Vocabulary | 131072 |
| RMSNorm epsilon | 1e-5 |
| FFN activation | ReLU² |

The generator is part of a Mixture-of-Transformers architecture, with understanding and generation streams.

The streams have distinct normalization and projection weights.

### Critical mathematical differences from π0/Gemma

- ReLU² FFNs, not GeGLU.
- Separate understanding (`und`) and generation (`gen`) weights.
- Query/key normalization within the generation attention path.
- Special normalization of understanding keys consumed by the generator.
- Multidimensional RoPE with modality-dependent position IDs.
- 16 query heads / 8 KV heads.
- UniPC diffusion scheduler, not Euler integration.
- Different output heads for action and vision modalities.

Do not accidentally inherit π0's GeGLU, diffusion scheduler, attention masks or state caching assumptions.

### Primary AV inverse-dynamics workload

Use FlashRT's validated geometry:

- Vision latent: `(1, 48, 16, 30, 52)`.
- Spatial patches: 2×2.
- Vision tokens: 6,240.
- Action tokens: 60.
- Padded model action dimension: 64.
- Generator tokens: approximately 6,300.
- Flat diffusion state: 1,201,920 elements.
- Denoising iterations: 30.
- Scheduler: Flow UniPC.

Verify exact token ordering, conditioning masks, padding and domain-dependent action dimensions against the checkpoint and boundary fixtures.

This AV workload is the primary performance target because FlashRT already supplies a measurable optimized reference.

---

# 3. Phase 0 — Exhaustive FlashRT and official-model audit

Before implementation, audit all relevant FlashRT optimizations at the source-code level.

### Generator sources

- `flash_rt/models/cosmos3_edge/pipeline_thor.py`
- `flash_rt/models/cosmos3_edge/static_engine.py`
- `flash_rt/models/cosmos3_edge/static_unipc.py`
- `flash_rt/models/cosmos3_edge/layer_ref.py`
- `flash_rt/models/cosmos3_edge/denoise_ref.py`
- `flash_rt/models/cosmos3_edge/weights.py`
- `flash_rt/frontends/torch/_cosmos3_edge_thor_spec.py`

### FlashRT optimized CUDA sources

- `csrc/fused_fp4/cosmos3_edge_fp4.cu`
- `csrc/gemm/fp4/cosmos3_edge_fp4_gemm_relu2_fp4out.cu`
- `csrc/kernels/cosmos3_edge_misc.cu`
- `csrc/gemm/fp4/cutlass_fp4_gemm_variants.cu`
- `csrc/attention/`
- `csrc/kernels/fusion.cu`
- `csrc/kernels/attention_seqused_fused.cu`

### Reasoner sources

- `flash_rt/models/cosmos3_reasoner/pipeline_thor.py`
- `csrc/kernels/cosmos3_reasoner_gemv.cu`
- `csrc/kernels/cosmos3_reasoner_attn.cu`
- `benchmarks/cosmos3_reasoner_thor.py`

### Other transferable algorithms

Inspect the optimized FlashRT π0/π0.5 and FP4 implementations, especially:

- Native FP4 block-scaled GEMMs.
- FP8 quantization and descale techniques.
- Fused normalization and quantization.
- Specialized small-M GEMMs.
- Fused attention.
- Weight packing and AWQ.
- Stream-K / split-K execution.
- Tensor-core epilogue fusion.

Only transfer operations that are semantically applicable to Cosmos 3 Edge.

### Deliverable

Create `cosmos3_edge/OPTIMIZATION_MATRIX.md`.

For every relevant FlashRT optimization document:

1. Source file and implementation symbol.
2. Operator and intended workload.
3. Supported hardware.
4. Mathematical behavior.
5. Precision and layout requirements.
6. Transferability to Cosmos 3 Edge on B200.
7. Native VM integration design.
8. Numerical validation.
9. Primitive-level latency.
10. End-to-end performance impact.

Implement and evaluate all technically transferable candidates. Retain only the variants that win under the required fidelity gates.

Do not stop after producing the audit. Proceed directly into implementation.

---

# 4. Phase 1 — Native Blackwell compute primitives

Develop the optimized compute foundation within the VM.

## 4.1 Native BF16 GEMM

Implement a native tensor-core GEMM with:

- BF16 input and weights.
- FP32 accumulation.
- TMA loading.
- Warp-specialized pipeline.
- Configurable tile sizes.
- Fused output epilogues.
- Efficient scratch-buffer handling.

Support large sequence dimensions and small-M work.

The full BF16 path is the correctness reference and fallback within the same native runtime.

## 4.2 Native FP8 GEMM

Port FlashRT's FP8 E4M3 implementations:

- W8A8 GEMM.
- Prequantized weights.
- Calibrated activation scales.
- Static per-operation descale.
- FP32 accumulation.
- BF16 outputs.
- Fused normalization and quantization.
- Shape-aware tile tuning.

Implement native FP8 paths for all six generated-stream projections:

- Q.
- K.
- V.
- O.
- FFN up.
- FFN down.

Search fused versus separate QKV projections on B200. FlashRT found fused-wide QKV slower on Thor, but that does not establish the optimal B200 choice.

## 4.3 Native NVFP4 GEMM

Implement Blackwell block-scaled W4A4 GEMMs directly in the VM.

Port the relevant design from FlashRT's SM100 CUTLASS kernels.

Support:

- E2M1 packed weights and activations.
- Per-16 UE4M3 scale factors.
- TMA-compatible scale layouts.
- Correct scale/descale arithmetic.
- FP32 accumulation.
- BF16 output.
- FP4-producing epilogues.
- Static weight preprocessing.

Audit the FlashRT tile configurations and autotune for B200.

Do not invoke CUTLASS GEMMs externally. Reproduce their optimized device execution using ThunderKittens/CuTe or native CUDA inside the persistent VM.

## 4.4 Fused ReLU² epilogue

Port:

`csrc/gemm/fp4/cosmos3_edge_fp4_gemm_relu2_fp4out.cu`

Implement:

`FP4 GEMM → ReLU(x)² → FP4 quantization → SFA generation`

The output must feed the following down-projection without a BF16 global-memory intermediate where feasible.

This is one of the highest-priority model-specific optimizations.

## 4.5 Fused residual + RMSNorm + FP4

Port:

`csrc/fused_fp4/cosmos3_edge_fp4.cu`

Implement:

`Residual add → RMSNorm → FP4 quantization`

Use register-level fusion with efficient reductions, vectorized memory accesses and native scale-factor output.

Support FP8 and BF16 variants as needed.

### Acceptance criteria

The VM can execute all required BF16, FP8 and NVFP4 projection shapes with correct numerical behavior and fused epilogues.

Every operator must be callable from the native instruction stream without another CUDA launch.

---

# 5. Phase 2 — Native generator attention

Implement a high-performance attention operator tailored to Cosmos 3 Edge.

## 5.1 Attention structure

For each of 28 layers:

- Generator Q uses 16 heads × 128 dimensions.
- Generator K/V use 8 heads × 128 dimensions.
- The generator attends over static understanding K/V plus current generation K/V.
- Current generation tokens use noncausal attention where specified by the reference.
- Q/K normalization and RoPE must occur in the correct order.

Use the exact combined attention semantics from the official model and FlashRT's reference implementation.

## 5.2 Fused Q/K normalization and RoPE

Port:

`cosmos3_edge_qk_norm_rope_bf16`

Especially the optimized warp-per-head, head-dimension-128 implementation.

Extend it to write K/V directly into the desired native attention layout.

Fuse:

- Q/K projection epilogues where profitable.
- Q/K RMS normalization.
- Multidimensional RoPE.
- K/V cache publication.

Avoid writing intermediate Q/K tensors to global memory unless measurements justify it.

## 5.3 Native FA4-style attention

Implement native FlashAttention-style computation using:

- Tiled QK and PV.
- Online softmax.
- FP32 softmax accumulation.
- Tensor-core matrix multiplication.
- TMA K/V loading.
- Efficient GQA.
- Split-KV where beneficial.
- Asynchronous producer-consumer scheduling.

The native attention must handle roughly 6,300 generator queries and the additional understanding K/V prefix.

Support long sequence lengths without materializing the quadratic attention-score matrix.

Explore:

- FP8 Q/K/V.
- BF16 Q/K/V.
- Selective FP8-KV caching.
- Alternative tile sizes and split counts.
- Online softmax across understanding and generation segments.

The precision policy must be calibrated on real model activations.

## 5.4 Cache handling

The understanding tower's K/V is invariant across denoising steps for fixed conditioning.

Store it once and reuse it across all 30 iterations.

For each generator layer:

1. Read static understanding K/V.
2. Compute current generated-stream K/V.
3. Perform joint attention.
4. Update generator hidden states.

Do not recompute understanding K/V within each denoising step.

### Acceptance criteria

A complete native generator attention layer produces numerically consistent results for the real AV token geometry.

The attention implementation must be benchmarked both independently and inside the VM.

---

# 6. Phase 3 — Native 28-layer Generator

Implement the complete generator transformer using the optimized native operators.

The per-layer execution should approximately follow:

```text
Generation hidden states
      |
      v
RMSNorm → FP8
      |
      v
Q/K/V projections
      |
      v
Q/K norm + 3D RoPE
      |
      v
Native joint GQA attention
  (cached und K/V + gen K/V)
      |
      v
O projection + residual
      |
      v
RMSNorm → NVFP4
      |
      v
FP4 FFN up projection
      |
      v
ReLU² + FP4 epilogue
      |
      v
FP4 FFN down projection
      |
      v
Residual / next-layer RMSNorm
```

All 28 layers execute natively.

### Optimization requirements

- FP8 for attention projections where accurate and fast.
- Native NVFP4 FFN.
- Fused residual and norm.
- Direct FP4 producer-consumer paths.
- Static understanding K/V reuse.
- Fused Q/K normalization and RoPE.
- Native attention without external FA4 launch.
- Layer-specific tile choices.
- Minimal global-memory intermediate materialization.

### Last-layer action-only optimization

FlashRT's AV path exploits the fact that only 60 action-token outputs are needed from the last layer.

Port this optimization.

At the final layer:

- Compute K/V for all necessary generated tokens.
- Compute Q, attention, O projection and FFN only for the 60 action tokens.
- Run the action head on those outputs.

This optimization is workload-specific.

Do not apply the 60-row shortcut to video generation or a workflow requiring full vision outputs.

### Acceptance criteria

A fully native 28-layer generator computes the correct action-token outputs for one timestep, given cached understanding K/V.

---

# 7. Phase 4 — Native conditioning and modality projections

The eventual native Generator must not depend on externally computed Transformer conditioning.

## 7.1 Native understanding tower

Implement the understanding (`und`) transformer using native operators.

Support:

- 28 layers.
- RMSNorm.
- Native causal GQA.
- QKV/O projections.
- ReLU² FFN.
- 3D RoPE.
- Generation-compatible per-layer K/V.

Compute the understanding stream once for each new conditioning request.

Store the appropriate normalized and RoPE-transformed K/V required by the generator.

## 7.2 Native action embedding

Implement the domain-selected action input and output heads.

Support:

- 64-dimensional padded action representation.
- Domain-specific weight selection.
- Action modality embedding.
- Timestep conditioning.
- 60 action tokens.
- Correct zero-padding of unused output dimensions.

Fuse:

- Action tail extraction.
- FP32→BF16 conversion.
- Action input projection.
- Domain bias.
- Modality bias.
- Timestep embedding addition.
- Scatter into the generation sequence.

Use domain-specific packed weights when known at setup time.

## 7.3 Native vision-token preparation

Port the relevant model-side operations:

- Latent patchification.
- Patch embedding/projection.
- Timestep embedding for noisy vision tokens.
- Vision/action token assembly.
- Modality-specific indexing.
- Spatial and temporal positional encoding.

For the AV inverse-dynamics workload, precompute vision embeddings that are invariant across denoising steps.

For full video generation, recompute timestep-dependent vision tokens correctly.

Do not reuse static AV assumptions when vision latents are being generated or denoised.

## 7.4 Single-launch cold-conditioning mode

Support two modes within the same native execution system:

- **Preconditioned:** already-computed understanding K/V supplied at launch.
- **Cold conditioning:** native understanding computation followed by full generator denoising inside one launch.

The preconditioned mode measures the optimized repeated-inference boundary.

The cold-conditioning mode measures the full native Transformer boundary for a fresh request.

Both must use the same VM, shared kernels and weight representations.

### Acceptance criteria

The native Generator can process model-ready conditioning inputs without framework Transformer calls and produce the appropriate K/V, generator embeddings and outputs.

---

# 8. Phase 5 — Native 30-step UniPC denoising loop

Implement the full diffusion process within one persistent kernel launch.

This is mandatory.

### Loop

```text
ONE CUDA LAUNCH

Initialize latent + scheduler state

Compute or load conditioning K/V

FOR step = 0..29:
    Prepare timestep embeddings
    Build current action / vision tokens
    Execute generator layers 0..27
    Compute action / vision velocity
    Run native UniPC update
    Update scheduler history

Write final latents / actions

RETURN
```

No host synchronization between denoising steps.

## Native UniPC

Port:

`cosmos3_edge_unipc_step_f32_bf16`

And its coefficient-precomputation logic from:

`flash_rt/models/cosmos3_edge/static_unipc.py`

Implement:

- Predictor/corrector coefficients.
- Scheduler history buffers.
- FP32 latent-state updates.
- BF16 model velocity input.
- Correct order-dependent history handling.
- Static timestep schedule.
- Precomputed coefficient tables.

Fuse the end-of-step velocity transformation with the scheduler where beneficial.

The complete flat latent state must be updated, including the correct modality regions.

For AV inverse dynamics, reproduce FlashRT's zero-velocity behavior for the fixed vision segment.

## Device-side iteration control

Implement a 30-step device-side schedule.

Either:

- Statically expand the complete instruction DAG, or
- Use a compact repeated device-side instruction schedule with generation-aware counters.

Prefer the latter if it substantially reduces instruction memory and improves scheduling.

All synchronization must remain safe across repeated steps.

Implement epoch-aware producer counters, buffer ownership and progress guarantees.

### Acceptance criteria

One native kernel launch performs all 30 generator evaluations and all 30 UniPC updates.

The final action result passes the numerical gates against the official implementation.

---

# 9. Phase 6 — Native Cosmos 3 Reasoner

Extend the same VM to support Cosmos 3 Edge autoregressive multimodal reasoning.

This is not a separate backend.

## 9.1 Native SigLIP2 + PatchMerger

Port the Reasoner's actual vision pipeline:

- 27-layer SigLIP2.
- 1152 hidden size.
- 16 attention heads.
- 72 head dimension.
- 4304 FFN dimension.
- Vision positional interpolation.
- Per-media attention windows.
- 2×2 spatial merging.
- Multimodal projector into hidden width 2048.

Reuse applicable optimized SigLIP kernels from FlashRT's π0/π0.5 work.

Implement native BF16/FP8/NVFP4 projections and fused normalization/GELU.

Do not assume this vision tower has the same patch-token interface as π0.

## 9.2 Native parallel prefill

The current FlashRT Reasoner uses PyTorch for vision and prefill. Replace that path with true native multi-row Transformer execution.

Implement:

- Token embedding.
- Multimodal token insertion.
- 3D mRoPE.
- Causal attention.
- Native KV-cache construction.
- 28-layer Transformer.
- Final norm and logits.

Do not copy the old Gemma megakernel's token-by-token prefill approach.

Use tensor-core GEMM for multi-token prefill.

## 9.3 Native W4A16 decode

Port FlashRT's Reasoner kernels:

- `cosmos3_reasoner_gemv_w4a16_bf16`
- `cosmos3_reasoner_rope_kv_fp8_bf16`
- `cosmos3_reasoner_decode_attn_fp8kv_bf16`

The initial design should support:

- Packed E2M1 weight codes.
- Per-16 BF16 weight scales.
- BF16 activations.
- FP32 accumulation.
- Fused QKV.
- Fused RoPE/KV append.
- FP8 KV-cache storage.
- Native split-KV attention.
- Native sampling.

FlashRT's Reasoner is dominated by W4A16 GEMV and attention bandwidth.

Do not assume a generic large-M tensor-core GEMM is optimal for single-token decode.

Benchmark:

- Native W4A16 GEMV.
- Native tensor-core W4 strategies where feasible.
- Different output-row assignments.
- Weight prefetching and L2 locality.
- Fused norm/GEMV.
- Reduced intermediate stores.

## 9.4 Native decode loop

Build the native Reasoner using the same instruction VM.

Implement:

- Single-token decode.
- Device-side token sampling.
- KV-cache append.
- Position update.
- Optional persistent multi-token generation.
- EOS handling.
- Maximum generation length.
- Deterministic fixed-length benchmark mode.

The target is a single launch per full fixed-length decode request, with no Python loop dispatching each token.

The implementation should support text, image and video prompts using the native vision/prefill path.

### Acceptance criteria

Native prefill and decode match the official Reasoner's reference numerically within documented tolerances.

Quantized generation is evaluated on representative multimodal prompts.

---

# 10. Phase 7 — Model-specific quantization and calibration

Integrate FlashRT's precision optimizations directly into native weight preparation and execution.

## Weight packing

Implement offline preparation for:

- BF16.
- FP8 E4M3.
- NVFP4 E2M1.
- Per-16 block scales.
- W4A16 Reasoner weights.
- Fused QKV layouts.
- ReLU² fused-epilogue output layouts.
- Native TMA-compatible matrix layouts.

## Calibration

Port FlashRT's calibration principles:

- Per-site activation-scale collection.
- Representative real-input calibration.
- Scale aggregation across denoising steps.
- Static scales for the optimized inference path.
- Quantization-error measurement.
- Checkpoint/shape-specific calibration caches.

Evaluate static versus dynamic quantization where relevant.

## Mixed precision

Perform an offline search over:

- BF16 attention.
- FP8 attention projections.
- FP8 KV cache.
- NVFP4 FFNs.
- W4A16 Reasoner decode.
- Selected FP4 attention projections.
- Sensitive operations retained in BF16.

Optimize end-to-end latency subject to numerical and task-level fidelity constraints.

The chosen policy must be encoded directly into the VM instruction schedule.

No separate inference backend or external precision dispatch.

---

# 11. Phase 8 — Megakernel-specific optimization

After porting the FlashRT optimizations, improve them through native persistence and cross-operation scheduling.

## 11.1 Cross-layer fusion

Explore:

- RMSNorm → QKV.
- Q/K normalization → RoPE → KV write.
- Attention → output projection.
- Output projection → residual → RMSNorm.
- FFN up → ReLU² → FP4.
- FFN down → residual → next-layer RMSNorm.
- Action projection → timestep conditioning.
- Action head → UniPC update.

Eliminate unnecessary global memory intermediates.

Retain fusion only when it improves end-to-end performance.

## 11.2 Persistent weight scheduling

The generator reuses the same layer weights 30 times.

Investigate:

- Cache policies that improve repeated FFN weight access.
- TMA prefetch scheduling.
- Overlap of weight loading with attention.
- Reduced redundant weight reads.
- Reuse of on-chip weight tiles across nearby work.
- Offline weight layouts optimized for the device-side instruction schedule.

Do not assume entire layers can be kept on chip.

Use B200 hardware counters to identify achievable reuse.

## 11.3 Critical-path scheduling

Develop a device-side DAG scheduler that prioritizes downstream readiness.

Explore:

- Tile-level producer-consumer dependencies.
- Overlap of independent modality operations.
- Early attention scheduling.
- Fine-grained pipeline staging.
- Cost-balanced assignment to persistent CTAs.
- Reduced synchronization traffic.
- Reduced VM instruction-fetch overhead.

Guarantee deadlock freedom and forward progress.

## 11.4 Attention/FFN balance

The AV workload has long-sequence attention and wide FFNs.

Profile whether performance is dominated by:

- QKV/O projections.
- Attention QK/PV.
- FFN up/down.
- HBM traffic.
- Shared-memory staging.
- VM synchronization.
- UniPC updates.

Optimize according to actual B200 measurements, not FlashRT's Thor bottleneck ordering.

## 11.5 Dynamic-shape specialization

Generate optimized native schedules for specific sequence lengths and modalities.

Prioritize:

- AV inverse dynamics with 60 action tokens.
- Text-only Reasoner.
- Image-conditioned Reasoner.
- Video-conditioned Reasoner.
- Supported image/video generation configurations.

Each compiled specialization must still execute within the same native VM architecture.

---

# 12. Phase 9 — Optional diffusion-step caching

Port FlashRT's TeaCache-style acceleration as an optional native scheduling optimization.

This must not replace the primary no-cache SOTA implementation.

Support a selected set of model-compute steps while UniPC advances on every required timestep using the intended velocity reuse.

Implement device-side static schedules.

Examples:

- 30 full model evaluations.
- 15 evaluations.
- 6 evaluations.
- 3 evaluations.
- 2 evaluations.

Only enable reduced-compute configurations after verifying:

- Final action similarity.
- Per-step trajectory divergence.
- Numerical stability.
- Multiple seeds.
- Multiple observations.
- Downstream policy success.

Report step-caching speedups separately from kernel-execution speedups.

A fast implementation that simply skips most model evaluations must not be represented as a faster full 30-evaluation megakernel.

---

# 13. Phase 10 — Full image/video generation

Extend the Generator beyond FlashRT's fixed-vision AV inverse-dynamics workload.

Support the official model's compatible:

- Text-to-image.
- Text-to-video.
- Image-to-video.
- Vision/action-conditioned generation where the checkpoint supports it.

This requires handling dynamic video latents rather than assuming fixed vision tokens.

Implement:

- Modality-dependent sequence construction.
- Dynamic timestep-conditioned vision tokens.
- Video latent projection and output heads.
- Correct multimodal attention masks.
- 3D positional encoding.
- Full latent velocity output.
- UniPC updates over all generated modalities.

Do not incorrectly apply action-only last-layer pruning to vision generation.

## VAE boundary

The Wan2.2 VAE is an external codec component of the generation pipeline.

Initially, benchmark the native Transformer on GPU-resident latents.

Then optimize the full input-to-output workflow, including VAE encode/decode.

Inspect FlashRT's `flash_rt/models/cosmos3_edge/vae_native.py`, including its:

- RMSNorm + SiLU fusion.
- 3D average downsampling.
- Causal Conv3D optimizations.
- T=1 Conv3D→Conv2D specialization.
- Channels-last memory layout.

Port useful VAE optimizations into native CUDA operators.

A separate VAE launch is permissible for an explicitly labeled Transformer-only single-launch result, but not for a claim that the entire user-visible video workflow runs in one CUDA launch.

Keep Transformer-only and full-workflow benchmark contracts separate.

---

# 14. Benchmarking and correctness

## Primary benchmark: AV inverse dynamics

Use FlashRT's exact 30-step AV inverse-dynamics workload and NVIDIA's official reference.

Performance contract:

- GPU-resident, prepared conditioning and input latent.
- All 30 velocity evaluations.
- Native UniPC updates.
- No TeaCache.
- No VAE preprocessing.
- No host-stage dispatch.
- One native CUDA launch.

Measure:

- GPU execution latency.
- Synchronized wall latency.
- p10/p50/p90.
- Memory usage.
- Tensor-core utilization.
- HBM throughput.
- Attention latency.
- GEMM latency.
- Native instruction overhead.
- Numerical parity.

Use NVIDIA B200 for the principal comparison.

Re-run FlashRT on B200 if supported or after a controlled B200 port. Do not compare B200 times directly against the published Thor result as evidence of a kernel speedup.

## Reasoner benchmark

Use FlashRT's reproducible profiles:

- Text prompt, 1705 tokens.
- Image prompt, 911 tokens.
- Video prompt, 1263 tokens.
- Batch 1.
- Greedy decode.
- 128 generated tokens.

Report separately:

- Vision encoding.
- Prompt prefill.
- Decode ms/token.
- Tokens/second.
- Complete request latency.

Compare native BF16 and W4A16/FP8-KV variants against official and FlashRT paths using identical hardware.

## Video-generation benchmark

Use the official supported configurations.

Measure Transformer denoising independently and full video generation including VAE separately.

Use fixed seeds, prompts, frame counts and resolution.

Compare against official PyTorch, vLLM-Omni and TensorRT-LLM results when the same hardware and inference contracts are available.

## Numerical validation

Implement:

- BF16 native versus official reference.
- FP8 native versus BF16.
- NVFP4 native versus BF16.
- Per-layer activation parity.
- Attention-output parity.
- Understanding KV-cache parity.
- Generator per-step velocity parity.
- Native UniPC parity.
- Final action output parity.
- Reasoner logits and generated-token checks.

FlashRT's AV reference uses gates of cosine ≥ 0.999 and relative L2 error below 3%.

Use these as initial compatibility gates, then add stricter model- and application-level checks where appropriate.

Validate closed-loop action behavior and generation quality rather than relying only on tensor similarity.

## Profiling

Collect Nsight Systems and Nsight Compute profiles.

Instrument the VM to measure:

- Instruction queue latency.
- Producer wait time.
- Consumer stall time.
- TMA bandwidth.
- Tensor-core active cycles.
- Resident CTA occupancy.
- Shared-memory page reuse.
- Register usage and spills.
- Barrier/counter overhead.
- Per-layer and per-denoising-step critical paths.

Use this evidence to direct optimization.

---

# 15. Repository implementation structure

Suggested layout:

```text
Megakernels/
├── cosmos3_edge/
│   ├── README.md
│   ├── ARCHITECTURE.md
│   ├── OPTIMIZATION_MATRIX.md
│   ├── Makefile
│   ├── native.cu
│   ├── config.cuh
│   ├── ops.cuh
│   ├── gemm_bf16.cuh
│   ├── gemm_fp8.cuh
│   ├── gemm_nvfp4.cuh
│   ├── gemv_w4a16.cuh
│   ├── norm.cuh
│   ├── relu2.cuh
│   ├── rope.cuh
│   ├── attention.cuh
│   ├── vision.cuh
│   ├── generator.cuh
│   ├── reasoner.cuh
│   ├── scheduler_unipc.cuh
│   ├── model.py
│   ├── schedule.py
│   ├── weights.py
│   ├── calibrate.py
│   ├── tests/
│   ├── benchmarks/
│   └── results/
```

Reuse existing Megakernels headers and runtime infrastructure rather than duplicating them.

Build for `sm_100a`.

Keep the original native Gemma implementation functional.

---

# 16. Implementation milestones

| Milestone | Deliverable |
|---|---|
| M0 | Full official/FlashRT audit and optimization matrix |
| M1 | Native optimized BF16/FP8/NVFP4 tensor-core operators |
| M2 | Native long-sequence GQA with Q/K norm + 3D RoPE |
| M3 | Optimized native 28-layer Generator, one timestep |
| M4 | Native 30-step Generator + UniPC in one launch |
| M5 | Native conditioning tower and AV embeddings |
| M6 | Complete native no-cache AV inverse-dynamics inference |
| M7 | Calibrated mixed precision and fused FP4 ReLU² |
| M8 | Native Reasoner vision + parallel prefill |
| M9 | Native Reasoner W4A16/FP8-KV decode |
| M10 | Persistent multi-token Reasoner decode |
| M11 | Cross-layer, memory locality and scheduling optimization |
| M12 | Native video-generation path and extended VAE optimization |
| M13 | Comprehensive SOTA benchmarking and reproducibility |

At each milestone:

1. Implement the actual native code.
2. Compile for B200.
3. Execute numerical tests.
4. Benchmark the operation and full available path.
5. Profile regressions.
6. Commit the working implementation.
7. Continue to the next milestone.

Avoid leaving parallel experimental runtimes in the final inference architecture.

---

# 17. Final acceptance criteria

The project is complete when:

- The complete AV inverse-dynamics Generator runs in one native CUDA launch.
- All 30 denoising evaluations execute inside that launch.
- UniPC executes natively.
- Static understanding K/V is reused correctly.
- All 28 generation layers execute using optimized native operators.
- Every transferable FlashRT Cosmos-specific optimization has been implemented, tested and accounted for.
- FP8 and NVFP4 execution is natively integrated.
- Native fused ReLU²/FP4 epilogues operate correctly.
- The Reasoner supports native multimodal vision, prefill and decode.
- Full video generation uses the same native infrastructure and correct dynamic-latent semantics.
- No host-dispatched framework model operators occur inside the claimed native execution boundaries.
- Numerical and application-level correctness gates pass.
- Actual latency is measured against comparable available baselines.
- Performance optimizations are documented and reproducible.

## Instructions for Codex

This is an **implementation and performance research project**, not merely a port or wrapper.

Use Megakernels as the execution engine and FlashRT as the source of optimized low-level algorithms.

Do not write a separate FlashRT backend.

Do not settle for a BF16-only implementation or an external CUTLASS wrapper.

Port the strongest relevant optimizations **into the native VM itself**.

Begin with the AV Generator, since it already has a strong optimized FlashRT reference and a fixed 30-step workload. Then complete the native understanding tower, Reasoner, and full generation pathways.

For the AV Generator, prioritize:

1. Native SM100 FP8/NVFP4 GEMMs.
2. Native FA-style long-sequence attention.
3. Fused Q/K norm + 3D RoPE + KV operations.
4. Fused FP4 up GEMM + ReLU² + FP4 output.
5. Fused residual + RMSNorm + FP4.
6. Complete device-side UniPC integration.
7. Static understanding KV reuse.
8. Final-layer action-only computation.
9. Cross-layer native scheduling.
10. Weight-cache optimization across 30 denoising steps.

For the Reasoner, prioritize native parallel prefill, W4A16 GEMV, FP8-KV attention, fused RoPE/KV append, and persistent greedy decoding.

**The ultimate objective is the fastest numerically trustworthy native Cosmos 3 Edge inference implementation achievable on B200.**

Do not declare success simply because the entire model runs in one launch. Profile the implementation, identify bottlenecks, and continue optimizing until no clearly profitable supported optimization remains within the available resources.

Record actual results; never fabricate benchmarks or claim SOTA without comparing against suitable contemporary implementations.
