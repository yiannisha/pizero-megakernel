# SOTA Native π0 Megakernel — Unified Megakernels × FlashRT Implementation

## Mission

Build the fastest possible **single-launch, fully native NVIDIA Blackwell π0 inference megakernel**, combining:

- The persistent native GPU execution engine from `yiannisha/Megakernels`.
- **All applicable optimized π0 and π0.5 implementations from FlashRT**, ported directly into the native VM execution path.
- Additional megakernel-specific optimizations made possible by executing the entire model inside one persistent kernel.

**This is one implementation, not two competing approaches.**

FlashRT's optimized primitives must become native Megakernels operators, fusion epilogues, memory layouts, precision paths, and scheduling strategies.

Target hardware: NVIDIA B200 (SM100a).

Primary performance objective: achieve state-of-the-art end-to-end single-request π0 inference latency on B200, with numerical fidelity sufficient for real robot execution.

The native execution must cover:

1. SigLIP: 27 vision transformer layers.
2. Gemma 2B: 18 prefix layers.
3. Gemma 300M: 18 action-expert layers.
4. All 10 Euler denoising iterations.
5. Final action chunk.

The entire model must execute inside **one CUDA kernel launch**.

### Repositories

- Target: https://github.com/yiannisha/pizero-megakernel
- Native VM: https://github.com/yiannisha/Megakernels
- Optimization source: https://github.com/flashrt-project/FlashRT
- Correctness reference: OpenPI pinned by the target repository
- Performance comparison: realtime-vla and FlashRT where configurations match

Inspected reference revisions:

- Megakernels: `08a7136cdfaa7e40629b65234d46cffcd7d48b00`
- FlashRT: `60054d253dcada1dab5e7c0ff68111f935ee41ca`

Pin exact revisions in the implementation and record any updates.

---

# 1. Non-negotiable architecture

There must be **one persistent native inference runtime**, with FlashRT optimizations embedded into it.

The completed implementation must NOT:

- Call FlashRT's externally launched CUDA kernels during inference.
- Invoke cuBLASLt from the native inference path.
- Invoke separate CUTLASS kernels.
- Dispatch Triton kernels.
- Use PyTorch model operators.
- Execute a CUDA Graph containing multiple model kernels.
- Implement GEMMs as external library calls from the host.

The implementation MUST:

- Use `megakernel::mk` and the existing native VM architecture.
- Execute FlashRT-derived GEMM, attention, normalization, quantization, fusion, and sampling algorithms as native VM instructions.
- Support native async tensor-core execution and TMA.
- Support producer-consumer pipelining, persistent CTAs, and shared-memory page management.
- Execute all model stages and denoising steps under one CUDA launch.
- Produce a fully integrated mixed-precision execution schedule.

Existing FlashRT kernels may be used as independent correctness and microbenchmark references during development, but cannot remain external execution dependencies in the final kernel.

**Critical porting rule:** A CUTLASS `GemmUniversalAdapter` wrapper cannot simply be invoked as an operation from within a persistent CUDA kernel. Port its device-level math, tiling, MMA pipeline, block-scale handling, and epilogue behavior into native VM operators. Adapt or extend the VM's warp-specialization and resource management to make that possible.

---

# 2. Phase 0 — Exhaustive FlashRT optimization audit

Before writing model kernels, audit FlashRT's actual optimized π0 and π0.5 execution paths.

Do not stop at reading documentation.

Inspect every model-relevant CUDA implementation and every optimization it enables, including implementations referenced indirectly by the model frontends.

### Primary FlashRT sources

Model pipelines:

- `flash_rt/models/pi0/pipeline_rtx.py`
- `flash_rt/models/pi0/pipeline_thor.py`
- `flash_rt/models/pi05/pipeline_rtx.py`
- `flash_rt/models/pi05/pipeline_thor_fp4.py`

Optimization documentation:

- `docs/kernel_catalog.md`
- `docs/kernel_fusion.md`
- `docs/optimization-details.md`
- `docs/pi05_prefix_nvfp4.md`
- `docs/pi05_decoder_skinny.md`
- `docs/pi05_thor_decoder_fp4_e2e.md`
- `docs/calibration.md`

Include the relevant optimized implementation families:

- `csrc/gemm/fp4/`
- `csrc/gemm/mega/`
- `csrc/gemm/fp8_smallM_*`
- `csrc/kernels/pi05/`
- `csrc/fused_fp4/`
- `csrc/attention/`
- `csrc/kernels/fused_qkv_attention.*`
- `csrc/kernels/fusion.*`
- `csrc/kernels/attention_seqused_fused.*`

### Required deliverable

Create `native_pi0/FLASHRT_PORTING_MATRIX.md`.

For every relevant FlashRT optimization, record:

1. FlashRT source and exact symbol.
2. Operation and model stage.
3. Input shapes and precision.
4. Hardware-specific assumptions.
5. Fused operations.
6. Quantization and scaling requirements.
7. Whether applicable to π0.
8. What must change for B200.
9. How it will be implemented inside the VM.
10. Numerical reference and microbenchmark.
11. Port status and final performance outcome.

Classify each optimization as:

- Transfer directly in concept and math.
- Transfer with native VM redesign.
- π0.5-specific; requires π0 semantic adaptation.
- Not transferable, with an explicit technical reason.

**Every technically transferable optimization must be implemented and evaluated.** A candidate may be omitted from the final optimized schedule only when correctness or measured performance demonstrates why another implementation is superior.

Do not assume the existing FlashRT default is optimal on B200.

---

# 3. Phase 1 — Blackwell native compute engine

Before implementing the entire model, extend Megakernels with a high-performance native operator foundation.

The Gemma native implementation is primarily optimized for GEMV. π0 needs much stronger multi-row compute primitives.

## 3.1 Native BF16 GEMM

Implement BF16 GEMM using Blackwell tensor cores with FP32 accumulation.

Use suitable ThunderKittens/CUDA primitives inside the VM.

Implement tiled GEMM with:

- Asynchronous TMA loading.
- Warp-specialized producers and consumers.
- Double/multi-buffered shared-memory stages.
- Efficient accumulation and epilogues.
- Shape-dependent tile configurations.

Support all vision, prefix, and expert projection shapes.

This provides the numerical reference within the native runtime.

## 3.2 Native FP8 GEMM

Port FlashRT's optimized FP8 projection strategies:

- Static activation scaling.
- Prequantized weight tensors.
- FP32 descale arithmetic.
- BF16/FP16 output paths.
- Shape-specialized GEMM tactics.
- Small-M and split-K variants.
- Fused norm/quantization producers.

Port the relevant small-M algorithms into the VM rather than dispatching the SM120 standalone kernels.

Implement FP8 attention-projection and feed-forward variants.

## 3.3 Native NVFP4 GEMM — top priority

Port FlashRT's block-scaled NVFP4 implementations.

Inspect especially:

- `csrc/gemm/fp4/cutlass_fp4_gemm.cu`
- `csrc/gemm/fp4/cutlass_fp4_gemm_variants.cu`
- `csrc/gemm/fp4/cutlass_fp4_gemm_geglu_il_sm100.cu`
- `csrc/gemm/fp4/cutlass_fp4_gemm_siglip_ffn_sm100.cu`
- `csrc/gemm/fp4/cutlass_fp4_gemm_fp4out.cu`
- `csrc/gemm/fp4/sm100_gelu_mul_blockscale_visitor.hpp`
- `csrc/gemm/fp4/cutlass_fp4_gemm_bias_bf16_sm100.cu`

Implement native Blackwell block-scaled tensor-core GEMM with:

- E2M1 packed FP4 weights and activations.
- UE4M3 block scales (16-element groups).
- Supported Blackwell scale-factor layouts.
- Native SM100 tensor-core instructions.
- TMA-compatible packed tensor layouts.
- FP32 accumulation.
- Correct global/descale factors.
- Quantized and BF16 output epilogues.

Support FlashRT's optimized tile families, then search for better configurations specifically within the persistent VM.

Initial candidates should include the SM100 families corresponding to:

- 128×128×128
- 128×256×128
- 128×256×256
- 128×64×256

Verify legal MMA layouts and instruction support rather than assuming every tile can be reused.

### Important: small-M performance

The expert uses approximately 51 rows for π0's default horizon.

A naive SM100 NVFP4 GEMM padded to 128 rows may waste significant computation.

Therefore, develop a native skinny GEMM family inspired by FlashRT's specialized π0.5 decoders:

- Split-K.
- Register-prefetched weight tiles.
- Multiple output tiles per CTA.
- Efficient partial reduction.
- Fused consumer operations.
- Specialized small-M tensor-core execution.
- Optional matvec-style paths for very small projections.

Benchmark these against regular SM100 FP4 GEMM inside the native VM.

Do not assume a native FP4 GEMM is the fastest choice for every shape; implement the best-performing precision-compatible primitive.

## 3.4 Native FP4 producer fusion

Support direct FP4 outputs from preceding operators.

Examples:

- RMSNorm → quantized FP4.
- LayerNorm → quantized FP4.
- GEMM + GELU → quantized FP4.
- GEMM + GeGLU → quantized FP4.
- GEMM + residual → quantized FP4.

Where possible, quantize from registers or MMA accumulators without writing an intermediate BF16 tensor to global memory.

This is essential: FlashRT's optimized epilogues must become *native VM epilogues*, not separate instructions that unnecessarily round-trip through global memory.

### Acceptance

The VM can perform BF16, FP8, and NVFP4 GEMMs with fused epilogues, including representative 51-row, 256-row, and 768-row workloads, entirely inside one persistent invocation.

---

# 4. Phase 2 — Highly optimized native SigLIP

Implement the complete 27-layer SigLIP using the fastest transferable FlashRT algorithms.

Target the best calibrated FP4/FP8 precision that meets fidelity requirements.

## Mandatory FlashRT optimizations

### Patch embedding

- Optimized patch extraction.
- Fused bias and positional embedding.
- Eliminate unnecessary im2col materialization if a direct native patch projection is faster.

### Vectorized LayerNorm

Port the vectorized SigLIP LayerNorm approach from:

`csrc/fused_fp4/siglip_ln_vec.cu`

In particular:

- Register-cached input loads.
- Vectorized memory accesses.
- Warp-level reductions.
- FP32 mean and variance.
- FP4/FP8 quantization from normalized values.
- Native scale-factor writes.

### Native FP4 attention projections

- LayerNorm → FP4.
- FP4 QKV projection with bias epilogue.
- Fused QKV attention layout handling.
- FP4 output projection with bias and residual.

Study FlashRT's fused SigLIP QKV attention algorithm, including head-dimension padding.

Port the compute into the VM, not as a separate FlashRT attention launch.

### Native FP4 SigLIP FFN

Port:

`csrc/gemm/fp4/cutlass_fp4_gemm_siglip_ffn_sm100.cu`

Implement:

- LayerNorm → FP4.
- NVFP4 up GEMM.
- Fused bias + GELU + FP4 output.
- NVFP4 down GEMM.
- Fused bias + residual.

Preserve padding and compaction behavior for hidden width 4304.

### AWQ

Investigate FlashRT's activation-aware SigLIP quantization.

Port offline AWQ weight transforms and any required runtime inverse scaling into the native operators.

Validate against representative observations, not only random inputs.

### Native scheduling

Schedule independent camera views concurrently where profitable.

Batch same-shaped GEMM work across camera views when it increases tensor-core utilization, while preserving independent image attention.

Avoid global barriers between camera views.

**Acceptance:** Fully native SigLIP with the best validated FlashRT-derived FP4/FP8 kernel strategies, outperforming the unoptimized native BF16 implementation under the same conditions.

---

# 5. Phase 3 — Highly optimized native Gemma 2B prefix

Implement all 18 Gemma prefix layers using the optimized FlashRT projection, normalization, attention and FFN algorithms.

## 5.1 Optimized RMSNorm and weight folding

Port FlashRT's RMSNorm-weight folding.

Fold `(1 + norm_weight)` into QKV and gate/up weights where numerically valid.

Perform transformations at weight-loading time.

Preserve the correct rounding semantics and validate against the untransformed model.

## 5.2 Mixed-precision attention projections

Implement native FP8 QKV/O projections and evaluate NVFP4 variants selectively.

FlashRT's π0.5 results identify Q/K quantization as precision-sensitive: broad NVFP4 conversion can introduce significant action error.

Therefore, start with:

- FP8 attention projections.
- High-precision attention accumulation.
- Native fused QKV + RoPE + KV-cache store.

Evaluate additional FP4 coverage using precision-aware, layer-specific selection.

Do not force FP4 QKV when it harms action fidelity.

## 5.3 Native optimized attention

Port the useful algorithmic aspects of FlashRT's FA-style attention:

- Tiled QK and PV matrix multiplication.
- Online softmax.
- Split-KV.
- GQA specialization.
- Sequence-length masking.
- Fused attention input/output layouts.

Support bidirectional prefix attention, rather than an autoregressive causal decode assumption.

Implement attention as a native VM operation with tile-level producer dependencies.

## 5.4 NVFP4 encoder FFN

This is a major performance target.

Port FlashRT's optimized fused NVFP4 gate/up path:

`csrc/gemm/fp4/cutlass_fp4_gemm_geglu_il_sm100.cu`

And its compact GeGLU epilogue.

Implement:

1. Residual + RMSNorm → FP4.
2. Interleaved gate/up NVFP4 GEMM.
3. GELU(gate) × up inside the epilogue.
4. Direct compact FP4 output with block scales.
5. NVFP4 down projection.
6. Residual update, optionally fused into the next normalization.

Optimize the data layout so the gate and up results are adjacent and can be combined directly without a global intermediate.

Use the optimized FlashRT half-width compact-store design where advantageous.

## 5.5 KV-cache generation

Fuse RoPE and K/V writes into projection consumers.

Avoid separate QKV rearrangement and KV-cache append instructions when data can be written directly to its final layout.

**Acceptance:** Native prefix executes all 18 layers, builds compatible K/V caches, and uses FlashRT-derived mixed-precision kernels within the single VM schedule.

---

# 6. Phase 4 — Highly optimized native π0 action expert

This stage should receive the most aggressive optimization because it executes 18 layers × 10 steps.

Target the best native skinny GEMM design and maximum practical intra-layer fusion.

## 6.1 Port FlashRT's skinny decoder architecture

Inspect:

- `csrc/kernels/pi05/pi05_decoder_skinny_fp8_sm120.cu`
- `csrc/kernels/pi05/pi05_decoder_skinny_fp4_sm120.cu`
- `csrc/gemm/fp4/nvfp4_skinny_gemm_sm120.cu`
- `csrc/gemm/fp4/nvfp4_skinny_consumers_sm120.cu`
- `csrc/gemm/fp4/nvfp4_skinny_attention_sm120.cu`

Extract transferable optimizations:

- Weight streaming with K-splitting.
- Asynchronous weight prefetch.
- Split partial accumulation.
- Fused partial reduction.
- Fused QKV + RoPE.
- Fused normalization, residual and quantization.
- Fused GeGLU output.
- Specialized small-M attention.
- Final action projection + Euler update.

Adapt the algorithms to B200 SM100a and the Megakernels native scheduler.

SM120-specific instructions or PDL mechanisms must not be assumed to work unchanged on B200.

## 6.2 π0-specific normalization

The π0 expert uses standard RMSNorm, unlike π0.5's adaptive RMSNorm.

Do NOT port π0.5 AdaRMSNorm unmodified.

Instead, apply its optimizations to the actual π0 operations:

- RMSNorm + FP8/FP4 quantization.
- Residual + RMSNorm + quantization.
- Fused gate/up GeGLU.
- Fused residual updates.
- Weight folding.
- Correct timestep/action embedding.

Exclude unnecessary π0.5 adaptive-conditioning work.

## 6.3 Native skinny NVFP4 GEMMs

Optimize all four principal projection families:

- QKV: 1024 → 2560.
- Attention O: 2048 → 1024.
- Gate/up: 1024 → 8192.
- Down: 4096 → 1024.

Build specialized configurations for π0's action horizon.

Implement optional FP8 fallback where it improves latency or fidelity.

All decisions must be made at offline planning/weight-conversion time, not by dispatching to another runtime.

## 6.4 Expert attention

Use specialized native GQA over:

- Cached prefix K/V.
- State-token K/V.
- Current-step action-token K/V.

Implement the exact π0 suffix mask.

Optimize for approximately 51 queries and up to 768+ prefix tokens.

Avoid materializing the full attention-score tensor.

Port the useful split-attention, small-query and online-softmax techniques from FlashRT.

## 6.5 Fused final projection and Euler integration

Fuse:

- Final normalization.
- Action output projection.
- Velocity scaling.
- Euler update.

Precompute constants when valid.

Keep the action buffer device-resident and reuse it across all ten steps.

Avoid redundant reads, writes, and conversions.

**Acceptance:** The full 10-step action expert executes inside one native VM launch with optimized small-M FP8/NVFP4 operators, supplied prefix K/V, and correct final actions.

---

# 7. Phase 5 — Quantization, calibration and precision optimization

Build FlashRT's quantization expertise directly into the native implementation.

This is not a separate inference strategy. It determines the data formats and instructions used by the native VM.

## 7.1 Weight preprocessing

Implement native-friendly offline weight packing for:

- BF16.
- FP8 E4M3.
- NVFP4 E2M1 + UE4M3 block scales.

Support:

- QKV packing.
- Gate/up interleaving.
- TMA-compatible tile layouts.
- AWQ transforms.
- Scale-factor layout generation.
- Quantization-aware weight folding.

Weight conversion must occur outside the timed model inference.

## 7.2 Activation calibration

Port FlashRT's calibration approach:

- Collect activation distributions from representative inputs.
- Generate static scales where appropriate.
- Evaluate saturation and quantization error.
- Aggregate denoising-step distributions when sharing scales.
- Cache calibration by checkpoint and configuration.
- Support recalibration with real observations.

For NVFP4, validate per-block scales and any global scale factors.

Handle padded or unwritten scale-factor regions deterministically; never allow uninitialized scale bytes to become NaNs.

## 7.3 Precision-aware model search

Implement an automatic offline search over per-layer/per-operation precision:

- BF16.
- FP8.
- NVFP4.
- Selected W4A16 paths.
- Optional activation-aware quantization variants.

Optimize **end-to-end latency under a fidelity constraint**, not isolated GEMM throughput alone.

The search should be able to choose, for example:

- SigLIP FP4.
- Prefix attention FP8.
- Prefix FFN FP4.
- Expert GEMMs FP4 or FP8.
- Selected BF16 operations where precision requires it.

The final precision choices must be compiled or encoded into the native VM instruction schedule.

No host-side precision dispatch during inference.

## 7.4 Quantized numerical gates

Validate:

- Per-layer intermediate activations.
- Per-denoising-step velocity.
- Final action chunk.
- Different camera observations.
- Different prompts.
- Different initial noise.
- Pretrained model checkpoints.

Measure cosine similarity, normalized RMSE, relative/absolute errors and task-level effects.

Run real policy rollouts in simulation (for example, LIBERO) when a checkpoint and environment are available.

Do not use action cosine alone as a proxy for unchanged closed-loop policy behavior.

**Acceptance:** The native megakernel uses the fastest calibrated mixed-precision schedule that satisfies explicit numerical and policy-quality gates.

---

# 8. Phase 6 — Megakernel-specific SOTA optimizations

After incorporating the transferable FlashRT optimizations, exploit the fact that the entire model now shares a single persistent GPU runtime.

These optimizations must be applied on top of the integrated optimized kernels.

## 8.1 Cross-layer producer-consumer fusion

Where correct and profitable, allow output fragments to move directly between VM operations without global-memory materialization.

Especially:

- RMSNorm → QKV.
- GEMM → RoPE/KV write.
- Attention → O projection.
- Residual → next-layer RMSNorm.
- Gate/up → GeGLU → down projection.
- Expert output → Euler update.

Measure whether fusion reduces memory traffic without harming occupancy or tensor-core throughput.

## 8.2 Device-side DAG scheduling

Use fine-grained producer counters to schedule ready work.

Explore:

- Overlap of independent camera streams.
- Earlier execution of downstream tiles.
- Balanced distribution across resident CTAs.
- Priority scheduling based on critical path.
- Tile-level dependencies where profitable.
- Reduced counter and instruction-fetch overhead.

No unsafe grid-wide barriers.

Guarantee forward progress even when all persistent CTAs are resident.

## 8.3 Denoising-invariant computation reuse

Cache computations that remain unchanged across the ten iterations.

Potential candidates:

- Prefix K/V.
- State projection.
- State-token attention representations.
- State-token K/V at individual expert layers.
- RoPE tables.
- Static masks.
- Timestep embeddings.
- Selected input transforms and scales.

Critically, verify dependencies against the exact π0 attention mask before reusing state-token intermediate results.

## 8.4 Cross-step weight reuse

Investigate whether the persistent VM can exploit repeated expert weights across denoising steps.

Potential approaches:

- Cache useful weight tiles in L2.
- Tune cache policy for expert versus prefix weights.
- Pipeline weight prefetch with current computation.
- Reuse resident/shared weight fragments where feasible.
- Reorder independent work to improve locality.

Do not assume entire expert layers fit in shared memory or registers.

Profile actual B200 cache hit rates and memory bandwidth.

## 8.5 Resident-worker and resource tuning

Search:

- Resident CTA counts.
- Shared-memory page budget.
- Tensor-memory requirements.
- Register allocation.
- Number of consumer warps.
- Instruction pipeline depth.
- GEMM tile shapes.
- Attention split count.
- Scratch-buffer layouts.

Importantly, extending the VM with TCGen05 operators may require modifying its current warp-specialization and shared-memory assumptions.

Make those changes explicitly and ensure the VM remains functional.

## 8.6 Specialized whole-model scheduling

Optimize the full fixed-shape π0 workload rather than individual generic operators.

Explore a static shape-specialized schedule for:

- Batch 1.
- Three camera views.
- Prompt length 0 or 20.
- Horizon 50.
- Ten denoising steps.

Allow separate compiled specializations for other workloads, but each inference must use one native megakernel.

The final schedule should maximize hardware utilization across the entire forward pass.

---

# 9. Phase 7 — Integrated correctness and performance

## Native inference contract

The public interface should remain:

```python
model = NativePi0.from_openpi(
    checkpoint_path,
    num_views=3,
    prompt_len=0,
    chunk_size=50,
    flow_steps=10,
    precision="auto",
)

actions = model.forward(
    images,
    state,
    noise,
    tokens,
)
```

The `forward()` execution must invoke exactly one native model kernel.

All input tensors and scratch buffers should already be device-resident.

### Correctness gates

Test in increasing scope:

1. Native GEMM primitives.
2. Fused FP8/NVFP4 epilogues.
3. SigLIP layers.
4. Prefix transformer layers.
5. Expert layers.
6. One expert denoising step.
7. Ten-step expert.
8. Complete BF16 π0.
9. Complete calibrated FP8/FP4 π0.
10. Closed-loop policy evaluation.

Use identical checkpoints and noise when comparing implementations.

### Benchmarking

Extend the existing benchmark suite with the optimized native backend.

Run on B200 with:

- Three 224×224 camera views.
- Prompt length 0.
- 50 action tokens.
- 10 Euler steps.
- Batch 1.

Compare against the existing realtime-vla baseline and any comparable FlashRT π0 implementation available for B200.

Also test multiple camera counts, prompt lengths and action horizons.

Keep input-processing and output-handling boundaries consistent.

Report:

- Median GPU latency.
- p10/p90 latency.
- Synchronized wall latency.
- Peak GPU memory.
- Actual CUDA launch count.
- Model numerical fidelity.
- Quantization policy.
- Per-stage timing.
- GPU occupancy.
- Tensor-core utilization.
- Achieved memory bandwidth.
- Register/shared-memory utilization.

Use Nsight Systems, Nsight Compute and VM instrumentation.

### SOTA optimization loop

For every significant optimization:

1. Profile the current whole-model bottleneck.
2. Identify the relevant FlashRT-derived implementation or novel VM optimization.
3. Implement and integrate it natively.
4. Verify numerical correctness.
5. Benchmark the isolated primitive.
6. Benchmark complete π0 inference.
7. Retain the optimization only if it improves the appropriate performance/fidelity objective.
8. Record the result and implementation commit.

Avoid optimization based on synthetic isolated GEMM gains that disappear end-to-end.

---

# 10. Implementation milestones

| Milestone | Required outcome |
|---|---|
| M0 | Complete FlashRT optimization inventory and native porting matrix |
| M1 | Native SM100 BF16, FP8, NVFP4 compute foundation |
| M2 | Optimized native expert using FlashRT-derived skinny kernels |
| M3 | Full 10-step native expert in a single launch |
| M4 | Optimized native Gemma 2B prefix and K/V |
| M5 | Optimized native SigLIP |
| M6 | Complete integrated one-launch π0 model |
| M7 | Calibrated mixed-precision and full pretrained-weight parity |
| M8 | Cross-layer fusion, state reuse and device-side scheduling optimization |
| M9 | Final B200 SOTA benchmark and reproducibility report |

These are implementation checkpoints, not separate runtime backends.

Implement correctness and optimization together at each component. Do not implement a slow BF16 model first and then independently rewrite the entire thing using FlashRT.

Instead, use BF16 as the reference and fallback while building the optimized native operators from the beginning.

Each milestone must leave the repository in a working, testable state.

---

# 11. Definition of done

The implementation is complete only if:

1. The complete π0 inference executes in one native CUDA launch.
2. The original Megakernels instruction VM is genuinely used.
3. All transferable FlashRT optimizations have been ported, evaluated and accounted for.
4. The best-performing applicable FP8/NVFP4 GEMMs and fusion strategies are integrated natively.
5. No FlashRT/CUTLASS/cuBLAS host-level model kernels are called during inference.
6. The native expert has a specialized small-M execution path.
7. Fused quantization and epilogues eliminate unnecessary intermediate memory traffic.
8. The model computes prefix K/V once and reuses it.
9. The final schedule exploits appropriate cross-layer and cross-step optimizations.
10. The implementation passes pretrained-weight numerical fidelity and representative policy-quality tests.
11. End-to-end benchmarks demonstrate the achieved performance against current comparable implementations.

The target is SOTA latency. Do not assert that SOTA was achieved unless measurements against appropriate competitors establish it.

## Final Codex instructions

Work directly in the target repository.

Use the actual native Megakernels and FlashRT source code as references. Do not implement generic textbook operators when a specialized and relevant FlashRT implementation exists.

Port the optimization **into the Megakernels VM**, even if that requires restructuring its compute pipeline, memory handling or instruction ABI.

Prioritize the highest-impact optimizations:

1. Native SM100 FP4 tensor-core GEMM and fused epilogues.
2. Native small-M FP4/FP8 action-expert kernels.
3. Optimized fused attention and KV-cache operations.
4. FlashRT's optimized SigLIP and prefix FFN algorithms.
5. Quantization-aware data layouts and calibration.
6. Device-side cross-layer fusion and scheduling.
7. Denoising-invariant state reuse and weight locality.

Record the latency and numerical effect of every optimization.

**Do not stop at a working but slow megakernel. Continue profiling, optimizing and comparing the integrated implementation until no clearly profitable, technically feasible improvement remains within the available implementation and hardware budget.**

Do not fabricate numerical or performance results. If a specific optimization cannot be ported, document the exact blocker and evaluate the closest fully native alternative.
