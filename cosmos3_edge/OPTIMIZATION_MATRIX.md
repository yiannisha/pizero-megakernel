# Cosmos 3 Edge optimization audit (in progress)

FlashRT source revision: `60054d253dcada1dab5e7c0ff68111f935ee41ca`.
Official framework: `cc28c0c36bf8f39d5d0a7afe71e237178f180155`.
This is a growing source audit, not evidence that M0 or any operator is complete.
All paths below are relative to FlashRT. The shared VM now has a synthetic-validated ReLU²/FP4 GEMM epilogue, described
below. The rows remain source candidates; end-to-end impact is **unmeasured**.

| Source and symbol | Workload/math/layout | Hardware and transfer | Native VM integration design |
|---|---|---|---|
| `csrc/fused_fp4/cosmos3_edge_fp4.cu::edge_res_rms_fp4_sfa_bf16_kernel` | BF16 residual addition; FP32 sum-of-squares and normalization on the unrounded residual sum; residual stored BF16; weighted normalized values quantized in groups of 16 to E2M1 with UE4M3 scales in CUTLASS SFA layout | NVFP4 path, Blackwell; directly applicable to 2048-wide gen stream | Consumer row operation with explicit scale-layout contract and residual publication dependencies; compare unrounded-norm behavior to BF16 reference |
| `csrc/fused_fp4/cosmos3_edge_fp4.cu::edge_relu2_fp4_sfa_fp16_kernel` | FP16 input; FP32 `max(x,0)^2`; per-16 FP4 quantization and SFA | Blackwell; applicable only if the producing dtype matches | Fallback consumer epilogue; prefer direct accumulator conversion when fidelity permits |
| `csrc/gemm/fp4/cosmos3_edge_fp4_gemm_relu2_fp4out.cu::cosmos3_edge_fp4_gemm_relu2_fp4out` / `ReLU2` | SM100 block-scaled GEMM; FP32 accumulator; E2M1 A/B; ReLU² then E2M1 output with per-16 UE4M3 SFD; tile 128×256×256, cluster 1×1×1 | SM100; directly applicable gen FFN up | Port device mainloop and epilogue into VM consumer/storer; host `GemmUniversalAdapter` is forbidden for inference; autotune tile in persistent residency budget |
| `csrc/kernels/cosmos3_edge_misc.cu::unipc_step_f32_bf16_kernel` | FP32 sample/history; BF16 velocity; **BF16 rounding of sigma×velocity** before FP32 subtraction; order-1/2 corrector and predictor | Ordinary CUDA; applicable on B200 | Elementwise instruction with disjoint sample/history outputs and step dependency; all 30 steps remain device scheduled |
| `flash_rt/models/cosmos3_edge/static_unipc.py::_precompute_coefficients` | CPU FP32 coefficient tables; order ramps 1→2, final predictor order 1 | Setup-only, applicable | Offline tables uploaded once; no Python dispatch between steps |

## Confirmed semantic constraints

- Verified checkpoint metadata: 28 layers, hidden 2048, FFN 9216, Q=16,
  KV=8, head dimension 128, ReLU², RMS epsilon 1e-5.
- Understanding `to_*` and generation `add_*` projections are distinct.
  Generation has separate Q/K norms and `k_norm_und_for_gen`.
- RoPE axes `[24,20,20]`, theta 100000000; exact official coordinate and
  rotation mapping still needs audit before implementing device RoPE.
- FlashRT AV `dump_replay.py` sets shift **10**, while the pinned scheduler
  configuration lists `flow_shift: 1`; setup must reproduce the actual workload
  schedule rather than silently selecting the checkpoint default.
- The action domain table has 32 domains and padded action width 64. Domain
  output masks and token ordering still require boundary validation.

## Remaining audit scope

Read full generator static engine/pipeline/reference and official implementation;
FP8 projections and calibration; FP4 scale conversion and packing; attention
kernels and split-KV; Q/K norm and strided RoPE; action/vision preparation;
understanding caching; action-only final layer; Reasoner GEMV/KV/sampling;
SigLIP2/PatchMerger; related π0/π0.5 transferable primitives; TeaCache;
VAE; all required benchmark contracts. Each candidate needs numerical tests,
B200 latency, and measured end-to-end impact before retention.

## Native ReLU²/FP4 implementation evidence

`native_pi0/nvfp4_probe.cu::Fp4Matmul::consumer` now supports activation mode 3:
FP32 accumulator → ReLU² → global output scaling → per-16 UE4M3 scale and E2M1
codes. Output uses the existing native SFA layout. No BF16 global intermediate
is created. Existing identity/GELU/GeGLU modes keep their semantics.

`cosmos3_edge/tests/validate_relu2.py` compiles SM100a and checks byte-exact data
and scale output against CUTLASS ReLU + multiplication / FlashRT quant_block16,
including M=60 and 6300 with N=9216, K=2048, and optional bias/residual.
`results/relu2_fp4.json` records initial K64 measurements and profiler evidence
of one original `megakernel` launch. This is not yet a calibrated model policy,
complete FFN, full fused CUTLASS epilogue equivalence, or end-to-end benchmark.

## Checkpoint header audit

Pinned remote safetensor headers confirm all 549 indexed BF16 tensors in two
shards, totaling 6,739,314,048 tensor bytes. Explicit shapes are generated in
`weights.py` from checkpoint-derived configuration. Validation rejects missing,
duplicate, misassigned, incorrectly typed/shaped, and overlapping tensor ranges.
The complete weight payload has not yet been checked; headers alone do not
establish tensor-content integrity or numerical correctness.

Official `cosmos_framework/model/generator/mot/unified_mot.py:790` applies gen
Q/K normalization before RoPE. Lines 833–848 normalize raw und K separately
before RoPE for generator consumption, retaining raw post-RoPE und K for causal
understanding attention. These must remain distinct cache representations.

Initial synthetic tile exploration (`results/relu2_fp4_k256.json`) also passes
all byte-exact gates and one-launch profiling. With no bias/residual, K256
measured p50 0.562 ms for M=6300 and 0.068 ms for M=60 (N=9216, K=2048),
versus K64 0.846 ms and 0.106 ms. These are separate development runs, not
an isolated end-to-end comparison or a model-policy selection. The existing
`python scripts/validate_fp4_output.py` identity/GELU regression suite also passed.

## Integrated ReLU² FFN

`cosmos3_edge/native.cu` includes the shared VM operators; its Makefile builds
SM100a without a host GEMM adapter. `ffn.py::NativeFFN` schedules FP4 up,
fused ReLU²/FP4/SFA and down with optional residual through the same ready queue.
The down consumer binds directly to the producer's packed data and scales.
`results/ffn.json` covers M=3,129,60,6300 with repeated epochs, permuted nodes,
poisoned intermediates and nondefault streams. Packed producers match exactly;
final synthetic output relative L2 is at most 2.17e-6 versus the independent
quantized reference. CUPTI confirms one launch per FFN. Small-case memcheck
reports zero errors (`results/ffn_memcheck.txt`). Initial M6300 H2048 I9216
latency is 2.82 ms median; this still needs profiling and optimization.

The actual checkpoint shards have now been fetched and verified against pinned
upstream LFS SHA256 (`sources/weight_files.json`, `results/checkpoint_integrity.json`).
`download_checkpoint.py` verifies full file size/hash before publishing each
shard, and `CheckpointWeights` checks content hashes by default.

**Fidelity gate not met:** `results/ffn_checkpoint.json` uses actual generator
layer-27 weights with synthetic activations. The research packing policy
(per-16 ceil weight scales, global weight scale .125, hidden output scale 1)
has relative L2 7.44–7.55%, cosine 0.99738–0.99747 against BF16 over three
inputs. Native quantized-reference parity is exact, so this is a precision
policy error, not evidence of full-model correctness. Do not select this policy
for optimized inference yet. Model-activation calibration, weight-scale/AWQ
search and mixed-precision alternatives remain required. Synthetic input
fidelity is also insufficient to establish real-request fidelity.

## BF16 / FP8 dense native integration

`gemm.cuh` creates resident invocations of the existing shared `Matmul<E>`
operator (TMA, double-buffered shared pages, TCGen05, FP32 accumulation), now
with BF16 output and optional residual/bias/ReLU² epilogues. `gemm.py` accepts
prepared E4M3 operands with static descales. The six Cosmos projection shapes
are tested at 60 and 6300 rows for both formats (`results/projection.json`).
All pass relative L2 ≤0.001 against a dequantized FP32 projection reference,
and profiler checks show one original VM launch. This validates the operator,
not FP8 calibration on real model activations.

`dense_dag.cuh` adds BF16/FP8 operation specializations to the original VM,
using the existing `native_pi0::DagScheduler` rather than another runtime.
`NativeDenseFFN` links up output to BF16 down input by direct storage binding.
`results/dense_ffn_checkpoint.json` tests real layer-27 (60 rows) and layer-0
(6300 rows) weights, three changed synthetic inputs, poisoned intermediates,
permuted nodes, exact split-launch parity and queue/epoch diagnostics.
BF16 error is approximately 0.29% relative L2 and FP8-up/BF16-down error is
2.1–2.4%, with cosine ≥0.9997 against BF16. These pass the initial gates only
for the tested synthetic inputs; no whole-model precision policy is selected.

Small dense DAG memcheck is clean (`results/dense_ffn_memcheck.txt`). The
existing `scripts/validate_tcgen.py` BF16/FP8 regression suite also passes.
Nsight Compute was attempted with a demangled-kernel filter, but access to
hardware counters failed with `ERR_NVGPUCTRPERM`; see `results/ncu_status.json`.
Per-stage consumer timings remain available from native VM diagnostics;
hardware-counter profiling and full critical-path tuning remain outstanding.

### Dense metadata optimization

`EdgeDenseView` caches immutable scalar fields and remote TMA descriptor
addresses in the compute roles; the descriptors themselves stay in device stage
storage. The shared dense Matmul now accepts this view without changing its
math or the original standalone path. `results/dense_ffn_cached_metadata.json`
preserves the same checkpoint and changed-input gates, exact split parity and
one-launch tracing. Median M60 latency changed from 0.404→0.310 ms (BF16) and
0.366→0.277 ms (FP8-up/BF16-down). M6300 changed from 2.984→2.903 ms and
2.974→2.886 ms, respectively. These are sequential primitive development runs,
not full-model speedups. Small-case memcheck remains clean after this change.
The remaining large-M overhead needs scheduler/instruction-path investigation;
consumer spans and per-node intervals are recorded, with no unsupported
hardware-utilization inference from the unavailable counters.

## Q/K normalization, interleaved mRoPE, and cache publication

`csrc/kernels/cosmos3_edge_misc.cu::qk_norm_rope_bf16_warp128_kernel` is now
ported as `rope.cuh::EdgeRopeOp::vector`, called by a native VM consumer. It
uses one warp per head, vectorized BF16 loads, FP32 RMS reduction and weighted
normalization, rotate-half RoPE, and BF16 output. Optional V copying publishes
into the caller's cache rows without concatenation. Strided projection inputs
and K-only normalization are supported. K-only mode preserves the raw causal
understanding keys while producing the separately normalized generator keys.
This CUDA algorithm transfers directly to B200; native projection-epilogue
fusion and full-model instruction-stream integration remain pending.

`rope.py::mrope_tables` follows the pinned official
`MultiModalRotaryEmbedding.apply_interleaved_mrope`: spatial axes replace
frequency slots 1/2 modulo 3, and the remaining frequencies use the temporal
axis. Axes `[24,20,20]` do **not** mean contiguous frequency sections. Tables
match the actual pinned official class exactly on CPU at BF16. Tests execute
those source math classes through AST extraction, replacing only the default
configuration's frequency-update decorator with identity to avoid importing
the training framework.

`results/rope.json` covers 3,60,6300 rows and repeated changed inputs. Fused
mode matches FlashRT's original optimized strided CUDA kernel bit-for-bit.
Relative L2 against official BF16-intermediate arithmetic is ~0.00302. The
reference-rounding mode is within 1.92e-5 in the recorded cases. K/V cache prefix
and tail remain intact; V copies are exact. At 6300 rows, initial p50 is
0.317 ms fused / 0.331 ms reference including V copy and direct K/V placement.
Each operator call launches one original VM kernel. These initial measurements are standalone. The integrated schedule below
now composes this operation with projection and attention scheduling.

A 768-vector instruction batch was measured (`results/rope_batch768.json`) and
rejected as the default: 6300-row p50 rose to 0.458 / 0.497 ms. The retained
batch is 192. These are native primitive comparisons, not end-to-end speedups.

## Native head-128 joint GQA

The shared `native_pi0/attention.cu::native_attention::attend` now supports
head dimension 128; `attention.py::NativeJointAttention` specializes its boundary
for Q16/KV8 and explicitly noncausal joint attention. It reads the prepared
understanding+generation K/V cache directly, using tiled warp tensor-core QK/PV,
online FP32 softmax, bounded shared pages and optional device-side split/merge.
It does not materialize a quadratic score matrix. This is an initial native
attention baseline: FA4-style TCGen05/TMA attention optimization, FP8-KV
calibration, efficient cross-head KV reuse and full generator integration remain.

`results/joint_attention.json` covers all 6300 generated queries as well as
60 selected queries over 6300 generated keys, plus a synthetic 128-row static
understanding prefix. Prefix bytes remain exact over three generation updates.
Relative L2 versus PyTorch SDPA on the prepared inputs is <0.00357, cosine
>0.999993, with one native attention launch including split merging. This does
not prove fidelity against real model activations or exact AV boundary fixtures.

For 60 queries, split8 measured 0.206 ms versus split1 1.006 ms. For all 6300
queries, split1 measured 11.012 ms versus split8 17.220 ms. Splitting must
therefore be chosen by workload. This baseline still requires substantial
performance work, and is not a SOTA claim. QK preparation runs as a separate
native launch in these tests, so they do not establish a single-launch layer.
Small-case memcheck is clean (`results/joint_attention_memcheck.txt`).


## Integrated projection/preparation/joint-attention boundary

`attention_schedule.cuh` combines the existing dense VM roles, `EdgeRopeOp`,
and the shared native attention consumer in one `megakernel::mk` invocation.
The Python wrapper creates tile dependencies offline. Preparation waits for
its exact Q/K/V producer tiles; attention waits for a single shared KV join
and its Q tiles. Static understanding K/V is supplied at launch and remains
unchanged. No external host operator is called during forward.

`results/attention_schedule.json` covers synthetic 17-,60-,6300-row cases,
BF16 and FP8 projections, selected 60-row and all-row attention, and split-KV.
It verifies bit-exact intermediate/final output against separately launched
native components, queue and dependency diagnostics, changed inputs across
three epochs, poisoned buffers, permuted nodes and nondefault streams.
Interleaved standalone attention calls exercise advancing epochs beyond prior
counter generations. Independent dequantized-projection + official norm/RoPE
+ SDPA reference error is below 0.0052 relative L2 with cosine >0.999986.
This does not measure FP8 error against unquantized weights or real requests.

One-launch profiling passes. Initial full-query p50 is 12.601 ms BF16 and
12.568 ms FP8; 60 selected queries over full generated K/V measure 2.041 and
2.016 ms, respectively. These include all-row Q/K/V projections: last-layer
Q projection pruning is still pending. Hidden RMSNorm, O projection, residual
updates and FFN are outside this boundary. No full-layer/model acceptance or
SOTA claim follows from this result.

Integrated schedule small-case memcheck reports zero errors
(`results/attention_schedule_memcheck.txt`). Standalone preparation memcheck
also passes (`results/rope_memcheck.txt`); its validation harness explicitly
initializes CUDA before loading CUDA DSOs to avoid a sanitizer startup hang.
