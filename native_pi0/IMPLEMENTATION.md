# Native π0 implementation decision

The objective remains the complete PLAN.md. The user clarified the execution
architecture on 2026-10-28: adapt the Hazy VM as the Gemma implementation does,
with native tile-range instructions, static per-worker streams, and direct
producer counters. A generic task/DAG executor is not the model runtime.

## Keep and reuse

| Component | Use |
|---|---|
| Megakernels VM controller/loader/launcher/consumer/storer | Original runtime and static instruction fetch |
| Gemma `model.py`, `native.cu`, `fused.cuh`, `attention.cuh` | Scheduling, range-loop, fusion and dependency reference |
| BF16/FP8 TMA/TCGen05 and NVFP4 math | Reusable implementations; integrate into native range operators |
| RMSNorm/LayerNorm, attention, packing and epilogue references | Numerical algorithms and independent checks |
| OpenPI backend and BF16 model benchmarks | Same-weight model oracle and end-to-end comparison |
| Weight packing and calibration utilities | Offline setup; no host model operations in forward |
| Complete model wiring from the failed reference | Archived source to reuse for architecture, shapes and weights |

## Retire from the active path

| Component | Reason |
|---|---|
| Generic ready queue and dependency-group executors | Per-tile dispatch and serial fan-out replace native scheduling |
| `RefWork` stage-counter dispatcher | Another custom fetch loop; not the requested instruction program |
| Generic projection/attention, block and stack wrappers | Tied to the retired executor |
| Worker sweeps and synthetic-chain timing reports | Historical component evidence, not model acceptance |
| Failed full-model reference backend | Output parity failed; preserve as debugging evidence |

`archive/generic-executors-2026-10-28/MANIFEST.json` records exact retired paths
and source hashes. The complete snapshot includes helpers needed to reproduce
the old work. `restore.py NEW_DIRECTORY` restores independently.

`dag_probe.*`, `dag_scheduler.cuh` and `dag_matmul.*` remain at their original
paths because the separate Cosmos project imports them. New π0 native code must
not import or include them. Cosmos sources and dependency pins are untouched.
The optional dynamic-fetch VM patch remains installed for that compatibility;
the new π0 globals must use the original static fetch branch.

## Native instruction contract

The host emits fixed 32-word instructions into static worker streams. An
instruction owns an operation-specific tile range (`first`, `count`), rather
than reserving a new global queue ticket for each tile. The operation's loader,
launcher, consumer and storer loop through that range using the native page and
semaphore pipeline. Per-tile phases continue across the instruction; TMEM is
released once when the instruction finishes.

Producer counters describe tensor row bands, Q/K/V heads or FFN input chunks.
Consumers acquire the counters for data they actually read. Producers publish
completed chunks with release semantics after all writers and proxy fences.
There is no generic indegree/ready-list structure or global stage dispatcher.

The instruction-memory view indexes logical worker/CTA IDs. This avoids
depending on dense physical SM IDs while leaving the VM's static fetch code
unchanged. Worker count is bounded by measured native occupancy/SM capacity.
Static program and dependency validation must reject cycles and missing ranges.

## Execution order and gates

1. Implement and verify native range pipelines and direct data counters.
2. Reuse the model wiring to integrate the complete BF16 vision, prefix,
   expert, heads and ten Euler steps. Do not resume precision/worker sweeps
   before this full reference exists and passes numerical checks.
3. Compare the complete forward against the same OpenPI weights and inputs;
   verify one original VM launch, changed-input replay and memory safety.
4. Measure isolated full-model latency against eager, compiled and graph Torch,
   realtime-vla and a comparable FlashRT implementation. Then optimize and
   introduce calibrated precision using the PLAN.md requirements.

The previous 1.08 s run was contaminated by another CUDA context and failed
output parity (~3.4% action NRMSE). The last stage-counter trial returned zero
outputs (NRMSE 1.0), so its ~0.076 ms timing is invalid. Neither is acceptance.
Gemma's vLLM win is a one-token decode result; its scheduling and fusion design
is relevant, while π0's batched prefix/expert shapes still require adaptation.

No M0–M9 milestone or SOTA claim is complete.

## Full static-stream backend

`model.py` and `model.cu` now wire the full BF16 model through the static VM:
27 vision layers, 18 prefix layers, 18 expert layers for each of ten steps,
FP32 input/output/time heads, and Euler updates. `NativePi0` owns the fixed
instruction streams and data counters; forward passes only root input pointers
and launches `megakernel::mk<..., NativeModelGlobals, ...>` once.

The initial full run is recorded in `archive/static-native-bringup-2026-10-28/native_pi0_static_full.json`:
one original VM kernel, nonzero complete actions, action NRMSE 0.03443 against
same-weight OpenPI. This fails the existing numerical gate. It is evidence of
complete execution, not accuracy or performance acceptance. Vision attention
is the first material numerical discrepancy and is being corrected in the
complete model path. No standalone operator campaign is required to continue.

The initial program contains 2,544 operations and 109,510 native range
instructions covering 189,552 compute tiles. These are precompiled streams,
not dynamically dispatched tasks. Further fusion is still required; merely
removing a queue does not establish good end-to-end performance.

The later tensor-core Gemma attention path uses two QK passes to preserve
eager BF16 score/scale and normalized-probability rounding. The latest vision
trial uses descending key blocks. `results/native_pi0_static_flash.json`
records the current full-model attempt; numerical acceptance remains open.

## Integration target clarified

The completed backend must incorporate the transferable FlashRT and
realtime-vla optimizations **inside this single native VM forward**, not
merely benchmark or document them. Setup-time weight folding and calibration
feed native operators; the inference path keeps one original VM launch.
The π0 implementation needs fused GEMM epilogues, action/time projection
folding, QKV/RoPE/cache writes, RMS-factor staging, GeGLU, attention, and
resident cache dependencies. Calibrated FP8 and selected SM100 NVFP4 sites
follow full-model numerical gates. π0.5's AdaRMSNorm and style conditioning
require a separate native model variant; they are not π0 operations.

Fused BF16 residual GEMM epilogues are the first ported change. They must
preserve the standalone BF16 projection rounding before the residual add.
Any precision or operator fusion is accepted only after same-weight full-model
parity and isolated latency checks; the current reference still fails parity.

### First integrated fusion: residual GEMM epilogues

The full π0 path now combines attention output and FFN down projections with
BF16 residual addition inside `NativeLinearOp`'s epilogue. The host binds the
resident residual buffer; the native loader acquires its producer row-band
counter before reading it. For biased vision projections, the epilogue adds
bias, rounds the projection to BF16, then adds the residual, preserving the
separate operator's rounding boundary. This removes 450 pointwise operations
and 9,252 range instructions from the three-view/50-action/10-step program.
`results/native_pi0_fused_residual.json` measured 218.9 ms versus 238.5 ms
for the prior static reference under separate isolated runs. The same-weight
action NRMSE remains 0.03466, above the acceptance threshold; the timing is
therefore engineering evidence, not an accepted model result.

The one-view, eight-action, one-step fused program completed both
Compute Sanitizer memcheck and synccheck with zero reported errors. These
runs still exit nonzero on the independent action accuracy gate (2.03% NRMSE);
CUPTI tracing is disabled under the sanitizers because their CUDA subscribers
conflict. The separate full-shape run confirms one VM launch.

### Offline action/time projection fold

The native model setup now composes the π0 action input projection with the
action half of `action_time_mlp_in`: `W_a @ W_in` is packed once as a 32→1024
weight. The time half, both biases and the timestep embedding become a
step-specific resident FP32 bias. Forward executes one smaller native linear
operator followed by the existing SiLU and `action_time_mlp_out`, removing
three operations per Euler step from the instruction program. No Torch model
operator runs in forward. This transfers realtime-vla's algebraic fold while
retaining the actual OpenPI π0 FP32 head precision.

A direct FP32 comparison of the folded and separate expressions measured
3.57e-7 NRMSE. The complete same-weight run in
`results/native_pi0_action_time_fold.json` has 80,852 range instructions and
215.5 ms isolated native median versus 245.2 ms eager OpenPI. Action NRMSE is
0.03456; the full-model accuracy gate remains open. The reference replay and
single-launch checks passed in that run.

### Final prefix layer K/V pruning

The last prefix layer supplies K/V to the expert but its attention output and
FFN are unused. The native setup now packs only its K/V projection weights,
and a native VM cache instruction copies the rotated K/V into the resident
cache after the projection and RoPE producers complete. Expert instructions
still wait on the cache producer's row-band counters. This removes the final
prefix Q projection, attention, output projection, FFN, and unused final norm
from the single-launch program.

`results/native_pi0_prefix_kv_only.json` records 79,896 range instructions
(down from 80,852), 130,426 compute tiles, one native CUDA launch, and
bit-exact same-input replay. The complete action NRMSE remains 0.03455784;
the accuracy gate is still open. Isolated native median was 216.2 ms versus
215.5 ms in the earlier run, so these measurements do not establish a latency
gain. The one-view, eight-action, one-step version passed Compute Sanitizer
memcheck and synccheck with zero reported errors. Both validator processes
still exit on the separate full-action numerical gate.
The reports were regenerated from an isolated checkout of commit `f9fd3f1`
with no uncommitted native source changes; the full action error and timing
were unchanged by that clean rebuild.

### Model-derived rotary tables and numerical gate

The original setup regenerated Gemma rotary inverse frequencies in FP32.
The pinned OpenPI BF16 model stores those frequencies in BF16 and converts
them back to FP32 to form the angles. At 768 prefix rows, the regenerated
cosine and sine tables differed from OpenPI's by 0.1137 and 0.1460 NRMSE.
An input-matched attention diagnostic showed that the native Gemma attention
operator itself differed from a matching eager calculation by only about
5.7e-5 NRMSE; the rotary tables caused the material discrepancy.

Setup now asks the actual prefix and expert rotary modules for their tables,
then passes the resulting BF16 constants to the native VM RoPE instructions.
There are no Torch model operations in `NativePi0.forward`. The complete
three-view, 50-action, ten-step same-weight random-model report in
`results/native_pi0_model_rope_benchmark.json` passes the stated numerical
limits: action NRMSE 0.00889, normalized maximum error 0.02995, cosine
0.99996, changed-noise NRMSE 0.00871, bit-exact same-input replay, and one
native launch. Its isolated median was 216.15 ms versus 249.94 ms for OpenPI
eager on B200. These are random-initialized weights, not a robot checkpoint.

The one-view, eight-action, one-step checks in
`results/native_pi0_model_rope_memcheck.json` and
`results/native_pi0_model_rope_synccheck.json` also pass the numerical gate;
Compute Sanitizer reported zero memory and synchronization errors. The
sanitizer scope is smaller than the complete benchmark configuration.

### Native RoPE and cache producer fusion

The Gemma RoPE range instruction now writes each rotated K and copied V to
the resident layer cache as it writes the QKV output. Its completion counter
publishes both outputs. Prefix and expert attention read that cache without
rewriting it, and the final prefix layer no longer needs a separate cache-copy
instruction. The rotation is still performed once per token; moving it into
the attention loader was measured separately and reverted because it repeated
K rotation across query tiles.

The clean-source isolated run in
`results/native_pi0_rope_cache_producer_clean.json` preserves the complete
action NRMSE 0.00889, one CUDA launch, and changed-input replay. It has
79,884 range instructions over 130,414 tiles and measures 208.71 ms native
versus 242.77 ms OpenPI eager. The prior native median was 216.15 ms in a
separate isolated run, so the change produced about a 7.4 ms improvement in
these measurements. Reduced-case clean-build memcheck and synccheck reports
are recorded alongside the benchmark.
`results/native_pi0_rope_cache_producer_prompt16.json` also passes the full
three-view, 50-action, ten-step gate with a 16-token prompt (action NRMSE
0.01230), covering the changed cache offset.
