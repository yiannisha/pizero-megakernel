# Native Cosmos 3 Edge

Implementation of `../edge_cosmos_plan.md`, using the persistent VM in
`../third_party/Megakernels`. Work is in progress: there is no runnable native
Cosmos model yet, and no Cosmos latency or fidelity claim.

Source revisions are pinned in `../upstreams.json`. `sources/manifest.json`
pins the exact checkpoint revision and SHA256 of the downloaded configuration
and transformer shard index. These small metadata files are stored in this module; model
weights are not. `spec.py` derives projection dimensions from verified checkpoint
metadata without importing FlashRT or a framework inference engine.

```sh
python -c 'from cosmos3_edge.spec import EdgeSpec; print(EdgeSpec.from_checkpoint())'
```

The official `NVIDIA/cosmos` repository is a landing/documentation repository.
The actual official model code is in `NVIDIA/cosmos-framework`, separately pinned.
The checkpoint is `nvidia/Cosmos3-Edge` at
`344d602b128d1bbdacb43b08d0a3626f46343e29`.

Current milestone: M0 source audit. Configuration and source pinning are complete;
all tensor headers and checkpoint content hashes have been verified. Exhaustive
operator audit, model boundary fixtures, and M1–M13 completion remain pending.

## Implemented primitive

The shared native NVFP4 VM operator in `../native_pi0/nvfp4_probe.cu` now
supports a ReLU² epilogue producing packed FP4 and native swizzled per-16
UE4M3 scales directly from FP32 accumulators. The setup wrapper exposes
`ResidentNVFP4(..., output_dtype=torch.uint8, output_relu2=True)`.
This reuses the existing VM compute path; it does not launch CUTLASS GEMMs.
The primitive still lives in the existing shared development operator library;
`ffn.py` now schedules up and down together; full model scheduling remains pending.

```sh
python -m unittest cosmos3_edge.tests.test_weights -v
python cosmos3_edge/tests/validate_relu2.py --profile --benchmark-output cosmos3_edge/results/relu2_fp4.json
```

The numerical oracle uses CUTLASS FP32 ReLU followed by multiplication, then
FlashRT `quant_block16`; validation checks packed bytes and swizzled scale bytes
exactly. This does not yet establish equivalence to the complete CUTLASS fused
block-scale epilogue or pretrained model fidelity. Synthetic FFN up-projection
shapes include M=60 and M=6300, N=9216, K=2048. JSON results record the source
hashes, environment, launch count, CUDA-event timings and synchronized wall times.
Event timings include submission gaps; these are primitive measurements, not
end-to-end Cosmos performance or a speedup against FlashRT.

`weights.py` validates all 549 generator-checkpoint tensor shapes, dtypes and
byte ranges against the pinned index, and offers a lazy setup-only safetensor
loader. Remote header evidence is in `results/checkpoint_headers.json`; the checkpoint
shards have also been downloaded locally and their complete upstream LFS SHA256
hashes verified (`results/checkpoint_integrity.json`).

## Integrated native FFN

`ffn.py::NativeFFN` schedules up projection, ReLU²/FP4 quantization and down
projection with optional residual in one original VM launch. Producer tiles
publish FP4 values/scales; each down tile waits for its row's up tiles through
the shared DAG. Packed buffers are used directly without an intervening BF16
activation tensor. This is an FFN building block with prepacked input, not a
complete layer or denoising request.

```sh
make -C cosmos3_edge
python cosmos3_edge/tests/validate_ffn.py --output cosmos3_edge/results/ffn.json
compute-sanitizer --tool memcheck --error-exitcode 99 python cosmos3_edge/tests/validate_ffn.py --quick --sanitizer
python -m pip install -r cosmos3_edge/requirements.txt
python -m cosmos3_edge.download_checkpoint .cache/cosmos3-edge
python cosmos3_edge/tests/validate_ffn.py --quick --checkpoint .cache/cosmos3-edge --output cosmos3_edge/results/ffn_checkpoint.json
```

The synthetic FFN tests cover repeated epochs with new inputs, poisoned
intermediates, permuted DAG nodes, nondefault streams, TMA guard storage,
producer byte parity, independent down-projection math and split-launch parity.
CUPTI confirms one kernel; memcheck reports zero errors for the small cases.
The initial 6300×2048→9216→2048 FFN measured 2.82 ms median on B200. No
FlashRT comparison or performance-win claim has been established. The
checkpoint variant uses real layer-27 weights and synthetic activations;
its BF16 comparisons are diagnostic and do not substitute for calibration
on real model inputs or downstream application gates.

`CheckpointWeights` checks complete shard SHA256 by default, in addition to
header validation. `NativeFFN.from_checkpoint` packs the selected `gen` or
`und` FFN at setup time; supplied scales are research parameters until calibrated.

The initial checkpoint-weight FFN test **fails** the plan's BF16 compatibility
gates: relative L2 ≈7.5%, cosine ≈0.9974 with uncalibrated packing and synthetic
activations. It matches the quantized math reference exactly but is not an
accepted model precision configuration. Calibration and higher-precision
alternatives are still required.

## Dense precision reference

`gemm.py::NativeProjection` exposes resident BF16 and prepared-FP8 projections
with TMA and native tensor cores, FP32 accumulation and BF16 output.
`NativeDenseFFN` uses `schedule.py::DenseSchedule` to execute up/ReLU² and down
in one original VM launch. Dense and FP4 schedules reuse the same VM and DAG
scheduler; integrating all operators into full layers is still pending.

```sh
python cosmos3_edge/tests/validate_projection.py --output cosmos3_edge/results/projection.json
python cosmos3_edge/tests/validate_dense_ffn.py --checkpoint .cache/cosmos3-edge --output cosmos3_edge/results/dense_ffn_checkpoint.json
compute-sanitizer --tool memcheck --error-exitcode 99 python cosmos3_edge/tests/validate_dense_ffn.py --quick --sanitizer
```

The six Q/K/V/O/up/down projection shapes pass BF16 and FP8 operator parity
at 60 and 6300 rows. Real checkpoint FFNs with synthetic activations give
about 0.29% relative L2 error for BF16 and 2.1–2.4% for FP8-up/BF16-down
against framework BF16. These pass the initial compatibility gates on these
inputs, but do not establish a calibrated model policy or whole-model fidelity.
The fused dense ReLU² rounds after the FP32 square, unlike framework BF16
intermediate rounding; that difference is included in the reported error.

Dense FFN tests also compare split and single-launch outputs exactly, vary
inputs across epochs, poison buffers and permute DAG nodes. Small-case memcheck
is clean. Nsight Compute hardware-counter collection was attempted but denied
with `ERR_NVGPUCTRPERM`; `results/ncu_status.json` records the attempt. No
hardware utilization/throughput claim is made from that failed collection.

The retained dense metadata optimization stores immutable scalar fields and
TMA descriptor addresses in local views. `results/dense_ffn_cached_metadata.json`
records the repeated correctness gates and updated median timings: 0.310 ms
BF16 / 0.277 ms FP8-up for 60 rows, and 2.903 / 2.886 ms for 6300 rows.
These FFN primitive times do not represent complete Cosmos request latency.

## Native attention preparation and joint GQA

`rope.py::mrope_tables` prepares the official interleaved 3D RoPE tables at
setup time. `NativeQKNormRope` performs Q/K RMSNorm and RoPE in one original
VM launch, with optional V copying directly into a cache slice. K-only mode
creates the understanding keys needed by the generator without mutating the
understanding stream's causal keys. Fused mode matches the pinned FlashRT
head-128 kernel exactly; reference rounding closely matches official BF16 math.

`attention.py::NativeJointAttention` uses the shared native tiled GQA operator
at head dimension 128. It reads a prepared static understanding prefix and the
current generation K/V without concatenating them. Native split-KV merging
stays inside the attention launch.

```sh
python cosmos3_edge/tests/validate_rope.py --output cosmos3_edge/results/rope.json
python cosmos3_edge/tests/validate_joint_attention.py --output cosmos3_edge/results/joint_attention.json
compute-sanitizer --tool memcheck --error-exitcode 99 python cosmos3_edge/tests/validate_rope.py --quick --sanitizer
compute-sanitizer --tool memcheck --error-exitcode 99 python cosmos3_edge/tests/validate_joint_attention.py --quick --sanitizer
```

Synthetic 6300-query joint attention passes SDPA parity gates with a static
128-row prefix. Split1 measures ~11.0 ms, while 60 action queries over the same
keys measure ~0.206 ms with split8. The full-query path is still an initial
performance baseline. The standalone tests use separate QK and attention launches. The integrated
schedule below now combines them with QKV projections; real boundary fixtures
and complete 28-layer / 30-step generator execution are still pending.

## Integrated generation attention schedule

`attention_schedule.py::GenerationAttentionSchedule` executes Q/K/V projections,
Q/K RMSNorm+RoPE, V publication and joint GQA in one original VM launch. It
accepts prepared understanding K/V at launch. Each preparation tile waits on
its projection producers; one KV-readiness join and query-tile dependencies
control attention dispatch. No host concatenation or intermediate dispatch
occurs inside this boundary. Separate-component execution remains available
for validation, with epochs coordinated to avoid stale split-KV counters.

```sh
python cosmos3_edge/tests/validate_attention_schedule.py --output cosmos3_edge/results/attention_schedule.json
compute-sanitizer --tool memcheck --error-exitcode 99 python cosmos3_edge/tests/validate_attention_schedule.py --quick --sanitizer
```

Full-geometry BF16 and FP8 projection variants pass independent reference gates
and bit-exact comparison to separately launched components, with permuted DAG
nodes, changed inputs, poisoned buffers, nondefault streams, and interleaved
standalone/scheduled calls. The synthetic full-query case measures ~12.6 ms;
60 queries over all 6300 generated keys measure ~2.0 ms including all Q/K/V
projections. Q projection is still computed for all rows in that test, so this
is not yet the complete final-layer action-only optimization. The block excludes
hidden-state RMSNorm, O projection, residual updates and FFN; it is not a
complete layer, generator timestep, or denoising request.
