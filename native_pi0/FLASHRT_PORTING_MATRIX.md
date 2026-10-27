# FlashRT native porting matrix

Reference: FlashRT `60054d253dcada1dab5e7c0ff68111f935ee41ca`;
Megakernels `08a7136cdfaa7e40629b65234d46cffcd7d48b00`;
ThunderKittens `664c108d16f12707a73d3072ab525f26fb2b4f62` (Megakernels gitlink).

This is an **in-progress audit**, not the completed M0 deliverable. Native
BF16/FP8/NVFP4 development compute paths exist; model operators, model parity,
and performance results remain incomplete. Component-level progress is recorded
below; no row represents full production integration. The machine has a B200
and CUDA 12.8 compiler.

`python scripts/audit_flashrt.py` regenerates `flashrt_inventory.json`: hashes
and line counts for the required source families, plus source-located calls
from all four primary pipelines. This intentionally includes attention's
vendored implementations; file enumeration does not establish semantic review.
The inventory follows repository-local Python imports and quoted C++ includes,
records dependency edges and unresolved includes, and recursively includes Thor
subdirectories. External template include resolution, dynamic imports,
indirect frontend dependencies, calibration, weight transforms, build gates,
and individual variants still require review before M0 can be accepted.

## Reviewed implementation details

Paths below are relative to FlashRT. Shapes are π0 target shapes, not claims
that each source supports every target shape without modification. M denotes
rows; N output columns; K reduction columns. Each benchmark must run the
candidate inside `megakernel::mk`, including its dependency and resource costs.

| Source and exact symbol | Operation, stage, target shapes / precision | Hardware and fusion | Quantization / scaling | Classification, π0 applicability and B200 changes | Native implementation and numerical / benchmark gate | Status / outcome |
|---|---|---|---|---|---|---|
| `csrc/gemm/fp4/cutlass_fp4_gemm_geglu_il_sm100.cu`: `cutlass_fp4_gemm_geglu_il`, `cutlass_fp4_gemm_geglu_il_hw`, `cutlass_fp4_gemm_geglu_il_hw_v10` | Prefix gate/up K=2048, N=32768; expert K=1024, N=8192; packed FP4 operands, FP32 accumulators, FP4 result | SM100 block-scaled tensor ops; 128×256×256 and skinny 128×64×256 tiles; adjacent gate/up accumulator columns; GELU×up, compact half-width stores | E2M1, UE4M3 per 16; A row-major, B column-major, alignment 32 elements; SFD for compact output must agree with consumer layout | Transfer with native VM redesign; directly applicable GeGLU semantics. Host `GemmUniversalAdapter` cannot execute inside VM | Port MMA mainloop and compact epilogue; allocate TMEM and shared stages through VM. Compare dequantized output to BF16 GeGLU and FlashRT, then full FFN and end-to-end; M=51,256,768 | Unported; no measurements |
| `csrc/gemm/fp4/cutlass_fp4_gemm_siglip_ffn_sm100.cu`: `cutlass_fp4_gemm_bias_gelu_fp4out`, `cutlass_fp4_gemm_bias_res_fp16` | SigLIP up 1152→4304 and down 4304→1152, M=256 per view; FP4 inputs | SM100 CUTLASS device math; up bias+GELU+FP4, down bias+residual to FP16 | Block scales and output norm constant (source sets 1); preserve padded hidden columns and subsequent compaction | Transfer with native VM redesign. FP16 source output requires explicit comparison to chosen BF16 policy | Native GEMM epilogues, deterministic padded SF regions; compare complete FFN, including non-multiple tile widths, before timing | Unported; no measurements |
| `csrc/fused_fp4/siglip_ln_vec.cu`: `layer_norm_fp8_vec_kernel`, `layer_norm_mul_fp4_sfa_vec_kernel`, wrappers `layer_norm_fp8_vec_fp16`, `layer_norm_mul_fp4_sfa_vec_fp16` | SigLIP D=1152; FP16 input, affine LN to FP8 or FP4 | 128 source threads per row; one 16-element quant block per thread; two int4 input loads; register cached values; FP32 two-pass variance, warp/block reductions | FP4 path rounds normalized affine result to FP16 before quantization; optional AWQ inverse scale; amax/6 rounded to E4M3; source software E2M1 threshold ties use `<=` | Transfer with native VM redesign; adapt 128-thread reduction to VM consumer warp groups. Source scale clamp 1e-12 can round to zero: handle zero-scale blocks explicitly and test | Native norm producer with final scale layout writes; preserve or deliberately validate rounding changes. Compare constants, zero, tiny, outlier and representative activations; benchmark fused LN→projection | Unported; no measurements |
| `flash_rt/models/pi0/pipeline_rtx.py`: `Pi0Pipeline` | Complete π0 orchestration; vision 27 layers, prefix/expert 18 each; suffix horizon+1 | Source uses external GEMMs and graph capture, which cannot be retained | FP16 activation policy with static FP8 scales; norm weights folded into projections; action output weights pre-scaled by -1/steps | Math transfers; host orchestration is not transferable. Unlike π0.5, standard RMSNorm, state projection and action/time MLP required | Compile static native instruction schedule; state row sees prefix+self only, actions see all suffix. Reference identical OpenPI checkpoint, tokens and noise at each step | Reviewed dimensions and mask; no native model |
| `include/config.cuh` in Megakernels: `default_config` | VM resource foundation | 16 consumer warps plus controller/load/store/MMA warps; 16KiB pages, 13-page assertion; two instruction stages | No existing π0 precision policy | Requires explicit resource redesign for TCGen05, not an independent runtime | Prove TMEM lifetime, async completion, page ownership and forward progress before integrating tensor-core instructions | Resource assumptions inspected; no changes yet |

## Mandatory remaining audit families

These are open work, not omissions justified by evidence:

- All FP4 GEMM variants, FP4-output visitor math and scale-factor layouts.
- `gemm/mega`, FP8 small-M variants, split-K reductions and register prefetch.
- π0.5 skinny FP4/FP8 decoders, consumers and attention, adapting AdaRMSNorm
  to π0 RMSNorm; replace SM120 instructions and PDL with legal SM100 VM work.
- SigLIP patch extraction / bias-position embedding, QKV attention padding,
  camera batching, AWQ preprocessing and every vectorized norm variant.
- FA-style attention, split-KV, GQA, sequence-used masking and fused layouts.
- Prefix/expert RoPE and direct KV stores; residual/norm/quant fusion variants.
- Frontend weight folding, calibration distribution aggregation, static scales,
  precision search, W4A16 candidates and representative policy gates.
- Thor FP4 pipeline's alternative rowops, cache-touch and compact-store tactics;
  distinguish host stream/cache launch machinery from transferable device work.

## Acceptance evidence still required

M0 requires a reviewed entry for every relevant optimization, including exact
symbols, shapes, scale conventions, native design and eventual measured result.
M1–M9 require native implementation, pretrained parity, calibrated policy tests,
launch tracing, per-stage/resource profiling and comparable B200 benchmarks as
specified in PLAN.md. Inventory generation or an upstream build cannot satisfy
these gates. No SOTA claim is made.

## Skinny and cache source review

| Source and exact symbol | Operation / shapes / precision | Hardware, fusion and scales | Applicability and B200 native design | Numerical reference / experiment | Port status and outcome |
|---|---|---|---|---|---|
| `csrc/gemm/fp4/nvfp4_skinny_gemm_sm120.cu`: `gemm_kernel`, `gemm`, `gemm_bf16_act`, `pack_weight`, `quantize_weight` | Expert projection families; M grouped in 16×MT rows; BN weight rows, KC reduction slice; E2M1 operands, FP32 partials | W and SFB register prefetched before PDL wait; padded shared A and SFA; cp.async; source `nvfp4_skinny_common.cuh` uses SM120 `mma.sync...kind::mxf4nvf4...m16n8k64` | Transfer with VM redesign; SM120 block-scale MMA cannot be copied into SM100. Evaluate TCGen05 tile family versus native FP8 or W4A16 skinny math; preserve tail masking at M=51. VM counters replace PDL launch chain | Compare partials and final consumers against dequantized matmul, including K splits and M=1,16,48,51; measure complete expert latency and scratch traffic | Unported; no measurements |
| `csrc/kernels/pi05/pi05_decoder_skinny_fp4_sm120.cu`: `residual_ada_norm_kernel`, `gate_gelu_kernel`, `action_in_norm_kernel` | Decoder width 1024; merged gate/up, half=4096 for target; FP32 split partials reduced to BF16 and FP4 | Partial scale alpha=a_gs×w_gs; residual stored BF16; source RMS square sum uses pre-round residual float values; style scales/shifts and gated residual are π0.5-specific | π0.5 semantic adaptation required. Remove style and residual gate; standard π0 RMSNorm and action/time MLP required. Explicitly validate RMS rounding order against OpenPI rather than copying source | Source fused-consumer tests establish π0.5 semantics only; create π0 layer and denoising-step comparison with identical weights/noise; final projection must update actions inside VM | Unported; no measurements |
| `csrc/gemm/fp8_smallM_handtuned_splitk_sm120.cu`: `fp8_gemm_splitk_partial_kernel`, `mma_m16n8k32_e4m3` | Small-M O/down projections; FP8 E4M3 A/B, scratch [split,M,N] FP32 | Double-buffer shared A/B with K padding; SM120 `kind::f8f6f4` MMA; separate source reduction applies alpha and BF16 cast; empty K slices explicitly zeroed | Transfer with VM redesign. Use legal SM100 FP8 tensor-core instructions, fuse reduction into consumer, tune splits to B200 worker count. Scratch writes must be visible before ready counters | Compare split and unsplit output including empty slices; measure reduction traffic and full expert, rather than importing source's synthetic HBM claims | Unported; no measurements |
| `csrc/gemm/fp4/nvfp4_skinny_attention_sm120.cu`: `partials_attention`, `mma_bf16` | Small query attention; source 32-key chunks and split keys, BF16 Q/K/V and FP32 accumulation | Native `mma.sync.m16n8k16` BF16 math is distinct from SM120 FP4 MMA; integrates projection partial input, uses online softmax | Transfer with VM redesign. Adapt GQA head_dim=256 and exact π0 state-row mask; integrate direct RoPE/cache store and split combine without full score materialization | Compare prefix/state/action visibility to OpenPI, including extreme logits, empty splits and variable prefix lengths; benchmark M=51 with 768+ prefix keys | Unported; no measurements |
| `csrc/kernels/pi05/thor/pi05_l2_prefetch.cu`: `l2_touch_kernel`, `l2_prefetch_kernel`, `l2_touch_ex_kernel` | Weight locality, arbitrary byte regions | Real ld.global.cg loads and cp.async.bulk.prefetch.L2; optional evict-last hint, pipelined loads; source separate launches/streams | Device algorithm transferable with VM redesign; host fork/join is not transferable. Schedule prefetch as native work or producer prologue, limit cache pollution and preserve page lifetimes | Profile B200 L2 hit rate, bandwidth and end-to-end latency with and without prefetch; retain only measured benefit | Unported; no measurements |

Upstream `gemma/kernel.so` builds successfully with CUDA 12.8 for sm_100a.
This verifies dependency/compiler compatibility only, not model execution or
π0 correctness. Pretrained Gemma validation requires its separate checkpoint.

## Quantization preprocessing progress

`quantization.py::pack_nvfp4` implements an offline linear-layout packing oracle
for the `nvfp4_skinny_common.cuh::quant_block16` nearest activation-scale policy
and `ue4m3_ceil` conservative weight-scale policy. It supports explicit global
scales, low-nibble-first E2M1, deterministic row/column padding and zero-scale
blocks. `interleave_gate_up` prepares adjacent gate/up rows from [N,K] matrices.
This is preprocessing/reference code, not a runtime backend. Weight squared-error
scale search, native scale consumers, AWQ and checkpoint conversion remain open.

Validation on the available B200: `python scripts/validate_quantization.py`
compiles an independent CUDA conversion oracle and checks E2M1/E4M3 codes for
21,068 values, including positive/negative representable values, all format
midpoints and adjacent floats. All match. `python -m unittest
native_pi0.test_quantization` passes five tests covering exact nibble order,
zero/tiny blocks, padding, global scales, gate/up ordering, invalid inputs, and
quantization error for M=51,256,768 and SigLIP width 4304. These are format and
synthetic packing checks; pretrained action fidelity and latency are unmeasured.

Calibration source review: `core/quant/calibrator.py::_checkpoint_hash` hashes
full sharded weights but only the prefix+size of a single-file checkpoint.
The native cache should hash complete contents and configuration, including
precision policy, observation dataset identity and denoising settings, to avoid
reusing scales after a same-size late-file weight change. `docs/calibration.md`
and `core/thor_frontend_utils.py::quant_fp8` require static activation/weight
scales and FP32 descale multiplication. Source zero-input calibration results
are not evidence of representative native π0 policy quality; real observations
and per-step distributions are still required.

### Scale layout preprocessing

`quantization.py::scale_offsets`, `swizzle_scales`, and `unswizzle_scales`
implement the pinned `csrc/quantize/nvfp4_sf_reshape_sm120.cu` address map:
128 rows × four per-16 blocks per 512-byte atom, with four row bands
interleaved. All padded bytes are initialized to zero. The source header has
inconsistent prose mentioning 8192-byte atoms; the actual kernel and byte-count
helper both use 512. The implementation follows executable source behavior.

`scripts/validate_quantization.py` now builds and executes the original FlashRT
reshape source as an independent development oracle on B200. Full destination
buffers match byte-for-byte for (rows, scale-blocks) = (1,1), (51,64), (127,3),
(129,5), (256,72), (768,128), (4304,72), including padding. Inverse layout checks
also pass. This proves agreement with FlashRT's reshape, not yet compatibility
with the eventual native TCGen05 operand/TMEM descriptors. Direct CUTLASS
Sm1xxBlockScaledConfig verification is now complete as described below; native
MMA integration remains open.

The validation oracle additionally instantiates CUTLASS v4.4.2
`Sm1xxBlockScaledConfig<16>::tile_atom_to_shape_SFA/SFB` directly on the host.
Both operand offset arrays and `cosize` capacities match the offline layout
on all seven shapes. No CUTLASS GEMM is called. This establishes global scale
buffer compatibility; TMEM scale copying and MMA descriptor legality still
require native operator validation.

### Offline activation calibration

`calibration.py::ActivationCalibration` ports the percentile-of-amax aggregation
from FlashRT `core/calibration.py::accumulate_amax`, with named operation,
observation and denoising-step records. It emits static E4M3 activation scales,
per-step maxima and a count of amax records exceeding the selected range.
That count is not an elementwise saturation rate. Required step coverage can
be enforced for each observation; duplicate records and nonfinite values fail.
`fp32_descale` computes scale products in FP32. `checkpoint_digest` reads full
contents of all supplied checkpoint shards; `cache_key` combines that digest,
dataset identity and the supplied configuration. Cache persistence and model
activation collection are not yet integrated.

Three calibration tests pass, covering ten-step aggregation, incomplete step
coverage, duplicates, zero/nonfinite observations, late-file checkpoint changes,
configuration/dataset identity changes and FP32 descale. These checks establish
offline utility behavior, not calibrated pretrained inference or policy quality.

### Original VM resource lifecycle probe

`python scripts/validate_vm.py` compiles `vm_probe.cu` against the pinned
Megakernels/ThunderKittens headers and executes `megakernel::mk` on B200.
All 16 consumer warps record participation and a consistent tensor-memory base
per CTA across repeated instructions. Original NoOp loader/launcher behavior
exercises shared-page release and tensor semaphore phase transitions. Three
launches each pass for (CTAs,instructions) = (1,1), (1,257), (16,17), (148,65).
This verifies lifecycle only; no MMA, TMA transfers, precision epilogues or model
math are tested, so it does not satisfy M1.

The first probe failed because it populated only `workers` instruction streams.
`include/util.cuh::get_worker_id` returns physical `%smid`, which can differ
from `blockIdx.x` even with one CTA. Probe setup now queries `%nsmid` and
populates every SM-ID slot; the output remains indexed by CTA. The query is
a separate development setup kernel, not inference. A production schedule
must account for physical SM-ID indexing and disabled SM slots when allocating
instruction/timing tables. A one-launch model cannot rely on densely numbered
SM IDs or only initialize the streams equal to the number of launched CTAs.

Resource source review: the pinned `tensor_allocator<1,1>` reserves all 512
columns once per CTA and releases them at VM exit. `allocate<T>` creates a
view, not a dynamic allocation, and runtime column offsets only have debug
bounds checks. FP4 accumulator/SFA/SFB TMEM regions therefore need an explicit
nonoverlapping column plan and semaphore-protected reuse. Pinned TK has BF16
and FP8 TCGen05 helpers, while native block-scaled NVFP4 requires extensions.

### BF16 TMA and TCGen05 data-path validation

`tcgen_probe.cu` adds a development VM instruction with two asynchronous TMA
loads, BF16 `tcgen05` matrix multiplication through the pinned TK device
helpers, FP32 tensor-memory accumulators, warpgroup tensor-to-register loads
and direct FP32 stores. It uses `megakernel::mk`, the original controller,
page allocator and launcher/consumer specialization. Shared pages are released
only after MMA completion; accumulator reuse waits for all consumer warps.
No external GEMM is invoked. The separate setup query obtains physical SM-ID
capacity; each compute invocation launches one VM kernel.

`python scripts/validate_tcgen.py` passes on B200 for 1, 2, 17 and 65 consecutive
instructions, each computing [128,64] × [128,64]^T to [128,128] FP32. Every
instruction result matches CPU FP32 multiplication of identical BF16 inputs
within atol=3e-5, rtol=2e-5. This is a data-path test, not a production GEMM:
only one CTA, one K tile and BF16 are implemented; multi-stage K pipelines,
shape tails, FP8/NVFP4, fused epilogues and performance evaluation remain open.
M0 exhaustive audit and M1 compute-foundation acceptance remain incomplete.
The same four cases pass under `compute-sanitizer --tool memcheck
--error-exitcode 1 python scripts/validate_tcgen.py` with zero reported errors.

### FP8 and double-buffered K-loop extension

The same `tcgen_probe.cu` VM instruction is now templated over BF16 and FP8
E4M3 input tiles. Static activation and weight scales are multiplied in FP32
outside the VM dispatch and applied to FP32 register accumulators before stores.
Both precisions use asynchronous TMA and legal TK SM100 TCGen05 device helpers;
there is no standalone GEMM or host precision dispatch inside the VM.

The K loop uses two A/B shared-page pairs. Producer readiness and MMA completion
have separate phase-tracked semaphores for each ring slot; pages are released
after their final use. Tensor memory is retained across K tiles and reused only
after consumer completion. TMA can fetch the next slot while the launcher
processes the current one. This is a development primitive, not yet a tuned
projection family: MMA completion is waited per K tile, and one CTA computes
a fixed 128×128 output tile.

`validate_tcgen.py` passes BF16/FP8 K=64 for 1,2,17,65 instructions; FP8 checks
scale pairs (1,1), (.0123,.0456), (2,.25). Additional K=128,192,256,1024 cases
pass for both precisions with 17 instructions each. References multiply the
identical quantized operands in CPU FP32, then apply the FP32 descale product.
These are compute parity tests, not quantization fidelity or performance results.
Arbitrary M/N, tails, instruction-level dependency fusion, NVFP4 and optimized
epilogues remain open, as do M0 and M1 acceptance.
The expanded BF16/FP8 suite passes CUDA memcheck with zero reported errors.

### Persistent multi-tile BF16/FP8 work distribution

The development instruction now derives output-row/output-column tiles from
`instruction_index * workers + blockIdx.x`. Every physical SM instruction
stream holds the same opcode sequence, so CTA work assignment is independent
of sparse SM IDs. Each CTA computes its own tiles through the double-buffered
K loop. Out-of-range tasks run original NoOp page/tensor release paths and
still advance all instruction phases. Output stores use the complete matrix
stride and repetition offset. This is static work distribution, not the
required fine-grained device DAG scheduler.

`run_grid_probe` validates three repetitions inside one VM compute launch for
each BF16 and FP8 case: M=256,N=384,K=192 with four CTAs; M=768,N=1024,K=256
with 16 CTAs; and logical M=51,N=2560,K=1024 with 16 CTAs. The 51-row test
explicitly pads A to 128 rows in setup and checks all padded output rows as
well as logical rows. It is a correctness baseline, not a skinny GEMM solution.
The irregular task counts exercise inactive final work slots. All outputs
match the identical-operand FP32 reference and the complete suite passes CUDA
memcheck with zero reported errors.

Open: arbitrary output tails without host padding, BF16 output epilogues,
in-VM input quantization, multiple output tiles per CTA, shape tuning, timing,
NVFP4 MMA and all model-stage integration. These grid tests do not complete M1
or establish performance against any competing implementation.

### Native SM100 NVFP4 device primitive

`nvfp4_probe.cu::Fp4Matmul` implements a development block-scaled E2M1 GEMM
instruction in `megakernel::mk`. Packed FP4 bytes use opaque byte-sized TK
tiles for asynchronous TMA transfers. UE4M3 group scales use the previously
verified Sm1xx global layout, are copied into a shared page by the loader
warp, then transferred to tensor memory with
`tcgen05.cp.cta_group::1.32x128b.warpx4`. Accumulators occupy TMEM columns
0–127, SFA 128–131, SFB 132–135. Scale/accumulator lifetimes are protected
by the original VM tensor semaphore. Global descale multiplies FP32 register
accumulators before output stores.

Device math and descriptors follow the exact primitive used by FlashRT's
SM100 CUTLASS GEMMs, inspected in pinned CUTLASS
`cute/arch/mma_sm100_umma.hpp::SM100_MMA_MXF4_SS` (VS=16),
`cute/arch/copy_sm100.hpp::SM100_UTCCP_4x32dp128bit_1cta`, and
`cute/arch/mma_sm100_desc.hpp::make_instr_desc_block_scaled`. The native
launcher issues the CUDA-12.8 SM100 instruction spelling directly:
`tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::4X`.
It does not invoke a CUTLASS GEMM adapter, library GEMM, SM120 MMA or PDL.
The K loop accumulates 64-element slices, refreshes per-slice scales, and
waits for MMA completion before overwriting shared/scales storage.

`python scripts/validate_nvfp4.py` compares the native result to CPU FP32
matmul of the identical dequantized packed operands. B200 cases pass for
K=64 with 1,2,17,65 consecutive VM instructions, plus zero-scale and varied
row/block-scale cases with 17 instructions. K=128,256,1024 each pass with
17 consecutive instructions. The full nine-case suite passes CUDA memcheck
with zero reported errors. These checks prove primitive math and resource
reuse for the tested shapes, not pretrained quantization fidelity.

Open: this FP4 development path is single-CTA, fixed M=N=128, uses a single
input buffer, and stages scales with loader-warp stores rather than scale TMA.
It has no fused FP4 output epilogue, skinny specialization, tile search or
latency measurements. FlashRT optimized GEMM/epilogue families remain unported
as complete operators; M1 and all end-to-end milestones remain incomplete.

### Double-buffered and multi-CTA NVFP4 extension

The FP4 path now uses two three-page operand/scale slots (A, B, scale bytes).
Each has independent TMA-ready, scale-ready and MMA-finished semaphore phases.
The loader prepares the next slot while the launcher consumes the current one;
it waits for the slot's prior MMA completion before reuse. TMEM SFA/SFB columns
are shared across slots and overwritten only after the preceding MMA completes.
Unused pages and inactive final tasks use original VM release paths. This
supersedes the single-buffer/single-CTA limitation recorded above.

The work assignment and complete-matrix output strides now match the BF16/FP8
development path. Scale source offsets account for both the output tile's
128-row group and the current 64-element K slice in the Sm1xx global layout.
`validate_nvfp4.py` passes the original cases plus K=192 (an odd pipeline
iteration count), M=256,N=384,K=192 on four CTAs, M=768,N=1024,K=256 on 16 CTAs,
and logical M=51,N=2560,K=1024 on 16 CTAs. Grid cases execute three matrix
repetitions inside one VM compute launch. M=51 is explicitly padded to 128;
all padded outputs are checked to equal zero. Every case passes CUDA memcheck
with zero reported errors and matches the dequantized-operand FP32 reference.

No latency benefit is claimed: resident benchmark setup, tuned tile families,
skinny MMA, fused FP4/BF16 epilogues, scale TMA, and full model integration
remain required. M0 exhaustive audit and M1 acceptance remain incomplete.

### Resident NVFP4 execution and primitive timing

`create_resident_fp4`, `launch_resident_fp4`, and `destroy_resident_fp4` separate
setup from compute. Setup constructs the physical-SM instruction table and TMA
descriptors, allocates VM program/timing storage, and configures shared memory.
Inputs/scales/output are caller-owned device pointers retained by the Python
`primitives.py::ResidentNVFP4` wrapper. Each `forward()` submits one native VM
kernel to the current PyTorch stream, with no allocation, input transfer,
quantization or framework model operator. Context-manager teardown releases VM
storage after pending users finish. Independent instances are required for
concurrent calls on different streams because output and VM scratch are reused.

`benchmarks/native_primitives.py` measures synthetic expert/vision/prefix QKV
projections after offline packing and warmup. It checks dequantized-operand
parity and deterministic execution on a non-default stream. CUPTI profiling
requires exactly one CUDA kernel event whose name contains `megakernel`.
Raw CUDA-event and synchronized-wall samples, median/p10/p90, shape/padding,
precision policy, source hashes and pinned upstreams are saved to JSON.
This is a primitive harness, not the optimized native model backend required
by Phase 7. It does not establish pretrained policy fidelity or end-to-end speed.

Resident expert validation for 4,16,148 workers passes CUDA memcheck with zero
reported errors using `--validation-only`; timing and CUPTI are explicitly
skipped in that mode because compute-sanitizer conflicts with CUPTI subscribers.
The sanitizer/profiler incompatibility initially caused a failed launch-count
assertion with no CUDA events; it was not treated as a successful trace.
One worker sweep overlapped validation and was discarded. The saved worker
sweep was rerun alone after validation completed.

`results/native_fp4_expert_workers.json` records 30 samples per worker count for
logical M=51 (padded128), N=2560, K=1024: GPU medians 0.5096 ms (4 workers),
0.2240 (16), 0.1212 (20), 0.1235 (32), 0.1208 (64), 0.1236 (148).
Each profiled call has exactly one native VM launch. These are initial
development-path measurements with FP32 output, not evidence that this tile
family is competitive or optimal. The few-microsecond differences among
20–148 workers do not justify a universal scheduling choice.

The final resident wrapper validates byte/scaling formats during setup and
selects its device during launch. All three saved worker reports were
regenerated sequentially with that version; their source hashes describe that
revision. Later ablations below record separate source hashes.
`results/native_fp4_prefix_workers.json` (M768,N2560,K2048)
records GPU medians 1.5818, 0.4143, 0.2255 ms for 16,64,148 workers.
`results/native_fp4_vision_workers.json` (M256,N3456,K1152) records 0.4629,
0.1349, 0.1342 ms for those counts. Each has 30 samples per count and a
one-native-kernel profile gate. CUDA-event intervals include wrapper submission
gaps; separate synchronized wall samples are retained. No external comparison,
whole-model speedup, calibrated fidelity, or optimal tile/schedule is established.

### MMA issue overlap ablation

`NVFP4_OVERLAP_MMA=1` uses separate TMEM scale banks for the two shared input
slots (128–135 and 136–143) and permits MMA issue to advance without a wait
after every operation. The loader's per-slot completion wait protects shared
reuse; the launcher drains the final completion phase of both slots before
publishing output or releasing pages. `--mma-pipeline overlap` selects this at
compile time in the primitive tools; there is no host precision/runtime dispatch
in `forward`. Serial issue remains the default.

The full NVFP4 numerical suite and CUDA memcheck pass for overlap. Matching
50-sample serial/overlap sweeps were run sequentially for expert, prefix and
vision with identical packed inputs and source hashes; the six
`results/native_fp4_{expert,prefix,vision}_{serial,overlap}.json` files record
the compile option and every sample. Representative medians (serial→overlap):
expert/20 workers 0.1199→0.1202 ms; prefix/148 0.2235→0.2240 ms;
vision/148 0.1349→0.1347 ms. There is no broad, decisive latency improvement,
so overlap remains an experimental tactic rather than the default schedule.
No numerical drift against the dequantized reference was observed in the sweeps.

### Kernel duration and resource evidence

Nsight Compute is installed but hardware-counter profiling fails with
`ERR_NVGPUCTRPERM`. No Nsight hardware utilization, cache, bandwidth or achieved
occupancy result was obtained. The first filter attempt matched no kernel;
the corrected `--kernel-name regex:^mk$` attempt reached the permission error.
These attempts are not counted as successful profiling gates.

CUPTI still supplies kernel events. The primitive harness now records a
separate `profiled_kernel_duration_us` from its one-call launch-count trace;
this single profiled call is not a latency distribution. Expert profiled kernel
durations were about 107–110 µs versus 120–125 µs CUDA-event medians, and prefix
about 210 µs versus 225 µs. Sources and samples are saved in
`native_fp4_{expert,prefix}_profiled.json`. Host submission gaps therefore do
not explain most of this development kernel's observed interval.

`resident_fp4_resources` queries CUDA function attributes and the occupancy
API during setup, outside timing. The 16-consumer-warp kernel reports 96
registers/thread, 10,400 static + 216,504 dynamic shared bytes, 808 local
bytes/thread, 640 threads/CTA and predicted maximum one CTA/SM. CUDA local
size includes stack/compiler local storage and is not a measured spill count;
predicted occupancy is not achieved occupancy. The resource report is in
`results/native_fp4_expert_resources.json`.

### Consumer-warp/register-budget ablation

`ComputeConfig` now explicitly supports compile-time 16- or 4-consumer-warp
configurations in the same original VM. Four consumers use 256 total threads
and a 200-register consumer budget; the legacy configuration uses 640 threads
and a 104-register consumer budget. Page completion counts derive from the
configuration, including unused-page and inactive-task release paths.
`--consumer-warps 4` selects the candidate in validation/benchmark tools.
Shared-page and tensor-memory budgets remain unchanged.

The four-warp kernel reports 200 registers/thread and 16 local bytes/thread,
with unchanged shared memory and predicted one CTA/SM. Its 50-sample sweeps
are in `results/native_fp4_{expert,prefix,vision}_warps4.json`. Expert GPU-event
medians are 0.1218 ms (20 workers), 0.1338 (148); prefix/148 is 0.2194 ms;
vision is 0.1277 (64), 0.1295 (148). Kernel-duration samples are separately
recorded. Relative to the preceding sweeps this is roughly 5% lower vision
latency, but no consistent expert interval improvement. These unpaired process
sweeps do not prove a universally superior configuration; defaults remain
unchanged and no final whole-model schedule has been selected.

M0 completeness, fused quantization/epilogues, skinny kernels and M1 model-shape
acceptance remain open despite the primitive/resource experiments above.
The four-consumer BF16/FP8 suite and the combined four-consumer/overlapping-MMA
NVFP4 suite both pass CUDA memcheck with zero reported errors, including
repeated instructions, odd K-loop counts, multi-CTA grids and padded rows.

### Native BF16 output and bias/residual fusion

Source reviewed: FlashRT
`csrc/gemm/fp4/cutlass_fp4_gemm_bias_bf16_sm100.cu` symbols
`cutlass_fp4_gemm_bias_bf16` and `cutlass_fp4_gemm_bias_res_bf16`, using
CUTLASS `LinCombPerColBias` with BF16 bias/residual/output and FP32 compute.
The math is `D = BF16(alpha * accumulator + residual + bias)`, with optional
terms and one final rounding; this applies to π0 projection/residual boundaries.
The port classification is transfer with native VM redesign: native MMA and
accumulator consumers replace the host GEMM/epilogue adapter.

`Fp4Matmul::consumer` now performs those optional terms in FP32 registers and
stores paired BF16 values directly to final global memory, using the exact
row/column mapping of TK warpgroup stores. It does not write a global FP32
projection or launch a separate add/cast kernel. The original FP32 output
mode remains available as the numerical development reference.
`ResidentNVFP4(output_dtype=torch.bfloat16, bias=..., residual=...)` prepares
BF16 inputs/output at setup; `forward()` still submits only the native VM.
Bias shape is N; residual shape includes padded M rows. No π0.5 gated or
adaptive residual arithmetic was introduced.

`scripts/validate_epilogues.py` first checks the native FP32 projection against
dequantized-operand matmul, then checks plain BF16 conversion, bias, residual,
and bias+residual against FP32 operations followed by BF16 rounding. All 16
shape/mode cases are bit-exact across three repeated launches, for M/N/K of
128/128/64, 256/384/192, logical51/2560/1024 (padded128), and 768/1024/256.
The four-consumer suite passes CUDA memcheck with zero reported errors. The
16-consumer suite passes separately with `--profile`, requiring exactly one
native CUDA kernel for every fused case. CUPTI and sanitizer runs are separate.

The primitive benchmark now supports `--epilogue fp32|bf16|bias|bias-residual`.
`results/native_fp4_expert_bf16.json` records GPU-event medians 0.1158/0.1185 ms
for 20/148 workers, versus 0.1243/0.1279 ms for bias+residual in
`results/native_fp4_expert_bias_residual.json` (30 samples each, four consumers).
Both preserve one-launch execution and zero NRMSE against their respective
dequantized/epilogue references. The fused variant performs additional math
and residual reads; this comparison measures its cost, not a speedup over an
equivalent unfused native schedule. Whole-model latency and pretrained fidelity
remain unmeasured. Direct GELU/GeGLU-to-FP4 epilogues and normalization producers
are still required before M1 acceptance.

### Direct FP4 output and fused GELU

`Fp4Matmul::consumer` now writes packed E2M1 outputs and Sm1xx-layout UE4M3
per-16 scales directly from its FP32 register accumulators, with optional
residual, BF16 bias, and GELU. The four lanes owning one row's 16-column
fragment reduce its maximum using quad shuffles. They generate the rounded
block scale, convert values with hardware nearest-even/saturating E2M1, and
write packed bytes without any global BF16/FP32 intermediate. One lane writes
each scale at the verified global layout offset. Zero scales explicitly emit
positive zero codes; scale buffers are deterministically initialized at setup.

Source math: pinned CUTLASS `GELU_taylor<float>` (FP32 FMA ordering and
`tanh.approx.f32`) and FlashRT
`csrc/gemm/fp4/nvfp4_skinny_common.cuh::quant_block16` (nearest UE4M3 per-16
activation scales, global descale, hardware E2M1 conversion). This ports the
transferable GEMM→GELU→FP4 producer math used by the FlashRT SM100 SigLIP
epilogue family into the original VM consumer. It does not yet port the full
optimized tile/visitor family or compact GeGLU store.

The development wrapper accepts `output_dtype=torch.uint8`, optional
`output_gelu=True`, and `output_global_scale`. `output` holds packed data;
`output_scales` holds swizzled scale bytes. Their allocations and bias/residual
preparation happen at setup. `forward()` remains one native VM launch. The
next VM operator's dependency handling and use of those device pointers are
not yet integrated into a model schedule.

`fp4_output_reference.cu` is an independent development oracle that invokes
the actual pinned CUTLASS GELU and FlashRT scalar block quantizer on the
independently checked FP32 projection. It is never called by native forward.
The oracle canonicalizes zero-scale block payloads to positive zero, matching
the native deterministic rule; all other bytes use the original quantizer.
`scripts/validate_fp4_output.py` compares both packed data and every scale byte.
Its 25 cases cover identity quantization, residual+bias quantization, GELU,
GELU+residual+bias, and bias+GELU without residual. Tested shapes are
M/N/K=128/128/64,256/384/192,logical51/2560/1024,768/1024/256, and the SigLIP
FFN-up shape 256/4304/1152 with N padded to 4352. Bias/residual padded columns
are zero; packed/scaling padding is checked against the complete oracle buffer.

All cases are byte-exact across three repeated forwards. Four-consumer
execution passes CUDA memcheck with zero reported errors. Separate
four- and sixteen-consumer `--profile` runs require exactly one native kernel
per forward and pass every case. `results/native_fp4_output_epilogues.json`
contains 30-sample CUDA-event/wall distributions per case, resource attributes,
one-launch gates and source hashes. For example, logical51/N2560/K1024 on
16 CTAs measures 0.2213 ms for plain FP4 output and 0.2290 ms for bias+GELU
FP4 output; the 4304-wide SigLIP-shape bias+GELU case on 64 CTAs measures
0.2460 ms. These fixture worker counts are not tuned, and these intervals
cannot establish a speedup against an equivalent unfused native schedule.
CUDA events include wrapper submission gaps; oracle work is excluded.

This establishes a tested native FP4 producer epilogue, not pretrained
quantization fidelity, full SigLIP FFN integration, or M1 completion. Compact
GeGLU, normalization→quantization producers, optimized shapes/tiling and
single-launch producer/consumer model scheduling remain required.

### Compact interleaved GeGLU-to-FP4

`Fp4Matmul::consumer` now implements half-width output for adjacent
gate/up accumulator pairs: `GELU(gate) * up`, packed directly to compact E2M1
data and per-16 UE4M3 scales. Two neighboring 16-column register fragments
provide a complete 16-element compact scale block. Quad reductions generate
the scale; even lanes combine neighboring nibble codes without duplicate
stores. Output data/scales use the compact width, not the interleaved GEMM
width. There is no global gate/up or BF16 activation buffer.

Source: FlashRT `sm100_gelu_mul_blockscale_visitor.hpp` symbol
`Sm100GeluMulCompactBlockScaleFactorRowStore`, used by
`cutlass_fp4_gemm_geglu_il_hw` / `_hw_v10`. Its logistic GELU formula differs
from regular CUTLASS `GELU_taylor`; the native path preserves that distinction,
the `/6` scale rule and nearest-even hardware conversion. The actual source
`Visitor::gelu_tanh_` and CUTLASS `NumericArrayConverter` run in the independent
reference kernel. Zero-scale payloads are canonically zero in both references
and native output. A positive global output scale is a native extension;
unit scale reproduces the source compact-store math. Runtime per-channel
AWQ scaling is not added: as in FlashRT, any such transform must be folded
into up weights before quantization.

`ResidentNVFP4(output_dtype=torch.uint8, output_geglu=True)` requires pairwise
interleaved weights (`interleave_gate_up`) without bias/residual. Its output
has half as many logical columns as the projection; `output_width` and the
scale allocation reflect that compact layout. `scripts/validate_geglu.py`
checks M/N_interleaved/K=128/128/64,256/384/192,51/8192/1024 and
768/32768/2048, with global output scales 1 and 8. These include the actual
expert and prefix gate/up dimensions. All eight cases match packed data and
scale bytes exactly across three forwards; padded M rows are zero. The full
suite passes CUDA memcheck. Four- and sixteen-consumer launch-trace runs pass.

### Independent driver launch tracing

Kineto/CUPTI activity profiling repeatedly returned no kernel event for the
large prefix compact case despite correct synchronized results; those runs
failed their gate and produced no accepted timing report. The new
`launch_trace.cpp` / `launch_trace.py` utility subscribes to CUPTI driver API
callbacks instead of relying on activity-buffer delivery. It counts launch
API entries, copies kernel symbols during the callback, and unsubscribes before
timing. A one-call scope must contain exactly one native `megakernel` symbol.
Extra legacy/cooperative, host-function or graph-launch calls also fail the
strict scope. An explicit two-forward control must count two launches.
This uses no privileged performance counters and is separate from sanitizer
and Kineto sessions. Nsight hardware-counter profiling remains unavailable.

### Wider K tiles and vectorized scale loading

`NATIVE_FP4_K_TILE` now supports 64,128,256. These are CTA reduction tiles,
implemented with one,two,four legal SM100 K64 block-scaled MMA instructions,
respectively. A/B TMA tiles grow to at most 16 KiB per input page; each K64
subtile has distinct scale banks in TMEM until completion. One commit covers
the complete K tile. Double-buffer reuse and final drains track K-tile phases.
TMA zero-fills partial final K tiles; corresponding absent scale atoms are
explicitly zeroed. The public K contract remains multiples of 64.

Scale loading now caches page/source addresses and transfers uint4 vectors
instead of repeatedly calculating page addresses and copying individual bytes.
This is retained as a native memory/scheduling improvement. With four consumers,
K128 passes the complete primitive suite under CUDA memcheck; K256 with overlapping issue also
passes, including K64/K192 tails and 65 consecutive VM instructions. K256
passes all 25 regular FP4-output cases under memcheck (including SigLIP
K1152) and all 16 BF16-output cases with launch profiling. No error is reported.

`native_fp4_compact_geglu.json` records the initial scalar-load compact path.
The subsequent `native_fp4_compact_geglu_k{64,128,256}.json` reports record
matching inputs, source hashes, compiler K-tile selection, 30-sample GPU/wall
distributions, exact numerical gates and one-launch driver traces. For global
scale 1, observed GPU-event medians are:

| Shape | Initial scalar K64 | Vector K64 | Vector K128 | Vector K256 |
|---|---:|---:|---:|---:|
| Expert M51,N8192→4096,K1024 | 0.1556 ms | 0.0676 ms | 0.0451 ms | 0.0439 ms |
| Prefix M768,N32768→16384,K2048 | 2.1995 ms | 0.4088 ms | 0.3265 ms | 0.2856 ms |

These are sequential synthetic primitive sweeps with identical precision and
no numerical drift, not pretrained model or policy results. Larger tiles do
not help every small workload: the 128/128/64 case is about 0.028 ms for K64
versus 0.032 ms for K256. Default development K64 remains available; final
per-operation tactic selection awaits integrated planning and model timing.
N256/N64 tile families, skinny split-K, normalization producers, model DAG
dependencies, calibrated pretrained parity and the M0–M9 acceptance gates
remain incomplete.

### Native normalization producer and VM startup barriers

`norm.cu` embeds register-cached vector normalization in the original VM:
FP16/BF16 RMSNorm and LayerNorm, optional residual addition/output, FP32
statistics, affine parameters and AWQ inverse scales, explicit source rounding,
and direct FP4 or FP8 output. No normalized global intermediate is produced.
RMS affine gamma is supplied as the already formed `(1 + weight)` parameter.
The BF16 reference follows pinned OpenPI GemmaRMSNorm's FP32 statistics and
final input-dtype rounding; it does not establish pretrained policy parity.
FP4 scale bytes use the same padded Sm1xx layout as native GEMM.

`validate_norm.py` covers 41 cases including expert/vision/prefix widths,
partial rows and partial scale atoms, zero/constant/outlier inputs, folded
parameters, residual rounding variants, two global scales, repeated launches,
and a nondefault stream. Six FP16 cases match the actual pinned FlashRT
`siglip_ln_vec.cu` producer byte for byte. BF16 normalization is independently
computed on CPU and passed through the actual FlashRT GPU quantizer; maximum
observed NRMSE is 0.000279045 against the unchanged 0.005 gate. CPU-packed
reference payloads and scale bytes are additionally compared to the GPU
quantizer. Nine quantization/calibration unit tests pass.

This exposed an offline packing midpoint discrepancy: division by the decoded
scale is not bit equivalent to FlashRT's FP32 reciprocal multiplication.
`pack_nvfp4` now reproduces reciprocal multiplication for both global and block
scales and canonicalizes zero-scale payloads. A regression case fixes scale
0.9375 and value 1.171875, where the two arithmetic paths select different
E2M1 codes. Prior result files retain their historical source hashes.

Synchronization checking initially failed at parallel per-lane VM mbarrier
initialization, including the unchanged NoOp probe. The standalone
`barrier_init_probe.cu` reproduces 12 missing-init errors for 13 parallel lanes;
the same executable with a `serial` argument passes. The recorded
`patches/megakernels-serial-barrier-init.patch` serializes startup initialization
in thread 0, retains the existing visibility fences, and is applied
idempotently by setup. With this patch, synccheck reports zero errors for all
NoOp lifecycle cases (1/16/148 CTAs, repeated launches and up to 257
instructions), and all 41 normalization cases with eight rows per instruction.
This is evidence about initialization and checked executions, not a claim
that every model synchronization path is validated. Reports generated after
the patch hash both its contents and the resulting VM header.

`results/native_normalization_rows{1,4,8}.json` contain matched 41-case
synthetic sweeps, each gated by one native launch using CUPTI driver callbacks.
With 16 resident CTAs, BF16 RMSNorm→FP4 at global scale 1 has GPU-event median
latencies of 31.78/31.94/46.48 µs for M51,D1024;
82.34/79.58/79.09 µs for M256,D1152; and
215.68/209.39/207.41 µs for M768,D2048 (row tiles 1/4/8 respectively).
FP16 source parity cases use up to 148 resident CTAs and show regressions with
larger row tiles. These data do not justify changing the default from one row.
Event intervals include host submission gaps; full sample and synchronized wall
distributions are retained. BF16 worker-count selection is exposed separately
by `--bf16-workers` for subsequent resident scheduling experiments.

A follow-up row-tile-1 sweep, `native_normalization_workers148.json`, passes
all 41 numerical and one-launch gates with explicit worker counts recorded.
For BF16 RMSNorm→FP4 at global scale 1, 148 CTAs give 42.37 µs (M51,D1024),
25.38 µs (M256,D1152), and 41.87 µs (M768,D2048), versus
31.78/82.34/215.68 µs with 16 CTAs. Large-row performance improves materially;
extra workers regress the expert shape. This motivates per-stage scheduling,
not a universal 148-worker default. All inputs and precision policies match.
The patched VM also passes synccheck for the existing four-consumer K256
NVFP4 suite, covering 1–65 instructions, zero/varied scales, K tails,
multi-CTA matrix tasks, repeated launches and the M51 expert-shaped case.
No full-model, pretrained, rollout, attention or end-to-end acceptance gate
is established by these producer and GEMM checks.

### One-launch normalization → projection and on-demand producer handoff

`norm_projection.cu` connects the vector producer to native NVFP4
TMA/TCGen05 projection within one `megakernel::mk` invocation. The Python
`ResidentNormProjection` wrapper accepts BF16/FP16 input and offline-packed
weights, retains GPU buffers, and returns FP32 or BF16 projection output.
K must be a multiple of 64 up to 2048; N is padded to a multiple of 128.
The source producer math is reused through a state-type template, rather than
copied into a separate normalization implementation.

Each 1/4/8-row group has a 64-bit epoch tag. A resident consumer claims an
unclaimed group with CAS, computes normalization and quantization, and
publishes readiness. First-instruction consumers in every CTA help with
strided producer hints, including CTAs with no projection tile. Projection
consumers also claim any unclaimed input groups they need. The loader waits
for readiness before initiating TMA; consumers never wait on another CTA
while holding uncompleted producer ownership. Hints are not ownership:
not-yet-resident CTAs cannot reserve work and prevent resident consumers
from producing it. This avoids a grid-wide barrier and does not require all
workers, or all workers across concurrent independent instances, to be resident.
It is a limited producer/projection schedule, not the complete model DAG.

All writing consumer threads execute the global async proxy fence, rendezvous
on the consumer barrier, and thread 0 performs the GPU-scope release store.
Loader GPU-scope acquire loads order its subsequent TMA reads. The barrier
and release/acquire chain carries prior writers' accesses transitively; a
second per-thread GPU fence was removed. The ordering rationale follows
[PTX ISA 8.7 memory synchronization and causality](https://docs.nvidia.com/cuda/archive/12.8.0/parallel-thread-execution/index.html#memory-consistency-model)
and its async-proxy fence rules. This reasoning accompanies runtime checking;
sanitizer success alone is not a proof of all GPU memory ordering.

Epochs advance through a scalar kernel argument, without a reset kernel or
GPU copy in forward. Setup synchronizes initialization of the readiness tags.
Repeated calls on one instance require stream ordering; independent instances
have independent tags and buffers. Graph capture is explicitly rejected by
this development wrapper because captured host epochs would be reused.
The validation-only split baseline is compiled behind
`NATIVE_NORM_PROJECTION_VALIDATION`; production forward does not call it.

`validate_norm_projection.py` covers 14 chain cases: RMS/BF16→FP32 and
LayerNorm/FP16→BF16 for M1/K64/N128, M3/K192/N384,
M51/K1024/N8192, M129/K192/N384, M256/K1152/N4352,
M768/K2048/N2560, and a single-worker M256/K192/N384 case.
Worker counts span 1/4/16/148, with tails, repeated VM instructions,
three changed-input epochs on nondefault streams, poisoned logical activation
bytes before every replay, residual output, and padded rows. Producer payloads
and scale bytes match standalone normalization exactly. Projection output has
observed NRMSE 0 against dequantized producer/weight operands. An independent
CPU normalization/packing/matmul reference has maximum whole-chain NRMSE
0.000775412 against the unchanged 0.005 gate (FP32 projection has a separate
1e-5 gate). Concurrent 148-CTA instances with different inputs reproduce their
separate reference outputs exactly, including poisoned-buffer replay.

For the current four-row producer configuration, both CUDA memcheck and
synccheck report zero errors on the complete chain suite and concurrent case.
The profiled run records exactly one native launch for every integrated case,
exactly two for every matched split baseline, and two for a two-forward
negative control. `native_norm_projection_rows4_release.json` records current
source/patch/header hashes, upstream pins, environment, 30 GPU-event and wall
samples, and all numerical/launch gates; all recorded source hashes were
checked against the worktree after the run.

Historical reports retain their own hashes: `native_norm_projection_initial`
is the initial per-row pull prototype; `rows{1,4,8}_pull` precede global helper
hints; `rows{1,4,8}` add helpers and still include the extra GPU fence.
The helper schedule cuts the RMS M129/K192/N384 case from about 490 µs in
row-1 pull scheduling to about 96 µs with row-1 helper scheduling. Larger row
groups reduce some coordination costs but do not improve every shape.

Current four-row release-path BF16 RMS chain medians are:

| M / K / N | Integrated one launch | Matched two native launches |
|---|---:|---:|
| 51 / 1024 / 8192 | 128.66 µs | 101.95 µs |
| 129 / 192 / 384 | 95.58 µs | 75.50 µs |
| 256 / 1152 / 4352 | 92.42 µs | 58.98 µs |
| 768 / 2048 / 2560 | 108.53 µs | 70.46 µs |

These sequential synthetic event medians include host submission gaps.
The integrated path still regresses the matched split baseline on these
shapes. This is a verified one-launch native dependency handoff with tuning
work remaining, not an accepted SOTA speedup. General multi-stage ready-work
scheduling, native attention, fused complete FFNs, all model layers/denoising
steps, pretrained fidelity/rollouts and end-to-end performance gates remain
incomplete; no M0–M9 milestone is promoted to complete by this chain.

### Device ready queue and dynamic native VM retirement

`patches/megakernels-dynamic-instructions.patch` adds an explicitly opted-in
`DYNAMIC_INSTRUCTIONS` interface to the pinned VM. The original controller
calls the program's warp-cooperative fetch callback instead of loading a
static instruction row. Opcode -1 publishes a terminal instruction. Worker
warps acknowledge its completion, and the controller retires that slot and
its remaining live predecessor slot using the existing instruction-finished
barriers. It does not simply abandon the pipeline. Static programs retain
the original instruction fetch and dispatch behavior. Both recorded patches
were applied to pristine files extracted from the pinned revision in a
temporary directory; the resulting four VM headers match the worktree exactly.
Setup applies both patches idempotently. New primitive reports also hash the
dynamic patch and the modified fetch/controller/util headers; existing reports
retain their historical source hashes.

`dag_scheduler.cuh` implements a bounded MPMC ready queue, CSR successor
lists, and dependency counters. Each node enters the queue once. Only ready
nodes are claimed, and only resident controllers claim them. A queue ticket
may briefly wait for publication by its resident producer; no task is reserved
for a not-yet-resident CTA. One elected resident controller seeds the root
queue inside the native invocation. Epoch tags distinguish reused queue slots
and lazily initialize dependency counts on their first decrement. Replays
need no GPU reset kernel or host-to-device copy in forward. A single instance
requires stream ordering; independent instances have separate queues and
buffers. Host-generated epochs currently exclude graph capture.

Dependency decrements use GPU-scope acquire/release CAS, rather than relaxed
atomic decrements. This carries preceding parents' writes into the last
parent's ready-queue publication. Queue slots use release stores and acquire
loads; the controller then publishes the native instruction to the worker
warps through the VM's existing barrier. Model operators with multiple output
writers or async-proxy readers must perform their corresponding writer
rendezvous and proxy fences before calling completion. The scalar probe does
not establish that future TMA-producing operators have done this correctly.

`dag_probe.cu` exercises that scheduler inside `megakernel::mk`, using exact
uint32 dependency math and variable node work. It is a scheduler/resource
probe, not a model kernel. A two-row timing ring per physical SM accommodates
the VM's final timing stores without allocating a full static program table.
The Python wrapper validates dependencies, duplicate edges and cycles before
launch. It computes an independent topological reference, including a closed
form of the delay recurrence checked against direct iteration.

`validate_dag.py` passes 18 graph cases: chains through 1024 nodes, 513
independent roots, fan-out/fan-in through 511 siblings, permuted 512-node
layered graphs, and permuted random DAGs through 4096 nodes. Worker counts
span 1/16/148. Three changed-seed epochs on nondefault streams per case check
exact values, one execution per node, queue head/tail/completed counts equal
to node count, no duplicate-execution errors, valid owners and ordinals,
and parent-end/child-start timestamp ordering. The sum of per-CTA terminal
ordinals equals node count, verifying actual early termination rather than
running every CTA through the static upper bound. Invalid graph controls
are rejected on the host.

The complete graph suite passes CUDA memcheck and synccheck with zero errors.
The original static VM lifecycle probe still passes, and the existing native
four-consumer K256 NVFP4 suite passes synccheck after the extension, including
repeated instructions, K tails, multi-CTA tasks, and expert-shaped input.

`results/native_dag_scheduler.json` records the refreshed unsanitized probe,
source/patch/header hashes, pins, GPU/compiler environment, queue diagnostics,
and one native launch for every case. A two-forward negative control records
two launches. Actual configured-kernel occupancy is one CTA per SM on the
148-SM B200 (10,520 static and 216,504 dynamic shared bytes, 128 registers,
8 local bytes). Two independent 100-CTA grids therefore cannot both be fully
resident. Their delayed root work overlaps by 881,664 ns in the recorded
run, and both produce exact results and epoch/count diagnostics. Overlap is
measured with B200 global timer timestamps, not inferred from two stream
submissions. All current report source hashes were checked against the
worktree after the run.

This establishes a tested native scheduling/retirement mechanism and removes
the need to rely on full-grid residency for future ready-task scheduling.
Native GEMM, normalization, attention and complete model stages are not yet
connected to this general queue. Their device descriptors, data dependencies,
completion publication, performance, checkpoint parity and all full-model
M0–M9 acceptance gates remain incomplete.
