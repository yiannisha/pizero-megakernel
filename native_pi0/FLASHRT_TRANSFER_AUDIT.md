# FlashRT π0 / π0.5 transfer audit (pinned 60054d2, B200 SM100a)

This audit follows executable paths, not `flash_rt/configs/pi0.yaml` or
`pi05.yaml`: those generic YAML shapes (for example, π0 action dimension 24 and
chunk size 1) do not describe the OpenPI π0 checkpoint and fixed shapes used by
our full-model validator. The relevant RTX π0 implementation is
`third_party/FlashRT/flash_rt/frontends/torch/pi0_rtx.py` plus
`flash_rt/models/pi0/pipeline_rtx.py`. The π0.5 RTX implementation is the
corresponding `pi05_rtx.py` frontend and `models/pi05/pipeline_rtx.py`.
SM100/Thor alternatives are in `frontends/torch/pi0_thor.py`,
`frontends/torch/pi05_thor_fp4.py`, and `models/pi05/pipeline_thor_fp4.py`.

## Model behavior to keep distinct

| Behavior | π0 RTX implementation | π0.5 RTX implementation | Native impact |
|---|---|---|---|
| Action inputs | Project state as an extra decoder token; action/time MLP; state query sees only prefix and itself | Uses adaptive RMS normalization with precomputed timestep style; optional state prompt | Separate native model variants; do not transplant AdaRMSNorm into π0 |
| Default numeric path | FP16 activations and FP8 E4M3 large GEMMs after calibration | BF16 activations and calibrated FP8 GEMMs, with optional NVFP4/INT8 paths | BF16 reference stays independent; π0 FP16 is a precision candidate requiring end-to-end gate |
| Temporal work | 18 decoder layers per step; ten default steps; action output accumulation | Same depth and default step count, different conditioning | Keep Euler update and step-dependent data in one VM launch |
| Caching | Prefix KV once; decoder KV appends each step | Same basic prefix/decoder split | Retain static data counters and resident caches |

Sources: [π0 pipeline](../third_party/FlashRT/flash_rt/models/pi0/pipeline_rtx.py),
[π0 frontend](../third_party/FlashRT/flash_rt/frontends/torch/pi0_rtx.py),
[π0.5 pipeline](../third_party/FlashRT/flash_rt/models/pi05/pipeline_rtx.py).
The π0 source explicitly reports FP16 improves ten-step FP8 stability compared
with BF16; this is a FlashRT design claim, not a validation result for our B200
megakernel.

## Transfer order for the native megakernel

| Priority | FlashRT mechanism and source | Native adaptation | Gate |
|---|---|---|---|
| 1 | Fold Gemma `(1 + RMS weight)` into QKV and gate/up weights; merge QKV and gate/up; precompute per-step timestep projection (`pi0_rtx.py`, `pipeline_rtx.py`) | Do immutable transforms at setup; eliminate repeated norm, concat, and FP32 timestep work from the 2,544-op program, while preserving π0 BF16 rounding in the reference tier | Same-weight full π0 actions and intermediate landmarks |
| 2 | `qkv_split_rope_kvcache`, attention reads resident KV, fused residual/norm/quantized activation, fused GeGLU (`pipeline_rtx.py`) | Fold RoPE and cache writes into native QKV or attention producers; combine residual + RMSNorm and GeGLU with adjacent native range epilogues, then remove the separate pointwise instructions | One VM launch, row-band counters, full replay, memcheck/synccheck |
| 3 | Calibrated FP8 E4M3 weight/activation GEMMs, separate scales for QKV/O/gate-up/down (`pi0_rtx.py`, `pi05_rtx.py`) | Pack weights offline; emit FP8 TMA/TCGen05 ranges and producer-side quantization with static scales. Retain action heads and sensitive norm/attention paths at higher precision as needed | Real-frame action accuracy plus isolated end-to-end latency against BF16 |
| 4 | π0.5 optional NVFP4 for SigLIP and Gemma FFNs: e2m1 values, per-16 UE4M3 scales, 4304→4352 padding, row-interleaved gate/up, producer-epilogue quantization (`docs/pi05_prefix_nvfp4.md`, `frontends/torch/pi05_rtx.py`) | Use existing native NVFP4 math/packing but port SM100 TMA/TCGen05 producers and epilogues; keep Gemma Q/K/V/O at FP8 initially because FlashRT reports a severe Q/K fidelity loss at FP4 | Checkpoint and real-frame action gate; never infer B200 speedup from RTX 5090 results |
| 5 | Skinny decoder GEMMs split K and fuse attention/RoPE, residual+norm and GeGLU consumers (`docs/pi05_decoder_skinny.md`, `csrc/kernels/pi05/pi05_decoder_skinny_fp8_sm120.cu`) | Redesign as static SM100 range work with direct producer counters. The existing FlashRT family and programmatic dependent launch target SM120 and separate kernels; they cannot be copied into one SM100 VM kernel | Full π0 and π0.5 model latency with matching shape/precision |
| 6 | π0.5 Thor selective FP4, AWQ, and decoder formats (`frontends/torch/pi05_thor_fp4.py`, `docs/thor/pi05.md`) | Port only selected sites after calibration. AWQ inverse scales can be folded into up rows for gate/up interleaving; compare layer-specific error before broadening | Real-data action accuracy and long rollout; default-off until accepted |

The π0.5 optional NVFP4 documentation measures 20.88 ms default and 10.39 ms
full NVFP4 on RTX 5090 for a two-view, ten-action checkpoint. Those figures
include a different GPU, precision, chunk length and checkpoint from our
current three-view, 50-action BF16 B200 run; they are design evidence, not a
matched baseline. The π0.5 SM120 skinny and NVFP4 options are explicitly
opt-in. Thor has SM100 FP4 FFN kernels, a more relevant starting point for
B200 than the SM120 decoder binary, but the Thor pipeline uses a different
scheduling/runtime boundary.

## Current gap

`model.cu` now executes the entire BF16 π0 reference in one static VM launch.
`results/native_pi0_model_rope_benchmark.json` passes the same-weight
random-model action gate at 0.00889 NRMSE and measures 216.15 ms versus
249.94 ms eager OpenPI for three views, 50 actions and ten denoising steps.
The program has 79,896 native range instructions over 130,426 tiles. The
remaining gates include checkpoint fidelity, calibrated precision, the π0.5
model variant, and matched compiled/graph/realtime-vla/FlashRT comparisons.

## Realtime-VLA fusion audit (pinned eb46c8d)

Realtime-VLA is another important source for **π0 fusion**, especially its
small action-expert path. The executable sources are
[`pi0_infer.py`](../third_party/realtime-vla/pi0_infer.py) and
[`convert_from_jax.py`](../third_party/realtime-vla/convert_from_jax.py).
It uses BF16 Triton kernels and a captured CUDA graph; it does not implement
the Hazy VM. Its `optimized` scheduling and tile branch is enabled only for
SM120, 2–3 views, chunk 63 and an empty prompt. Our B200/SM100 test takes its
general branch, so the README's RTX 5090 optimized timings do not describe our
matched baseline.

| Realtime-VLA fusion | Source | Transfer to native π0 |
|---|---|---|
| Offline action input projection × action half of the first time MLP; precompute step-dependent time bias | `convert_from_jax.py` around `fused_weight`, `time_dependent_biases`; decoder in `pi0_infer.py` | Highest priority for the 10-step loop: replace action projection, concat and first time MLP with one small GEMM plus fused SiLU. This changes BF16 rounding order, so compare against same-weight OpenPI before accepting. |
| Fold final RMS weight and `-1/steps` into output projection; accumulate velocity into noise | `convert_from_jax.py` final projection transform; `pi0_infer.py::rms_matmul_k_1024_32_bias_res` | Native output epilogue can do normalization, projection, bias and Euler update; preserve the reference's explicit FP32 multiply/add rounding until parity is measured. |
| QKV projection with RoPE and direct Q/K/V writes; gate/up matmul with GeGLU; GEMM bias+residual and bias+GELU epilogues | `pi0_infer.py::matmul_n_2048_2560_qkv_rope`, `scaled_matmul_rope_qkv`, `scaled_matmul_grouped_gate`, `matmul_small_bias_res`, `matmul_small_bias_gelu` | Emit native range operators with fused epilogues and publish source row-band counters only after all writes. Avoid separate RoPE, GeGLU, residual and GELU instructions. |
| Compute an RMS row factor separately, consume it within the QKV/gate GEMM instead of materializing a full normalized activation | `pi0_infer.py::rmsnorm_factor_kernel`, `scaled_matmul_rope_qkv`, `scaled_matmul_grouped_gate` | Keep a compact factor buffer or per-row shared state, then scale A during native TMA/GEMM staging. This can remove large intermediate writes but needs a tile dependency and exact rounding gate. |
| Last prefix layer writes only K/V needed by the expert; encoder/decoder overlap via streams and per-layer events | `pi0_infer.py::transformer_encoder`, `record_run`, `transformer_decoder` | Prune unused last-layer prefix Q/O/FFN work and express prefix-cache readiness as direct data counters inside the single VM. The native scheduler can overlap independent work without copying the CUDA-stream design. |
| Grouped small-row decoder GEMMs and fused down-projection residual | `pi0_infer.py::matmul_grouped_res`, `matmul_k_4096_1024_res` | Adapt tile shape and K partition to SM100, then keep the residual add in the native matrix epilogue. The exact SM120 tile parameters and PDL launches are not portable. |

The residual GEMM epilogues, offline action/time fold and last-prefix K/V
pruning are now in the native path. A trial that moved Gemma RoPE from its
once-per-token native producer into the attention loader kept the complete
action NRMSE exactly at 0.00889 and removed 198 separate RoPE stages, but
isolated native median increased from 216.15 to 224.32 ms. Attention revisits
K for multiple query tiles, so that placement repeats the rotation. The trial
was reverted. The accepted native RoPE producer now writes each rotated K
and V cache row once before attention reads it. The clean isolated result in
`results/native_pi0_rope_cache_producer_clean.json` is 208.71 ms native with
the same 0.00889 action NRMSE, compared with 216.15 ms for the preceding
native path in a separate isolated run. QKV projection and RoPE remain
distinct native instructions; projection-epilogue fusion remains open.
The gate/up projection and GeGLU are now fused in the native GEMM consumer
using setup-time interleaved gate/up weight rows. The first complete run kept
0.00889 action NRMSE and measured 194.17 ms native, with 71,608 range
instructions. The clean-source replay measured 194.09 ms native in
`results/native_pi0_geglu_epilogue_clean.json`. This is a BF16 result;
quantized FFN paths remain open.

The subsequent realtime-vla native pass adds compact RMS row factors with
folded Gemma weights, 64-row expert GEMMs, vision bias+GELU, action/time SiLU
and final action/Euler epilogues, and QKV+RoPE+cache writes in one native range
instruction. The complete random-model gate passes with one launch at
0.00848 action NRMSE and a 174.45 ms isolated B200 median. Each of the
vision GELU, 64-row expert, and QKV/RoPE fusions improves matched native
latency in `results/native_pi0_realtime_variants.json`. The action path still
has a large gap to the matched OpenPI CUDA graph reference (~40.23 ms), so
further expert scheduling and kernel work remain performance priorities.
