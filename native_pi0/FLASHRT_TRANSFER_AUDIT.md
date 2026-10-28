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

`model.cu` now executes the entire BF16 π0 reference in one static VM launch,
but `results/native_pi0_static_flash.json` fails action parity (3.47% NRMSE)
and takes 238.5 ms versus 240.3 ms eager OpenPI for the same random-weight
three-view/50-action test. It has 91,654 native range instructions over
146,712 tiles. Fuse before measuring precision variants; a static fetch path
alone does not remove repeated norm, pointwise, metadata and skinny GEMM work.
The next implementation gate is same-weight BF16 numerical parity, followed
by source-backed fusion and calibrated precision porting above.
