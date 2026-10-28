# π0 megakernel evaluation on NVIDIA B200

## Workload and implementation

This evaluation uses ordinary π0 inference with three 224×224 camera views, an empty prompt, a 50-action chunk, and ten Euler denoising steps. The native path uses BF16 transformer projections with FP32 accumulation and FP32 action heads. Measurements use one NVIDIA B200 with 148 SMs and 126 MB of L2 cache. Model weights and inputs are fixed across the native and OpenPI CUDA-graph runs; the benchmark constructs a random model with seed 101 because no trained checkpoint was supplied.

Both native designs use the original Hazy megakernel VM. The host compiles fixed per-worker instruction streams of contiguous tile ranges before inference. The VM's controller, loader, launcher, consumer, and storer execute these ranges; row-band producer counters enforce data dependencies. There is no dynamic task queue or generic DAG dispatcher in the native inference path.

| Design | Launches per inference | Boundary |
|---|---:|---|
| Single megakernel | 1 | Vision encoder, prefix transformer and cache construction, then all ten action-expert steps |
| Three component megakernels | 3 | Vision encoder; prefix transformer and cache construction; ten-step action expert |

The three launches use the same operator implementations and buffers as the single launch. Splitting lets each component have its own static instruction stream, but does not change the 180 serial action-expert layers or their tile geometry. Both designs run the complete model on the GPU; setup-time weight preparation and instruction construction are outside the timed forward.

The optimized schedule contains 1,627 operator stages, 62,072 nonempty range instructions across 148 workers, and 93,854 assigned compute tiles. The three-launch version partitions this schedule into vision, prefix, and expert streams without adding a dynamic scheduler.

## Fusions and graph simplifications implemented

- Fold each Gemma RMS gain into the QKV and gate/up weights at setup. Compute compact FP32 row factors and apply them in projection epilogues instead of materializing full normalized activations.
- Merge gate and up projections into one tensor-core GEMM and apply GeGLU in its BF16 output epilogue.
- Fuse QKV projection with Q/K rotary embedding and direct K/V cache writes. Producer counters become ready after the rotated keys and cached values are visible.
- Fold bias and residual addition into attention-output and FFN-down GEMM epilogues, preserving the BF16 rounding boundary needed by the reference.
- Fuse vision FFN-up bias and GELU into its GEMM epilogue. The optimized vision variant also converts image patches once, performs patch projection on tensor cores, and adds patch bias and position embeddings in the native program.
- Skip the unused final prefix-layer attention output and FFN: that layer only produces the K/V cache needed by the action expert. The optimized prefix variant uses one-pass tiled attention in the earlier prefix layers.
- Compose the action-input projection with the action half of the first time MLP at setup; precompute the time-dependent bias for each Euler step and fuse SiLU into the native FP32 projection.
- Fold the final expert RMS gain into the action-output projection and fuse the Euler update into that projection's epilogue.
- Use a 64-row tensor-core path for the 51-row action-expert projections to avoid computing the padded upper half of a 128-row tile.

These are actual native operators or setup-time transforms, not host calls to an external kernel sequence. The action-expert attention remains a two-pass BF16-compatible kernel for numerical parity. The measured full-model action error against the same-weight OpenPI reference is 0.00876 NRMSE, 0.02785 normalized maximum absolute error, and 0.99996 cosine similarity for both native launch designs.

## End-to-end B200 latency

GPU timings are medians of five CUDA-event samples after warmup, with no other compute PID observed. All three rows below use the same random weights and inputs; the native rows use the optimized vision patch and prefix attention variants described above.

| Execution | GPU median | Relative to CUDA graph |
|---|---:|---:|
| OpenPI CUDA graph | **40.25 ms** | 1.00× |
| One native megakernel | 169.54 ms | 4.21× slower |
| Three native megakernels | 169.03 ms | 4.20× slower |

The 0.51 ms difference between one and three native launches is small compared with their roughly 129 ms gap to the CUDA graph. Before the two later component changes, the three-launch path took 174.01 ms; separating the earlier 174.45 ms single kernel had likewise changed little. The optimized fusions reduced some work, but launch count is not the dominant cost.

The earlier combined fusion pass reduced the single native forward from about 194.1 to 174.4 ms. Same-input ablations measured 176.3 ms without fused vision GELU, 181.9 ms without the 64-row expert GEMMs, and 181.3 ms without fused QKV/RoPE/cache production, versus 174.4 ms with all three enabled. These changes helped individually; none removed the gap to the graph baseline.

| Three-launch component, optimized path | GPU median | Fraction of native total |
|---|---:|---:|
| Vision | 10.02 ms | 5.9% |
| Prefix and cache construction | 18.44 ms | 10.9% |
| Ten-step action expert | 140.73 ms | 83.3% |

Component times are measured as isolated native launches after buffers and caches are populated. Their sum can differ slightly from the complete forward because the timings were collected as separate samples. The vision patch change reduced vision from 12.89 to about 10.02 ms; one-pass prefix attention reduced prefix from 21.81 to about 18.44 ms. A four-way split-K experiment on the expert down projection preserved similar output error but *increased* the three-launch total to 183.36 ms, so it was archived rather than enabled.

## What the device profile shows

A timing-enabled build recorded the resident VM's consumer instruction timestamps, and CUDA events measured each component. Instrumentation increased the expert's measured phase time to 152.04 ms; the uninstrumented 140.73 ms above is the performance result. The timestamps include dependency waits, so they identify long resident instruction intervals but are not pure arithmetic times. Active CTA counts come directly from the static instruction streams.

| Median active CTAs per layer | Vision, 27 layers | Prefix, 17–18 layers | Expert, 180 layers |
|---|---:|---:|---:|
| QKV projection | 148 | 120 | **20** |
| Attention | 148 | 96 | **8** |
| Output projection | 54 | 96 | **8** |
| Gate/up or FFN-up | 148 | 148 | **64** |
| FFN-down | 54 | 96 | **8** |

Each expert layer has only 51 query rows. The native 64×128 tensor-core tiles give its attention, output projection, and down projection eight independent CTA assignments while the GPU has 148 SMs. The expert attention scans the key sequence twice to preserve BF16 rounding. These low-parallelism, dependent stages repeat for 18 layers across ten denoising steps. Vision and prefix expose more parallelism, but 54-CTA vision output/down projections and 96-CTA prefix projections still leave many SMs without work at those stages. The split-K down-projection regression is evidence that merely adding CTAs to one short stage does not repair the overall schedule.

The measured parameter storage is approximately 827 MB for vision, 5,017 MB for the prefix language model, and 623 MB for the expert, versus 126 MB of B200 L2. Thus, the explanation is **not** that the entire model fits in cache or that weight traffic is negligible. The three regions use different weights, and the expert's weights exceed L2; placing them in one kernel does not keep all weights on chip across layers or denoising steps. Fusion can remove some activation traffic and launch boundaries, but this implementation pays much larger costs from insufficient tile parallelism and repeated synchronization along the dependency chain. CUDA graph replay already removes the CPU's per-operator dispatch overhead while allowing each operator to use its own grid and tile shape.

No DRAM throughput or tensor-core utilization percentage was measured: Nsight Compute returned `ERR_NVGPUCTRPERM` for this device. The profile therefore supports the scheduling and occupancy explanation directly, while the claim that a specific stage saturates or does not saturate memory bandwidth remains unverified. It is reasonable to expect limited *cross-operator memory-traffic savings* from a whole-model megakernel here; it is not reasonable to infer that π0 has no memory-sensitive operators.

## Conclusion and evidence

For this B200 workload, the current one- and three-launch native megakernels have nearly identical latency and are about 4.2× slower than the same-weight CUDA graph. The action expert accounts for most native time. The strongest measured mechanism is that its small-row operators expose very little parallel work to the 148 SMs, while the megakernel's fused boundaries do not eliminate the model's serial layer dependencies or large weight set. A faster native design would need different per-component and per-operator tile schedules; reducing the number of launches alone has not produced a win.

- [Matched one-launch and CUDA-graph benchmark](results/native_pi0_report_single_b200.json)
- [Matched three-launch benchmark with component times](results/native_pi0_report_three_b200.json)
- [Resident VM instruction profile and parameter sizes](results/native_pi0_component_vm_profile_b200.json)
- [Initial three-launch and split-K ablation](results/native_pi0_partitioned_b200.json), [split-K result](results/native_pi0_partitioned_splitk4_b200.json)
- [Native model construction](native_pi0/model.py), [VM operators and launch](native_pi0/model.cu), [static range scheduler](native_pi0/program.py)
- [Benchmark harness](benchmarks/matched_native.py), [profile harness](benchmarks/profile_native_components.py)
- [Timing-enabled native library builder](scripts/build_native_profile.py)

To reproduce the instruction profile, build the timing-enabled library with `python scripts/build_native_profile.py --output /tmp/pi0-profile.so`, then run `python -m benchmarks.profile_native_components --library /tmp/pi0-profile.so --report results/native_pi0_component_vm_profile_b200.json`. Use an isolated B200 and the repository's Python requirements. Instrumented component times should be reported separately from uninstrumented latency.

The benchmarks use random weights. They establish implementation correctness against that same-weight reference and latency for the stated shape; they do not establish policy quality on a trained checkpoint or a parity-matched result against every optimized inference implementation.
