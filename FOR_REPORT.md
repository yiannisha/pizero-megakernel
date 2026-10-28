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

The 0.51 ms difference between one and three native launches is small compared with their roughly 129 ms gap to the CUDA graph. Launch count is not the dominant cost in this implementation.

## What we tried and what it gained

The following are **GPU CUDA-event medians**, with lower values better. A positive gain is a reduction in latency; a negative gain is a regression. The reference for each row is shown explicitly because the trials were made at different stages of development. The initial, intermediate, and final builds use the same workload and seed but were not all measured in one simultaneous A/B run, so their differences should be read as observed changes rather than isolated causal effects.

| Trial | Comparison | Measured latency | Observed gain |
|---|---|---:|---:|
| Later fusion bundle (RMS factor folding, vision GELU epilogue, QKV/RoPE/cache epilogue, 64-row expert GEMMs, fused action SiLU and final Euler projection) | Earlier native single kernel, 194.02 ms → fused single kernel | 174.45 ms | **19.57 ms (10.1%)** |
| Split the earlier fused program into vision, prefix, and expert launches | Earlier fused single kernel, 174.45 ms → three kernels | 174.01 ms | **0.44 ms (0.3%)** |
| One-pass attention in the earlier prefix layers | Three-kernel total, 174.01 ms → prefix variant | 170.93 ms | **3.08 ms (1.8% total)**; prefix phase saves 3.42 ms (15.7%) |
| Vision patch conversion and tensor-core projection, plus position/bias addition | Prefix variant, 170.93 ms → patch variant | 169.03 ms | **1.90 ms (1.1% total)**; vision phase saves 2.44 ms (19.6%) |
| Combined later component changes | Earlier three-kernel total, 174.01 ms → final three-kernel total | 169.03 ms | **4.98 ms (2.9%)** |
| Use one launch instead of three for the final operators | Final three-kernel total, 169.03 ms → final one-kernel total | 169.54 ms | **−0.51 ms** |
| Four-way split-K for expert FFN-down projection | Earlier three-kernel total, 174.01 ms → split-K variant | 183.36 ms | **−9.35 ms (5.4% slower)** |

The first row is a *bundle* of later changes: gate/up + GeGLU, residual epilogues, and action-input/time composition were already present in its 194.02 ms baseline. It does not measure a separate contribution for every fusion listed above. The prefix and patch rows follow the actual sequence of measured variants. Component phases are isolated launches in separate samples, so their medians need not add to the end-to-end median; across the earlier and final three-kernel builds, the expert phase changed from 139.44 to 140.73 ms. The split-K variant preserved similar output error but was slower, so it was archived.

Three same-run, same-input **leave-one-out ablations** isolate the effect of particular choices within an earlier 174.35 ms fused single-kernel build:

| Omitted optimization | Latency with it omitted | Gain from keeping it |
|---|---:|---:|
| Vision FFN-up bias + GELU epilogue | 176.25 ms | **1.90 ms (1.1%)** |
| 64-row expert GEMM path for 51 query rows | 181.86 ms | **7.51 ms (4.1%)** |
| QKV projection + RoPE + direct K/V cache production | 181.29 ms | **6.93 ms (3.8%)** |

These ablation gains overlap and cannot be summed to estimate the bundle's gain. No separate latency ablation is available for each remaining fusion.

| Three-launch component, optimized path | GPU median | Fraction of native total |
|---|---:|---:|
| Vision | 10.02 ms | 5.9% |
| Prefix and cache construction | 18.44 ms | 10.9% |
| Ten-step action expert | 140.73 ms | 83.3% |

Component times are measured as isolated native launches after buffers and caches are populated. Their sum can differ slightly from the complete forward because the timings were collected as separate samples.

## Why the megakernels are slower on B200

A timing-enabled build recorded the resident VM's consumer instruction timestamps, and CUDA events measured each component. Instrumentation increased the expert's measured phase time to 152.04 ms; the uninstrumented 140.73 ms above is the performance result. The timestamps include dependency waits, so they identify long resident instruction intervals but are not pure arithmetic times. Active CTA counts come directly from the static instruction streams; they describe *available scheduled parallelism*, not a hardware-counter measurement of achieved occupancy.

| Median active CTAs per layer | Vision, 27 layers | Prefix, 17–18 layers | Expert, 180 layers |
|---|---:|---:|---:|
| QKV projection | 148 | 120 | **20** |
| Attention | 148 | 96 | **8** |
| Output projection | 54 | 96 | **8** |
| Gate/up or FFN-up | 148 | 148 | **64** |
| FFN-down | 54 | 96 | **8** |

### Small action matrices expose too few independent CTAs

The action expert processes a 50-action chunk plus one state token: only **51 query rows** per denoising step. For the native schedule, one QKV projection exposes 20 CTA assignments, gate/up 64, and attention, output projection, and FFN-down only **8 each**. Eight assignments are at most eight simultaneously working CTAs *for that stage* on a 148-SM GPU; each assignment can loop over multiple tiles but cannot turn that stage into 148 independent CTAs. Other stages may overlap through row-band pipelining, so this is a bound on stage parallelism rather than an exact count of idle SMs. Attention is divided across eight query heads, while the narrow output/down projection grids have few output tiles. The 64-row GEMM path avoids padded work and saves 7.51 ms in the earlier ablation, yet still leaves the small-output stages with limited grid parallelism. Vision and prefix expose more work, though their 54-CTA and 96-CTA stages also fall short of 148 assignments.

This is a **shape problem**, not a statement that the whole network is small. The expert contains 18 transformer layers repeated across ten Euler steps, giving **180 dependent layer executions**. Each layer needs its projection outputs before attention, attention before its output projection, and the residual before the FFN. The native VM uses row-band producer counters to preserve these dependencies; consumers wait for required counters, then publish completion after the output is visible. Because the next operation depends on the preceding one, the GPU cannot simply combine the spare CTAs from adjacent stages into parallel work. The two-pass expert attention also reads the key sequence twice to match the BF16 reference behavior. Its median recorded consumer interval was about 1.22 million cycles, versus about 0.085 million for expert QKV, but these intervals include waits and must not be read as pure kernel arithmetic times.

### Fusion saves less than the schedule costs

One persistent Hazy VM launch removes many host-visible launches and lets the fused epilogues avoid intermediate activation reads/writes. The measurements show that those gains are real: the later fusion bundle saved 19.57 ms, and removing QKV/RoPE/cache fusion alone cost 6.93 ms in the earlier ablation. Yet the final one- and three-launch paths differ by only 0.51 ms. A CUDA graph already replays the operator sequence without paying Python dispatch at every operator, while its individual kernels can select their own grids, tile sizes, and specialization for the 768-row vision path, prefix path, and 51-row expert path. The native launch keeps 148 worker CTAs resident and executes **62,072 nonempty range instructions** over **1,627 stages**; each range follows the static stream and uses dependency counters when its inputs are needed. This machinery enables a correct single launch but does not change the serial mathematical order of the 180 expert layers. The measured 140.73 ms expert phase accounts for **83.3%** of native end-to-end latency and dominates the 4.2× gap to the same-weight graph.

Adding more output tiles is not automatically a win. Four-way split-K on expert FFN-down increased the earlier three-launch total by 9.35 ms. Splitting K creates partial results and additional reduction/synchronization work, so extra parallelism must outweigh that overhead in each short layer. This experiment rules out that particular split-K schedule as a solution; it does not rule out different operator-specific schedules.

### Large weights limit whole-model reuse

Measured parameter storage is approximately **827 MB** for vision, **5,017 MB** for the prefix language model, and **623 MB** for the expert, versus **126 MB** of B200 L2. Even the expert weights alone exceed L2. A single launch cannot keep all component weights on chip across the full sequence, and fusing adjacent layers mostly joins operations with *different* weight matrices. The ten expert steps can reuse the same weight set, but its size prevents assuming that L2 retains it all between steps. For a 51-row BF16 projection, the idealized per-invocation arithmetic from a `M×K` by `K×N` multiply is `2MKN` FLOPs for `2KN` weight bytes, or roughly **51 FLOPs per weight byte** at `M=51`, even if each weight is fetched only once for that invocation. Actual traffic, cache hits across steps, and tile-level rereads require measurement. The practical memory benefit of this megakernel is therefore concentrated in intermediate activations and launch overhead, not in making the multi-gigabyte weight set resident in cache. The model is small in **query-row count** at the expert, which hurts parallelism; it is large in **weight footprint**, which still makes memory traffic relevant.

These are bounds and measured schedule facts, not a full roofline analysis. Nsight Compute returned `ERR_NVGPUCTRPERM`, so DRAM bytes, achieved bandwidth, tensor-core utilization, and achieved SM occupancy were unavailable. The profile directly supports the low-parallelism and long-dependency-chain explanation. It does **not** establish that bandwidth is unsaturated, that all weights are reread from DRAM every step, or that each long consumer interval is compute time. The 40.25 ms graph result shows that this workload can run much faster on the same B200 with the same weights and inputs; attributing the entire gap to one hardware bottleneck would require hardware counters or additional controlled kernel-level experiments.

## Conclusion and evidence

For this B200 workload, the current one- and three-launch native megakernels have nearly identical latency and are about 4.2× slower than the same-weight CUDA graph. The action expert accounts for most native time. The strongest measured mechanism is that its small-row operators expose very little parallel work to the 148 SMs, while the megakernel's fused boundaries do not eliminate the model's serial layer dependencies or large weight set. A faster native design would need different per-component and per-operator tile schedules; reducing the number of launches alone has not produced a win.

- [Matched one-launch and CUDA-graph benchmark](results/native_pi0_report_single_b200.json)
- [Matched three-launch benchmark with component times](results/native_pi0_report_three_b200.json)
- [Earlier native baseline](results/native_pi0_matched_baselines.json), [fused variant and same-run ablations](results/native_pi0_realtime_variants.json)
- [Prefix attention variant](results/native_pi0_prefix_flash_b200.json), [vision patch variant](results/native_pi0_vision_patch_b200.json)
- [Resident VM instruction profile and parameter sizes](results/native_pi0_component_vm_profile_b200.json)
- [Initial three-launch and split-K ablation](results/native_pi0_partitioned_b200.json), [split-K result](results/native_pi0_partitioned_splitk4_b200.json)
- [Native model construction](native_pi0/model.py), [VM operators and launch](native_pi0/model.cu), [static range scheduler](native_pi0/program.py)
- [Benchmark harness](benchmarks/matched_native.py), [profile harness](benchmarks/profile_native_components.py)
- [Timing-enabled native library builder](scripts/build_native_profile.py)

To reproduce the instruction profile, build the timing-enabled library with `python scripts/build_native_profile.py --output /tmp/pi0-profile.so`, then run `python -m benchmarks.profile_native_components --library /tmp/pi0-profile.so --report results/native_pi0_component_vm_profile_b200.json`. Use an isolated B200 and the repository's Python requirements. Instrumented component times should be reported separately from uninstrumented latency.

The benchmarks use random weights. They establish implementation correctness against that same-weight reference and latency for the stated shape; they do not establish policy quality on a trained checkpoint or a parity-matched result against every optimized inference implementation.
