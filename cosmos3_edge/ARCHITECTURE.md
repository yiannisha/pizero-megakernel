# Native execution architecture

## Shared VM

`native.cu` compiles the existing operators in `native_pi0/dag_matmul.cu`,
which include the original `third_party/Megakernels/include/megakernel.cuh`.
The Cosmos headers add operators and setup functions to this engine. They do
not introduce another persistent runtime. The existing VM patches for dynamic
instruction fetching and serial barrier initialization remain prerequisites.

Projection CTAs retain controller, loader, tensor-core launcher, consumer and
storer roles. TMA loads stage shared-memory pages in a double buffer; TCGen05
accumulates FP32 in tensor memory. Consumers store BF16 or packed NVFP4/scales.
The resident setup API owns program/diagnostic allocations; Python owns tensor
storage. Setup and teardown may allocate/synchronize; `forward()` only dispatches
one original VM kernel and returns a view created during setup.

## Projection formats

- BF16: 128×128 output tiles, K64 TMA stages, FP32 accumulators and BF16 stores.
- FP8: E4M3 operands, same tile/pipeline, static activation and weight descales
  multiplied into the accumulator before BF16 output.
- NVFP4: packed E2M1 operands and swizzled per-16 UE4M3 block scales, native
  SM100 block-scaled tensor-core instructions. The shared K tile defaults to 256.
- Epilogues optionally add residual then bias, and apply ReLU². Dense ReLU²
  operates on FP32 accumulators before BF16 output rounding; framework BF16
  intermediate rounding differs, and its measured error is recorded separately.
- NVFP4 ReLU² writes packed data and scales directly; there is no BF16 hidden
  activation between its FFN projections.

Current projections require K divisible by 64 and pad M/N to 128. The host
weight-preparation API validates Cosmos stream names and checkpoint tensor
shapes. FP8 callers must supply prepared E4M3 operands and descales. Runtime
normalization-to-FP8 integration, site calibration and final policy selection
are still pending.

## Tile dependencies

The same `native_pi0::DagScheduler` schedules dense and FP4 operators. A node
is one matrix output tile. Each down-projection tile depends on all up tiles
for its 128-row block. The controller fetches ready nodes on device, and exits
only when the entire DAG retires. Producer writers issue the global async proxy
fence, synchronize all consumer threads, then release successor readiness.
Consumer TMA descriptors in device stage storage are explicitly acquired into
the tensor-map proxy. Epoch-tagged counters prevent old invocation state from
satisfying new dependencies.

`DenseSchedule` supports BF16 outputs feeding BF16 inputs, including FP8-up
projections feeding a BF16 down projection. A BF16 producer cannot yet directly
feed an FP8 consumer: a quantizing producer must be added first. The FP4
schedule supports direct packed-output/scales handoff. Both schedules currently
share VM infrastructure but are development operator schedules; integrating
all precision operators into a complete model instruction stream remains work.

## Validation boundaries

Current tests cover standalone projections and two-projection FFNs. They
include independent math references, actual checkpoint weights with synthetic
activations, reordered DAG nodes, changed inputs across epochs, poisoned output
storage, nondefault streams and one-launch tracing. They do not prove model
layer, conditioning-cache, attention, denoising, Reasoner or video-generation
correctness. No full-model latency or SOTA comparison is established.

## Cosmos attention preparation

`EdgeRopeOp` is a VM consumer operation with one warp per 128-dimensional head.
It supports separate reference and fused rounding policies, row-strided input,
and direct output into K/V cache slices. Understanding keys for generator
consumption are normalized before RoPE, independently of causal understanding
keys. Setup tables use the official interleaved multimodal frequency mapping.

`NativeJointAttention` wraps the existing tiled online-softmax VM operator,
now instantiated for head dimension 128 and Cosmos Q16/KV8. The complete
understanding+generation cache is visible to every query; π0's special suffix
mask is not applied. Split partials and their merge remain device-side in the
same attention launch. The standalone APIs execute these operators separately. The integrated
generation-attention schedule now composes dense projections, preparation and
attention in one launch. Full layer and model integration remain pending.


## Integrated generation attention

`GenerationAttentionGlobals` extends the existing dense schedule globals;
`GenerationAttentionSchedule` uses the same epoch-aware DAG queue. Device
instructions dispatch the existing BF16/FP8 projection roles, the warp-per-head
preparation consumer, a KV publication join, and the existing online-softmax
attention consumer. The attention consumer reuses the same shared-memory pages;
no second runtime or externally dispatched attention kernel is involved.

Preparation nodes depend on the precise 128-row/128-column projection tiles
that produce their head vectors. A shared join represents complete generated
K/V readiness, avoiding repetition of the entire key dependency set for every
query tile. Attention also waits on its query preparation tiles. The static
understanding prefix has no per-step producers and is read unchanged.

Each producer fences its stores and synchronizes writers before publishing
readiness. Schedule epochs advance beyond both the queue and the component
attention epoch, so interleaving separately launched components cannot reuse
old split-counter generations. Components are caller-owned and require stream
ordering; overlapping execution on shared buffers is unsupported.
