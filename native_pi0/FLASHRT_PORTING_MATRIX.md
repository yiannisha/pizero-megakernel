# Active native porting status

Execution architecture is defined in [IMPLEMENTATION.md](IMPLEMENTATION.md):
static Hazy VM streams, range-loop native operators and direct data counters.
The generic DAG and stage-counter executors are retired. Their detailed source
inventory, porting notes and results remain in
[the archive](../archive/generic-executors-2026-10-28/native_pi0/FLASHRT_PORTING_MATRIX.md).

| Area | Current status |
|---|---|
| Original VM static controller/page/TMA/consumer/storer path | Used by new `native.cu`; no dynamic instruction callback |
| Native BF16 matrix ranges | Implemented; per-instruction tile loops, continuous TMA/MMA phases and one TMEM retirement |
| Static stream planner | Gemma-style contiguous ranges assigned to least-loaded worker streams |
| Direct readiness counters | Implemented per produced row band; publish completed range contributions |
| BF16/FP8/NVFP4 primitive math | Preserved; standalone correctness evidence is component evidence |
| RMSNorm/LayerNorm and attention algorithms | Preserved for integration into range operators |
| Packing/calibration/inventory tools | Preserved for offline setup and later precision work |
| Complete BF16 Pi0 | Reintegrating onto native instruction streams; no passing complete model yet |
| Full mixed-precision model, checkpoint/rollout parity | Open |
| Eager/compiled/graph Torch, realtime-vla, FlashRT model wins | Unproven |

`validate_native_linear.py` checks one-worker and cross-row-band ranges,
expert/prefix matrix shapes, changed-input replays, exact direct counters,
bit-exact split-native output parity, independent Torch GEMMs and one VM launch.
These are execution-foundation checks, not a substitute for the complete model.

Shared `dag_probe.*`, `dag_scheduler.cuh` and `dag_matmul.*` remain solely for
Cosmos compatibility; the new Pi0 build does not include them. All M0-M9 model
acceptance gates remain open. The original PLAN.md objective is unchanged,
subject to the user's explicit scheduling clarification.
