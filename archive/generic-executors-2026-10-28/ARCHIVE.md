This is a source-and-results snapshot of the retired generic π0 executors.
It is historical evidence, not the active implementation or a passing model.
MANIFEST.json records every snapshot hash and each path removed from the active
tree. No Cosmos sources were moved. Shared DAG helpers remain at their original
paths solely because Cosmos still imports them.

Restore the snapshot into a separate directory with:

    python archive/generic-executors-2026-10-28/restore.py /tmp/pi0-old-experiments

The destination must not exist. The restore script verifies hashes and leaves
the current implementation untouched. CUDA dependencies are linked from the
workspace's third_party directory. The archived reference requires the pinned
OpenPI Python environment. Its output parity failed; the final stage-counter
trial's short timing is invalid because it returned zero outputs.
