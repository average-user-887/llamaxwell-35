# Publication manifest

Prepared against `804349dc98991765af2f3cce80e4554fc60335a2`.

| Files | Contents |
| --- | --- |
| `README.md`, `REPRODUCE.md` | Result, trace semantics and exact compile instructions |
| `depth2.receipt.json` | Verified runtime, slot/cache profile, request times and status |
| `cycle-timing.json`, `cycle-timing.jsonl` | All measured cycle clocks, stage offsets/counts, trim groups and exact output parity |
| `trace-v2.jsonl`, `trace-v2-meta.json` | Complete typed verifier chronology and public trace digest |
| `warm-deltas.json` | Separately derived stage means from previous uninstrumented host counters |
| `server-context.patch`, `server-context.base.cpp` | Exact single-TU modification and its pinned preimage |
| `rebuild-in-container.sh`, `lineage.json`, `abi.json`, `loader-check.txt` | Exact compile/link recipe, build and runtime hashes, ABI and device-free loader checks |
| `SHA256SUMS` | Whole package checksums |
