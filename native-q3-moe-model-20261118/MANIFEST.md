# Publication allowlist

Prepared against `b9c6598b8f0b04eeff3e8ae4f34beafc30d2c287`.

| Files | Contents |
|---|---|
| README.md, REPRODUCE.md, MANIFEST.md | Results, scope, reproduction, file allowlist |
| {smoke-off,smoke-on,serial-off,depth2-off,depth2-on,serial-on}.receipt.json | Six actual statuses; separately labeled reconstruction for the refused native control; target settings, mapped software identifiers, measured cache, timing, observed EOS/EOG counts, structural repetition, score, bounds, acceptance and layer-level dispatch |
| {same six labels}.trace-v2.jsonl and {same six labels}.trace-v2-meta.json | Chronological event records and checksummed exporter metadata, including all available cumulative cost snapshots and incomplete-record markers |
| comparisons.json, q3-warm-comparison.json | Four matched pairs, first drift positions and counts; independently recomputable 1,025-round component comparison |
| trace_export.py, warm_compare.py, verify-package.py | Actual chronology converter and component comparator plus standalone public checks |
| probe.base, release.base, observation.base | Portable bootstrap source for the three experiment modules |
| initial.probe.patch, r1.probe.patch, r2.probe.patch; initial.release.patch, r1.release.patch, r2.release.patch; r1.observation.patch, r2.observation.patch | Complete sequential deltas. Exact release and observation changes, and exact parser changes with private historical literals carried as typed substitutions |
| source-pins.json, private-input-keys.json, assemble-source.py | Typed private-input names, public software SHA-256 pins and source reconstruction |
| audit.cpp, match.inc, build.sh | Exact common host observer source, matching-structure ABI and CPU build script |
| SHA256SUMS | Every file above except itself |

Only public software and sanitized-trace fingerprints are published. Measured records omit data fingerprints, token IDs, response text, host paths and device identifiers. Model flags and source use portable in-container names where needed. The per-layer array index is the target architecture layer. Neither frequency counts nor first-unequal positions expose the compared token values.
