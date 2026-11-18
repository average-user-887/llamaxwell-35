# Package manifest

| File | Contents |
|---|---|
| `candidate.patch` | Byte-identical single-file CUDA patch used in the rank-6 experiment. |
| `README.md` | Dispatch guards, physical work reductions, build and replay settings, and measured limits. |
| `evidence.json` | Measured library fingerprints, allowlisted timings and counts, and diagnostic and recall verdicts. |
| `SHA256SUMS` | Checksums for the patch, README, evidence, and manifest. |

The allowlist covers public source commits, source/model/image/library/input and receipt digests, public GPU details, typed flags, bounded workload and cache counts, physical and observed dispatch shape descriptors, draft counts, checker comparisons, timing and rate scalars, wrapper status, and exactness or mismatch-position verdicts. The short recall result is represented by its verified equality outcome.
