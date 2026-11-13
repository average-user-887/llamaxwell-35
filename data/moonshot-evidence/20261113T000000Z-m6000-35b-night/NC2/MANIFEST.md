# NC2 logit-vector manifest

Generated read-only from existing files; nothing re-run. `<M>` = the m6000-llama-cpp root. Full per-file list with sha256: manifest.json.

## Totals

| set | files | bytes | MB |
|---|---|---|---|
| G1 | 69 | 68545152 | 68.5 |
| G1c | 53 | 3369205760 | 3369.2 |
| all | 122 | 3437750912 | 3437.8 |

Git push fit (each file <50 MB, total <300 MB): **NO**. G1c files are 63.6 MB each (64 rows x 248320 x 4 B): 53 of 122 files exceed 50 MB; total 3438 MB vs 300 MB. G1 .npy files (993.3 KB each, 1 row) alone fit: 68.55 MB.

## G1c (rows = teacher-forced decode rows [64 x 248320 f32] at positions B..B+63; run-20261112T202254Z)

| arm | bucket | docs present | rows each | dims | bytes each |
|---|---|---|---|---|---|
| A | 32768 | 0,1,2,3,4,5,6,7 | 64 | [64, 248320] | 63569920 |
| A | 65536 | 0,1,2,3,4,5,6,7 | 64 | [64, 248320] | 63569920 |
| A | 98304 | 0,1,2,3,4,5,6,7 | 64 | [64, 248320] | 63569920 |
| Ap | 32768 | 0 | 64 | [64, 248320] | 63569920 |
| Ap | 65536 | 0 | 64 | [64, 248320] | 63569920 |
| Ap | 98304 | 0 | 64 | [64, 248320] | 63569920 |
| B | 32768 | 0,1,2,3,4,5,6,7 | 64 | [64, 248320] | 63569920 |
| B | 65536 | 0,1,2,3,4,5,6,7 | 64 | [64, 248320] | 63569920 |
| B | 98304 | 0,1,2,3,4,5,6,7 | 64 | [64, 248320] | 63569920 |
| C(F32 ref) | 32768 | 0 | 64 | [64, 248320] | 63569920 |
| C(F32 ref) | 65536 | 0 | 64 | [64, 248320] | 63569920 |
| C(F32 ref) | 98304 | none | - | - | - |

## G1 (1 row per file: single position, `<doc>_r<rep>_<checkpoint>.npy`, dims (248320,) float32, 993280 bytes + header; run 20261112T171109Z)

| arm | docs | checkpoints (file count) |
|---|---|---|
| A | codex,codexs,d0,d1,fx1,u0 | 16384,32768,65536,65836 (18 files) |
| Ap | codex,codexs | 16384,32768,65536,65836 (6 files) |
| B | codex,codexs,d0,d1,fx1,fx2,u0 | 16384,32768,65536,65836 (21 files) |
| C(F32 ref) | codex,codexs,d0,d1,fx1,u0 | 16384,32768,65536,65836 (18 files) |
| A2(repeat) | codex | 16384,32768,65536 (3 files) |
| B2(repeat) | codex | 16384,32768,65536 (3 files) |

Earlier G1 attempts (20261112T170708Z, T170929Z, T170954Z) hold no logit vectors (only logs/rows.csv).

## Missing

| set | arm | doc | bucket/checkpoint | reason |
|---|---|---|---|---|
| G1c | Ap | doc1 | 32768 | by design: noise arm ran doc 0 only (tier 2) |
| G1c | Ap | doc1 | 65536 | by design: noise arm ran doc 0 only (tier 2) |
| G1c | Ap | doc1 | 98304 | by design: noise arm ran doc 0 only (tier 2) |
| G1c | Ap | doc2 | 32768 | by design: noise arm ran doc 0 only (tier 2) |
| G1c | Ap | doc2 | 65536 | by design: noise arm ran doc 0 only (tier 2) |
| G1c | Ap | doc2 | 98304 | by design: noise arm ran doc 0 only (tier 2) |
| G1c | Ap | doc3 | 32768 | by design: noise arm ran doc 0 only (tier 2) |
| G1c | Ap | doc3 | 65536 | by design: noise arm ran doc 0 only (tier 2) |
| G1c | Ap | doc3 | 98304 | by design: noise arm ran doc 0 only (tier 2) |
| G1c | Ap | doc4 | 32768 | by design: noise arm ran doc 0 only (tier 2) |
| G1c | Ap | doc4 | 65536 | by design: noise arm ran doc 0 only (tier 2) |
| G1c | Ap | doc4 | 98304 | by design: noise arm ran doc 0 only (tier 2) |
| G1c | Ap | doc5 | 32768 | by design: noise arm ran doc 0 only (tier 2) |
| G1c | Ap | doc5 | 65536 | by design: noise arm ran doc 0 only (tier 2) |
| G1c | Ap | doc5 | 98304 | by design: noise arm ran doc 0 only (tier 2) |
| G1c | Ap | doc6 | 32768 | by design: noise arm ran doc 0 only (tier 2) |
| G1c | Ap | doc6 | 65536 | by design: noise arm ran doc 0 only (tier 2) |
| G1c | Ap | doc6 | 98304 | by design: noise arm ran doc 0 only (tier 2) |
| G1c | Ap | doc7 | 32768 | by design: noise arm ran doc 0 only (tier 2) |
| G1c | Ap | doc7 | 65536 | by design: noise arm ran doc 0 only (tier 2) |
| G1c | Ap | doc7 | 98304 | by design: noise arm ran doc 0 only (tier 2) |
| G1c | C | doc0 | 98304 | by design: F32 ref ran doc 0 only (tier 2); 98K unreachable (cudaMalloc OOM at ctx 98560 with fa off/f32 KV; retry ctx 66000 covers 32K/65K only) |
| G1c | C | doc1 | 32768 | by design: F32 ref ran doc 0 only (tier 2) |
| G1c | C | doc1 | 65536 | by design: F32 ref ran doc 0 only (tier 2) |
| G1c | C | doc1 | 98304 | by design: F32 ref ran doc 0 only (tier 2); 98K unreachable (cudaMalloc OOM at ctx 98560 with fa off/f32 KV; retry ctx 66000 covers 32K/65K only) |
| G1c | C | doc2 | 32768 | by design: F32 ref ran doc 0 only (tier 2) |
| G1c | C | doc2 | 65536 | by design: F32 ref ran doc 0 only (tier 2) |
| G1c | C | doc2 | 98304 | by design: F32 ref ran doc 0 only (tier 2); 98K unreachable (cudaMalloc OOM at ctx 98560 with fa off/f32 KV; retry ctx 66000 covers 32K/65K only) |
| G1c | C | doc3 | 32768 | by design: F32 ref ran doc 0 only (tier 2) |
| G1c | C | doc3 | 65536 | by design: F32 ref ran doc 0 only (tier 2) |
| G1c | C | doc3 | 98304 | by design: F32 ref ran doc 0 only (tier 2); 98K unreachable (cudaMalloc OOM at ctx 98560 with fa off/f32 KV; retry ctx 66000 covers 32K/65K only) |
| G1c | C | doc4 | 32768 | by design: F32 ref ran doc 0 only (tier 2) |
| G1c | C | doc4 | 65536 | by design: F32 ref ran doc 0 only (tier 2) |
| G1c | C | doc4 | 98304 | by design: F32 ref ran doc 0 only (tier 2); 98K unreachable (cudaMalloc OOM at ctx 98560 with fa off/f32 KV; retry ctx 66000 covers 32K/65K only) |
| G1c | C | doc5 | 32768 | by design: F32 ref ran doc 0 only (tier 2) |
| G1c | C | doc5 | 65536 | by design: F32 ref ran doc 0 only (tier 2) |
| G1c | C | doc5 | 98304 | by design: F32 ref ran doc 0 only (tier 2); 98K unreachable (cudaMalloc OOM at ctx 98560 with fa off/f32 KV; retry ctx 66000 covers 32K/65K only) |
| G1c | C | doc6 | 32768 | by design: F32 ref ran doc 0 only (tier 2) |
| G1c | C | doc6 | 65536 | by design: F32 ref ran doc 0 only (tier 2) |
| G1c | C | doc6 | 98304 | by design: F32 ref ran doc 0 only (tier 2); 98K unreachable (cudaMalloc OOM at ctx 98560 with fa off/f32 KV; retry ctx 66000 covers 32K/65K only) |
| G1c | C | doc7 | 32768 | by design: F32 ref ran doc 0 only (tier 2) |
| G1c | C | doc7 | 65536 | by design: F32 ref ran doc 0 only (tier 2) |
| G1c | C | doc7 | 98304 | by design: F32 ref ran doc 0 only (tier 2); 98K unreachable (cudaMalloc OOM at ctx 98560 with fa off/f32 KV; retry ctx 66000 covers 32K/65K only) |
| G1 | A | fx2 | 16384 | absent (arm did not run this doc/checkpoint) |
| G1 | A | fx2 | 32768 | absent (arm did not run this doc/checkpoint) |
| G1 | A | fx2 | 65536 | absent (arm did not run this doc/checkpoint) |
| G1 | Ap | d0 | 16384 | absent (arm did not run this doc/checkpoint) |
| G1 | Ap | d0 | 32768 | absent (arm did not run this doc/checkpoint) |
| G1 | Ap | d0 | 65536 | absent (arm did not run this doc/checkpoint) |
| G1 | Ap | d1 | 16384 | absent (arm did not run this doc/checkpoint) |
| G1 | Ap | d1 | 32768 | absent (arm did not run this doc/checkpoint) |
| G1 | Ap | d1 | 65536 | absent (arm did not run this doc/checkpoint) |
| G1 | Ap | fx1 | 16384 | absent (arm did not run this doc/checkpoint) |
| G1 | Ap | fx1 | 32768 | absent (arm did not run this doc/checkpoint) |
| G1 | Ap | fx1 | 65536 | absent (arm did not run this doc/checkpoint) |
| G1 | Ap | fx2 | 16384 | absent (arm did not run this doc/checkpoint) |
| G1 | Ap | fx2 | 32768 | absent (arm did not run this doc/checkpoint) |
| G1 | Ap | fx2 | 65536 | absent (arm did not run this doc/checkpoint) |
| G1 | Ap | u0 | 16384 | absent (arm did not run this doc/checkpoint) |
| G1 | Ap | u0 | 32768 | absent (arm did not run this doc/checkpoint) |
| G1 | Ap | u0 | 65536 | absent (arm did not run this doc/checkpoint) |
| G1 | C | fx2 | 16384 | absent (arm did not run this doc/checkpoint) |
| G1 | C | fx2 | 32768 | absent (arm did not run this doc/checkpoint) |
| G1 | C | fx2 | 65536 | absent (arm did not run this doc/checkpoint) |
