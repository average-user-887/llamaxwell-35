# Q8S: q8_0 KV cache quality screen (generic cache-type path, quality only)

Gate per bucket: dmean KL <= +.0010 and same-top >= -0.3 pt. Pass/fail reported, marginal marked, not judged.

## Long set (G1c harness, docs 0-3 only = 256 rows/bucket; baseline = G1c arm A vectors, lib sha confirmed)
| arm | 32K dmean / top | 65K dmean / top | 98K dmean / top |
|---|---|---|---|
| Q8KV | +.00305 / -1.17 FAIL | +.00126 / -0.39 FAIL | +.00140 / -0.78 FAIL |
| Q8K  | +.00216 / -0.39 FAIL | +.00090 / -0.39 FAIL | +.00097 / +0.00 PASS |
| Q8V  | not run (time cut reached) | | |
All cells marginal: doc-level CI95 spans zero in every bucket (4 docs). Max row KL 0.0233-0.1007.

## Shallow set (e1,ea,eb,ec two-slot cases, 272 rows, 8 units; baseline = stock F16 D1B dumps)
| arm | dmean | CI95 (unit) | top | max | gate |
|---|---|---|---|---|---|
| Q8KV | +.00520 | [-.00067,+.01107] | -1.47 | .289 | FAIL |
| Q8K  | +.00261 | [-.00103,+.00625] | -0.74 | .215 | FAIL |
| Q8V  | +.00248 | [+.00032,+.00465] | -0.74 | .193 | FAIL |

## VRAM (-np 2 -c 196608, fa on, post-load; KV preallocated)
| variant | KV MiB | saving | nvidia-smi used MiB |
|---|---|---|---|
| F16 | 3840 | - | 22900 |
| K-only q8 | 2940 | 900 | 22086 |
| V-only q8 | 2940 | 900 | 22070 |
| K+V q8 | 2040 | 1800 | 21378 |

## Not done (reported honestly)
- Rotation: tree has upstream attention rotation, ON by default for quantised cache in all Q8 arms; rotation-OFF control arm (LLAMA_ATTN_ROT_DISABLE=1) not run.
- Two-slot independence, q8 slot save/restore bit-equality, slot/checkpoint file size and restore time, per-arm peak VRAM under fill: not run (GPU budget exceeded; lock then taken by another session).

## Method / caveats
- Arms differ from F16 only by -ctk/-ctv; stock q8 path disables occamy fast decode and GEMM prefill (quality only, no timing).
- Long arms on docs 0-3 (not 8) for time; wide CIs. Shallow baseline lib (D1B, N2SPLIT=0) differs from G1c lib; ed case skipped (F16 slot restore incompatible).
- VRAM numbers are post-load, not filled-slot peaks. Server runs used --ctx-checkpoints 0.

## Files
Q8S/: long/Q8S_TABLE.md, SHALLOW_TABLE.md, vram_*/load.log, vram.log, steps.log, long.log, shallow.log, run_q8s.sh, harness/analyzers, bin/, dumps/.
