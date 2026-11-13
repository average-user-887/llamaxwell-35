# N2 result: multi-doc noise floor for G1c GEMM gate

Verdict: Ap (arm A + FA_PARALLEL_BLOCKS_MULT=2) is NOT a zero floor: A-vs-A repeat is exactly 0 (shallow), but Ap-vs-A is +.0014..+.0033 dmean, -0.4..-1.0 pt long, +.0022 / -1.10 pt shallow. B-vs-A (+.0025..+.0035, -0.8..-1.2 pt) fails the gate (dmean<=+.0010, top>=-.3) in all 3 buckets, but is statistically indistinguishable from the Ap-vs-A floor (every B CI95 overlaps Ap's; point estimates within ~.0002 at 32K/98K, .0012 at 65K). So the GEMM B result is not separable from a pure FA-reordering perturbation. A zero floor does not exist: the "noise" is real numeric drift from any reduction-order change. Same holds for q8: the shallow A-repeat is exactly 0, so q8 shallow drift (+.0026..+.0052, -0.7..-1.5 pt) is real drift, not run-to-run noise, but it is the same size as an innocuous FA reorder (Ap shallow +.0022 / -1.10 pt), i.e. no larger than what a benign kernel change produces. Gate as defined (dmean<=+.0010) is not passable by any perturbation of this kind (Ap itself fails it).

## Shallow set (272 rows, e1/ea/eb/ec, 8 units; baseline = stock F16 D1B pair0 dumps; CI95 over units)
| comparison | dmean | CI95 | top pt | max |
|---|---|---|---|---|
| A repeat vs A | +0.00000 | [0,0] | +0.00 | 0 (bit-exact) |
| Ap vs A | +0.00224 | [-0.00022,+0.00471] | -1.10 | .124 |
| Q8KV vs A (Q8S) | +0.00520 | [-0.00067,+0.01107] | -1.47 | .289 |
| Q8K vs A (Q8S) | +0.00261 | [-0.00103,+0.00625] | -0.74 | .215 |
| Q8V vs A (Q8S) | +0.00248 | [+0.00032,+0.00465] | -0.74 | .193 |

## Long set, pooled (CI95 over docs; Ap and B docs 0-7 = 512 rows/bucket; Q8 docs 0-3 = 256 rows)

## Ap vs A (per doc: dmean / top pt)

| doc | 32K | 65K | 98K |
|---|---|---|---|
| 0 | +0.00447 / -1.56 | +0.00119 / +0.00 | +0.00168 / +0.00 |
| 1 | +0.00656 / +0.00 | +0.00002 / +0.00 | +0.00000 / +0.00 |
| 2 | +0.00024 / +0.00 | +0.00074 / +0.00 | +0.00194 / +0.00 |
| 3 | +0.00227 / +0.00 | +0.00126 / +0.00 | +0.00261 / -1.56 |
| 4 | +0.00165 / -1.56 | +0.00051 / +0.00 | +0.00465 / -1.56 |
| 5 | +0.00653 / -3.12 | +0.00190 / +0.00 | +0.00385 / -3.12 |
| 6 | +0.00323 / +0.00 | +0.00409 / -1.56 | +0.00446 / -1.56 |
| 7 | +0.00152 / -1.56 | +0.00125 / -1.56 | +0.00182 / +0.00 |

| pooled | docs | rows | dmean | CI95 doc | top pt | CI95 doc top |
|---|---|---|---|---|---|---|
| 32768 | 8 | 512 | +0.00331 | [+0.00134,+0.00528] | -0.98 | [-1.95,-0.00] |
| 65536 | 8 | 512 | +0.00137 | [+0.00034,+0.00241] | -0.39 | [-1.00,+0.21] |
| 98304 | 8 | 512 | +0.00263 | [+0.00129,+0.00396] | -0.98 | [-1.95,-0.00] |

## B vs A (per doc: dmean / top pt)

| doc | 32K | 65K | 98K |
|---|---|---|---|
| 0 | +0.00423 / -1.56 | +0.00081 / +0.00 | +0.00348 / -3.12 |
| 1 | +0.00326 / +0.00 | +0.00001 / +0.00 | +0.00000 / +0.00 |
| 2 | +0.00046 / -1.56 | +0.00089 / +0.00 | +0.00121 / +0.00 |
| 3 | +0.00636 / -1.56 | +0.00193 / +0.00 | +0.00105 / +0.00 |
| 4 | +0.00148 / -1.56 | +0.00033 / +0.00 | +0.00848 / +0.00 |
| 5 | +0.00705 / -1.56 | +0.00408 / -1.56 | +0.00475 / -3.12 |
| 6 | +0.00342 / -1.56 | +0.01074 / -4.69 | +0.00557 / -1.56 |
| 7 | +0.00132 / +0.00 | +0.00143 / +0.00 | +0.00204 / +0.00 |

| pooled | docs | rows | dmean | CI95 doc | top pt | CI95 doc top |
|---|---|---|---|---|---|---|
| 32768 | 8 | 512 | +0.00345 | [+0.00146,+0.00543] | -1.17 | [-1.78,-0.57] |
| 65536 | 8 | 512 | +0.00253 | [-0.00044,+0.00549] | -0.78 | [-2.18,+0.62] |
| 98304 | 8 | 512 | +0.00332 | [+0.00096,+0.00569] | -0.98 | [-2.17,+0.22] |

## Q8KV vs A (per doc: dmean / top pt)

| doc | 32K | 65K | 98K |
|---|---|---|---|
| 0 | +0.00253 / -1.56 | +0.00136 / +0.00 | +0.00300 / -1.56 |
| 1 | +0.00570 / +0.00 | +0.00002 / +0.00 | +0.00000 / +0.00 |
| 2 | +0.00022 / -1.56 | +0.00110 / -1.56 | +0.00161 / +0.00 |
| 3 | +0.00375 / -1.56 | +0.00258 / +0.00 | +0.00099 / -1.56 |
| 4 | - | - | - |
| 5 | - | - | - |
| 6 | - | - | - |
| 7 | - | - | - |

| pooled | docs | rows | dmean | CI95 doc | top pt | CI95 doc top |
|---|---|---|---|---|---|---|
| 32768 | 4 | 256 | +0.00305 | [-0.00060,+0.00669] | -1.17 | [-2.41,+0.07] |
| 65536 | 4 | 256 | +0.00126 | [-0.00041,+0.00293] | -0.39 | [-1.63,+0.85] |
| 98304 | 4 | 256 | +0.00140 | [-0.00060,+0.00340] | -0.78 | [-2.22,+0.65] |

## Q8K vs A (per doc: dmean / top pt)

| doc | 32K | 65K | 98K |
|---|---|---|---|
| 0 | +0.00290 / -1.56 | +0.00080 / +0.00 | +0.00124 / +0.00 |
| 1 | +0.00353 / +0.00 | +0.00002 / +0.00 | +0.00000 / +0.00 |
| 2 | +0.00025 / +0.00 | +0.00062 / +0.00 | +0.00111 / +0.00 |
| 3 | +0.00196 / +0.00 | +0.00216 / -1.56 | +0.00152 / +0.00 |
| 4 | - | - | - |
| 5 | - | - | - |
| 6 | - | - | - |
| 7 | - | - | - |

| pooled | docs | rows | dmean | CI95 doc | top pt | CI95 doc top |
|---|---|---|---|---|---|---|
| 32768 | 4 | 256 | +0.00216 | [-0.00011,+0.00443] | -0.39 | [-1.63,+0.85] |
| 65536 | 4 | 256 | +0.00090 | [-0.00054,+0.00234] | -0.39 | [-1.63,+0.85] |
| 98304 | 4 | 256 | +0.00097 | [-0.00009,+0.00203] | +0.00 | [+0.00,+0.00] |


Gate reading B-vs-A: FAIL all buckets (dmean +.0035/+.0025/+.0033 > +.0010; top -1.17/-0.78/-0.98 < -0.3). Hardened analyzer exit 0 (complete, finite). Its own verdict line: "A-vs-Ap noise is of similar size or larger".
B vs multi-doc floor (Ap-vs-A): dmean diff +.0001 (32K), +.0012 (65K), +.0007 (98K); top diff -0.19/-0.39/0.00 pt; all within doc CIs. Per-doc B exceeds Ap mainly on doc 6 (65K .0107 vs .0041) and doc 4 (98K .0085 vs .0047).
Note: doc 1 gives ~0 KL at 65K/98K for every arm (likely a low-entropy/repetitive slice); it dilutes pooled means equally across arms.

## Method / caveats
- Long: fixed harness copy (container name n2-*), NC2/bin/g1c_rows, lib v4 sha256 0c7ffbde...330d5 (verified), NR=64, PER=2, docs 0-7, 4 chunks ~16 min each (65 min GPU, within cap). A and B vectors are the existing G1c run (symlinked, unchanged). First launch failed (relative OUT path, rc=125, nothing ran) and was rerun with an absolute path.
- Shallow: Q8S shallow_q8.sh variant (D1B lib, N2SPLIT=0, dx2seq_q8 binary, kt=vt=f16) for A2 and Ap, baseline = D1B pair0 dumps. A2 bit-exact to baseline confirms binary/env equivalence.
- Q8 long rows: from Q8S, docs 0-3 only; Q8V not run there. CI with 4 docs is wide.
- Ap shallow uses same env addition only; lib differs from long set (shallow baseline lib D1B vs v4).
- Lock released after each run; unload via POST qwen36-35b-moe only; containers n2-* removed; no MAXWELL flags; nothing pushed.

## Files
N2/: run.sh, run2.sh, shallow_n2.sh, shallow_an.py, tables.py, g1c_harness.py (n2- names), shallow_table.txt, shallow/, run/{A,B symlinks,Ap/*.f32,steps.log}, analyze.out, tables.md.
