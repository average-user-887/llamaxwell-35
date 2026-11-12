# G1c run card (CPU prep; nothing here has touched the GPU)

## Run
Env (never stored in files): `G1C_GPU` (device id), `G1C_UNLOAD_URL` (full POST URL naming qwen36-35b-moe; unload-all URLs are refused), `G1C_PUBLISH` + `G1C_PORT` (FGDIV container publish spec and its port; endpoints per ~/LAB_FACTS.md). Optional: `MODE=full|cut`, `TIER2=0`, `SKIP_FGDIV=1`, `G1C_NR` (rows per bucket).
`bash run_all.sh` : flock on gpu-bench.lock, POST unload, abort (rc 91) if compute apps present, containers `g1c-*` only, MAXWELL_GRAPHS unset, no power/clock changes. Output `run-<utc>/` (steps.log, run.log, G1C_TABLE.md, fgdiv_analyze.out).
Order: A, B, FGDIV, tier 2 (Ap doc0, C doc0) only before the 22:30 CET cut.

## Design
B (A + GGML_CUDA_FA_MAXWELL_GEMM=1) vs A, Codex v4 lib, teacher-forced full-vocab FP32 logits (`g1c_rows`, single seq). Buckets 32K/65K/98K, rows at n_kv = B+256 (tag 256, see TAIL.md). Noise: A-vs-Ap, A-vs-C (C: -fa off, f32 KV, ub 512; likely cannot reach 98K, harness retries ctx 66000 with 32K/65K only).
Gate: PASS only if B-vs-A dmean <= +.0010 and same-top >= -.3 pt in every bucket. Fail with A-vs-Ap of similar size: "gate not meaningful at this depth (owner decision)". Gate never relaxed.

## GPU minutes (from G1 speeds: A ~6.5 min/98K doc, B ~5.2, ~1 min load per container)
- Full (8 docs x 64 rows): A+B 94-102, FGDIV ~19, overhead ~3 = ~121-124 min. Over 120 and over the ~83 min left at last check.
- Minimal cut (default when <130 min remain): 4 docs x 128 rows = 512 rows/bucket: A ~28, B ~23, FGDIV ~19, overhead 3 = ~73 min. Caveat: only 4 docs, CI by doc is wide, rows within a doc are correlated; analyzer warns when docs < 8.
- Tier 2 (Ap, C on doc 0) only if time remains.

## FGDIV (<=20 GPU min)
CPU result (G1b units.jsonl, 32 fg tokens): workload a timed, all 10 A and 10 B units identical (no divergence). Workload a identity pass (fixed 4 s arrival): A vs B first divergent index 26; B identity vs B timed also 26, A identity == A timed. Workload b: the "divergence at 0" is a confound (A and B never got the same u-segment), so b is unmeasured.
GPU plan: teacher-forced fg stream (cap0 ids) under real two-slot co-batch, cap0 vs cap1024, same IDs; rows 0..31 (row 0 = prefill). Scale: stock two-slot-vs-solo KL .00306 (C2), deep .00506. Verdict text: "within gate" (max mean KL <= .0010) else "outside gate - reported, not judged". Hypothesis: row-0 KL >> rows 1..31 means prefill-borne (chunk/ubatch composition) divergence.
Limit: no nsys/ncu; kernel names cannot be traced. Evidence is -lv 4 batch logs plus source-level FA dispatch (by Q->ne[1] and n_kv). nvprof in the devel image is untested on this arch.

## Caveats
g1c_rows compiled, not run on GPU. Analyzer selftest reproduced G1 A-vs-B KL .2936 (codex 65536). G1's `n<64` tail rule is not source-backed (TAIL.md).
