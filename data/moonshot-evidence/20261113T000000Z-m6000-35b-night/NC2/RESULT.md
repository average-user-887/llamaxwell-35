# NC2 result (CPU only; nothing run on GPU, nothing re-run)

| item | outcome |
|---|---|
| 128-row cut-mode bug | found + fixed (doc slice 98368 tokens < 98304+128 needed); test fails on original, passes on fix |
| hardening | analyzer rejects partial runs, wrong vocab/rows, NaN/Inf (exit 2); harness validates plan; C++ bounds-checks |
| tests | `python3 -B test_nc2.py`: 15 tests OK (fixed); original: bounds test FAILS (98432 > 98368) |
| inventory | 122 vectors, 3437.8 MB (G1 69 files 68.5 MB; G1c 53 files 3369.2 MB); 53 files >50 MB; git push: NO |

## Bug
DOCLEN=98368 = 98304+64. Harness wrote exactly DOCLEN tokens per doc; with G1C_NR=128 (cut mode in run_all.sh) the last bucket reads tokens up to index 98431, 64 past the file end. g1c_rows.cpp did `tok[pos]` with no bounds check (UB: garbage/zero token teacher-forced for rows 64..127 at 98K, wrong rows silently). Buckets 32K/65K were fine. The real run (run-20261112T202254Z, steps.log: mode=full nr=64) never hit it, so run.log shows nothing; the defect is latent in the cut-mode path (RUN.md "4 docs x 128 rows").
Fix: slice length = max(DOCLEN, maxB+NR) (doc starts stay k*98368, so A/B 64-row vectors remain comparable); validate_plan() rejects docs that run past the fixture; C++ returns 16/17 on P/R beyond token count; analyzer reads NR/docs from env and run_all.sh exports G1C_DOCS.

## Files
g1c_harness.py, g1c_rows.cpp (recompile needed before GPU use), g1c_analyze.py, run_all.sh (copies), test_nc2.py, inventory.py, manifest.json, MANIFEST.md.
Test usage: `NC2_DIR=<G1c evidence dir> python3 -B test_nc2.py` shows the original failing.
Caveat: the g1c_rows.cpp edit is not compiled (no llama.h here); analyzer not run on the real 3.4 GB set (it would write G1C_TABLE.md into evidence).

## N2 reuse
Arm A and B 64-row vectors, all 8 docs x 3 buckets (24 files each): `<M>/occamy35-day-20261112/evidence/G1c/run-20261112T202254Z/{A,B}/doc{0..7}_b{32768,65536,98304}.f32` (63569920 B each = 64 x 248320 f32). Valid only if N2 uses same lib (sha256 0c7ffbde...330d5, Codex v4), image, model, fixture and base env. Ap doc0 (3 files) also exists there and is reusable; N2 needs Ap for docs 0..7 (or 1..7).
Ap = arm A (BASE_ENV + FA_ENV: GGML_CUDA_OCCAMY_FA=1 FUSED=1 FA_PIPE=3 FA_ADDR=1; fa on, f16 KV, ub 1024, ctx 98560) plus the single extra env `GGML_CUDA_FA_PARALLEL_BLOCKS_MULT=2`. Command (under flock gpu-bench.lock, after unload+idle check, MAXWELL_GRAPHS unset): `G1C_GPU=<gpu> G1C_NR=64 G1C_PER=2 python3 -B g1c_harness.py Ap <OUT> 0,1,2,3,4,5,6,7` (harness in G1c dir or this fixed copy with G1C_DIR set to dir holding g1c_rows). Exact env lists are in G1c run-*/Ap/launch_c0.json.
