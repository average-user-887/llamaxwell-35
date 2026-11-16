# HANDOFF (16 Nov 2026) - M6000 Occamy 35B 48h card set
Root: $W = occamy35-48h-20261115 (on the Storage disk). Decisions: $W/DECISIONS.md.
Publish: $W/publish-all/<card>/ (sanitized, 32 cards, ~8 MB); MV1 bundle: $W/MV1/publish/. Assembler: $W/tools-publish/assemble_publish_all.sh <W>.
Prod: untouched, stays batch8 (d4c13eb). Bundle9 base 52c9e84. No pushes made.

## Results vs gate (1% N3 makespan; KLD dmean<=+.0010, top>=-.3)
- MV1: MTP loses in every arm (np1 and np2) on step cost, not acceptance. Measured ~53 ms/round vs ~34 ms break-even.
  Split: target verify 41.1 ms (FA 16.35), draft-ctx re-eval 8.9, head draft 8.2, sampling 0.4. Confidence HIGH on step cost, MEDIUM on per-op split.
  Greedy B diverges from A (token 26 np1; 18 and 35 np2); 5 rejections/100 rounds, restores=0.
  Peak VRAM MiB np1 A 21174 / B 22165; np2 A 22882 / B 24187.
  Q->ne[1]==1 hypothesis NOT confirmed: FA counters never flushed (atexit lost on container removal); only indirect timing (1.63 vs ~0.6 ms/layer).
  Ranked: 1 multi-row F16 FA for target (new kernel), 2 narrow draft-ctx re-eval (server), 3 fused-MoE window, 4 multi-row q8 FA for head, 5 GDN n-row (<1 ms).
- D1b: run crashed (CUDA error: operation not permitted when stream is capturing; gemm arm not engaged). RESULT.md still says PENDING. Needs diagnosis.
- KQ1: GPU run failed at once (test-backend-ops usage text; micro table all n/a). Needs rerun.
- DC0, FA0, H1, HI0, BX, L0-L2, FK1 and the rest: see each card's RESULT.md in publish-all (DC0/FA0 RESULT.md may lag newer run summary.md).

## Rules
- Owner 16 Nov: VRAM only needs to fit; KV quants OK at equal quality; graphs OK in tests only; Fable never for kernels (Opus codes, Fable advises); no 1660 Ti test without direction.
- Never set GGML_CUDA_MAXWELL_GRAPHS. New behaviour behind flag/env, default off. Placeholders for URL/IP/port/GPU UUID in anything published.
- Keep GPU jobs queued behind the lock; no sleep/poll loops.

## Hard cut: about 20:4x CET 17 Nov. Stop and publish what exists.

## Next
1. Diagnose D1b graph-capture abort; rerun KQ1 with correct test-backend-ops args.
2. Rerun MV1 with FA counter flush (clean shutdown) to settle Q->ne[1]==1.
3. Scope multi-row F16 FA kernel for the target verify (Opus card) and the draft-ctx narrowing (patcher card).
4. Refresh stale RESULT.md files from run summaries, rerun assembler, rerun privacy grep.
Privacy grep: only known false positive is an epoch value (…822172.21) in A5 telemetry.csv.
