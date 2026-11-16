# FA0 result (pre-gate; no GPU run done; CPU build only)

| item | value |
|---|---|
| Task 1 FA share of N3 wall | 18.8% (L2 summary) vs 13.5% (raw 21.5 s / 159 s): right basis is ~13.5-13.8% (see below) |
| Task 2a split of FA time | NOT MEASURED (GPU not run). Instrument: GGML_CUDA_FA_SPLIT=<file> (cudaEvents per phase) |
| Task 2b knob sweep | NOT MEASURED. Rows: conv block 128/512, conv grid 2048, TILEQ 32, SPLITK 2048/8192, CHUNKQ 512, FUSED softmax, FUSED+SPLITK 2048 |
| Go rule (non-GEMM >=25% of FA time, or any config >=1.05x) | pending GPU run |

## Task 1: denominators
- 18.8% = 69.9% (FA share of prefill-phase node time, 21478 / 30745 ms) x 40 s (ASSUMED prefill-only wall) / 149 s (prod N3 wall, assumed 40+60+49).
  It mixes bases: a node-time share from the timing arm (sync per graph, non-captured, slower) is applied to wall seconds of the prod arm. The same 69.9% x 40 s = 28.0 s, but only 21.5 s of FA was actually measured.
- 13.5% = 21.478 s / 159 s. Measured timing-arm run: makespan 163.2 s; summed timed node ms = 13642 + 30745 + 111534 = 155.9 s. So 21.478/155.9 = 13.8% of timed node time, 13.2% of the 163.2 s makespan.
- Right denominator: same-basis measured total (timed node time 155.9 s or timing-arm makespan 163 s) -> FA = 13.2-13.8% of N3 wall in the timing arm. 18.8% is an over-estimate (assumed phase walls are not measured). Caveat: timing arm inflates small/mixed (111.5 s timed vs 60 s assumed) so on prod the FA share of wall is probably somewhat higher than 13.5% but cannot exceed ~19%; the 25%-of-FA go rule is not affected, the ">=15% of N3 wall" statement in ANALYSIS is borderline-NO on the measured basis.
- Within the prefill phase, FA = 69.9% of prefill graph time is a clean same-basis number.

## What "FA" is in prod prefill (important)
Prod sets GGML_CUDA_FA_MAXWELL_GEMM=1: the prefill FA node is NOT one kernel. It is per (kv-head group x 64-query tile): pack, convert F16 K/V to F32 (per chunk per group), cublasSgemm QK (FP32), masked softmax kernel, cublasSgemmBatched/Sgemm PV (FP32), reduce. So the achieved 4.2 TFLOPS (62% fp32 peak) is mostly cuBLAS SGEMM; "QK GEMM / softmax+mask / PV GEMM / F16 K,V load-convert" map 1:1 to launches, so per-phase cudaEvent timing is used instead of in-kernel clock64.

## Build and run
- branch fa0-split (worktree FA0/src) off l2-nodetime 3173bef; knobs default to prior behaviour; split timing default off.
- run: N3_GPU=<gpu> N3_UNLOAD_URL=<lab unload endpoint for qwen36-35b-moe, POST> N3_PORT=<p> N3_HOST=<h> bash FA0/fa0_gpu.sh ; then FA0/run-*/summary.md (LATEST). ~12 N3 short replays (A, 9 configs, A2).
- Scratch VRAM: TILEQ 128 rejected (would raise score scratch); TILEQ 32 lowers it; others unchanged. SPLITK changes only fp32 summation grouping of PV partials (reorder-level; needs the KLD/reorder-floor check before any adoption).
