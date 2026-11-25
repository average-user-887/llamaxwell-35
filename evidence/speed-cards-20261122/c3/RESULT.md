# C3 RESULT - Maxwell FA GEMM tile/split tuning (append prefill @80K)

Verdict: REJECT (no env change). Best arm +0.71% at n=1, below the 1% kill line; chain stopped by kill rule at 07:29 CET.
Source: runs/chain/driver.log, runs/chain/*/results.jsonl, scripts/c3_score.py (run with python3 -I; scorer output matches the table below).

## Arm table (append prefill, 4096-token chunk at 81920 ctx; tok/s = mean of 3 appends @80-90K)

| arm | env | append ms @81920 | append tok/s | delta vs B0 (control mean 190.10) | n | decode tok/s (1-slot, single run) | decode delta | tmax C | valid | VRAM peak MiB |
|---|---|---|---|---|---|---|---|---|---|---|
| a0 B0 (card1 lib) | none | 21047 | 189.95 | -0.08% (ref) | 1 | 97.72 | ref | 77 | yes | 22524 |
| d (v1 lib, defaults 64/4096) | none | 20981 | 190.27 | +0.09% | 1 | 97.87 | +0.2% | 77 | yes | 22524 |
| d2 (v1 lib, defaults) | none | 21017 | 190.07 | -0.02% | 1 | 98.02 | +0.3% | 77 | yes | 22528 |
| s2048 | SPLIT=2048 | 20760 | 191.52 | +0.71% | 1 | 90.91 | -7.0% | 77 | yes | 22536 |
| t128 | TQ=128 | 21430 | 186.10 | -2.14% | 1 | 88.10 | -9.8% | 78 | yes | 22728 |
| t128s2048 | TQ=128 SPLIT=2048 | 21192 | 187.63 | -1.34% | 1 | 90.40 | -7.5% | 78 | yes | 22752 |
| s8192 | SPLIT=8192 | 22015 | 181.04 | -4.80% | 1 | 84.93 | -13.1% | 77 | yes | 22522 |

Deltas for sweep arms are vs the v1-default control mean (d,d2 = 190.17), as the driver computes. Noise: three default-config runs (a0,d,d2) span 189.95-190.27 tok/s = 0.17% peak-to-peak. s2048's +0.71% is ~4x that, but n=1 and under the 1% kill line, so not adoptable. All tmax 77-78 C (<80 C), so no run dropped. SPLIT=4096 is not swept as an arm; it is the default. TQ=64/SPLIT=2048 = s2048; TQ=128/SPLIT=4096 = t128.

## Gate verdicts

| gate | result |
|---|---|
| Append prefill @80K >= +3% | FAIL. Best s2048 +0.71%; best TQ arm -1.3%/-2.1%. |
| Kill if <1% after first sweep | TRIGGERED (driver 07:29:17). |
| No regression (decode / 2-slot) | NOT EVALUATED for 2-slot; decode single-run drops of 7-13% on all non-default arms are not conclusive (greedy text changed with SPLIT=2048 and TQ=128 per REVIEW item 6, so token streams differ), but no evidence of a gain. |
| KLD/top-8 proxy (dmean<=+.0010, top>=-.3) | NOT RUN (skipped by kill; proxy would be top-8 prob compare, not true FP64 KLD). Greedy-text change on s2048/t128 means the gate would have been a real question. |
| Attention-share gate >=15% | PASS on measured share: 45.2% measured (FLASH_ATTN_EXT 152470 ms of 337 s per-node-synced setup+appends; per-node sync inflates totals, shares only). The driver gate uses max(measured, projected); projected 80-88K window share is 60.6%, not used here. |

## Steps run vs skipped
- Ran: profile (NODE_TIMING), b0/a0, sweep1 x6 (d, s2048, t128, t128s2048, s8192, d2).
- Skipped by kill rule: ABBA n=2 vs B0, 2-slot AB(BA) at 40K, logprob top-8 proxy, any 2-slot check at production depth.

## Risks / notes
- TQ=128 adds ~200 MB scratch (VRAM 22524 -> 22728/22752 MiB) and is slower; two-slot check never ran at production depth (earlier only 2x48K/40K). Do not adopt.
- SPLIT=2048 changes greedy output and lowers decode in this single run; if revisited, needs n>=3 ABBA and true KLD.
- Scorer/driver quality check is a top-8 proxy only.

## Adoption
REJECT. Keep defaults (no GGML_CUDA_FA_MAXWELL_GEMM_TQ / GGML_CUDA_FA_MAXWELL_GEMM_SPLIT set; i.e. 64 / 4096). If anyone retests: GGML_CUDA_FA_MAXWELL_GEMM_SPLIT=2048 is the only candidate (+0.71%).

## Provenance
Branch claude/speed-c3-20261122 (local), base 7ac90aa, commits 7afb2d3, 2592658.
Build: c3-fa-tune/build.sh (nvcc fattn.cu; link = link-card1.sh with fattn.cu.o swapped).
sha256:
- 20ef19be7eb6903337c28126b7b8f165aff1aef04e83ce52cbcd435e16e2bda6  ctl/out/libggml-cuda.so (functionally B0)
- 5afeaeb484062ac3153e7dd174125a0025c31109e755232ce98604d858140639  v1/out/libggml-cuda.so (tuning lib)
- c08bb3f08af3c0385f438836188845ff535587dad0c4ca63b5310806a65cd59f  orig/out/libggml-cuda.so (card1 lib, reproduced)
- 7efa7a719abe2451f1c123aecba08a52a8c778ef0d84c517b78c1e6dc48ebe36  ctl/fattn.cu.o
- 3d3efe37d6ae1ab4bf2fc82695a8aa1a3e419ea0f991afa87f8fa18e7b0a429f  v1/fattn.cu.o
Note: a0 ran on the card1 lib (c08bb3f0); d/d2 and sweep arms on v1.
