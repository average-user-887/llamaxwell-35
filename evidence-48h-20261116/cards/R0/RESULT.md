# R0 bundle9 (build only, no GPU run)

| item | value |
|---|---|
| branch | bundle9 (worktree R0/src, off night15-gdn28h 54efa6d) |
| commit | 52c9e84 |
| libggml-cuda.so sha256 | 2bf2df8a2340ae842458cac43569b2d275bc93fc98d3c4aef3605b64bd7de538 |
| llama-server sha256 | 592908f59ff8496ed5800493cbe1ca175a5d7aa447ce84abff632874196813c5 |
| other shas | R0/RECEIPT.txt |

| flag | engagement counter (stderr, needs GGML_CUDA_OCCAMY_DEBUG=1) | merged |
|---|---|---|
| GGML_CUDA_MOE_V3=1 (already in batch8) | "occamy-fused: MOE_V3 launches N" | yes (base) |
| GGML_CUDA_GDN2_A2ROW=1 (28h) | "occamy-gdn: n2gdn variant A2ROW ...", "n2gdn launches N" | yes (base) |
| GGML_CUDA_OCCAMY_GDN18=1 (card 18) | "gdn18 engaged launches N", "gdn18 variant K blocks/SM ... on/OFF" | yes (conflict in occamy-gdn.cu: gdn_fused2 now <A2, IL>, template plumbing only) |
| GGML_CUDA_OFA_RAGGED=1 (card 14) | "occamy FA ragged on: engaged=N" | yes |
| GGML_CUDA_OFA_RAGGED_RATIO=2 (new guard, 0=off) | "occamy FA ragged guard: Lmax= Lmin= ... -> ragged\|stock split", "ragged guard fallbacks=N" | yes (new, 52c9e84; host sync every 64th ragged call) |
| --mixed-batch-prefill-max N / LLAMA_ARG_MIXED_BATCH_PREFILL_MAX (card 15, default off) | mixed-batch size stats log (server-context.cpp) | yes |
| 28b overlap | - | NO: no in-tree flag/branch (microbench ~+1%, stopped) |
| 28c fused exp-on-load PV | - | NO: microbench 0.92-0.99 vs gate <=0.90, not integrated |
| 28e MoE FP32 SGEMM | - | NO: 1.09x vs 1.25x gate, no branch (plan row A4) |

Build: `bash R0/build-r0.sh` (incremental, docker, sm_52). 0 errors. Strings for ragged/guard/gdn18/A2ROW verified in lib. GPU not run.

## Run commands (orchestrator; takes gpu-bench.lock)
N3 ABBA x3, A=batch8 prod libs, B=bundle all flags (ragged, ratio 2, GDN18, A2ROW):
`N3_GPU=<id> N3_UNLOAD_URL=<.../api/models/unload/qwen36-35b-moe> bash R0/gpu_r0.sh`
Add the cap2048 + gap arm C (PMAX default 1152): `R0_ARM_C=1 ... bash R0/gpu_r0.sh`
both_mixed (98K+30K) decode row: card 14 harness $N2/14/n14_dec.py with N14_UNITS=both_mixed; it hardcodes arm env (B = RAGGED only, one lib), so adapt ARM/lib to the R0 build and flags before use (not done).

## Risks
- gdn_fused2 template merge not GPU-verified; use GGML_CUDA_OCCAMY_GDN18_CHECK=N bit-equality.
- ragged guard adds a periodic stream sync.
