# DOWNX extra verification (card C5, 2026-11-10)

Source: main 124599c (DOWNX), model occamy-1.0-Q4_0-headq40b.gguf, Quadro M6000, 250 W cap and clocks untouched.
Image sha256:3a0caa9a (prod), all prod GGML_CUDA_* flags (scripts/c5.sh `E`), GPU bench lock held for every run.
Timing and A/D/G-prod used the prod lib `libggml-cuda-downx35-20261110.so` (sha256 7b4f304ae3fa...).
The debug lib (`src/dxcount.patch` on 124599c, sha256 2b89b13df89b...) was used only for B and G-dbg, never for timing.
Raw logit dumps (63 MB each) stay outside git; `hashes/` has per-step SHA-256 and whole-file hashes.

## A. Raw FP32 logits replay: PASS
`src/dxlogits.cpp`: 64 teacher-forced tokens (BOS + Austen, Pride and Prejudice ch. 1), batch 1, n_ctx 4096, FA on.
Each step dumps the full vocab of FP32 logits (248320). Each arm runs in a fresh process, so there is no sampling and no seed.
The arms dx0, dx1, dx2, dx3 and a dx0 repeat all use the same binary with only the env changed.
In every arm all 64 steps are bit-identical: 15,892,480 values per arm, 0 mismatches, no NaN or Inf.
Every arm's dump has the same file hash, bfba6277...; see `logs/A-cmp.txt` and `hashes/`.

## B. Dispatch counters: PASS
Block 0 / thread 0 of each `moe_v2<PHASE,DX>` increments a zero-copy host-mapped counter (dev). Host launch calls are counted separately (host).
- Harness, 64 steps x 40 MoE layers: dx0..3 each give only `moe_v2<0,dx>` = dev 2560 / host 2560. With OCCAMY_3L=1 and dx1 the result is `<1,0>,<2,0>,<3,1>` = 2560 each.
- llama-bench tg64 r2, dx0..3: only `<0,dx>`, 5160/5160.
- llama-server with prod args, dx1, quick client: `<0,1>`, 7565/7565.
- Finding: the prod M6000 config runs with no CUDA graphs. `ggml_cuda_graph_set_enabled` turns graphs off for cc < 7.0 unless GGML_CUDA_MAXWELL_GRAPHS is set, and prod does not set it. That is why dev == host.
- Opt-in graphs (G, GGML_CUDA_MAXWELL_GRAPHS=1): harness `<0,dx>` dev 2560 / host 120; bench dev 5160 / host 280.
  The graph replays do execute the selected DX variant.

## C. ABBA x5, llama-bench tg128 -r3, prod lib: PASS (+1.2-1.3 %)
| pair | dx0 mean (min-max) | dxN mean (min-max) | n |
|---|---|---|---|
| dx0 vs dx2 | 79.245 (79.191-79.278) | 80.216 (79.958-80.275) | 10/10 |
| dx0 vs dx1 | 79.261 (79.221-79.302) | 80.150 (79.900-80.302) | 10/10 |
SM clock held at 1151 MHz in every run. Max temperature was 66-68 C, mean power was 139-152 W, and throttle reasons were 0x0.
The one exception is a single 0x2 sample (sw power cap). The per-run lines are in `logs/C-abba-*.txt` and the raw nvidia-smi 500 ms traces are in `smi/`.

## D. Server coverage, prod args except --cache-ram 0, dx0 vs dx1: PASS
The greedy 128-token output sha is identical between dx0 and dx1 for every test (`logs/D-srv-dx*.client.txt`):
- short: 4ee1f467
- ~16K depth (15681 prompt tokens): abe5ff19, tg 70.0 / 70.5
- ~64K depth (62991): 5e3941d4, tg 54.0 / 54.3
- two concurrent slots x2 reps: slot0 d31165a9 / slot1 0435a5cc in both modes, about 46-47 tok/s per slot (client agg_tok_s 69.4/71.3 dx0, 71.5/71.5 dx1)
  Caveat: the fused MoE path is batch-1 only, so concurrent two-slot decode probably does not run DOWNX. The test shows equality and no regression there.
- slot save/erase/restore: the continuation after restore (cache_n 226) gives 860ff48b, the same as a direct run, in both modes.
- graph replay equality: prod has no graphs (see B). With opt-in graphs, prod-lib dx0..3 and dbg-lib dumps are bit-identical to A (bfba6277...).

## Reproduce
`scripts/locked.sh "bash scripts/c5.sh A" ...`, with steps A B C D (CDX=1 for mode 1) and step G in `scripts/c5g.sh`.
The endpoints are redacted to SWAP_URL / SRV_PORT; get them from the host's lab facts.
