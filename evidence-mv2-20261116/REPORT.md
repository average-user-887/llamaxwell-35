# MV2: why 35B native MTP1 is slower than no-spec at ~80K (Quadro M6000 24G, sm_52)

Branch: mv2-verification-diagnosis-20261116 (evidence only, not merged). Date: 2026-11-16.
Builds on MV1 (evidence-48h-20261116/MV1). All numbers below come from raw/ in this directory.

## 1. Setup (exact)
- Fork llamaxwell. Base MV1 commit a12c0fa; MV2 = 931462b (counter registry, periodic flush) + 424828b (include fix). Patch: patch/mv2-counters.patch (a12c0fa..424828b). Prod reference build: batch8 d4c13eb (not touched).
- Binaries: libggml-cuda.so sha256 a0dea5eb561b277bc959a070f1cd5ff9474cbf436dbb2502a80e05a8f057a86e; libllama-server-impl.so sha256 332c6c1e57e3c4d19a5fd4a552014e781b69afd407a41483afa8c01e7ee3d4d2; launcher 592908f59ff8496ed5800493cbe1ca175a5d7aa447ce84abff632874196813c5. Build container image sha256:3a0caa9afea12f5ca5d982621cc2d86b1fa6ac614d735b262fa475d7dc6e2a6d.
- Model: occamy-1.0-Q4_0-headq40b-mtp.gguf (Qwen3.6-35B MoE, native MTP head), sha256 d5ff985eea0bcb3a7305a2033582bf63f756c9f4b3700d96b4939f81ee041eae.
- Hardware: Quadro M6000 24 GB (<GPU_UUID>), driver 580.178.04, power limit 250 W. Observed SM clock 1075-1151 MHz, mem clock 3304 MHz, power 75-77 W at idle sample points (vram.csv, 5 s cadence). Peak VRAM: np1 A 21174 MiB, np1 B 22165 MiB, np2 B 24187 MiB (of ~24576: tight).
- Server (non-production container, single dedicated GPU, harness lock held): `-ngl 999 -np N -c 84224*N -b 2048 -ub 1024 --mixed-batch-size 1024 --prefill-single-stream --checkpoint-min-step 2048 -fa on --jinja --metrics --cache-ram 12288 --ctx-checkpoints 2 -lv 4 --override-kv tokenizer.ggml.pre=str:qwen35`. MTP arm adds `--spec-type draft-mtp --spec-draft-n-max 1 --spec-draft-type-k q8_0 --spec-draft-type-v q8_0`. GGML_CUDA_* env identical to the prod batch8 launcher (GGML_CUDA_MAXWELL_GRAPHS unset). Full commands: harness/mv2_gpu.sh.
- Fixtures: long = ~80000-token prompt (public stdlib-corpus text), greedy, 384 new tokens, 1 cold + 3-4 cached repeats (prompt_n=4, decode only). np2 = two slots x 84224 ctx, two 80K prompts. short = 2000-token stdlib corpus ("corp") and seeded random hex words ("rand", random.Random(7)). Rejection-heavy "hot" = corp prompt, temperature 1.5, top_k 0, seeds 1234/99, 192 tokens. Client: harness/mv2_client.py.
- Instrumentation: GGML_CUDA_MV1=1 counts only; =2 adds per-FA-call cudaEvent timing; GGML_CUDA_MV1_OUT file rewritten every 256 events and on exit (proof: 297-line file after `docker stop`, both smoke arms and all long arms). LLAMA_MV1_TRACE=1 host timers every 25 rounds. GGML_CUDA_NODE_TIMING(+_SPLIT=1) = per-op phase table (dec2 = 2-row target verify graphs, small = draft graphs).
- Timing definitions: "uninstrumented" = no counters or counters mode 1 (A0/B0 vs A1/B1 agree within 0.3%, so counters mode 1 is free). Instrumented (mode 2 + node timing, arm np1_trB, 33.2 tok/s) is used only for per-op shares, never for throughput. Graph tag gN/tM: M = max MUL_MAT src1 ne[1], not the token count (draft g103: t64 = 1-row head step, t128 = 2-row ctx re-eval; target g3936 t2 = 2-row verify).

## 2. Dispatch proof (counters, ~80K ctx; raw/run-long/np1_B1, np1_A1, np1_trB counters.tsv)
| Path | Rows | KV | FA kernel | calls |
|---|---|---|---|---|
| target verify (10 attn layers) | 2 | F16 | stock_tile (NOT occamy_splitk, NOT maxwell_gemm) | 8230 = 823 rounds x 10 |
| target no-spec decode | 1 | F16 | occamy_splitk | 16270 |
| draft head step (1 layer) | 1 | Q8_0 | stock_vec | 828 |
| draft ctx re-eval (1 layer) | 2 | Q8_0 | stock_vec | 823 |
| target/draft prefill | 1024 | F16 / Q8 | maxwell_gemm / stock_tile | - |
Also at 2 rows in the verify graph: MoE window matched (nt=2, 40/round) and fused GDN (30/round) are active. Per round: one 2-row draft graph + one 1-row draft graph. No occamy FA kernel is used by the draft nor by the 2-row verify.

## 3. Throughput (uninstrumented, tok/s)
| Config | No-spec A | MTP1 B | Acceptance |
|---|---|---|---|
| np1 80K greedy | 58.0-58.15 (5 arms) | 37.24-37.37 (3 arms) | 0.969 (1.969 tok/round) |
| np2 80K (aggregate) | 67.45 (33.4+33.7) | 38.67 (19.6+19.2) | 0.949 / 0.741 |
| 2K corp | 79.4 | 73.4 | 0.79 |
| 2K rand | 80.3 | 77.9 | 0.90 |
| hot, temp 1.5 (192 tok) | 77.6 / 77.5 | 39.5 / 39.4 | 0.0215 / 0.016 |
Break-even at np1: no-spec step 17.2 ms; with 1.969 tok/round a round must be <= ~33.9 ms. Measured round ~52.9 ms (= 1.969/37.3).

## 4. Where the round goes (np1, 80K, per round)
Uninstrumented host timers (mv1_host.txt, steady 250-round window): target decode 37.07 ms (1.02 calls/round), draft 14.85 ms, spec_process 0.60, sample/accept 0.36, restores 0. Sum 52.9 ms. np2: target 33.5 ms, draft 14.0 ms per round. Under the synchronising instrumented arm host attribution shifts (draft 8.1 / spec_process 8.9); only sums are comparable.
Node timing (instrumented, np1_trB/nt.tsv):
- Verify graph (308 graphs, 39.8 ms each): FLASH_ATTN_EXT 16.4 ms (1.63 ms/call x10, stock_tile), fused MoE window 8.0, attn_qkv 1.6, GDN fused 1.4, output 1.3, rest ~11.
- Draft graphs (623): FA = 4558 of 6414 ms (71%). Event-timed: 2-row Q8 stock_vec 7.64 ms/call, 1-row Q8 stock_vec 5.58 ms/call; sum 13.2 ms/round vs 14.85 ms measured uninstrumented.
- References: 1-row F16 occamy_splitk 0.62 ms/call (10 layers = 6.2 ms, the no-spec cost); 4-row stock_tile 2.22 ms/call. The 80K x 2-head Q8 KV is ~83 MB, ~0.3 ms at memory roofline, so the draft FA is ~20x off roofline.
- Hot arm: with ~2% acceptance B runs at ~half the no-spec rate, i.e. a round costs ~2x a no-spec step even when nothing is accepted. acc_hist=[430,355,0,0], restores=0 (no rollback restore path was exercised; rollback cost is not separately measured and is not claimed).
Conclusion supported by dispatch + timing: the loss is the verify FA (+~10 ms over a 1-row step) plus the draft attention (~13 ms) both running generic kernels; MoE/GDN at 2 rows are not the bottleneck. Even with a free draft, verify alone (37.1 ms) exceeds the 33.9 ms budget, so the target multi-row FA must also improve.

## 5. Output divergence (greedy, 80K, 1 slot)
- Determinism: A cold == A rep0 == A rep2; B cold == B rep0 == B rep2, B identical across servers. At np2, cold and cached outputs differ even for A, so output depends on cache/chunking path regardless of speculation.
- First differing token: index 26 (A id 12 '-', B id 10 '+'); 72/384 tokens match afterwards. A's candidates there (decode shape): '-' -0.418, '+' -1.104.
- Forced-prefix probe on the no-spec server with B's first 26 tokens as prefix (raw/run-div/dvA/result.json):
  | shape | '-' | '+' |
  |---|---|---|
  | full chunked prefill (80026 tok, repeated twice, identical) | -0.3826 | -1.2808 |
  | 2-row cached (verify shape) | -0.3655 | -1.2406 |
  | 1-row cached (decode shape) | -0.5257 | -1.005 |
  Shape alone moves the '+' logprob by ~0.28 nat and the gap by ~0.5 nat, but '-' wins in every shape: no-spec never selects B's '+'.
- 2K fixtures: corp first diff at index 41 (near-tie, decode -1.521 vs -1.654), rand at index 1 (gap 0.015 nat). Same-token logprob differs 0.05-0.5 nat between decode and prefill shapes at 2K.
- Verdict: batch-shape numerics are large enough (0.3-0.5 nat) to explain flips at near-ties, and the 2K flips are genuine near-ties. The 80K index-26 flip (gap 0.7 nat in decode shape) is NOT reproduced by any no-spec shape; so it is unresolved between (a) MTP-path numerics (different verify KV/GDN state rows) and (b) a spec state/KV/acceptance bug. B's top-n candidates are unavailable (n_probs under MTP returns 2 entries). Correctness of MTP1 output is NOT established. Needed: teacher-forced logprobs from the B server's verify path (dump verify-row logits) at index 26, and an MTP run with GDN/KV snapshot comparison vs no-spec at that position.

## 6. Fix ranking (estimates from per-call numbers; NOT measured end-to-end, NOT additive)
1. Draft KV to F16 (flag only: `--spec-draft-type-k f16 --spec-draft-type-v f16`). Should move the 1-row step to occamy_splitk (~0.62 vs 5.58 ms) and the 2-row re-eval to stock_tile F16 (~1.63 vs 7.64 ms): up to ~-11 ms/round event-timed, plausibly -8..-11 real; round ~42 ms, ~46 tok/s, still below 58. Memory +~77 MB/slot; np2 peak already 24187 MiB, so margin risk; fit not claimed.
2. Multi-row F16 target FA kernel (2-4 rows): 16.3 -> ~6-8 ms, saves ~8-10 ms; floor is the 1-row 6.2 ms. Needed to get under budget.
3. Remove the redundant 2-row draft ctx re-eval graph: ~3-4 ms after fix 1 (~8 ms before); overlaps with fix 1.
1+2: round ~32 ms (near break-even, ~+5% best case). 1+2+3: ~28.5 ms, ~69 tok/s (+19%) as an optimistic upper bound. Multi-row Q8 draft attention is an alternative to fix 1 if F16 draft KV does not fit at 2 slots.

## 7. Next experiment (priority order)
1. F16 draft-KV arm: np1 and np2, same fixtures, MV2 counters mode 1 then B0; confirm dispatch moves to occamy_splitk/stock_tile and measure real round time and VRAM peak (<= ~1 h GPU).
2. Log verify-row logits at divergence index 26 under MTP (correctness question, section 5).
3. Scope the multi-row F16 target FA kernel (kernel work) with break-even target <= ~8 ms for the 10-layer verify FA.

## 8. Limitations
Single model/GPU/prompt family; per-op shares come from the instrumented arm (33.2 tok/s) and may shift under uninstrumented overlap; rollback cost not isolated (restores=0 observed); fix savings are arithmetic, not measured; 2x96K not tested and not claimed to fit; the corpus prompts are regenerable from the public stdlib text and a seeded generator (see client).

## 9. Layout
raw/run-smoke, raw/run-long, raw/run-div: per arm result.json (outputs as token ids), counters.tsv, mv1_host.txt, vram.csv, sanitized server.log, steps.log; np1_trB/nt.tsv node timing. harness/: scripts (sanitized, env placeholders). patch/: MV2 counters patch.
