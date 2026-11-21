# QA prefill card 2 (R3 bundle, combined-r3-card1, ctx 2x65536)

| step | what ran (denominators) | gate | result |
|---|---|---|---|
| S1 OOM diag + ABBA | N3R 2 slots x 6 turns, start 45232 -> ~61.4K, -c 131072; runs B1,B2,A,B,B,A (6/6 valid, tmax 77-79 C, 0 trips) | B >= 1% faster makespan | FAIL: A 144.29 s (n=2, 144.22/144.35) vs B 149.41 s (n=2, 149.26/149.55) = -3.55%. B fits at 2x64K (0/6 OOM). Outputs deterministic per arm; B matches A on 3/12 turn texts (ub change diverges tokens) |
| S2 ctx-checkpoints 8 vs 2 | BFCL preview 40 native, paced, ub1024, -np 1 -c 84224; 471 vs 462 req; tmax 78/78 C, 0 trips, 0 infra | fewer forced reprocesses, parity explained | PASS: forced full reprocess 8/471 req (112.7 s) -> 0/462 (0 s); wall 1871 -> 1762 s (-5.8%, pauses 136/120 s). 32/40 samples byte-identical request sequences; the 8 that differ are exactly the 8 miss samples (full reprocess vs checkpoint restore numerics). Score 14/40 -> 15/40, discordant 3 (2 up, 1 down). Checkpoint size 67-146 MiB each (host RAM), up to 8 per slot |
| S3 logprob parity (PROXY, not KLD) | serial a1 path, 5 req (miss_param_1, base_5, long_context_74, base_63 T1+T2), MV2 top-8 logprobs decode steps 0-96, ref A1 (ub1024 ckpt2); tmax 75 C | top-1 agree, mean/max abs dlogprob vs same-build A | A2 floor: 471/471 steps agree, dlp 0/0. C8 (ckpt 8): 471/471, dlp 0/0, identical. B (ub2048+V5): diverges on 4/5 req (first div at steps 5, 9, 4, -, 61), 162/166 aligned steps agree, mean 0.0173 max 0.495 |
| S4 BFCL 160 | full 160 native, ckpt8 ub1024 -np 1, paced (208 pauses, 441 s), tmax 78 C, 0 trips; 1908 req; 160/160 resolved: 132 scored + 28 model_cap, 0 infra; worker wall 6732 s | rel drop <= 5% vs serial AND discordant excess <= floor (0, existing R2-vs-R3 serial evidence) | FAIL (strict): 69/160 vs serial 71/160, rel drop 2.82% (pass) but discordant 30 (16 down, 14 up) > floor 0. Same discordance as the R3 native ckpt2 reference (29 vs serial, 66/160, drop 7.04%), so it is native-vs-serial, not ckpt8: ckpt8 vs native ckpt2 = 7 discordant (2 down, 5 up), 69 vs 66. Forced full reprocess 44/1924 -> 0/1908. Caveat: reference native had GRAPHS=1, this arm strips it; prompt eval 3774 s vs 3674 s ref |

Recommendation: adopt --ctx-checkpoints 8 on arm A (ub1024, no V5); reject arm B (ub2048+V5). S4 strict gate fails on the discordant clause, but that clause also fails for the current native ckpt2 prod path; ckpt8 improves on it (69 vs 66, drop 2.82% vs 7.04%). The native-vs-serial discordance (about 30/160) is the open item.

S3 note: B single-request prefill is 8.1% faster (4 T1 prompts 32.64 s vs 35.51 s), but its tokens diverge and its two-slot makespan is worse (S1). This is a proxy parity gate; no KLD vs FP64 ran.
S1 OOM attribution: compute buffer is set by -ub, not V5: ub1024 344 MiB (A, B2), ub2048 688 MiB (B, B1). VRAM peak ub1024 23233 MiB, ub2048 23793 MiB (+560 MiB) at 2x64K. At 2x84224 the KV grows by about 730 MiB, which puts ub2048 past the 24 GB card, so the card-1 OOM comes from ub2048 working memory (compute buffer + FA pool), not from V5. No FA-scratch patch was built because B loses on speed even when it fits.
S1 split: deep setup prefill (45K tok/slot) B 112.8 s vs A 119.9 s (-5.9%), B1 113.7, B2 119.1 (so the setup gain is from ub2048; V5 adds 0). Two-slot append phase is slower with ub2048, and makespan is worse. B2 (ub1024+V5, n=1) 149.39 s is also slower than A.


---
# Card 1 (previous)

| step | what ran (denominators) | gate | result |
|---|---|---|---|
| 1 prefix reuse | BFCL R3 native log: 1924 req, 12.29M prompt tok, 1.31M processed (89% reused); serial 1931 req | reuse working, else locate | PARTIAL: 44/1924 req (13% of 3674 s prefill) are forced full reprocess (hybrid/recurrent, divergence before both kept checkpoints; ctx-checkpoints 2, min-step 2048). Not slot/id_slot (np=1), not cache-ram. Live N3R: every append turn cache_n = prior depth (12/12 turns, both A runs) |
| 2 two-slot makespan | ABBA on N3R (2 slots, 6 turns each, 63.9K->~80K, ctx 2x84224) | B >= 1% faster | FAIL: B not runnable. A 176.62 / 176.30 s (setup 2x191 s). B runs 2 and 3: CUDA OOM (cuMemCreate in ggml_cuda_pool_vmm::alloc from ggml_cuda_flash_attn_ext) at 2nd ubatch of the first 64K prefill. 0/2 B valid |
| 3 KLD | not run | - | - |
| 4 BFCL 160 | not run | - | - |

Arm A: BFCL native env minus GGML_CUDA_MAXWELL_GRAPHS, OCCAMY_DEBUG unset; -np 2 -c 168448 -ub 1024 --mixed-batch-size 1024 -b 2048 --cache-ram 0 --ctx-checkpoints 2, draft-mtp n1 f16. Arm B: same + -ub 2048 + GGML_CUDA_FA_MAXWELL_GEMM_V5=1.
A buffers: KV 3290 MiB, compute 381 MiB. B: compute 761 MiB, then the FA pool grows at runtime and OOMs (model 18867 MiB).
A-A: token run_sha identical (04e05e5605cb), makespan spread 0.18%. 8/12 turns are flagged void by N3R in both A runs (same fixture behaviour at n-predict 160; identical across runs).
Thermal: max 77 / 78 C in A runs (pacing cooldown <=55 C before each run); no run reached 80 C; watchdog did not trip.
Arm A FA GEMM counters: engaged=1500 fallback=4800 (ne3:4406, q_lt_64:334, keys_lt_4096:60).

## Step 1 detail (native; serial similar)
cold first turn 171 req 1884 s (51%); reuse-miss full reprocess 44 req 482 s (13%); warm partial >256 tok 87 req 360 s; warm small append <=256 tok 1622 req 948 s (26%, ~0.58 s/req for ~42 tok).
Minimal fix proposal (no code): --ctx-checkpoints 8 (about 70 MiB each) so an early checkpoint survives before the divergence point; verify by BFCL log replay.

## Risks
- GGML_CUDA_MAXWELL_GRAPHS removed from both arms (system rule), so arm A is not byte-identical to the BFCL recipe.
- The handoff says to finish BFCL quality before 2-slot work; the owner-approved card order was followed.
- No model-level KLD vs FP64 harness for the card1 libs found; existing rows tools (g1c_rows, kq1_rows) are linked to other builds, and the reference is prod/F16, not FP64.

## Files
step2/{steps.log,results.json,<n>_<arm>/{server.log,n3r/run.json,n3r/turns.md,temp.log}}, scripts/, libs.sha256
