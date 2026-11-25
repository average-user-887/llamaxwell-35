# QA prefill card (R3 bundle, combined-r3-card1): STOP at step 2

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
