# M2 RESULT: MTP long-context decode/prefill (run <W>/M2/run-20261115T213435Z)

Decode t/s (1 slot / 2-slot per-slot s1 | aggregate), prefill t/s cold 1-slot. Depths used: 26, 52000, 80000 (4096 for pp). Peak VRAM from 1 Hz nvidia-smi.
| arm | np x ctx | spec | draft_n>0 | dec26 1s | dec52K 1s | dec80K 1s | dec26 2s agg | dec52K 2s agg | dec80K 2s agg | pp4K | pp52K | pp80K | peak MiB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| prod batch8 (served GGUF, ub2048) | 2x98304 | none | n/a | 89.3 | 66.2 | 58.5 | 113.1 | 77.5 | 66.8 | 604.6 | 387.7 | 321.1 | 23867 |
| MTP off (-mtp GGUF, ub1024) | 2x84224 | none | n/a | 89.6 | 66.3 | 58.4 | 112.9 | 77.5 | 66.8 | 552.2 | 368.3 | 306.8 | 22882 |
| MTP n-max 1 | 2x84224 | draft-mtp | yes | 78.4 | 39.1 | 37.2 | 95.4 | 43.6 | 35.6 | 517.8 | 334.4 | 275.2 | 24289 |
| MTP n-max 2 | 2x84224 | draft-mtp | yes | 72.4 | 37.8 | 32.5 | 86.1 | 41.0 | 35.2 | 517.5 | 335.2 | 276.0 | 24314 |
| MTP n-max 3, all np1 arms, np1 98304 | - | - | NOT RUN | | | | | | | | | | time budget (70 min) hit |

Acceptance (drafted/accepted): n1 26:102/88, 52K:76/50, 80K:65/62; n2 26:162/110, 52K:119/67, 80K:112/70. Void rule passed on all run spec arms (drafted>0 every decode row).
Findings: MTP is slower than no-spec at every depth (1 slot -12..-45%, 2 slots -15..-47%); loss grows with depth. MTP prefill penalty vs off: -6% (4K), -9% (52K), -10% (80K); off vs prod: -9%/-5%/-4% (ub1024 vs ub2048). n1 beats n2 at depth.
Counters: no occamy_spec counter exists in the batch8 libs (checked source and .so). Debug arm (dbg_n3_np2) skipped, so no fused-kernel fallback lines at verify rows>1 captured; counters.txt per arm has log greps only (only occamy_fa_gemm fallback=0 line seen earlier).

## Method
m2.sh (detached, flock 9 on gpu-bench.lock, POST unload qwen36-35b-moe, containers nm2-, trap cleanup; 250 W untouched, MAXWELL_GRAPHS unset). Client m2_client.py: real stdlib python text, cache_prompt false, temp 0, ignore_eos; 1-slot rows n_predict 192/128 (pp and tg from same request); 2-slot rows stream both slots, tg = slot1 clean decode (slot0 still decoding), s0 window = slot0 tokens inside slot1's decode window; agg = sum. MTP arms: F16 target KV, q8 draft KV, ckpt 2, ub1024, cache-ram 12288.
## Caveats
Single run per arm, no repeats. 26 row is a 28-token prompt (pp there is meaningless). 2-slot window short (4-16 s). Skipped arms: n3_np2, dbg_n3_np2, all np1 arms (incl. 98304).
## Files
<W>/M2/run-20261115T213435Z/<arm>/{result.json,client.out,server.log,counters.txt,vram.csv}, steps.log; scripts m2.sh, m2_client.py in <W>/M2

# M2b (skipped arms + repeat; run in LATESTB dir, runb-*)
Decode t/s 1 slot at 26 / 52K / 80K [draft_n/acc at 80K]; 2-slot aggregate 26 / 52K / 80K; pp 4K / 52K / 80K; peak MiB.
| arm | dec 1s | 2s agg | pp 4K/52K/80K | peak |
|---|---|---|---|---|
| n3_np2 | 69.7 / 35.7 / 40.1 [114/88] | 57.7 / 32.2 / 27.2 | 519 / 335 / 276 | 24440 |
| n2_np2 repeat (M2: 72.4/37.8/32.5; 86.1/41.0/35.2) | 72.6 / 37.8 / 32.5 [112/70] | 86.3 / 41.1 / 35.4 | 524 / 337 / 277 | 24415 |
| off_np1_84224 | 88.5 / 66.3 / 58.3 | n/a | 547 / 367 / 306 | 21174 |
| n1_np1_84224 | 78.7 / 39.1 / 37.2 [65/62] | n/a | 523 / 336 / 277 | 22165 |
| n2_np1_84224 | 73.2 / 37.8 / 32.5 [112/70] | n/a | 524 / 335 / 276 | 22228 |
| n3_np1_84224 | 71.2 / 34.1 / 40.2 [114/88] | n/a | 527 / 336 / 276 | 22391 |
| n2_np1_98304 | 72.3 / 37.8 / 32.4 [112/70] | n/a | 519 / 336 / 277 | 22600 |
| off_np1_98304 | 88.7 / 66.1 / 58.5 | n/a | 555 / 368 / 307 | 21476 |
| dbg_n3_np2 (short rows) | 70.8 / - / - | 57.3 / - / - | 519 | 24440 |
Noise: n2_np2 repeat matches the first run within 0.3 t/s (<1%) on 1-slot and 2-slot rows; run-to-run noise is tiny, so MTP deficits are real. 98304 per slot fits np1 with MTP (22600 MiB). n3 is worse than n1/n2 at 26 and 52K, and in 2 slots (agg 27-58); at 80K 1-slot it is higher than n2 (40.1 vs 32.5, acceptance 88/114) but still below MTP off (58.3). Drafted>0 on every spec arm. No arm void.
Fallback/verify capture (GGML_CUDA_OCCAMY_DEBUG=1, dbg_n3_np2): occamy_fa_gemm engaged=0 fallback=0; log shows "occamy-gdn: window rejected: unexpected cpy" and occamy-sf2 fusion lines (mtp_gate node fused 212x); no per-kernel verify-row>1 fallback counter exists. Details in runb-*/dbg_n3_np2/server.log.
Fallback detail: GDN fusion window rejected at the MTP/verify graph ("unexpected cpy at node 19", and "n_rs != n_seqs or noncontiguous ids at node 7"); occamy-fused windows rejected (unexpected mul_mat, SCALE, ROPE, no residual ADD within 128 nodes). These are the fused-kernel fallbacks at the draft/verify graphs.
