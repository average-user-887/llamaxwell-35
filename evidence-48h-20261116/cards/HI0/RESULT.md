# HI0 host-idle attribution (CPU analysis of existing L1 + H1 + DC0 data; no new GPU run, no code change)

Verdict: the "~9 s post-decode gap" is NOT in the N3 wall. The "~2 s post-prefill gap" is ckpt_create, mostly GPU work. Real host-only cost in the wall is ~0.8 s (0.5%).

| item | s per replay | in N3 wall (160.5 s)? | split | evidence |
|---|---|---|---|---|
| 9 s "post-1-row-decode" idle window (L_2: 9.18 s) | 9.1 | NO | slot RESTORE http 3.56+3.52 = 7.08 s (task_proc max 3.56 s = restore) + 2.0 s t0 lead (t0 = now+2) | window ends at t0+0.1 s (run.json t_run0 vs ledger row 7); restore+warm is done before t0 (n3_replay_*.py run_session); makespan 160.49 = last t_end 160.5 from t0. L_1 18.4 s = cold-page-cache restore 12.7 s. |
| post-prefill gap (6/4/3 windows >50 ms, max 2.0 s) | 5.0-7.0 | YES | ckpt SAVE (ckpt_create) 7.65 s total/replay (20 ckpts, 62.8 MiB each): GPU sync wait 6.84 s + pageable copy 0.81 s | H1 ledger prod arms: ckpt_wait_gpu 6.84 (max 2.0), ckpt_copy 0.81-0.82 (max 0.05). Wait = last async no-output prefill ubatch still computing: GPU busy, not idle. |
| ckpt restore in wall | 0.00 | - | cache_match_ckpt n=9 sum 0.0 s; 0 "restored context checkpoint" in prod server.log | all 8 requests are append (cache hit) |
| host memcpy (pageable) | ~0.81 | YES | 20 x ~41 ms (~1.5 GB/s) | ckpt_copy |
| GPU sync wait | ~6.8 (GPU-busy) | yes but not idle | real GPU idle ~= host time between ubatch finish and next submit, ~0.1-0.2 s est. | ckpt_wait_gpu |
| scheduler/other | decode-after remainder ~2.0 s + post_decode 0.8-1.3 s over ~1900 decodes (~1 ms/step) | YES | per-step host work overlapped with async GPU decode; not separable as idle without GPU timeline | L1 summary |

## H1 coverage
H1 (pinned async batch copy, flag LLAMA_H1_CKPT=1) only attacks the 0.81 s copy. It does not touch: the sync wait (unavoidable GPU work), the number of ckpts, restore, or the 9 s fixture window. Measured so far (n=5 runs, ckpt_copy): prod 0.81/0.82, h1 1.04/1.03/1.26 s: H1 is not faster (extra copy into the vector still pageable memcpy), expect H1 net <=0 on N3.

## Candidate fixes (expected s saved of 160.5 s)
| fix | saves | note |
|---|---|---|
| A. skip ckpt saves nobody restores (N3: 0/20 restored; e.g. no save when prompt is pure append and ckpt exists <2048 tok earlier, or save lazily at slot release) | 0.8 s copy + ~0.1-0.2 s GPU bubble = ~1.0 s (0.6%) | default-off flag; must keep prod rollback correctness (ckpts needed after edits/regenerate). Largest, simplest |
| B. device-resident ckpt (D2D copy to a VRAM ring, DtoH deferred/never) | up to ~0.8 s | ~63 MiB each; 2 slots x 2 = ~250 MiB exceeds peak VRAM budget (prod +100 MiB); only 1 per slot fits |
| C. overlap restore with something | 0 in wall | restore is outside makespan in N3; in prod it blocks the slot ~3.5 s per 84K restore -> real-use win, not N3 |
| D. true async copy on side stream, sync only when ckpt is consumed | <=0.6 s | H1 did NOT achieve this (it syncs once then memcpys) |
Ceiling for all ckpt work in-wall: ~1 s (0.6%). Below the 1% gate: do not open an Opus card; fold A into a batch with other small gains.

## Not done
No timing-counter branch/worktree, no hi0_gpu.sh: existing L1+H1 ledger stages already separate save/copy/wait/restore. Not separable without GPU timeline (nsys): GPU idle share inside the 6.8 s ckpt_wait_gpu and the ~2 s decode "remainder".
