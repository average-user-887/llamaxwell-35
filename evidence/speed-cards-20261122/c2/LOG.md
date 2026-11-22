# C2 append scheduling LOG (append-only)
- 04:19 worktree wt-c2 branch claude/speed-c2-20261122 @7ac90aa. Source of card1 server-impl = r3-truncation-a2/server-context.cpp.
- Finding: append chunks already interleave with the other slot's decode (--mixed-batch-size 1024 caps prompt chunk to ub while a slot generates). Remaining lever: S1 owner pick is round-robin; when both slots append, 2048-token batches alternate and both finish late.
- Patch (env, default off): LLAMA_APPEND_SCHED=1 -> S1 owner = eligible slot with fewest est. remaining prompt tokens (STARTED: n_tokens - cached); LLAMA_APPEND_CHUNK=N optional smaller mixed chunk (not used in the A/B).
- 04:21 control rebuild of unpatched source reproduces card1 server-impl bdd39cf9... byte-exact; patched build 7a782656... (-Werror). Commit 997feee.
- Harness c2_abba.py: blocking flock per run + admit_idle_fleet per run, containers r3bench-c2-*, 80 C own-container watchdog, cooldown <=60 C, stop 08:15. Replay copy stores per-token gaps (p95 stall).
- 04:23 queued chain behind C1 lease: timing OFF96,ON96,ON96,OFF96,OFF64,ON64,ON64,OFF64 then MV2 logprob proxy OFF64,ON64.
- 04:26-04:37 runs 1-3 (OFF96, ON96, ON96) all CUDA OOM during the first setup prefill (both arms, so not the patch): -c 2x98304 with f16 KV (3840 MiB) + model 18867 MiB peaks at 24.4 GB. Not valid runs; moved to oom96/. Lease was held for each container's whole life.
- 04:47 chain stopped (no container left), harness switched to 2x84224 (start 63.9K -> ~80K, the largest two-slot depth known to fit, prior step-2 A 176.5 s). Restarted: OFF84,ON84,ON84,OFF84,OFF64,ON64,ON64,OFF64, then MV2 OFF64,ON64. Cooldown start <=60 C (inside the 72 C start rule).

## 04:55-06:14 all timing runs INVALID (80 C)
- 0455_1_OFF84, 0516_2_ON84, 0537_3_ON84, 0559_4_OFF84, 0614_5_OFF64 all reached tmax=80 C ~25 s into the run phase; watchdog removed own container. Moved to hot80/ (kept, excluded from all tables).
- Earlier rc=1 x3 = 2x96K CUDA OOM in setup prefill, both arms (oom96/). Not timing data.

## 06:20 reviewer fixes
- C++: STARTED-slot remaining = n_tokens - LCP(cache, task) (was stale prompt.n_tokens); a rollback can only add tokens, so it can still under-estimate. Anti-starvation: a slot passed over 8 consecutive S1 batches takes the batch (counter starve-overrides in log). Rebuilt ON lib sha256 f9aaae14cb24cc62454d14698dae06de382c3c2b31080b137fb67df2ac99bfab (-Werror clean).
- Harness: C1 pause pacing ported (docker pause own container >=77 C, unpause <=72 C or 15 s, every pause logged in temp.log and recorded). Per run: pause_s (all), pause_run_s (overlap with makespan window), makespan_raw_s, makespan_net_s. Decode-gap p95 still includes pauses (both arms alike).
- Commit c6e3cc2. Chain relaunched 06:24, pid 868630 (driver python 868667), order OFF84,ON84,ON84,OFF84,ON64,OFF64,OFF64,ON64, then MV2 OFF64,ON64; stop 08:15.

## 08:17 final
- Chain finished at the 08:15 stop with 4 valid 84K runs (n=2 per arm). 64K and MV2 not run. Gain +0.70% raw, -0.36% net of pause: REJECT. No r3bench-c2 containers or processes remain; lease free.
