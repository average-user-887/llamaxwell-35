# Adversarial review: speed cards C1/C2/C3 (read-only, 2026-11-22 ~06:20)

Ranked by severity. All numbers recomputed from raw run.json / results.json(l) / steps.log.

## S1 (critical) C2 has no thermal pacing; every 84K run touched the 80 C hard limit
- c2/scripts/c2_abba.py:65-78 Watchdog only kills at >=80 C; no 77/72 pause (C1 and C3 have it). Start cooldown <=60 C only (:81-87).
- c2/runs/steps.log: runs 1-4 OFF84/ON84/ON84/OFF84 all `tmax=80 invalid=True` (04:59-06:04). Zero valid C2 data; chain still running at 64K with the same watchdog.
- Fix: stop the C2 chain between runs, port C1's pause() (c1_mtp.py:72-79) or C3's Watch (c3_append.py:85-90), discard everything in c2/runs.

## S2 (critical) C1 64K "+2.95% makespan / +7.9% decode" is a pacing + output-divergence artifact
- makespan includes docker-pause time: pauses recorded in results.json fall inside [t_run0, utc_end].
  pause s inside makespan: OFF 33.3 / 35.5, ON 24.1 / 30.2. Makespan minus in-window pause: OFF 137.0 / 134.2 (mean 135.6), ON 134.6 / 141.4 (mean 138.0) => ON is ~1.8% SLOWER.
- abba3/2_ON64 generated 1412 tokens vs 1920 in every other run (agent1 turns end after 1-2 / ~122 tokens, void 'short(<64)'); its shorter makespan reflects less work. Gate agg tps (tokens / predicted_ms) = -9.1% ON.
- decode tps (c1_gate.py:12,36) is a per-turn mean that includes pause-frozen turns (OFF run3 non-void turns 22.9 / 12.3 tps; ON run2 6.9) and void turns.
- Order was OFF,ON,OFF,ON (abba3), i.e. ABAB, not ABBA (LOG.md line 4).
- Fix: subtract in-window pause from makespan (or freeze the client clock during pauses), require equal token counts per turn, use ABBA, report agg tps.

## S3 (high) C1 token-identity gate fails and the baseline itself is not deterministic
- per-turn token_sha ON==OFF 8/48 (gate output). OFF-vs-OFF also differ on every non-void agent0 turn (e.g. (0,1) 74712311 vs 888fc776), so 2-slot greedy is batch-order nondeterministic; the "identical to MTP-off" gate cannot be evaluated this way.
- ON run2 agent1 turns 2/5 returned 2 and 1 tokens (edit/copy-heavy), not reproduced in ON run4 -> possible 2-slot MTP early-stop bug; check stop_type/stopped_eos for those turns.
- Fix: test identity at -np 2 with one active slot, or against a fixed batch schedule; investigate the 1-2 token turns before any adoption.

## S4 (high) Fixture inflates MTP: 8/12 turns are void loops with ~100% draft acceptance
- void_reason 'loop(>50% repeated 4grams)' on 8/12 turns in every abba3 run; draft lines show acceptance 1.000, mean len 2.00 (results.json draft_lines). Only 4 non-void turns per run carry the real signal and they are pause-contaminated.
- Fix: score decode/makespan on non-void turns only, or use a fixture with <25% void.

## S5 (high) C1 gate does not drop invalid runs
- c1/scripts/c1_gate.py:7-21 globs run dirs and never reads `invalid`, `ok`, or tmax>=80. Pointed at c1/abba it would include 3_OFF64 (tmax 80, makespan 25 s, ok=False). abba3 is clean only by luck. Fix: filter on results.json invalid/ok, as c2_gate.py:4 does.

## S6 (medium) C3 numerics change greedy output; decode "regression" check confounded
- runs/chain sweep1: decode_text_sha 200722... for B0/d/nt, 13f2dc... for s2048 and t128. TQ/SPLIT changes summation order, so the text changes and MTP acceptance changes with it. Decode -7% (s2048 90.9) and -10% (t128 88.1) vs d 97.9 are n=1 runs on different text, so they cannot be read as regressions or as non-regressions.
- KLD row is a top-8 proxy (c3_score.py:51-61, 94), not FP64 KLD; same for C2 (c2_mv2cmp.py, n=1 AB). Neither meets the dmean gate as written.

## S7 (medium) C3 TQ=128 enlarges scratch; 2-slot VRAM is not tested at prod depth
- v1 fattn-maxwell-gemm.cuh:589-593: score_elements = 98304 * packed_columns, which doubles at TQ=128 (vram_peak 22524 -> 22728 MiB, 1 slot). The C3 2-slot arm is 2x48K (c3_driver.py:35, CTX 98304 total), not 2x78-84K. C1/C2 show 23.2-24.4 GB peaks there. Fix: 2-slot OOM check at prod ctx before adopting t128.
- Patch logic looked sound: key_capacity 98304 is guarded (:553), 98304 divides exactly by 2048/4096/8192, pv_pointers <<<1,256>>> covers split_count <= 48.

## S8 (low) Evidence and receipt drift
- c1/RESULT.md still says "RUNNING since 04:16 (pid 463074)"; the active driver changed twice (546494, now abba4). C1 has no commits; harness edited live (c1_mtp.py/c1_gate.py mtime 06:16).
- Committed wt-c2 m6000/speed-c2/c2_abba.py differs from the running script (CTX map, default order), and the committed wt-c3 c3_append.py differs from c3/scripts (home path).
- C1 COOL_C default 55 (c1_mtp.py:19) vs LOG "cool<=60"; C1 watchdog starts after container start (c1_mtp.py:136), same in C2/C3, so startup goes unmonitored (low heat).
- C3 attention-share gate uses max(measured 45.2%, projected 60.6%) (c3_driver.py:146); the projection is a model, not a measurement.
- C2 patch (997feee): shortest-first can starve the long-append slot under sustained short appends (bounded in N3R). Its rem for a STARTED slot uses the stale prompt.n_tokens(). Not a correctness bug, but watch p95 stall on the long slot.

## Lease / cleanup: OK
- All three take the flock before ServerSession.__enter__ and release it after __exit__ (c1_mtp.py:124-153, c2_abba.py:107-129, c3_append.py:171-211).
- r3_bench cleanup_container uses an exact name match (^/name$). The C3 driver final rm matches the 'r3bench-c3-' prefix plus its own pid. No cross-card kill path.
- C1 LOG admits a manual `docker rm -f r3bench-c1-*` that killed its own concurrent chain's container; that is limited to C1.

## Distrust
- C1: the 64K +2.95% / +7.9% claim (LOG.md 06:16 entry). 96K is untestable (ON96 OOM at 24410 MiB).
- C2: everything so far (0 valid runs).
- C3: sweep ranking is fine (noise ~0.2%). Decode and parity rows are pending and confounded.
