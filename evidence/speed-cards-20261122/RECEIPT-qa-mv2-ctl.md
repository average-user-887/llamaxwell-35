# QA bounded controls receipt - 2026-11-21 (combined-r3-card1, no rebuild, no push)
Sanitized: no payload text, no GPU ids, no raw dumps. Private files stay in /tmp/qa-mv2-ctl (*.private.json, x/).

## Roles (configured vs verified)
| role | configured | actual (self-reported ID) | outcome |
|---|---|---|---|
| Opus | m6k-coder | claude-opus-5-5 | read-only A+B done |
| Fable | m6k-advisor | NOT RUN (usage-credit error, rate_limit) | no independent advisory; no substitute labelled Fable |
| Haiku | m6k-runner, model override haiku | claude-haiku-5-5 | ran exact command, exit 0 |
| Sonnet | lead | claude-sonnet-5-5 | sequence, admission, reconciliation |

## GPU admission
- One executor, lease on gpu-bench.lock (released: 0 holders after), M6000 telemetry age/util check at start, DEBUG unset, 8 experts, existing build, 1 slot, 1660 Ti untouched.
- Fleetdash is retired (27 Oct); admission = harness lease + fleet telemetry check as in the replay. Disclosed deviation.
- No standing host 80 C watchdog found. Substitute: per-run watchdog (own r3bench containers only, kill at >=80 C) + cooldown <62 C before start. peak 62 C, 31 samples, 0 trips.
- Containers left: 0. Watchdog process stopped with the run.

## Card 1: identical-settings serial repeat (miss_param_1 T1, spec none, -ub 1024, temp 0, seed 0, cache-ram 0)
Settings, env, args and runtime library hashes identical to a1_serial (diff clean).
- 3 fresh servers (denominator 3): prompt_n 3025, predicted_n 115, rc 0 each; logprobs bit-identical across the 3 at every dumped step; output identical across the 3.
- vs a1_serial (predicted_n 104): 0/3 repeats reproduce it. logprob differences already at n_dec=4 (1716: -9.19 vs -9.39).
- vs native arm a2 output: 3/3 identical (hash).
- Of 4 identical-setting serial runs, 1 (a1) is the outlier. Cause of a1 vs repeats: unexplained.
- Control NOT clean against the reference -> -ub sensitivity NOT run (gated by the card).

## Card 2: n_dec=5 split, same prefix (token ids 760,1156,6587,728,310 in all arms), tokens 25 vs 1716
margin = lp(1716) - lp(25); positive = 1716 (a1 serial's token) wins
| arm | lp(1716) | lp(25) | margin | path at step | decision |
|---|---|---|---|---|---|
| a1 serial | -.68464 | -.72583 | +.04119 | M=1 | sample 1716 |
| ctl serial r1-r3 | -.82076 | -.59636 | -.22440 | M=1 | sample 25 |
| native | -.84264 | -.58068 | -.26196 | V rows=2 | accept draft 25 |
| Q8A=0 | -.69341 | -.71746 | +.02405 | V rows=2 | reject, emit 1716 |
| all-five-off | -.74276 | -.66253 | -.08023 | V rows=2 | accept draft 25 |
- All-five-off still ran an M=2 verify batch (rows=2) with stock GDN, no T2/MROW/DOWN4 lines, snapshot kept; Q2 cohorts and FA_Q2_R2 stayed on. It did not remove M=2 verification.
- Dumps cannot attribute the kernel per step (counters count graph captures). Proposed minimal instrumentation, NOT applied: per-dispatch bitmask (T2GDN|Q8A|DOWN4|MROW|Q2COH) kept with graph nodes, appended as path=<mask> on the V line (server-context.cpp ~L4753). Acceptance: native rows=2 -> T2GDN|Q8A; Q8A=0 -> T2GDN; all-off -> none of T2GDN/Q8A/DOWN4/MROW.
- predicted_n differences (104/108/115/115) come from reasoning-token length before </think> (43/47/53/53); tool_calls byte-identical in all arms.

## Card 3: base_63 T2 exact request
NOT VERIFIED. No recorded request bytes (log_model_api off; receipt keeps counts only), so no hash to compare. Offline reconstruction exists (T1 18005 B identical both arms; T2 18509 B, differing only in tool_call id), plus known replay defects (reasoning inlined as <think>attachment://..., no reasoning_content, missing leading newline, wrong tool_call id, seed/max_tokens/cache_prompt/id_slot differ). EOG boundary (4410 vs 4411): UNTESTED, replay skipped.

## Reproduced / not / untested
- Reproduced: serial determinism within a build (3/3); M=2 verify margin spread across arms.
- Not reproduced: a1_serial output from identical-setting fresh serial servers (0/3).
- Untested: -ub sensitivity, EOG boundary, base_5 and long_context_74 flips, any other prompt, 2-slot, MTP promotion, 17-case, BFCL.

## Next step (one)
Re-run the exact a1 serial arm (all 5 requests, fresh server) x2 under the same lease to see whether a1 was a one-off or a session-sequence/start-state effect; only then -ub sensitivity.

## Addendum 1 - a1 serial arm x2 (approved re-run), 21 Nov
- Exact a1 serial arm (5 requests, fresh server per repeat), 2 repeats, denominator 2: rc 0 both; predicted_n 104/101/118/83/109 and prompt_n 3025/4854/5433/4328/4324+113 in both; matches a1_serial T1 (4/4).
- mv2.log of both repeats is byte-identical to a1_serial's (same sha256). env.args and server.args identical to a1_serial and to my ctl runs.
- ctl_ub1024 (miss_param_1 alone) differs from a1 already at n_dec=0 prefill logprobs, with the same prompt_n 3025. So a1 is NOT a one-off: it is reproducible (3/3 incl. original). My ctl control was not a faithful replica of the a1 setup; the difference (request/body/sequence/harness path) is unidentified. The earlier "1 of 4 serial runs is the outlier" reading is withdrawn; serial is deterministic per harness path.
- Implication: the serial-vs-serial flip floor must be measured via the a1 harness path, not ctl.py. -ub sensitivity still not run (needs approval).
- Run safety: peak 72 C (watchdog kill at 80 C, 0 trips), 46 samples; 0 containers left, lock free, no watchdog process left. Note peak exceeded the 70 C abort used between runs; that rule applies to pre-run state, in-run kill is 80 C.
- Executor: claude-haiku-5-5.

## Addendum 2 - Gemini R3 inspection (branch gemini/r3-maxwell-cards-20261121, tip abee1c1, not on main)
- Verified: 121.3 tok/s only at 8K (Q8C); 80K warm +1.6% lossless / +5.9% Q8C; wall -9.5% IFEval native, -12.5% Q8C, -10.0% BFCL; BFCL gen +29.9%. Quality: IFEval -1.85pp (-2.3% rel); BFCL 71->66/160 (-7.0% rel, lower95 -15.5pp, unresolved); Q8C BFCL partial 49/160 (thermal kill).
- Fable (m6k-advisor, actually ran, no files read): claim inflated; fix BFCL gate to paired relative drop <=5% vs measured serial flip floor; best lever = prefill bundle ub2048+mixed-batch+V5 (~55% of BFCL wall is prefill); first log prefix-cache reuse; 3 serial-vs-serial BFCL repeats; Q8C only as later separate arm. Accept gates: >=1% two-slot makespan, KLD dmean<=+.0010 top>=-.3, 160/160 0 infra, discordant excess <= floor, no 80 C kill.
- Opus card NOT launched (needs owner go; BFCL runs are long GPU jobs).
- Thermal correction (owner): 70-79 C normal, 80 C hard limit; 62C/70C rules withdrawn.

## Addendum 3 - Opus prefill card 2 (autonomous window, ctx 2x64K, combined-r3-card1, no rebuild, no push)
Executor: claude-opus-5-5 (m6k-coder). Evidence: /tmp/qa-prefill-card/RESULT.md, LOG.md; local branch claude/qa-prefill-card-20261121 (a228ff3, 6c5ad14, 7ac90aa), not pushed.
- S1 two-slot ABBA (6 runs, 6/6 valid, 0 OOM at 2x64K): A (ub1024) 144.29 s vs B (ub2048+V5) 149.41 s = B 3.55% slower. FAIL (gate >=1% faster). Deep setup prefill -5.9% with ub2048, but append phase loses more. The 2x84K OOM comes from the ub2048 compute buffer (688 vs 344 MiB; +560 MiB peak), not V5.
- S2 --ctx-checkpoints 8 vs 2, BFCL preview n=40 native: forced full reprocess 8/471 req (112.7 s) -> 0/462; wall 1871 -> 1762 s (-5.8%); 14/40 -> 15/40, 3 discordant. PASS (small n).
- S3 proxy parity (not KLD; no KLD vs FP64 ran), 5 req: A repeat and ckpt8 match reference 471/471; B diverges on 4/5 (mean |dlogprob| 0.017, max 0.495).
- S4 BFCL 160 native, ckpt8+ub1024, paced: 160/160 resolved (132 scored, 28 model_cap), 0 infra, tmax 78 C, 0 trips (208 pauses, 441 s). 69/160 vs serial 71 = -2.82% rel (<=5% OK). Discordant 30 (16 down, 14 up) vs floor 0 taken from older serial-vs-serial runs, so strict FAIL; current ckpt2 native also fails (29 discordant, 66/160, -7.04%). ckpt8 vs ckpt2 native: 7 discordant. Forced reprocess 44/1924 -> 0/1908.
- Caveats: ckpt2 reference ran with GRAPHS=1, Opus arms without (confounded); floor not measured fresh; S2 n=40.
- Recommendation: adopt --ctx-checkpoints 8 with ub1024, no V5; reject ub2048+V5. Q8C, MTP promotion, push remain unapproved.
- Next (needs approval, ~2 h each): 2 fresh serial + 2 native ckpt8 BFCL 160 repeats, same build, no GRAPHS, to measure the native-vs-serial discordance floor.
- Cleanup: 0 r3bench containers, lock free, no leftover processes.

## Addendum 4 - speed campaign cards C1-C3 (autonomous window to 09:00 CET, 22 Nov, combined-r3-card1, no push, no prod change)
Sanitized. Workers: 3 fresh Opus coders (m6k-coder) + 1 Opus read-only adversarial reviewer + Sonnet/Haiku scoring/log helpers; Sonnet lead. GPU serialized via lease; per-run watchdog kills own containers at >=80 C; pacing pause >=77 C, resume <=72 C, 15 s max. Evidence: /tmp/qa-speed-cards/{c1,c2,c3,review}/RESULT.md|REVIEW.md.
| card | idea | verdict | key numbers |
|---|---|---|---|
| C1 | two-slot native MTP (draft-n 1) | REJECT | makespan net of pauses: 64K -1.73%, 78K -2.41% (ON slower); 2x96K OOM (24410 MiB), max fit 2x79872; lossless gate NOT JUDGED (OFF vs OFF only 8/12 turns identical); possible early-EOS bug in 1 ON run (open) |
| C2 | shortest-first append scheduling (c6e3cc2) | REJECT | 84K n=2/arm: +0.70% raw, -0.36% net of pauses (<1%); p95 stall same, max stall worse (5.6-7.0 s vs 0.6-5.0 s); 64K, KLD/MV2 NOT RUN |
| C3 | FA maxwell GEMM TQ/SPLIT sweep | REJECT | best SPLIT=2048 +0.71% append (kill <1%); TQ=128 -2.14% and +200 MB; attention share 45.2% measured; ABBA/2-slot/top-8 skipped by kill rule |
- Invalid and excluded: C2 5 runs touched 80 C (hot80/), C1 early runs unpaced/80 C; 2x96K OOM in both arms (f16 KV 3840 MiB + 18867 MiB model).
- Adversarial review caught: C1 +2.95% raw makespan was a pause artifact; C2 had no pacing; fixed and rerun. Fixture caveat: 8 of 12 turns void loops.
- No ADOPT candidate; baseline stays ub1024 + --ctx-checkpoints 8. Branches local only: claude/speed-c1-20261122 (7ac90aa, no commits), claude/speed-c2-20261122 (c6e3cc2), claude/speed-c3-20261122 (2592658).
- Cleanup: 0 r3bench containers, no card processes, lease free (checked 08:17).
