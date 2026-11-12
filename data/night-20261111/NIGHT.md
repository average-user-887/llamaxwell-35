# Occamy 35B night run, 11->12 Nov 2026 (lead: Claude Opus session b3b1df8c)

Plan: ~/Downloads/M6000-35B-optimization-plan-2026-11-10 (6).md (v6). Cut-line 07:30 CET 12 Nov:
prod llamacpp-m6000 on the best gate-passing lib, everything pushed, lock released.

## Baseline at start (22:00 CET)
- Prod: lib prod-libs/libggml-cuda-call35-20261111.so (c9825b1 = GH main), flags DOWNX=1 DOWNH=2 QKREUSE=1 PCSHARE=1,
  -np 2 -c 196608 -b 2048 -ub 1024. Launcher ../run-35b-call-20261111.sh.
- Alt build cand C24 = 6f239c9b (branch occamy35-moonshot-c10-c11-c07), same features, lib sha 8a2a37d4.
- Evidence that GH lacks: ../occamy35-moonshot-20261110/results/{GPU2,T1,C24,C25} (text only; dumps excluded).
- Harness: ../occamy35-moonshot-20261110/harness/c04/abba_run.py (lock = ../gpu-bench.lock, flock).
  App binaries (durable copy): ./app (llama-server sha 093f845f, llama 33f1f159).

## Gates (owner rules)
G0 build clean, no spills, flag-OFF SASS identical; G1 64-step teacher-forced logits+state BITEQUAL vs flag-OFF;
G3 dispatch counters show path engaged; G4 ABBA x5 warm_decode: >= +1.0% mean, CI95 low > 0; ctx-4k no regression >2%.
Serving smoke (C25-style: 2-slot, 16k, abort) before prod adoption. Numerics-changing candidates: KLD gate
dmean <= +.0010, same-top >= -.3 pt, else reject. Never change power/clocks (250 W).

## Round loop
Each candidate: Opus authors default-OFF flag in its own cand/ clone -> Sonnet runs gates under lock ->
pass => merge to GH main (fast-forward or merge commit, never force), tag main-night-<id>, text evidence under
data/night-20261111/<id>/, rebuild prod lib, adopt via docker create+rename (keep prev container for rollback).
Fail => keep ledger row with reason, no merge.

## Ledger
rounds/LEDGER.md (one row per round).
- 22:1x prod llamacpp-m6000 STOPPED (not removed) for night benches; restore = docker start llamacpp-m6000 (or adopt new)

## Advisor directions (Fable advisor session, owner-authorized, ~23:50)
Priorities: 1 ATTN-FUSE (a: fused QKV matvec + shared q8_1 quantize, bitexact; b: norm/rope/set_rows; c: combine/out/gate), 2 HOST-GAP (10 HtoD + 1 DtoH per token),
3 BUNDLE sub-gate wins (MOE1 +0.25%, GDN1, HOST-GAP), 4 MoE down phase. Not tonight: head, two-slot, KV quant, power/clocks.
Kill-early: G1 -> 1-cycle sanity + nvprof d0 busy delta; < 0.05 ms/token => park for bundle, skip full ABBA.
Final build + serving smoke + prod adoption START by 06:15.
- MOE2 queued (advisor): phase-C tail balancing on MOE1 11112ee, flag MOE_TAILBAL, GPU priority below AF1/HG1; no further MoE DRAM/prefetch work tonight.
- (~23:55 advisor) GDN1 accepted for bundle. GDN-C tail (GDN2) conditional: if AF1 1a clears 0.1 ms kill screen -> AF1 1b first, no GDN2 tonight; if AF1 1a misses -> start GDN2 immediately. GDN2: prefer exact whole-row rebalance (stays in bundle); if row split, own flag + own KLD gate, gated last, outside bitexact bundle. No more prefetch/load-hoist candidates tonight.
- (~00:10 advisor) HG1 accepted for bundle. GR1 = GGML_CUDA_MAXWELL_GRAPHS=1 env test on main+FA_PIPE=3: G1 -> capture-per-token check -> 1-cycle sanity warm+d4k -> nvprof gap. >0.5% sanity => joins bundle as prod env flag. Bundle (~+0.75%) gets no full ABBA until MOE2 or GR1 lands.
- (~00:15 advisor) Drop matvec fusion. SK1 small-kernel merge (AF1 author): T1 rms_norm+rope+set_rows Q/K per attn layer, T2 gate_copy+unary_gated+quantize_q8_1+add; flag OCCAMY_SMALLFUSE bitmask, off main 722b98f, T1 alone if T2 not ready 03:30, cut 04:15. Bundle ABBA x5: launch when any of MOE2/GDN2/SK1 passes kill screen, no later than 04:45. GR1 rejected (no gain).
- (~01:52 owner direction for 03:00 wake-up):
  Best next avenue: losslessly repack the GDN output-projection weights, then test whether the smaller shared-memory footprint allows more useful GPU occupancy.
  The current profile attributes 3.732 ms/token, about 32.6% of short-context GPU busy time, to GDN. Source analysis suggests this layout could reduce shared memory from 36,608 to 27,392 bytes per block and potentially raise residency from two to three blocks per SM. That is a hypothesis, not a measured speedup. A one-layer pilot needs about 4.5 MiB; all 30 layers would need about 135 MiB.
  For Claude, I’d compare the existing layout at grid 48 with the packed layout at grids 48 and 64, and grid 72 only if actual occupancy permits. Grid 48 isolates the layout change; grid 64 may balance the added occupancy against repeated work. Require byte-exact unpacking, logits and recurrent-state checks, compiler resource and occupancy measurements, then paired full-model timings and VRAM checks. Stop if the whole model slows down even when the isolated phase improves.
  One caveat: an earlier bandwidth proof of concept, commit 07d749a, isn’t accessible in the published repository, so its exact overlap needs checking before calling this idea new. (Note: 07d749a verified locally; it tested experimental MOE_BW/GDN_BW prefetching, so this smem-reduction weight repacking is distinct and new). Use bounded subagent cards, orchestrate lean.

- (03:05 owner + advisor) GDN2 stopped (WIP committed on night-gdn2). GDN3 = lossless ssm_out weight repack (smem 36608->~27392, 2->3 blocks/SM hypothesis), flag OCCAMY_GDN_REPACK, card A discriminator first (grid 24 vs 48; <10% cost => stop), then 1-layer pilot, all 30 layers only if >=5 us/call; cut 05:15. BUNDLE1 (MOE1+MOE2+GDN1+HG1 on 722b98f) integrate+G1 by 04:15, ABBA x5 by 05:15. SK1 resumed, cut 04:15. Priority: GDN3 > SK1 > bundle.
- (03:20 advisor) GDN3 card B constraints sent: C half-row/direct staging needed for smem saving, <=80 regs (drop PREF if over), occupancy-API check before grid 64/72 (grid barrier deadlock risk), test grid 64. Bundle expectation +0.5-0.8%. Report GDN3 pilot verdict before 06:15 build decision. Parked for later: lane-oriented phase-A repack.
- (03:40 advisor) Plan stands. SK1 kill screen judged on token period as well as busy sum. GDN3 grid72 2-3 us/call bitexact (~0.06-0.09 ms/token x30 layers, ~135 MiB VRAM) = next-session bundle candidate, not tonight. No new candidates after SK1; protect 06:15 adoption + rollback container. Report final prod state (lib, flags, smoke, total vs 22:00 baseline) to advisor.
- (04:05) PROD ADOPTED: BUNDLE2 lib c7d0455e (GH main 2ca7934), env prod + FA_PIPE=3 LOP3 TAILBAL GDN_PREF HOSTGAP SMALLFUSE=2. Rollback: docker stop llamacpp-m6000; docker rename llamacpp-m6000 llamacpp-m6000-night-failed; docker rename llamacpp-m6000-call-prev llamacpp-m6000. Next session: GDN3 grid72 repack (+0.06-0.09 ms/tok, 135 MiB), SK1 T1 one-block smem variant, bcast-add fold.
- (07:55 owner) Start GDN3 grid72 repack for all 30 layers -> card GDN4 (Opus): night-gdn3 merged onto night-bundle2 in cand/GDN4, G0/unpack/G1/kill-screen/ABBA x5 vs BUNDLE2/VRAM; cut 10:30. Lead pushes/adopts if it passes.
- (08:19) GDN4 PASS: +0.86% warm / +0.82% d4k vs BUNDLE2, bitexact, +135 MiB. Merge main, tag main-night-gdn4, smoke, adopt.
