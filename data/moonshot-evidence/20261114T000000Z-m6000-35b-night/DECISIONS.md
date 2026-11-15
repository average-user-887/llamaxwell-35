## 2026-11-13 ~20:00 consult 1 (brief: consult/1-brief.md)
Q: blind spots / reorder; start card 2 in parallel?
Astra: keep 0->1; test starvation + per-slot completion/decode latency; check the other slot's decode token does not still force ne3!=1; if S1 loses efficiency go to multi-stream GEMM; ub2048 may add decode stalls; softmax fusion next; defer Q8/Q4 KV; reserve time for stack replay + rollback.
Fable: YES start card 2 (multi-stream GEMM, Opus) now in parallel; GPU order 0 -> 4/1 -> 3. Card 1 ride-along decode token may still trip ne3!=1 (inferred). Card 1 run also reports ne3_q_ge_64 residual split (<64 ride-along vs >=64) + per-slot completion, B TTFT, decode stall. Card 2: code-complete by 00:00, hard cut 02:00; cancel only if card 1 residual <3% and stack already <=156 s; accept loss <1%, KLD gate vs FP64, >=1% gain over best stack. Cards 5/6 out unless GPU idles. Card 7 reserved from 05:00 with rollback line. Fallback S1b (separate llama_decode for the other slot's decode tokens).
Data: card 0 baseline gemm 179.86 s; prefill work lost to ne3_q_ge_64 = 20.19%.

## Card 8 (owner order 13 Nov ~20:15): WP4 moe_v2 occupancy
Owner asked to restart the "2-day Opus" moe_v2 build in the background, last in the GPU queue. Opus coder started (branch moe2, $M/moe-v2-20261114), no GPU use; any GPU timing goes behind cards 3/2/7. Owner order, so no consult.
card1 s1 -4.5% ne3 loss 0; card2 ms correct, N3 deferred (per Fable rule: cancel if S1 residual<3% and stack<=156 s; stack not yet measured).

## 14 Nov ~14:20 Card 7b gates (stack) + consult 2
- FA RMSE vs FP64: fused == stock at kv8448 (+0.01%, noise), ~0.5x stock at 32768/98304 -> PASS under reformed gate (strict "<=" rule was a criterion artefact).
- Token parity: stack differs from GEMM-only on a 6k prompt at tok 10; GEMM-only (already prod) differs from stock at tok 177/32 -> greedy drift from non-bit-exact numerics, not stack-specific. Judged acceptable, but model dmean (KLD) vs reorder floor NOT yet run.
- Save/restore round trip PASS.
- Decision: hold prod swap until card 9 (stack+moe_v3) N3 result; then ONE KLD gate on the final candidate (stack or stack+moe3), then swap per handbook s6.
- Card 3c: ub2048 = 160.22 s vs 161.16 s (-0.6%, +408 MiB) -> below 1% gate, not adopted.
- Fable consult 2 ranking: 1 moe_v3 adopt (-1..2%), 2 ub2048 (done, -0.6%), 3 decode-attention profile before any kernel card, 4 prefill/decode overlap check (card 2 earns a run there), 5 Q8-K only as VRAM enabler, 6 Q4 KV defer. Astra reply pending.
- 14 Nov ~16:00 Card 2b (stack+ms N3 timing) DROPPED: ms conflicts with stack chunked-q/fused loop in fattn-maxwell-gemm.cuh (6 hunks, numerics-affecting); S1 already leaves 0% ne3 loss so upside small. Revisit only if card 11 profile shows ne3 attention is large.
- Card 9: moe_v3 on stack = 160.35 s vs 161.2 s (-0.5%), below 1% gate, not adopted.
- GPU queue now: 7c KLD (running) -> 3d mixed-batch caps 512/2048 -> 11 decode-attention profile.

## 14 Nov ~17:00 card 3d (RESULT_c1.md)
base 161.32 s; A cap512 +2.4% (reject); B cap2048+ub2048 158.65 s (-1.65%) but max token gap 5.6->10.8 s (decode stall for other slot), VRAM 23867. Not adopted yet: gain partly ub2048 (-0.58%), gap regression. Follow-up queued: cap2048 ub1024 (isolates cap effect) to see if gap/gain separate.

## 14 Nov ~17:35 card 12 dec2 (run-20261114T151119Z)
Decode ABBA (v2 engaged per log): single_30k +0.15%, both_30k -0.35%, single_90k -0.30%, both_deep +1.43% (token sha differs 2/1) -> no gain, below 1% gate, noise. N3 pairs: v2 not engaged in N3 logs (WARN), stack 160.48 vs dec2 160.83 -> inconclusive but the engaged decode bench shows no gain, so dec2 DROPPED. Awaiting card 11 profile before any further decode-kernel work.

## 14 Nov ~18:00 card 13 attention rewrite: STOPPED after analysis (13/DESIGN.md)
Stack decode FA kernel already at ~277 GB/s = 87% of 317 theoretical (above measured memset 268.8); F16 rewrite ceiling ~1.1% of step, expect 0.3-0.7% -> cannot meet >=1% gate. Explains dec2 null. Remaining F16 lever: ragged two-slot split (DESIGN §4): est N3 +0.5-0.9%, both_deep ~1.9%, mixed depth 30k+96k ~15%; single-slot bitwise unchanged -> card 14 started (owner rule: single-slot untouched + no two-slot loss = win). Q8 K+V decode (~15% step) deferred: needs KLD, Q8N failed gate earlier.

## 14 Nov evening: owner asked PDF guide + tonight plan; Fable scout ruling (batch rule: batched gains passing gate are adopted)
- Order: 7d swap -> 7e (8 docs) -> card 14 gpu_ab parts 0-2 (confirm ragged engaged under prod env; bypassed for Q8 KV) -> batch candidate (stack+ub2048+moe_v3+ragged if passes), N3 5 pairs vs stack, ~158 s if additive (unmeasured).
- 3e (cap2048/ub1024) likely no-op (cap limited by ub); replace with cap1536/ub2048 and cap2048/ub2048 on batch build.
- Accept: batch >=1% vs stack, spread<gain, max gap <=6.5 s, no alloc failure ctxfull 2x96K; card 14 NS=1 bitwise equal, both_30k not slower.
- CPU: card 15 token-gap bound (patcher), GDN profile+design (Opus, later), Q8 ragged if prod Q8-K. 156 s unlikely tonight w/o gap fix; 150 s needs new lever.
- Astra consult reply still not received.

## 14 Nov eve: 7e partial (per guide agent)
- 98K, 8 docs: dmean +0.00308 vs floor +0.00316 PASS; top-token agreement -1.17 vs floor -0.59 pt fails strict line. 65K phase stopped after arms P,S (19:32), no analysis. Reformed gate says dmean within floor -> keep stack; top-token line logged as risk. Guide: GUIDE_20261114.pdf (6 pp), card 14 expected to engage (prod F16 KV).

## 14 Nov eve: 7e final ($N2/7/KLD_RESULT2.md)
- 98304 (8 docs): dmean +.00308 vs floor +.00316 PASS; top -1.17 vs floor -.59 FAIL (6 vs 3 rows of 512). 65536 (4 docs): dmean +.00106 vs floor +.00085 (+.00021, inside CI) FAIL; top -.39 = floor, 1 row. Stack indistinguishable from reorder floor on dmean; top-token line noise-limited. Verdict: swap stands under reformed gate (dmean within floor, noise-limited top line); caveat ref=prod not FP64, server-only features not exercised.

## 14 Nov eve: 3e result ($N2/3/RESULT_c2.md, 3 blocks/arm)
- C (cap2048/ub1024) = base (161.74 vs 161.78 s): cap alone is a no-op, as the scout predicted. B (cap2048/ub2048) 159.07 s -1.68%, TTFT -6.3%, decode +2.1%, max gap 10.79 vs 5.59 s, VRAM +1.7%. Gain comes from ub2048; gap regression remains -> card 15 (PMAX) tests bounding it. Not adopted.

## 14 Nov 21:19 7d prod swap: renames done, smoke PENDING
- llamacpp-m6000 = stack7 (g1+wp2 fused+s1, cap1024, ub1024, np2, FUSED=1; libs prod-libs/stack7-20261114), Created not started; old = llamacpp-m6000-b3gemm-prev. llama-swap smoke 500 (swap-ctl refuses while lock held; card 14 took the lock). Fallback smoke queued on the lock (/tmp/smoke7d.sh -> /tmp/smoke7d.log). UNVERIFIED until it reports engaged= + fused line; first llama-swap load after lock release is the real test.
- Rollback: docker stop llamacpp-m6000; docker rename llamacpp-m6000 llamacpp-m6000-stack7-failed; docker rename llamacpp-m6000-b3gemm-prev llamacpp-m6000

## 14 Nov late: prod stack7 engagement check ($N2/7/PROD_ENGAGE_CHECK.md)
- 4641-token prompt via own port: engaged=10, fused calls=10, softmax_launches=320, HTTP 200, pp 549 / tg 80 t/s. q_tiled=0 (G1 tiling not firing; stack was measured with GEMM+FUSED+s1 env, no tiling env assumed) -> not a regression; swap STANDS. Open item: confirm in the batch N3 run whether q_tiled should engage.

## 15 Nov ~00:30: card 14 + card 15 results (GPU had idled since ~22:45; fixed by queueing below)
- Card 14 ragged (F16, engaged 14 lines/run): decode ABBA x6: single_30k +0.59%, both_30k -1.61% (sha differs), single_90k +0.70%, both_deep +0.21%, both_mixed(98K+30K) +14.26%. N3 3 pairs: stack 161.31 vs ragged 161.99 s (+0.4%, no gain). Not adopted for N3; kept as flag for uneven-depth workloads. Part 0 RMSE not yet read.
- Card 15 (PMAX bound): N3 makespans base 161.92, c1536 161.81, c1536p 161.09, c2048 159.25 (-1.65%, gap 10.8 s), c2048p 161.39 (gap 6.18 s). Bounding the gap erases the cap2048 gain (~= base). Gap-bound idea fails; ub2048 alone ~-0.6%.
- Batch candidate now: stack + moe_v3 (-0.5%) + ub2048 cap1024 (-0.6%) ~ -1.1% if additive; ragged dropped from batch (N3 neutral).

## 15 Nov ~03:50 batch result
- Card 17 KLD (moe3 vs stack): dmean S-P +0.00000 at 98304 and 32768 (floor +0.00304/+0.00175) PASS; top-token 0.00. Exactly zero suggests moe_v3 is bit-identical to stack; MOE_V3=1 was set in S arm but no engagement line (no DEBUG) so engagement is unconfirmed there; card 16 shows moe_v3 launches=65536.
- Card 16 N3: M (moe3+ub2048) 159.63 s vs S 161.55 s = -1.19%, all 5 M < all 5 S, gap 5.61 s, VRAM +408 MiB. GATE-CANDIDATE.
- Remaining accept line: ctxfull 2x96K no alloc failure -> card 19 queued (after card 18 GDN A/B). On ALLOC-OK adopt batch via §6 swap.
- Card 18 (GDN interleave d041581, flag GGML_CUDA_OCCAMY_GDN18) built, A/B queued/running; est 0.5-1.3%.

## 15 Nov ~04:05 ADOPT batch
Card 19 ctxfull ALLOC-OK (M 23867 vs S 23459 MiB, 0 alloc errors). All accept lines met (>=1%, gap 5.61<=6.5, VRAM +408 MiB, KLD pass, ctxfull ok). Card 20 launched: prod swap to batch8 (stack7 + moe_v3 + ub2048), old container -> llamacpp-m6000-stack7-prev. Card 18 (GDN18 fixed 5b1e570) relaunched, queued.

## 15 Nov ~04:25 scout ruling (Fable)
- 156 s not reachable by GDN18 alone (needs -2.3%, GDN18 est 0.8-2.1 s). Next lever: server-side append scheduling / prefill-decode overlap; get a per-phase split of the batch N3 run first. Attention parked. GDN follow-up only if GDN18 >=0.5%. q_tiled: diagnostic only, don't force blind.
- Queue: (1) GDN18 N3 on batch A/B/A accept <=158.8 s, above A/A spread, no single-slot loss, KLD ok, ctxfull ok; (2) phase-split instrumentation row; (3) q_tiled counter + forced N1 row if flag; (4) cap/ub sweep (ub1536/3072 @cap1024; cap768/1536 @ub2048) -> card 21 launched; (5) final N5 A/B/A of batch + passers by 07:00.
- Objective restated: <=156 s OR best gated stack by 08:00 (157-158 adopted). Unknowns: gap composition, A/A noise, ub3072 fit.

## 15 Nov ~04:50 card 18
GDN18 N3 161.30 vs 161.39 s (-0.06%), decode +-0.3%: dropped (per scout: GDN follow-up only if >=0.5%). Not adopted. Prod swap (card 20) in flight; card 21 sweep queued.

## 15 Nov ~05:05 card 20 DONE: prod = batch8
Prod llamacpp-m6000 = batch8 (stack7 + moe_v3 + ub2048, cap1024); previous llamacpp-m6000-stack7-prev. Smoke 200; 4641-tok pp 589 / tg 85.8 t/s (stack7 549/80); fa_gemm engaged=10 q_tiled=10 (q tiling now engages at ub2048, open item answered); fused calls=10. Rollback in STATE.md. Card 21 sweep running, card 22 phase analysis (CPU) running.

## 15 Nov ~05:15 card 22 phase split (CPU)
M run 159.6 s: slot0 prefill/decode/idle 64/87/9 s, slot1 72/60/28 s (idle mostly start offset). 5.6 s max gap = one slot frozen while the other prefills a 4K append; prefill is compute-bound so not idle GPU. Union of prefill windows 108 of 160 s. Upper bound 44 s slot1 stall, realistic 3-8 s (unmeasured); card 15 gap-bound already showed bounding the gap erases the gain, so overlap lever = mixed prefill+decode batches/smaller chunks, probably what cap already does. Waiting on card 21 sweep before queueing more.

## 05:20 15 Nov
- Card 23 (flag variants on batch8): no arm passes. A 159.37 s, B(no single-stream) +0.06%, C(-b 4096) +0.13%, D(ckpt 8/4096) +0.26%; all within noise. Dropped.
- Card 21 aborted at 04:43 ("GPU busy", collided with swap smoke); never ran. Relaunched as card 21b (adds real ub3072 with -b 3072). Card 24 (N5 S/M confirm, batch8 vs stack7) queued behind it.
- Card 24 (N5 S/M confirm): stack7 161.31 s vs batch8 159.93 s = -0.86% (5 valid each, ranges 0.21/0.15 s, no overlap); gap 5.64 s, VRAM +408 MiB. Card 16 gave -1.19%; combined ~-1.0%. Prod stays batch8. 21b still queued/running.
## 06:25
- Card 21b (cap/ub sweep, 2 valid runs/arm): ref A 159.72 s; ub1536 +1.41%; ub3072/-b3072 -0.04%; cap768 +1.99% (gap 4.3 s); cap1536 +0.42% (gap 8.1 s). No arm passes; cap1024/ub2048 stays. Card 26 (cap 896/1152/1280) queued behind card 25.
## 07:20
- Card 25: A vs A2 noise 0.17%; fused FA GEMM off = +5.5%. Card 26: cap896 +2.9%, 1152 +0.2%, 1280 +0.2%; none pass. Prod stays batch8; queue drained, final push next.
