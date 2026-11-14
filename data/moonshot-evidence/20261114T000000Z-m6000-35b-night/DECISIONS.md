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
