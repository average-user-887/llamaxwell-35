# Night 13→14 Nov 2026: M6000 35B moonshot plan (cut 08:00 CET 14 Nov)

Reviewed by the Fable advisor 13 Nov ~16:30. Astra has **not** seen it: the orchestrator's first action is one
consult on this plan (handbook section 3), sent while card 0 builds. Cards below then start without further consults.

Objective: makespan of two agentic slots at 84K to 96K with tool-result appends (N3 replay, short profile).
Output dir `N2=$M/occamy35-night-20261114` (create; copy `RULES.md` from `occamy35-night-20261113/`, update the prod line).

## Where we are

| | N3 two-slot makespan | +4K at 90K, single | peak MiB |
|---|---|---|---|
| BUNDLE3 (prod until 16:10 today) | 210.68 s | 31.25 s | 23,109 |
| **Prod now**: BUNDLE3 + GEMM prefill, F16 KV, ub1024, cap 0 | **177.5 s (-15.76%)** | 23.09 s (-26%) | 23,459 |

**Target tonight: 156 s or lower (-12% vs prod now). Stretch: 150 s.** That is -26% to -29% against BUNDLE3.

Why this is reachable (advisor estimates, not measurements, unless marked):
- Single slot gained 26%, two slots only 15.8%. Measured: with both slots in one batch the GEMM guard `ne3 != 1`
  sends prefill back to the old tile kernel; `flash_attn_tile` is still 11.8% of profile time (N3B nvprof).
  Estimate for closing that gap: -7% to -14% makespan.
- `masked_softmax` is 7.3% of the GEMM profile (measured). Fusing it: estimate -3% to -5%.
- ub2048 was worth -1.5% before GEMM (measured) and is blocked today only by the `q > 1024` guard.
- The raw `ne3` counter (11,680) is mostly two-slot decode, because `ne3` is tested before `q_lt_64`
  (fattn-maxwell-gemm.cuh:236-238). Card 0 fixes the counter so the size of the loss is measured before card 1 or 2 is believed.

Source: `$N/WP0/src` branch `gemm-b3`; harness `$N/WP0/harness` (`n3_replay.py`, `wp0_report.py`, `wp0_gpu.sh`).

## Cards, in GPU order

| # | Card | Who | Accept | Cap |
|---|---|---|---|---|
| 0 | **K0 counters.** Split the reason `ne3` into `ne3_q_ge_64` and `ne3_decode`; add per-reason work totals (sum of q rows x keys). One N3 block on prod flags to size the loss. | m6k-patcher; run by script | counters print; prefill work share lost to `ne3_q_ge_64` is known | 30 min code, 15 min GPU |
| 1 | **S1 one stream per prefill ubatch** (server side, `tools/server/server-context.cpp`, next to the existing `use_mixed_batch()` cap code). When a slot has 64 or more prompt tokens pending, build the ubatch from that slot alone so `ne3 == 1` and GEMM engages; alternate slots per ubatch; decode-only batches unchanged. Lead check 16:50: plausible. With -np 2 the KV is 2 streams, so `llama-kv-cache.cpp:731` uses `split_equal(n_ubatch, true)` and a ubatch holds every seq present in the server batch; keep a second slot's prompt tokens out of the batch (gate next to `can_batch_with`, `server-context.cpp:3488`) and prefill ubatches become single-stream. A one-token decode of the other slot still forms a 1+1 ubatch first; that is fine (q<64). Behind a flag, default off. First 30 min: read-only feasibility; if a ubatch cannot be made single-stream in this fork, stop and go to card 2. | m6k-patcher | N3 3 paired blocks vs prod: `ne3_q_ge_64` work near 0; makespan at least 5% better; max token gap reported (not a gate); identical tokens vs prod on the restore round trip | 90 min code, 40 min GPU |
| 2 | **WP1 multi-stream GEMM** in the kernel path (one GEMM pass per `ne3` slice, scratch reused serially, mask per stream). **Only if** card 1 is infeasible, loses, or leaves more than 3% of profile time in `flash_attn_tile`. | m6k-coder (Opus) | same as card 1, plus operator RMSE vs FP64 no worse than stock per stream at 32K/98K | 3 h |
| 3 | **G1 Q tiling.** Replace the `q > 1024` fallback by a loop over Q chunks of at most 1024 rows through the existing path (scratch unchanged). Then flag rows in the same GPU session: ub2048, and **C1** = `--mixed-batch-size 1024`. | m6k-patcher; Opus only if the accept fails once | ub2048 single +4K at 90K under 23.09 s; two-slot better than the best so far; peak at most 23,976 MiB (600 MiB headroom); RMSE vs FP64 no worse than stock. C1: makespan not worse (CI), max gap down | 45 min code, 60 min GPU |
| 4 | **WP2 fused mask+softmax epilogue** (online softmax across key chunks; spec in `$N/WRAPUP_20261113.md` WP2; techniques in `KERNEL_INSPIRATION.md`). Start coding at 17:00 in parallel with cards 0 to 3; it is the long one. | m6k-coder (Opus) | `masked_softmax` share under 6% of profile; +4K at 90K at most 20.5 s; FP64 ratio not worse than prod | build by 01:00, 60 min GPU |
| 5 | **WP3b GEMM accepts a Q8 K cache** (dequantise the q8_0 K strip in the pack step). Completes the owner's adoption list; frees 812 MiB; small speed effect. Start only when card 4 is built. | m6k-coder (Opus) | with `-ctk q8_0 -ctv f16`: engaged above 0 on both slots; single +4K at 90K at most 3% slower than the F16 GEMM arm of the same build; RMSE vs FP64 no worse than stock; save/restore round trip gives identical tokens | 3 h, then park |
| 6 | **Q4 KV rows** (owner: go ahead and test). First nvprof `<1,1>` vs `<1,0>` to explain why Q8 K+V was flat, then K8V4 and K4V4 against F16. Slot it whenever the GPU is idle between cards. | script; m6k-coder only if a kernel fix is needed (consult first) | dmean vs F16 against the reorder floor, task parity, MiB saved, ms/tok | 90 min GPU total |
| 7 | **Stack confirmation.** Best passing combination vs prod, N3 5 paired blocks, plus single +4K at 90K and the round trip. Swap prod if it passes. | script | reformed gate and at least 1% | start by 06:00 |

Not tonight: WP4 `moe_v2` occupancy (decode side, about two days of Opus; it does not move this target) and WP5 k=2 rows.

Opus use tonight: card 4 for certain, card 5 if time, card 2 only if card 1 fails. Everything else is Sonnet or Haiku.

## Rules for the night

- One GPU job at a time under `gpu-bench.lock`; coding runs CPU-side meanwhile. Never leave the GPU idle while a
  built card or card 6 is waiting.
- Every arm's RESULT prints the counters and peak VRAM. `engaged=0` with the flag set is a void arm.
- A card that passes the reformed gate and gains at least 1% may replace prod with the swap procedure
  (handbook section 6). Named swappable tonight: cards 1, 2, 3, 4, 5 and the card 7 stack. Q4 KV is test only.
- Stop a card at its cap, write what is known to its RESULT.md and move on. Do not spend Opus on a parked card.
- A new idea that is not in this table is a new task: consult first (handbook section 3).
- Push each finished card's RESULT.md to gh-pub `evidence-20261112` under
  `data/moonshot-evidence/20261114T000000Z-m6000-35b-night/<card>/` after redaction and the privacy grep.
- 07:30: stop launching GPU work. 08:00: push `NIGHT_REPORT.md` (one table: card, makespan vs prod, adopted or
  not), the prod state and the rollback line, then the owner report.
