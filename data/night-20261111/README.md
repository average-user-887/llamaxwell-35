# Night run 11->12 Nov 2026: llamaxwell 35B on M6000 (audit index)

Baseline 22:00: main c9825b1 (prod lib libggml-cuda-call35-20261111). End state 04:03: prod = main 2ca7934
(merge night-bundle2), lib sha256 c7d0455e50a6342187e0699ebd08d98930f6e4f44070da8a9e7edeca9bb2c600,
env = old prod env + GGML_CUDA_OCCAMY_FA_PIPE=3 MOE_LOP3=1 MOE_TAILBAL=1 OCCAMY_GDN_PREF=1 OCCAMY_HOSTGAP=1 OCCAMY_SMALLFUSE=2.
All flags default-off; every merged change is bit-exact (G1: 64-step logits max_abs 0, state sha 652476ce477f / 5be5fc2d816d).

Gates: G0 build, 0 spills, flag-off SASS identical; G1 bitexact; G4 ABBA x5 >= +1% CI95 low > 0, d4k no regression > 2%;
bundle rule: sub-1% bitexact items merge when the stacked bundle clears +1% with CI > 0. Serving smoke before prod.

| Round | Branch / commit | Outcome | Evidence |
|---|---|---|---|
| R1 | c9825b1 vs C24 | keep c9825b1 (C24 -0.46%) | R1/ |
| E07 FA_PIPE=3 | merged 722b98f (tag main-night-e07) | +1.10% warm, +1.17% d4k, +2.34% d16k | E07/ |
| MOE1 LOP3 | night-moe1 11112ee | +0.25%, bundled | MOE1/ |
| MOE2 TAILBAL | night-moe2 d324789 | moe_v2 -2.4%, bundled | MOE2/ |
| GDN1 PREF | night-gdn1 1fc225d | ~0.05 ms/tok, bundled | GDN1/ |
| HG1 HOSTGAP | night-hostgap 930c1ca | -0.035 ms/tok, bundled | HG1/ |
| AF1 fused QKV | night-attnfuse 72198ab | slower, rejected | AF1/ |
| GR1 CUDA graphs env | (no code) | no gain, rejected | GR1/ |
| GDN2 phase-C tail | night-gdn2 59dfbfc | WIP, never built (redirected) | - |
| GDN3 ssm_out repack | night-gdn3 c253a85 | bitexact, 2-3 us/call at grid 72 (<5 us bar); 30-layer run in progress 12 Nov as GDN4 | GDN3/ |
| SK1 SMALLFUSE | night-sk1 16c49c3 | T2 -0.08..-0.13 ms/tok period, bundled; T1 inert | SK1/ |
| BUNDLE1 | night-bundle a25300c, merge 84a34e3 (tag main-night-bundle1) | vs E07: +2.28% warm, +2.13% d4k | BUNDLE1/ |
| BUNDLE2 | night-bundle2 b3c8c71, merge 2ca7934 (tag main-night-bundle2) | vs BUNDLE1 +0.65%/+0.73%; vs E07 +3.03%/+2.86% | BUNDLE2/ |
| SMOKE2 + adopt | prod container swap 04:03 | serving smoke pass (C25 caveat), VRAM equal | SMOKE2/ |

Total vs 22:00 baseline (chained ABBA, not a direct head-to-head): about +4.16% warm decode, +4.06% d4k.
Single-sample serving smoke vs old prod: short +2.9%, 16k +5.3%, 2-slot +1.6%.
Caveats for audit: chained totals; the 16k gain is E07-only ABBA plus the smoke sample; one post-cancel slow slot (31.75 t/s) in the
smoke did not reproduce in 3 cycles; the 2-slot text differs A vs B at batch-timing points where A also differs from itself.
Ledger: LEDGER.md. Plan/directions log: NIGHT.md. Text evidence only (dumps and binaries excluded).
