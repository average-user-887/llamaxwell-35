# Occamy35 night + day ledger (2026-11-11 .. 2026-11-12)

Files: rounds.csv / rounds.json (25 rows), telemetry_index.csv (524 files), telemetry_arm_summary.csv (39 run/arm groups).
Paths use `<LAB>` for the lab storage root. Numbers are copied from the round LEDGER.md/DAY.md and the g4 analysis.json files; empty cell = not in sources.

## Summary
Merged chain (warm decode, tok/s, ABBA, 95% CI): E07 +1.10% -> BUNDLE1 +2.28% -> BUNDLE2 +0.65% -> GDN4 +0.86% (+3.88% vs E07). All BITEQUAL.
Day (2-slot deep, ms/step, negative = faster): N2SPLIT -4.52% [-5.55,-3.49]; N2GDN -3.3% [-3.80,-2.72]. Both in BUNDLE3, whose gate is pending (numbers empty).
Serving smokes adopted: SMOKE2 (04:03 CET), SMOKE3 (~08:30 CET).

## Parked / rejected / record-only
| Round | Status | Mechanism |
|---|---|---|
| R1 (C24 vs prod) | rejected | LOSS: warm -0.457%, d4096 -0.342%, CIs exclude 0 |
| AF1 fused QKV | rejected | 0.672 vs 0.601 ms/tok, slower |
| GR1 Maxwell graphs | rejected | no gain (warm -0.15%, period 11.970->11.986 ms) |
| MOEN2 | rejected (closed) | 24% shared bytes < 25% rule; no real expert-id log |
| MOE1, MOE2, GDN1, HG1 | parked sub-gate, then in-bundle | each below its kill screen (+0.25%; -0.022 ms busy; <0.05 ms/tok; -0.035 ms); stacked in BUNDLE1 |
| SK1 T1 | parked (inert) | 0 engagements; alias check rejects, needs 1-block smem kernel |
| GDN3-B | parked -> GDN4 | pilot <5 us/call gate; phase C bandwidth-bound; grid72 later scaled to 30 layers |
| GDN5 phase-A | record-only/parked | ~1%, +407 MiB, VRAM margin ~1026 MiB; original ssm_in cannot be dropped |
| GDN3-A, D0, D2H, N2DIFF | record-only | measurements/reads; D2H ~0.4% (logits copy) |
| GDN2 | stopped 03:05 | WIP on branch night-gdn2, no row |
| ATT1 | in-bundle passenger | not significant alone (warm +0.09% CI includes 0) |

## Telemetry method
Max power/temp over all samples; SM clock min/max/median over busy samples (util>50; for files without util, clock>500 MHz). Excursion = power >= 245 W, min busy SM clock >5% below the file median, or temp >= 85 C.
Result: 3 of 524 files flagged, all sm_clock only (min 575 vs median 1151): B3 g4-b3 A warm c02, GDN4 g4-gdn4 A warm c00, D0 vram_deep.csv (also max power 244.03 W, just under limit). These look like ramp-up/transition samples inside busy windows, not sustained throttling; not otherwise investigated. No file reached 245 W or 85 C (max 244.03 W, 74 C).

## Gaps
- No lib sha256 for E07, MOE1, HG1, AF1, MOE2 (not recorded); GDN1/GDN3/SK1 shas matched to local lib files.
- Dates inferred from trial timestamps/mtimes; ledger carries none.
- ATT1: DAY.md CIs differ from analysis.json (both noted); deep ABBA has no CI.
- BUNDLE1 ledger says "main 961f104" but the candidate log shows that as an E07 evidence commit.
- Memory-only VRAM files (smoke_vram*, SMOKE2/3 vram) have no power/clock/temp and are not indexed. vram_speed.csv has power only.
- BUNDLE3 numbers left empty for the lead.
