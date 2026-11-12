# Interim evidence snapshot, 12 Nov 2026 (text only)

Interim snapshot taken at 2026-11-12 13:50 CET while the run is in progress; numbers may be superseded by the final push to main.
Rounds still being written (B3 especially) are copied as they were at that time.
Only text files (md txt log json csv sh py sass-summary) are included. In dumps/ only log.txt and prof.csv are kept.
Skipped (over 5 MB): rounds/D0/deep.csv (137 MB), rounds/D1BF/dumps/ks_p0/prof.csv (6.6 MB), rounds/D1BF/dumps/ks_p1/prof.csv (6.3 MB).
Not included: binaries, logits, state dumps, .so, .gguf, .sass disassembly.
Context and timeline: DAY.md. Figures below are as stated in DAY.md unless a raw file is cited.

| Item | Stated result | Pointers |
|---|---|---|
| D0 deep profile 16K/64K/96K | per-kernel tables at 16K, 64K, 96K (ofa_decode only depth-scaling, 7.43 -> 18.26 ms at 96K per DAY.md) | rounds/D0/deep_e_table.txt, rounds/D0/d0.ktable.txt; DAY.md 09:15 |
| D0 two-slot deep | batch-2 decode falls to stock mmvq/mmvq_moe/gdn_direct; margin 1433 MiB both slots deep | rounds/D0/vram_deep.csv, rounds/D0/deep2.log; margin figure: DAY.md only |
| D0 98K profile | not a separate table; profile is at 96K | rounds/D0/deep_e_table.txt |
| ATT1 (D1ATT) | bit-exact (G1 64/64, deep 64/64); warm +0.09% CI[-0.29,+0.46]; d4096 +0.24% CI[-0.15,+0.55]; deep two-slot 36.280 -> 36.147 ms (-0.37%, about 1 sd); none significant | rounds/D1ATT/g1-B.cmp.txt, gd-B.cmp.txt, deep_abba.log (36.280/36.147 confirmed), g4-att1-warm_decode.analysis.txt, g4-att1-d4096.analysis.txt (per-cycle values; the pooled +0.09/+0.24 are DAY.md figures, not re-derived here) |
| N2SPLIT (D1B) | two-slot deep 35.55 -> 33.94 ms/step, -4.52% CI[-5.55,-3.49] (DAY.md); KLD vs single-slot stock .00306 vs N2SPLIT .00082 | rounds/D1B/speed.log (its own t-CI line reads -4.54% [-5.63,-3.44], n=3 cycles; differs slightly from DAY.md, probably a different tool/rounding, not reconciled), rounds/D1B/kld_vs_solo.txt |
| N2GDN (D1BF) | two-slot deep 33.955 -> 32.822 ms/pair, -3.3% CI[-3.80,-2.72] | rounds/D1BF/speed.log (cycle3 33.965 -> 32.822 visible; CI line is DAY.md only) |
| N2GDN kill-screen trace 60 x 115.6 us vs 30 x 196.4 us per step | reported by the run lead, raw file not yet present (prof.csv files are over 5 MB and skipped here; ks.log shows BITEQUAL states) | rounds/D1BF/ks.log |
| GDN5 phase-A pilot (D1) | record only; 1-layer drift-dominated, +407 MiB for 30 layers, not adopted | rounds/D1/pilot.log, rounds/D1/prof/ |
| DtoH desk check (D2H) | closed | rounds/D2H/REPORT.md |
| MoE n=2 desk check (MOEN2) | closed (uniform-overlap estimate, no observed expert-id log) | rounds/MOEN2/REPORT.md |
| N2DIFF | CPU read of residual n=2 vs n=1 differences; no fix today | rounds/N2DIFF/REPORT.md |
| BUNDLE3 gate (B3) | IN PROGRESS. Raw results present at snapshot: G1 BITEQUAL 64/64 (g1-B.cmp.txt); warm_decode analysis verdict INSUFFICIENT_CYCLES, mean +0.05% (g4-b3-warm_decode.analysis.json); server deep ABBA raw output "DELTA period B vs A -8.98%" (b3_deep.log) and "-1.00%" (b3_deep1.log), not interpreted here; smoke logs smoke_A/B.log. Build 421f2fe logs: build-421f2fe.log. Adoption has NOT been decided in this snapshot. | rounds/B3/ |

## Not claimed
- Chained percentages across candidates are not direct measurements; only the bundle gate measures the stack.
- A serving smoke run is not proof of correctness.
- 64-step exactness does not cover all contexts.
- Stock two-slot output already differs from single-slot output (rounds/D1B/kld_vs_solo.txt); N2SPLIT is closer to single-slot than stock, not identical.
