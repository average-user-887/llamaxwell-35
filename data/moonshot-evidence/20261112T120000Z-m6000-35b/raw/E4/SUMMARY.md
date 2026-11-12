# E4 phase trace + E1 engagement (nvprof, 12 Nov, single run per arm)
Arms: A = GDN4 (prev prod), B = BUNDLE3 (prod since ~14:20). Run dirs 20261112T132117Z-A, 20261112T131846Z-B; per-phase kernel CSVs in */analysis/.
| phase | A ms/tok plain | B ms/tok plain | B vs A | A gap ms | B gap ms |
|---|---|---|---|---|---|
| prefill 4K (ms/tok) | 1.830 | 1.836 | +0.3% | 119 | 114 |
| N1 shallow | 13.42 | 12.99 | -3.2% | 1.25 | 1.07 |
| N2 shallow | 21.77 | 18.62 | -14.5% | 3.53 | 1.76 |
| N2 deep (D0 slots ~98K+90K) | 36.79 | 33.38 | -9.3% | 5.32 | 2.34 |
Single nvprof+plain run per arm. Indicative only, not ABBA; the timed claim stays with the G1 deep ABBA (-8.98% +-1.21, provisional).
E1 engagement (B, kernel names, per decode step): N2SPLIT -> n2gather_copy 10 + gdn_fused2 30 (gdn_fused n1 = 0); FA_ADDR -> ofa_decode_hs_ar 10 (fa_hs/fa_base 0); N2GDN -> gdn_fused2 replaces gdn_fused in N2 phases. All three treatments ENGAGED.
Note: the A classifier also counts n2gather_copy 10/step in N2 phases; name-match check pending (may be the generic copy kernel).
B deep busy split: ofa_decode_hs_ar 14.24 ms (45%), moe_v2 8.57, gdn_fused2 5.85, mmvq q4_0 int=2 2.25 (41 launches/step).
B N1: moe_v2 4.61, gdn_fused 3.61, head_rowpack 1.19, mmvq q4_0 0.88 (40 launches/step).
Prefill 4K (B): sgemm 28%, mul_mat_q q4_0 ~42%, flash_attn_tile 8.3%, gated_delta_net 5.8%, dequantize_block_q4_0 3.9%.
