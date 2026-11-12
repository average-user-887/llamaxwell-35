# Night ledger 11->12 Nov
| round | candidate | gates | result | merged |
|---|---|---|---|---|
| R1 | h2h prod c9825b1 vs C24 6f239c9b; profile | G4 ABBA x5 | C24 -0.46% warm, -0.34% d4k (CI excl 0): keep c9825b1. Profile d0: moe_v2 4.77ms 41.6%, gdn 3.73 32.6%, head 1.19, attn .42 (1.86 @16k) | evidence 6f5e9d5, 0a832be |
| E07 | OCCAMY_FA_PIPE=3 head-split attn decode (215fe8e) | G0 G1 bitexact G3 G4 | +2.34% d16k, +1.10% warm, +1.17% d4k, CI>0 | merged main, tag main-night-e07 |
| note | BUNDLE rule (advisor interpretation, owner may overrule): stacked sub-gate wins qualify if bundle passes G1 bitexact + G4 >=+1% CI>0 | - | - | - |
| note | BUNDLE rule OWNER-APPROVED (relayed by advisor session, 12 Nov ~00:00 CET): stack of sub-1% bitexact changes may merge if bundle clears +1% CI>0 | - | - | - |
| MOE1 | MOE_LOP3=1 nibble unpack lop3 (11112ee, parent c9825b1) | G0 G1 bitexact G4 | +0.252% warm CI low +0.10, +0.236% d4k: sub-gate, PARKED for bundle | no |
| GDN1 | OCCAMY_GDN_PREF=1 load-issue hoist/prefetch in gdn_fused<0,1,1,1> (1fc225d, parent 722b98f) | G0 G1 bitexact, kill-screen | nvprof -1.5us/call x30 = ~0.045-0.054 ms/tok (<0.05): 1-cycle sanity +0.20% warm; PARKED for bundle | no |
| HG1 | OCCAMY_HOSTGAP=1 pinned staging + 1 gather kernel for graph inputs (4598cc9, 930c1ca) | G0 G1 bitexact, kill-screen | nvprof d0 period -0.035 ms (trimmed mean, median ~0); gap 240.7->169.6us median (noise ~80us); PARKED for bundle | no |
| AF1 | OCCAMY_ATTN_QKV fused Q/K/V q4_0 matvec (72198ab) | G0 G1 bitexact, kill-screen | fused QKV 0.672 ms/tok vs 0.601 unfused (+0.07 slower); busy deltas noise: REJECT (not bundled). 1b not started | no |
| GR1 | env GGML_CUDA_MAXWELL_GRAPHS=1 on main+FA_PIPE=3 (no code) | G1 bitexact, no per-token recapture, 1-cycle sanity, nvprof | warm -0.15%, d4k +0.17% (1 cycle); period median 11.970->11.986 ms: no gain, REJECT (not bundled) | no |
| MOE2 | MOE_TAILBAL=1 phase-C tail balance moe_v2<0,61> (d324789 on 11112ee) | G0 G1 bitexact, kill-screen | busy mean -0.022 ms (<0.05), busy_min -0.113, moe_v2 -0.115 ms (-2.4%) consistent x4: PARKED for bundle | no |
| GDN3-A | occupancy discriminator gdn_fused grid 24 vs 48 (3995600, measurement-only GRID caps) | nvprof d0 | fused 125.7->145.6 us (+16%), B +34%, C +7%; moe_v2 +2%: occupancy matters for GDN -> card B pilot started | no |
| BUNDLE1 | MOE1+MOE2+GDN1+HG1 merged on main 961f104 (night-bundle a25300c), lib b1c263ab | G0 0 spills, G1 bitexact (OFF=E07=ON), G4 ABBA x5 vs E07 lib | warm 84.88->86.81 t/s +2.28%±0.28, d4k 82.36->84.11 +2.13%±0.29: PASS bundle gate -> merge main | yes |
