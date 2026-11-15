# Card 18 RESULT: GDN phase-B two-sequence interleave (GGML_CUDA_OCCAMY_GDN18, night14-gdn18 5b1e570)
| check | result |
|---|---|
| gate (CHECK, 600 windows) | bit-equal, 0 mismatches, 0 server crashes |
| N3 stack (3 runs) | 161.12 / 161.61 / 161.45 s, mean 161.39 |
| N3 gdn18 (3 runs) | 161.31 / 161.31 / 161.27 s, mean 161.30 (-0.06%) |
| decode ABBA x12 | single_30k +0.11%, both_30k -0.26%, both_deep +0.11% (token sha differs in some pairs) |
Verdict: NO GAIN (inside noise, below 1% gate and below scout's 0.5% line). Not adopted; flag stays default-off. Earlier run hit misaligned-address bug (fixed 5b1e570).
Run dir: 18/run-20261115T025426Z
