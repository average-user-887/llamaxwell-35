# Card 28f: decode bandwidth analysis (CPU only, from card 28d trace)
Two-slot decode, ~90K ctx, batch8 libs, 257 steps, 2 tokens/step, bound 317 GB/s, target 85% = 269.5 GB/s.
Source: 28d/run-20261115T163429Z/trace.csv (nvprof), kernel time / 257 steps.

| op | ms/step | bytes/step (analytic) | achieved GB/s | % of 317 | ms at 85% | ms saved/step at 85% | rank |
|---|---|---|---|---|---|---|---|
| gdn_fused2 (state only, lower-bound bytes) | 5.81 | 264 MB | 45 | 14% | 0.98 | 4.83 | see note |
| gdn_fused2 (state + qkv/z/out weights, likely) | 5.81 | 830 MB | 143 | 45% | 3.08 | 2.73 | 2 |
| moe_v3 (routed experts, 15.75 unique/layer) | 7.92 | 1115 MB | 141 | 44% | 4.14 | 3.78 | 1 |
| mul_mat_vec_q (dense: head + attn q/k/v/o) | 2.23 | 439 MB | 197 | 62% | 1.63 | 0.60 | 3 |
|   of which lm_head | 1.29 | 286 MB | 221 | 70% | 1.06 | 0.23 | |
|   of which attn q/k/v/o | 0.93 | 153 MB | 165 | 52% | 0.57 | 0.36 | |
| ofa_decode_hs_ar (+combine) | 13.13 | 3692 MB | 281 | 89% | 13.70 | 0 (already above 85%) | 4 |

Total potential if every op below 85% reached it: ~7.1 ms/step (MoE 3.8 + GDN 2.7 + dense 0.6) of 31 ms wall (~23%), i.e. roughly 32 -> ~41 tok/s per slot. Realistic fraction far lower.

## Verdict per op
- ofa_decode_hs_ar: at the bandwidth wall (89%); only fewer KV bytes (Q8/Q4 KV, K-only quant) can help, not kernel work.
- moe_v3: 44%, biggest kernel-side headroom (~3.8 ms); per-launch ~0.1 ms with only 48 blocks on 24 SMs is the suspect (occupancy/latency, not bytes).
- gdn_fused2: 45% if it includes the projections (likely), 14% if state only; 72 blocks, ~0.195 ms per launch; headroom 2.7-4.8 ms, tied with MoE for first.
- mul_mat_vec_q: 62% overall, lm_head 70%; small headroom (0.6 ms), attn small-matrix projections (grids 256-4096) at 52% are launch-latency-limited.

## Bytes and assumptions
- Dims (server log): 40 layers, 10 full-attn (recurrent_layers pattern, interval 4), n_embd 2048, 16 Q heads, 2 KV heads, head_dim 256, expert ff 512, 256 experts/8 used, ssm inner 4096, state 128, 32 v-heads, 16 groups, conv 4. KV buffer F16 3840 MiB (K 1920 + V 1920) for 98304 cells x 10 layers x 2 seqs; RS buffer 125.6 MiB.
- Attention: 2 slots x 90128 tokens (mean of 90000..90256) x 10 layers x (512 K + 512 V) x 2 B = 3.69 GB. Assumes every KV byte read exactly once, no L2 reuse across the 8 Q heads sharing a KV head (reuse would make bytes lower and achieved GB/s lower; +-1% from context length). Time includes ofa_combine (17 ms total).
- GDN state: fp32 32x128x128 per layer per slot = 2.10 MB, read + write, x30 layers x2 slots = 252 MB, plus conv state 12 MB. Confirmed by RS buffer size (65.8 MB per seq).
- GDN projection weights (qkv 2048->8192, z 2048->4096, out 4096->2048, Q4_0) = 566 MB. ASSUMPTION: these run inside gdn_fused2. Evidence: the GDN layers in the trace show only moe_v3 x2 and gdn_fused2 between norms, no mmvq; attention-layer mmvq launches (q 4096 blocks, o 1024, k/v 256) reproduce 10 layers exactly. If wrong, state-only row applies. Biggest uncertainty in this table (factor ~3 in GDN bytes).
- MoE: expected unique experts per layer for 2 independent tokens, uniform routing = 256*(1-(248/256)^2) = 15.75; per expert 3 x 2048 x 512 x 18/32 = 1.77 MB. Real routing is skewed and the two slots are different texts, so true unique count is likely 12-16 (bytes 0.85-1.13 GB, achieved 107-143 GB/s, saved 3.7-4.8 ms). Shared expert/router (~71 MB) not counted; if inside moe_v3 achieved is 150 GB/s, saved 3.5 ms. moe_v3 has two launches per layer (0.104 + 0.091 ms); assumed gate/up and down.
- Dense: all mmvq are ggml_type 2 (Q4_0); head assumed Q4_0 (248320 x 2048, 286 MB; model file name suggests q4_0 head). Head time = the 124160-block launches (255 of 257 steps).
- Time per step uses 257 steps for all kernels; trace is nvprof, serialised kernel times, ~95% of the wall step. Launch gaps and the 1.4 ms of non-listed kernels not attributed.
- 317 GB/s is the datasheet bound; measured practical peak on this card is not in the trace, so "% of 317" slightly understates attainable fraction and the 85% target may itself be above what sm_52 sustains. Attention at 89% suggests ~285-290 GB/s is reachable.
