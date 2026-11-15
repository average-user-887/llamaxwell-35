# Card 28h: gdn_fused2 two-slot decode (microbench, grid 72, M6000 sm_52, 250 W)

| variant (gdn_fused2, 2 seqs) | us/launch | GB/s | A-end (min/max) | B (bar1-out to B1-end max) | C (bar2-out to C-end max) | bit-equal vs base | vs base |
|---|---|---|---|---|---|---|---|
| base (batch8 kernel) | 200.2 / 201.6 | 138 | 101.1 / 105.9 | 45 us | 41 us | ref | 1.00x |
| **A2ROW** (phase-A row pairs, committed, default off) | **177.0** | 156 | 73.7 / 82.8 | 46 us | 40 us | out, conv, ssm, scratch: equal | **1.131x** |
| A2ROW + phase-B state tile prefetch (one item ahead) | 178.2 | 156 | 73.7 / 82.9 | 46 us | 40 us | equal | 1.126x (no gain, reverted) |
| A2ROW + alpha/beta hoist in phase B | 178.5 | 155 | 73.7 / 82.8 | 46 us | 41 us | equal | 1.130x (no gain, reverted) |
| A2ROW + alpha/beta hoist + phase-C pass-2 first-row prefetch | 179.4 | 155 | 74.1 / 83.6 | 46 us | 41 us | equal | 1.124x (no gain, reverted) |
| A2ROW at grid 66 (base 206.3-207.2) | 180.7-181.7 | 153 | | | | equal | 1.14x |

Gate (card): >= 1.15x on the op, i.e. <= 174 us. **Not met: best 1.131x.** A provisional N3 row was run anyway on coordinator request (below).

## Bytes per launch (settled from the source)

| item | size |
|---|---|
| phase A weights (qkv 8192 + gate/z 4096 + alpha 32 + beta 32 = 12352 rows x 1152 B, Q4_0-style) | 14.23 MB |
| phase C ssm_out (2048 rows x 2304 B, repacked) | 4.72 MB |
| SSM state, 2 seqs x 2.1 MB, read + write | 8.4 MB |
| conv state | small |
| total | ~27.7 MB; base 138 GB/s, A2ROW 156 GB/s |

The projections sit inside gdn_fused2 (whole layer window). The split2 host path can also issue a gather
cudaMemcpyAsync of the states when s_copy is not (head, head+1); that is outside the kernel and not timed here.

## Diagnosis

- Base phase A (~100 us for 14.2 MB, 142 GB/s) was bound by shared-memory bandwidth. Each weight row reloads
  the lane's 64-float x chunk for both seqs (16 KB of LDS.128 per row per warp), about 57 us at 128 B/clk,
  against ~14 us of FMA.
- The fixable stall: A2ROW dots two rows per activation load (dot2v2x2 keeps each row's fmaf chains in the
  dot2v2 order, so the output is bit-identical). Phase A drops by 27 us to 74-83 us (~180 GB/s). It is now
  close to co-bound by the nibble decode/FMA ALU work (~3.7 us per pair-iteration) and LDS (~2.7 us). The
  remaining 9 us spread is the tail of 21 vs 22 rows per warp.
- Phase B (~46 us for both seqs) and phase C (~40 us, two K-halves) did not respond to load-latency fixes
  (state-tile prefetch, alpha/beta hoist, pass-2 first-row prefetch were all null or slightly worse, at 80 regs
  with 56 B spill loads). Phase B is a serial per-sequence chain: ~3.6 items per block, each with 3-4
  __syncthreads, plus a per-seq tail of threadfence, atomics and gated-norm finalisation. The next real
  lever is to interleave seq0 and seq1 inside each phase-B item, which would halve the serialised chain
  (estimate -15 to -20 us). That change needs double phase-B smem buffers and touches the reduction order
  (reorder-level, not bit-equal). It did not fit before the 18:35 cut.

## Status

- Branch night15-gdn28h, commit 54efa6d (parent d4c13eb = batch8 source). Env GGML_CUDA_GDN2_A2ROW=1
  selects gdn_fused2<1>; default off, so it adds no change to prod.
- The result is provisional/partial: gate missed by about 3 us; one provisional N3 ABBA row (2 runs per arm).
- Microbench rerun of the committed build (reproducibility): grid 72 201.5 -> 177.7 us (1.134x), bit-equal.
- Bench files: bench/mb.cu, mb2.cu, kern*.inc, gen_ts.py, run.sh, result_mb.txt, result_mb2_*.txt.
- The production 35B model was unloaded (POST, one model) for the bench runs and was not restored
  (no-restore rule).

## N3 ABBA short row (provisional, 2 slots)

| arm | config | n valid | makespan mean s | delta vs A | max gap mean / max s | VRAM peak mean MiB |
|---|---|---|---|---|---|---|
| A | batch8 libs, batch8 env (ref) | 2 | 159.11 | - | 5.61 / 5.61 | 23867.00 |
| B | 28h libs 54efa6d + GGML_CUDA_GDN2_A2ROW=1 | 2 | 158.25 | -0.54% | 5.59 / 5.60 | 23867.00 |

| run | makespan s | max gap s | VRAM MiB | engaged | valid |
|---|---|---|---|---|---|
| A_1 | 159.18 | 5.61 | 23867.00 | a2row=0 n2gdn=33000 moe_v3=65536 | yes |
| A_4 | 159.04 | 5.60 | 23867.00 | a2row=0 n2gdn=33000 moe_v3=65536 | yes |
| B_2 | 158.24 | 5.59 | 23867.00 | a2row=1 n2gdn=33000 moe_v3=65536 | yes |
| B_3 | 158.27 | 5.60 | 23867.00 | a2row=1 n2gdn=33000 moe_v3=65536 | yes |

Method: n3_replay.py short profile (card 26 pattern), prod arm, --no-sweep --no-ctxfull, order A B B A, cap1024 ub2048 b2048,
env from run-35b-batch8-20261115.sh + LLAMA_PREFILL_SINGLE_STREAM=1 + GGML_CUDA_OCCAMY_DEBUG=1; MAXWELL_GRAPHS unset. A run is void unless
n2gdn launches and moe_v3 launches are > 0 and the "n2gdn variant A2ROW" line is present (B) / absent (A). Noise ~0.5%. Provisional.

Within-arm spread is 0.14 s (A) and 0.03 s (B), so B is faster in every run, by -0.86 s (-0.54%). That matches the expected size: about 23 us x 30 GDN layers per two-slot decode step. It is at the gate noise level and below the 1% release gate. KLD was not run; the variant is bit-equal on the op. Run dir: run-20261115T172404Z.
