# Kernel inspiration (research card, 13 Nov 2026; web reads only, no GPU)
Verified = read on the page; (know) = from general knowledge/paper, not re-read this session (WebFetch of the PDF failed) - treat as unverified.

| Technique | Source (URL, file) | Card | Expected effect | CUDA-C? | Risk |
|---|---|---|---|---|---|
| Bank = reg%4; keep C accumulators and A/B fragments in disjoint banks; max 1 same-bank source per FFMA | github.com/NervanaSystems/maxas/wiki/SGEMM (verified) | WP2 / GEMM inner loop | Avoids 1-2 clk stalls per conflicting FFMA; cuBLAS-like loss est. ~28 GFLOPS of ~4000 (<1%) | Partly: ptxas chooses regs, no direct control; indirect via fragment layout/unroll | Low gain, cannot force |
| .reuse flags (3 operand slots, 8 B cache each); back-and-forth "swirl" FFMA order gives 39-47% reuse | same page; control-code bits: wiki/Control-Codes | GEMM micro-kernel | Wiki: --noreuse costs only "a couple hundred GFLOPS" (~5%) | ptxas emits reuse itself; serpentine (boustrophedon) order of the 8x8 FFMA loop in C helps it | Low |
| LDS.128 A/B fragments, 32 LDS : 512 FFMA (1:16) per 8-deep k step; double-buffered regs (R64-71/80-87 A, R72-79/88-95 B) | SGEMM wiki (verified) | WP2 | Hides LDS latency; ~98% FFMA issue bound | Yes: float4 smem loads + explicit register double buffer in C | Med: reg pressure (needs ~128 regs, 8x8 per thread) |
| Tile 64x64 (64 thr, 8x8/thread) vs 128x128 (256 thr); 8-wide k unroll; 8/16 KB smem double buffered with XOR toggle, ONE bar.sync per iteration | SGEMM wiki (verified) | WP2 tiling | 64x64 wins small/skinny (up to +25% vs cuBLAS), 128 wins large (+4.8% at 4096) | Yes | Low |
| Stay <128 regs: >128 drops occupancy (4->3 or 4->2 warps/sched) | SGEMM wiki | WP2, moe_v2 | Occupancy cliff at 128 (256 thr) | Yes: __launch_bounds__ / -maxrregcount | Low |
| Epilogue through smem in chunks of 8 regs, warp-synchronous shuffle, coalesced store | SGEMM wiki | WP2 | Cheap (~12 clk) epilogue; template for fused softmax: do it per 8-reg chunk | Yes | Med |
| maxDNN: 64x64 blocking, loses efficiency when N or minibatch not multiple of 64; author suggests other block sizes | arxiv.org/abs/1501.06633; github.com/ebay/maxdnn (MIT) | WP2 odd shapes (GQA 8 rows per kv head, head_dim) | Pad/choose tile variant per shape instead of one tile | Yes: template tiles (e.g. 64x32, 32x64) + predicated edge | Low |
| Online softmax fused into QK epilogue (FlashAttention-2 style: row max m, row sum l, rescale O) | Dao-AILab/flash-attention (know), flash-decoding blog below | WP2 | Removes S global round-trip; target most of the 16.8% pass | Yes | Med: rescale of PV accumulators; see note |
| Flash-Decoding: split KV into chunks, per-split partial O + 1 logsumexp scalar per row, tiny 2nd reduce kernel | pytorch.org/blog/flash-decoding/ (verified) | ofa_decode deep ctx | Fills SMs when slots=2 and ctx 90K (M6000 has 24 SMs) | Yes | Low |
| Dequant in registers: byte->float via magic number (0x4B000000 \| b) - 8388608.0f, or PRMT + int8 scale in half pairs, instead of I2F | CUDA C Programming Guide throughput table: 32-bit conversions are ~32/clk/SM on cc5.x vs 128 FFMA (know, re-check) | WP3b, ofa_decode V | I2F is quarter-rate and competes with FFMA; magic-number is 1 LOP + 1 FADD | Yes (PTX/CUDA C intrinsics __byte_perm, __float_as_uint) | Low |
| Delta rule chunkwise: chunk 64 (supports 16/32/64), UT transform T=(I+tril(diag(b)KK^T,-1))^-1 diag(b), w=T k, u=T v, then state update S += k^T (u - w S) | github.com/fla-org/flash-linear-attention fla/ops/gated_delta_rule/chunk.py (verified names); arxiv.org/abs/2406.06484 (know) | GDN | GDN is 1.5-2% share: max 1% win | Yes (small 64x64 triangular solve) | Not worth it now |

## Licenses
maxas: MIT (README + GitHub sidebar), archived Jan 2023. maxDNN: MIT (LICENSE.txt). FLA: MIT. PyTorch blog: idea only. All permit code reuse with notice kept; nothing copied here.

## SASS-only vs CUDA-C
- SASS-only: explicit bank assignment of R0-R95 blocks, exact .reuse flags, control codes (stall/yield/barriers, 17 bits per instr: stall 0-3, yield 4, wbar 5-7, rbar 8-10, wait mask 11-16), dual-issue (stall 0). maxas is Perl, targets sm_50/52, and needs a cubin round trip (cuobjdump -> edit -> assemble). Our env: ptxas only, so out of scope unless the owner wants the maxas toolchain; its own page says the gain over good ptxas code is ~2-5%.
- CUDA-C level: tile choice, serpentine FFMA order, float4 LDS, register double buffering, XOR smem toggle, one-barrier loop, launch bounds, smem epilogue, split-K, magic-number dequant.

## WP2 (fused mask + softmax)
- Per-thread 8x8 C tile means a row of S is spread over 8 threads (columns) in a 64-wide tile: row max/sum = in-thread reduce over 8 cols, then 3 shfl_xor steps over 8 lanes if mapped contiguously. Choose thread layout so a row's 8 owners are lanes within one warp.
- Do not keep full S rows resident. Per 64-key tile: S tile in 64 regs, compute m_new, p=exp2(s-m_new) (fold log2e into the Q scale), write P tile to smem (not global) as the A operand of the PV sgemm; O accumulators (head_dim 128 x 8 q rows) rescale by exp2(m_old-m_new) only when m changes (skip branch is warp-uniform often).
- Cheaper stepping stone if regs explode: two-pass without rescale: pass 1 QK computes only row max/sum (S discarded), pass 2 recomputes QK and multiplies by 1/l into P. Costs 2x QK flops but zero S traffic; QK is the small GEMM at 8 q rows (GQA packing) - likely still a win versus a global round trip at 90K keys. Verify by ncu-free timing.
- Mask: causal/slot mask is a per-tile predicate; skip fully masked tiles entirely (loop bound), only edge tiles apply -inf.
- Register budget from SGEMM page: 64 C + 16 frag + addressing fits <128; softmax stats add ~4 regs per row-owner; keep P in smem to avoid extra regs.

## WP3b (q8_0 K strip in pack step)
- q8_0 block = 32 int8 + fp16 scale. Pack step is bandwidth-bound; load as int4 (16 B = 16 values), expand with PRMT/magic-number into floats, one FMUL by the block scale (hoist scale: multiply once per 32-strip, or fold into Q-side scaling per key if packing keeps blocks aligned: s_k = scale_k * dot(q, int8) so dequant scale applies post-dot, 1 mul per key not per element).
- That fold (scale after dot) only works if the QK sgemm accumulates over exactly one q8 block; with head_dim 128 = 4 blocks, apply scale per 32-wide k slice - cheap if the k-loop step is 8 or 16 aligned to 32.

## ofa_decode / Q8 V
- K-only Q8 helped (-3.2%), K+V flat "issue-bound": suspect I2F-class conversions plus per-element scale on V. Fixes: (1) magic-number byte-to-float (no I2F); (2) V scale applies to the weight p, not the element: o += (p*scale_blk) * int8v -> 1 FMA/element after a cheap int->float; (3) convert 4 bytes with one 32-bit load and 4 PRMT/LOP; (4) half2 path unavailable fast on sm_52 (no packed FP16 FMA), so stay FP32.
- Split-KV: 8 q heads share one KV head, so one block reads KV once for 8 heads (already done). Split ctx 90K into >= 24-48 chunks per (kv head, slot); partial O (8x128 fp32) + m + l (8 floats each) to a workspace, small reduce kernel: O = sum_i exp(m_i - M) l_i-weighted... use logsumexp form per blog. Ensure chunks >= ~2K keys so the reduction stays <5%.

## moe_v2 occupancy (104 regs/40.75 KB -> <=64/<=24 KB)
- No direct source gives this; applicable: SGEMM page says occupancy cliff is register-count driven (128 regs); 64 regs on sm_52 gives 4 blocks of 128 thr (64K regs/SM). Tactics: smaller per-thread output tile (4x4 not 8x8: 16 acc regs instead of 64), stage only one buffer (drop double buffer smem, keep register-side prefetch), __launch_bounds__(128,4), -maxrregcount=64 compare spill count in ptxas -v. Dequant fragments just-in-time with magic-number, not pre-expanded.

## GDN
- FLA default chunk 64, pipeline: cumsum gate -> intra (kkt + solve_tril + recompute_w_u fused) -> sequential fwd_h state recurrence -> fwd_o. Fused intra is the part to port; state recurrence is inherently sequential across chunks. At 1.5-2% share, defer; revisit only if the other items finish.
