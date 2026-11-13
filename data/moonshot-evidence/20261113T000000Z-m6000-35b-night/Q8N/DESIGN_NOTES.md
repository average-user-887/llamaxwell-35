# Q8N design notes (supporting arithmetic and source references)
Source: E/G1b/src = 2436f75 + cap patch (57076fe); build E/G1b/build (sm_52-real, -Xptxas -v). CPU only, no GPU used.
Evidence in this dir: sass_hs_ar_bundle3src.txt (cuobjdump -sass of ofa_decode_hs_ar_kernel), resusage_hs_ar.txt.

## Geometry (E/Q8 sizing, BUNDLE3 log)
40 layers, 10 full-attention, n_head_kv 2, 16 q heads (GQA 8), D 256. KV bytes/token over all layers: F16 20,480; q8 K+V 10,880; one-sided 15,680.
DRAM bytes/step (sum of both streams' n_kv): 2x98,304 = 196,608 tok gives F16 4.03 GB, K+V 2.14 GB, one-sided 3.08 GB. E4 fixture 98K+90K (~188.4K tok) gives 3.86 / 2.05 / 2.95 GB.
Measured 14.24 ms on the E4 fixture means 271 GB/s achieved, 85% of 317. The projections assume the same efficiency.

## Kernel facts (fattn-occamy.cu, hs_ar = lines 416-562)
Block (split, kvh, stream) has 256 threads = 2 head groups x 4 unit-warps. A unit is 4 tokens. Lane owns 8 dims, and the hg warp owns 4 q heads.
Both hgs re-read the same K/V rows (L1/L2), so every element is loaded+converted twice. FMAs are not duplicated.
ptxas: hs_ar 118 regs / 0 spill / 33,664 B smem; flag-off ofa_decode_kernel<0> 200 regs x 128 thr. Both give 2 blocks/SM (65,536/(120*256)=2; 96 KB/33.6 KB=2). The resident count for nsplit comes from the flag-off kernel (2*24 SM = 48).
Loop SASS (0x1510-0x2948), per warp per unit: 486 instr = 242 FFMA, 65 F2F, 48 FMUL, 30 SEL, 18 SHFL, 18 FADD, 13 IADD, 9 LDG (8x .128 + mask), 6 DEPBAR, 5 LDS, rest small.

## Pipe estimate (24 SM, 4 schedulers, ~1.114 GHz; per element = 2 x 486/2048 warp-instr)
Issue: 2.013e9 elem (2x98K) x 0.475 = 9.56e8 warp-instr / 1.07e11 per s = 8.9 ms (8.5 ms on the E4 fixture).
Conversion pipe (8 lanes/partition, 4 cyc/warp-op): 65 F2F gives 4.8 ms. FMA pipe (308 FP ops) gives 5.7 ms. So today is BW-bound (12.2-14.2 ms) with issue at about 60%.
Layout A additions per warp-unit: loads 8 rows x (3 LDG.32 + 1 LDG.U16) - 8 = +24; PRMT realign 16; scale F2F 8; K fold FMUL 16; V p*d FMUL 16. Total +80, so 566 (+16.5%).
  Issue becomes 9.9 ms (E4 fixture) against a BW floor of 7.6 ms, so it is issue-bound. With 85-95% issue efficiency: 10.4-11.6 ms.
  One-sided: +40, so 526 -> 9.2 ms issue against 10.9 ms BW. That is BW-bound, ~11 ms.
  Conversion pipe: (65+8)/65 x 4.8 = 5.4 ms, not binding. That is why I2F is preferred over magic-number FADD (more issue).
Layout B (dim split: hg0 = dims 0-127, hg1 = 128-255, lane = 4 dims x 8 heads, so q/o stay 32 regs each): each element is loaded and converted once.
  Est. per warp-unit: 24 loads + 8 PRMT + 32 I2F + 8 F2F + 256 FFMA + 64 FMUL fold + 32 rescale + tree/softmax ~70, about 500 (but over half the dims).
  The issue estimate is ~8.7 ms on the fixture. Cost: a cross-warp score sum per unit (smem + 64-thread named barrier), which is unmeasured.
Convert cost against bytes saved (kill rule): convert delta = +8 F2F per 64 elem (scale) plus +16 PRMT, about +0.6 ms. BW saving = 6.6 ms (K+V) or 3.3 ms (one-sided). No kill.
Step impact: N2 deep is 33.38 ms/step (E4 B). A K+V saves ~3-4 ms, about -9..-12%. B saves up to ~5 ms.

## Alignment detail (layout A)
o = 34b + 2 + 8q (b = lane>>2, q = lane&3). b even: o%4 = 2, so the window is words at o-2, o+2, o+6 (12 B), with the bytes in w0.B2..w2.B1. b odd: o%4 = 0, 2 words, w2 address := w1 (no overrun at b=7, q=3).
PRMT selectors are per lane and precomputed (b-parity). The I2F byte selects are then fixed .B0-.B3 for every lane, so the code stays uniform.
The scale d is at 34b; read it by LDG.U16 (or from w0.H0 when b is even, which is not worth the divergence). The generic vec path does the same 2-B memcpy (ggml_cuda_memcpy_1<4,2>).

## Generic q8 path facts (stock reference)
fattn-vec Q_q8_1 = true for quantized K: Q is quantised to q8_1 in smem; KQ = int dot (dp4a emulated on sm_52) x dK x dQ. V: dequantize_V_q8_0 in fp32 (FP16_AVAILABLE needs cc>=600).
Instances without FA_ALL_QUANTS: f16-f16, q8_0-q8_0 (and q4_0). Mixed q8/f16 means f16_fallback, a full K+V to_fp16 every call (decode included).
Tile/MMA: need_f16_K = need_f16_V = true, so ggml_cuda_flash_attn_ext_get_f16_extra_data converts ggml_nelements(K/V) per call. Scratch is n_kv x 512 x 2 B per tensor.
GEMM (GGML_CUDA_FA_MAXWELL_GEMM) guard requires F16 K/V and K count <=98,304. It is off in prod and irrelevant at 128K anyway.

## KV write
llama-kv-cache cpy_k/cpy_v use ggml_set_rows. CUDA set-rows.cu:300 dispatches set_rows_cuda_quant<block_q8_0, quantize_f32_q8_0_block> (cpy-utils.cuh:137).
d = amax/127 is stored f16; q = round(x/d_fp32). This is identical for both gate arms, so both read the same cache bytes.

## Open checks before coding (phase 2 start)
- Probe-compile a 10-line kernel to confirm that I2F.F32.S8 with a byte selector exists in sm_52 SASS (else BFE+I2F).
- Confirm that the q8 view passes the strides %16 and data %16 checks in the real graph (K view nb1=544, nb2=272); log once.
- Q8S quality screen result picks the combo; build only that instance.
