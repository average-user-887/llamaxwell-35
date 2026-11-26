# LlamaXwell-35 R3 v5 benchmark receipts

These results come from a Quadro M6000 24 GiB (`sm_52`) with CUDA 12.8. Each model pair uses the same executable, host library and CUDA library in both arms. The fixture uses Occamy Q4_0 MTP, one slot, 84,224 context cells, `-b 2048 -ub 1024`, maximum draft depth 1, FP16 draft K/V, greedy sampling and a fixed seed. The first request has no cached prefix. Successive requests reuse the prefix in the same server process. Rates count generated tokens during decode. Cold TTFT measures elapsed time to the first token.

[Machine-readable receipts](receipts.json) include SHA-256 digests for the model, executable, measured libraries, source manifest and private source receipts. The export contains aggregate measurements and parity checks. The evidence covers the measured greedy fixtures and the MTP profile.

## Shipped v5 runtime

Both selector arms enable fixed-stride Q2 attention. The enabled arm also sets `GGML_CUDA_OCCAMY_FA_Q2_R2_MID_TOP16=1`. The order reverses between the two fixtures. Each pair emits 192 cold tokens and two 3,072-token warm runs. All 6,336 generated token IDs per arm, streamed text, prompt/cache counts and MTP proposal/acceptance counts matched across arms and against the qualified v4 reference.

| Prompt tokens | Order | Warm control tok/s | Warm enabled tok/s | Gain |
| ---: | --- | ---: | ---: | ---: |
| 16,384 | OFF then ON | 108.158 / 108.117 | 110.266 / 110.226 | 1.95% / 1.95% |
| 40,000 | ON then OFF | 107.196 / 107.214 | 109.548 / 109.542 | 2.19% / 2.17% |

Warm intervals had 8–10 samples each at the same median graphics and memory clocks of 1,151 and 3,304 MHz, all P0. The enabled 40K cold run had a median graphics clock 19 MHz higher than its control. The table reports decode gains.

A separate held-out 16,512-token chat-templated prompt matched all 768 token IDs per arm and the v4 reference, with exact text and counters. It used one 384-token cold request and one 384-token cache replay. This is a parity check on another fixture.

The selector applies to padded widths 16,640–19,456 and 40,192–43,264 in multiples of 256. The Q2 revision-2 fixed-stride path also requires two KV heads, 12 splits, one resident block per SM, 512-byte K/V head strides and matched token pitches of 1,024 or 2,048 bytes. Top16 changes block selection while retaining the generic scan and fixed-stride attention compute. The standalone operator compared both pitches across eight shapes and 104 exact cases. At the model's observed 1,024-byte pitch, complete attention latency fell 19.23–19.90%. The 1,024-byte dispatch was observed in the model runs.

## Earlier build comparisons

Each row measures its own toggle on an earlier CUDA binary identified in the receipts. The v4 stack rows toggle dense K/Q mask, KV tail trim, Gate Q8X2, GDN TILE8, J48 expert tiles and mid-context fixed stride together. The other v4 rows isolate fixed stride. Keep these comparisons separate from the shipped v5 selector measurements.

| Comparison | Prompt tokens | Warm control tok/s | Warm enabled tok/s | Gain |
| --- | ---: | ---: | ---: | ---: |
| v4 stack | 2,000 | 122.315 / 121.953 | 123.562 / 123.131 | 1.02% / 0.97% |
| v4 stack | 40,000 | 103.831 / 103.798 | 107.182 / 107.122 | 3.23% / 3.20% |
| v4 fixed stride | 16,384 | 107.086 / 106.962 | 108.012 / 108.015 | 0.86% / 0.98% |
| v4 fixed stride | 40,000 | 105.726 / 105.670 | 107.175 / 107.115 | 1.37% / 1.37% |

Each pair matched all 6,336 generated token IDs per arm. The 40K v4 stack enabled arm had a 13 MHz median cold-prefill graphics clock advantage, and the enabled 40K fixed-stride arm had a 31.5 MHz advantage. The released-profile measurements do not establish performance for serial, Q8C or additional slots.

The J48 prefill comparison isolated 48-column expert tail tiles on an earlier CUDA binary. Cold TTFT at 16,384 tokens was 34.237 → 33.943 s in the forward pair and 34.256 → 33.953 s in the reverse pair. Each pair matched 576 generated token IDs and the compared text and counters. The prefill intervals had matched median clocks of 1,151/3,304 MHz, with 11 P0 samples per arm in each comparison.

## Experimental stride range

An independent experimental CUDA binary isolated `GGML_CUDA_OCCAMY_FA_Q2_R2_EXTEND_STRIDE`. At 8,192 tokens the warm runs improved 1.10–1.24%. At 56,000 tokens the warm runs improved 1.25–1.28%. Each pair matched 2,240 generated token IDs per arm, streamed text, cache and MTP counts. The receipt identifies this separate binary. The v5 launcher uses its two measured mid-context bands.

## Receipt export

Run `python3 export_receipts.py /path/to/campaign` with the saved comparison and plan files to recreate `receipts.json`. The exporter checks matched runtime hashes and output parity before serializing explicit aggregate fields. Raw prompts, outputs and machine-specific runtime details remain in the source records.
