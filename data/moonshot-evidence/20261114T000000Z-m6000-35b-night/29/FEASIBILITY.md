# Card 29 feasibility: V-side Q8 KV on batch8 (CPU investigation only)

Verdict: NOT feasible as a speed row. Q8 V (and Q8 K) makes the Maxwell fused FA GEMM path reject; no GPU row queued.

## Findings
- Prod (run-35b-batch8-20261115.sh): no -ctk/-ctv => F16/F16 KV; -fa on, -np 2, -c 196608, -ub 2048, --mixed-batch-size 1024; env GGML_CUDA_FA_MAXWELL_GEMM=1, _FUSED=1, OCCAMY_FA=1, FA_PIPE=3, FA_ADDR=1. No KVQ8 env set.
- "K-only Q8 adopted 13 Nov" = owner accepted it on the adoption list, but it was NOT adopted: HANDOFF/STATE.md says GEMM rejects q8_0 K (engaged=0, reason ktype; +4K@90K 30.47 s vs 23.09 s F16 GEMM). Prod stayed F16 KV. The "-3.2% K-only helped" note was the older non-GEMM decode path.
- Source ($M/moe-v2-20261114/src/ggml/src/ggml-cuda/fattn-maxwell-gemm.cuh:229-231): requires Q F32, K==F16 (OFG_ktype), V==F16 (OFG_vtype). No in-kernel dequant; q8 K or V => fallback for the whole op, so prefill/append loses the GEMM (the -26% gain) and goes to stock/tile.
- Only q8 consumer: Q8N native read (fattn-occamy.cu ofa_launch kvq8, fattn-occamy-kvq8.cuh), needs GGML_CUDA_OCCAMY_KVQ8=1 exactly, FA=1, PIPE=3, ADDR=1, and applies to decode only (Q->ne[1]==1, ne[0]==256 dim). It reads q8 K and/or V in-kernel (V scale not folded; K+V measured flat "issue-bound"). RULES.md: Q8S strict FAIL all arms, Q8N measurement-only; DECISIONS.md: Q8 K+V decode deferred, needs KLD.
- Net: with q8 V, prefill falls back from GEMM (large loss, cf. -0.46% vs b3 earlier), decode could use Q8N (small, unproven). Gain would need WP3b-style kernel work (GEMM pack step dequant of q8 K and V) = new kernel, out of card scope.

## Arm flags (if a kernel ever lands; none engages today)
- A: batch8 as prod (F16/F16).
- B: A + -ctk q8_0  (expect counters: engaged=0, reasons=ktype)
- C: A + -ctk q8_0 -ctv q8_0 [+ -e GGML_CUDA_OCCAMY_KVQ8=1 for decode-only native read] (expect engaged=0, reasons=ktype,vtype)
- Engagement check: log line "occamy_fa_gemm: engaged=N fallback=M reasons=..." must show engaged>0 and no ktype/vtype.
## Step 2: skipped (no arm engages the fused path).
