# MOEN2: would a true n=2 MoE kernel pay? (CPU/GGUF metadata only)
Model: qwen35moe, 40 layers, E=256, k=8, expert ffn 512, shexp ffn 512 (GGUF metadata).
Observed ids: NONE found. grep hits under day/night/moonshot are profiler CSVs, SASS, build logs, no router/expert-id dump.
So overlap below is the uniform-random BASELINE k^2/E = 0.25 shared experts of 8 per layer (3.1%), NOT observed.
Real routing is usually more correlated for similar prompts, so true overlap is likely >= baseline; unmeasured.
Bytes per layer (Q4_0 head-q40b), from tensor sizes:
- one routed expert (gate+up+down) = 1,769,472 B; 8 selected = 14,155,776 B
- shexp (gate+up+down) = 1,769,472 B; router ffn_gate_inp 2,097,152 B + shexp gate 8,192 B
- total per row per layer = 18,030,592 B (~721 MB over 40 layers, ~18.0 MB/layer)
Shares of per-row moe bytes:
- (b) shexp + router = 3,874,816 B = 21.5%
- (a) expected overlap 0.25 x 1,769,472 = 442,368 B = 2.5%
- sum = 24.0% per row, just under the 25% rule
Actual n=2 step saving (both rows' bytes = 36.06 MB/layer; shexp+router read once instead of twice, shared experts once):
- saved 4.32 MB / 36.06 MB = 12.0% of two-row bytes (~1.0 ms of 8.6 ms if fully bandwidth-bound and moe_v2 includes shexp/router)
- routed experts only (if shexp/router are separate kernels): 3.1% = ~0.27 ms/step
Caveat: if the second row's shexp/router reads already hit L2 (3.9 MB/layer, fits M6000 L2 of 3 MB only partly), real gain is lower still.
Recommendation: CLOSE MoE n=2 as not worth a kernel. 24% per-row (12% of step) is below the ~25% bar, and
the only part that is certain (shexp/router sharing, ~1 ms max) is a cheap fusion, not a new n=2 kernel.
Reopen only if an observed expert-id log (e.g. 2 slots of real agentic traffic) shows coincidence >~30% (would add ~+8 pts).
Script: /tmp/g.py (read-only GGUF metadata, python3 -I; no tensors loaded).
- Next-session step 1 (advisor 13:xx): measure real 2-slot expert coincidence. Debug-only host print of topk_moe ids (no existing flag; GGML_CUDA_OCCAMY_DEBUG prints fusion plans only), ~50 steps on the D0 deep slots. Reopen if >30%.
