# Q8 KV as a capacity project (2x128K): paper sizing, no GPU (12 Nov)
Geometry (BUNDLE3 server log): 40 layers, 10 full-attention (KV) layers, n_head_kv=2, head_dim 256 (K and V); GDN recurrent state 125.6 MiB (2 seqs), fixed.
KV bytes/token (all 10 layers, K+V): F16 20,480 B; q8_0 (34 B per 32 elems) 10,880 B.
Today: -np 2 -c 196608 (2x96K) F16 KV = 3840 MiB; measured peak 23,108 MiB of 24,463 MiB total -> margin ~1,355 MiB.
| config (2x128K = 262,144 cells) | KV MiB | delta vs today | est. margin |
|---|---|---|---|
| F16 K + F16 V | 5,120 | +1,280 | ~75 (no-go: compute buffer + Codex GEMM scratch +412 do not fit) |
| q8_0 K + F16 V | 3,920 | +80 | ~1,275 |
| F16 K + q8_0 V | 3,920 | +80 | ~1,275 |
| q8_0 K + q8_0 V | 2,720 | -1,120 | ~2,475 |
Margins exclude compute-buffer growth with n_ctx (408 MiB today, partly n_kv dependent). Confirm with a load-only check before any plan.
Stock status (Codex Q8_FORWARD_LOOK): -ctk/-ctv q8_0 disables the occamy fast decode and the GEMM prefill. Speed is flat or slightly worse (+0.39%/-0.70%), so Q8 is capacity only.
Required native scope to keep today's speed at 2x128K:
1. Decode: ofa_decode_hs_ar (FA_ADDR) reads q8_0 K and/or V blocks with FP32 dequant in-register (no dp4a). It is ~45% of deep N2 busy, and halving bytes there may also help speed.
2. KV write: k_set_rows f32->q8_0 already exists upstream; check it is dispatched for both K and V.
3. Prefill: flash_attn_tile and Codex GEMM need F16 KV. Options are a per-chunk q8->f16 dequant scratch (~2 heads x 256 x 2 B x n_kv per K or V tensor, = 256 MiB at 128K for K+V) or native q8 loads in the GEMM path.
4. Quality: KLD gate dmean <= +.0010, same-top >= -.3 pt, shallow and deep. E10 (the quality screen) was NOT_RUN today. An external report gives KL 0.039 on a similar 35B, so a fail is plausible.
Recommendation: run E10 as a quality screen only (no timing) first. Only on a pass, open an Opus design card for (1)+(3), starting with the variant that halves the most bytes that passes.
