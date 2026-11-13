# N1 phase-separated kernel profile, deep append at 90K (BUNDLE3)

Arm A = GEMM off (prod BUNDLE3 lib). Arm B = GEMM on (FA_MAXWELL_GEMM=1, Codex v4 cand lib, NOT the BUNDLE3 lib: not a pure flag flip; it also carries N2GDN).

| arm | append | rep | wall s (HTTP) | prefill s (server) | GPU span s | GPU busy s | kernels |
|---|---|---|---|---|---|---|---|
| A | +1K | 1 | 8.00 | 7.97 | 7.94 | 7.91 | 4186 |
| A | +4K | 1 | 32.01 | 31.98 | 31.96 | 31.90 | 11443 |
| A | +1K | 2 | 8.01 | 7.99 | 7.95 | 7.92 | 4186 |
| A | +4K | 2 | 32.00 | 31.98 | 31.96 | 31.92 | 11443 |
| B | +1K | 1 | 5.87 | 5.84 | 5.81 | 5.77 | 5806 |
| B | +4K | 1 | 23.42 | 23.39 | 23.36 | 23.30 | 17573 |
| B | +1K | 2 | 5.90 | 5.88 | 5.84 | 5.81 | 5806 |
| B | +4K | 2 | 23.47 | 23.44 | 23.42 | 23.37 | 17573 |

### Grouped shares of GPU busy time in prefill (% ; ms)

| arm | append | rep | FA/attention | MoE matmul | dense mmq/mmvq | GDN/ssm | quantize/copy | other |
|---|---|---|---|---|---|---|---|---|
| A | +1K | 1 | 77.7% (6143) | 10.5% (830) | 7.5% (593) | 1.5% (121) | 1.7% (133) | 1.1% (87) |
| A | +1K | 2 | 77.7% (6154) | 10.5% (830) | 7.5% (594) | 1.5% (121) | 1.7% (132) | 1.1% (88) |
| A | +4K | 1 | 78.3% (24968) | 10.3% (3282) | 7.3% (2341) | 1.5% (480) | 1.5% (490) | 1.1% (343) |
| A | +4K | 2 | 78.3% (24994) | 10.3% (3276) | 7.3% (2333) | 1.5% (480) | 1.5% (490) | 1.1% (343) |
| B | +1K | 1 | 0.4% (25) | 14.2% (820) | 79.5% (4586) | 2.1% (121) | 2.3% (133) | 1.5% (88) |
| B | +1K | 2 | 0.4% (25) | 14.3% (829) | 79.4% (4615) | 2.1% (121) | 2.3% (133) | 1.5% (88) |
| B | +4K | 1 | 0.1% (26) | 14.0% (3263) | 80.2% (18697) | 2.1% (481) | 2.1% (492) | 1.5% (344) |
| B | +4K | 2 | 0.1% (26) | 14.0% (3281) | 80.2% (18747) | 2.1% (482) | 2.1% (492) | 1.5% (344) |

Full top-10 tables per arm x append (rep1 and rep2 ms, % of prefill GPU busy, % of wall, calls), restore and 1-token phases: REPORT_TABLES.md (same dir).

### Headline
- Arm A prefill: stock `flash_attn_tile<256,256,4,8>` owns 78% (+4K: 24.95 s of 31.9 s, 50 calls). MoE mmq 10%, cuBLAS dense 7%, GDN 1.5%.
- Arm B (GEMM on): prefill 23.4 s at +4K (-27%), 5.85 s at +1K (-27%). Attention is now GEMM kernels, classified as "dense" by the name-based grouper: ~1280 extra `sgemm_128x128x8_NT_vec` calls (QK, ~6.2 s, inferred from call-count delta vs A), `maxwell_sgemm_128x64_nn` 5.9 s (PV), `maxwell_gemm_masked_softmax` 3.9 s, `sgemm_32x32x32_NN` 0.14 s => ~16.2 s, ~69% of B prefill is attention GEMM. Non-attention work is unchanged vs A (MoE ~3.3 s, GDN 0.48 s).
- Restore ~0.56 s HTTP (0.15 s GPU, 80 copies); 1-token step ~0.10 s HTTP, 23 ms GPU, top kernel `ofa_decode_hs_ar_kernel` ~7 ms. Same in both arms. Prefill is 98% of the append cost.

## Method
- Container (own, removed after) runs llama-server under nvprof (--print-gpu-trace, CSV), -np 2 -c 196608 -b 2048 -ub 1024 --cache-ram 0 --ctx-checkpoints 16 -fa on, F16 KV, BUNDLE3 env copied verbatim, MAXWELL_GRAPHS absent. Held the bench lock; prod model unloaded via the proper POST; GPU confirmed idle.
- Per cell: restore the 90K slot (n_restored 90000), prefill the prompt with +1K or +4K fresh tokens (n_predict 1, temp 0), then a probe = same prompt plus one extra token (1-token batch) as a stand-in for the first decode step (n_predict 1 itself has no decode step). 5 s idle between steps; phases are split by those idle gaps in the nvprof trace and cluster order. Order: warm(1K), r1_1K, r1_4K, r2_1K, r2_4K.
- Percentages are of prefill GPU busy time; "other" includes norms, rope, elementwise.

## Caveats
- nvprof, not nsys; no NVTX. Phase split via idle gaps. Profiling overhead looks small (wall 32.0 s matches unprofiled ~32 s).
- Arm B uses the Codex v4 lib; differences may include N2GDN or other build changes, not only the GEMM flag (kernel counts for GDN/MoE are identical to A, so the effect shown is attention).
- FA call counts (20 at 1K, 50 at 4K) are consistent with 10 attention layers x ubatch count; inferred.
- QK/PV attribution of sgemm calls in B is by call-count delta, not by tracing launch sites.
- Restore was page-cached. Only slot 0 used. Two repeats agree within ~0.3%. Used the G1b build/bin llama-server.

## Files (N1/)
RESULT.md, REPORT_TABLES.md, n1_run.sh, n1_all.sh, n1_drive.py, n1_an.py, n1_report.py, run/armA/{trace.csv,steps.json,server.log}, run/armB/{...}, run-failed-notrace-1/ (first attempt, empty trace, kept), all_A.log, all_A.failed1.log, all_B.log.
