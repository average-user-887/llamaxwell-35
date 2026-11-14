**DRAFT: results of cards 7d, 7e, 11 and 3e are pending.**

# Night report 13-14 Nov (M6000 35B), written after the 08:00 cut; nothing pushed, prod unchanged

| card | what | N3 makespan vs prod-now (177.5 s) | adopted |
|---|---|---|---|
| 1 | S1 single-stream prefill | 169.7 s, -4.5% (3 pairs) | not yet (in stack) |
| 4 | WP2 fused split softmax | 165.97 s, -7.1% (n=1); single +4K@90K 21.21 s vs 23.22 | not yet (in stack); FP64 RMSE pending |
| 3 | Q tiling >1024 rows | tests 32/32 OK on GPU in stack | in stack |
| 2 | multi-stream GEMM | correct, bit-identical; no N3 (S1 leaves 0% ne3 loss) | no |
| 7 | stack = 1+3+4 | **161.2 s mean, -9.3%** (161.0/161.4/161.3 vs 177.5/177.7/178.1); TTFT 17.8 vs 22.2 s; gap 10.9 vs 11.8 s | not swapped |
| 8 | moe_v3 decode occupancy | decode +3-4% single, hashes equal; no N3/KLD | no |

Target 156 s missed (161.2 s); 1% gate passed. Prod unchanged: BUNDLE3 + GEMM, F16 KV, ub1024. Rollback line stays as in STATE.md.
Stack: branch stack 246209f, $N2/7/bin, flags GGML_CUDA_FA_MAXWELL_GEMM=1 GGML_CUDA_FA_MAXWELL_GEMM_FUSED=1 --prefill-single-stream.
Open: FP64 RMSE of fused vs stock, round-trip identical-token check, ub2048/C1 rows for card 3, publication of RESULT.md files.
