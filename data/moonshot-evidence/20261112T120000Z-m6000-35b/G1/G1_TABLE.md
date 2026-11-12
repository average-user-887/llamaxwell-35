# G1 numerics (KL(ref||test) nats, ref first-named; window=(doc,ckpt), full-vocab row0; nkv%4096 at last bulk ubatch)
| bucket | pair | n win | meanKL | maxKL | same-top1 row0 | same-top1 16 rows | meanTV |
|---|---|---|---|---|---|---|---|
| 16384 (nkv%4096=0) | C-vs-A | 6 | 0.00102 | 0.00295 | 100.0% | 99.0% | 0.0061 |
| 16384 (nkv%4096=0) | C-vs-B | 6 | 0.00316 | 0.00921 | 100.0% | 97.9% | 0.0206 |
| 16384 (nkv%4096=0) | A-vs-B | 6 | 0.0025 | 0.00737 | 100.0% | 99.0% | 0.0192 |
| 16384 (nkv%4096=0) | A-vs-Ap | 2 | 0.0029 | 0.0029 | 100.0% | 93.8% | 0.0194 |
| 32768 (nkv%4096=0) | C-vs-A | 6 | 0.00254 | 0.00965 | 100.0% | 99.0% | 0.0111 |
| 32768 (nkv%4096=0) | C-vs-B | 6 | 0.00146 | 0.00722 | 100.0% | 97.9% | 0.0056 |
| 32768 (nkv%4096=0) | A-vs-B | 6 | 0.00243 | 0.00702 | 100.0% | 96.9% | 0.0097 |
| 32768 (nkv%4096=0) | A-vs-Ap | 2 | 0.000407 | 0.000407 | 100.0% | 100.0% | 0.0012 |
| 65536 (nkv%4096=0) | C-vs-A | 5 | 0.0108 | 0.0436 | 100.0% | 100.0% | 0.0261 |
| 65536 (nkv%4096=0) | C-vs-B | 5 | 0.0387 | 0.182 | 100.0% | 100.0% | 0.0433 |
| 65536 (nkv%4096=0) | A-vs-B | 5 | 0.066 | 0.294 | 100.0% | 100.0% | 0.0610 |
| 65536 (nkv%4096=0) | A-vs-Ap | 1 | 0.259 | 0.259 | 100.0% | 100.0% | 0.2400 |
| shift65836 (nkv%4096=512) | C-vs-A | 1 | 0.00114 | 0.00114 | 100.0% | 100.0% | 0.0152 |
| shift65836 (nkv%4096=512) | C-vs-B | 1 | 0.0265 | 0.0265 | 100.0% | 100.0% | 0.0993 |
| shift65836 (nkv%4096=512) | A-vs-B | 1 | 0.0259 | 0.0259 | 100.0% | 100.0% | 0.1027 |
| shift65836 (nkv%4096=512) | A-vs-Ap | 1 | 0.00368 | 0.00368 | 100.0% | 100.0% | 0.0334 |
- 16384: matched n=6 dmean=+0.00214 top delta=+0.0 / -1.0 pt
- 32768: matched n=6 dmean=-0.00107 top delta=+0.0 / -1.0 pt
- 65536: matched n=5 dmean=+0.02795 top delta=+0.0 / +0.0 pt
- 65836: matched n=1 dmean=+0.02536 top delta=+0.0 / +0.0 pt
Controls (codex window, full vocab): A-vs-A2 and B-vs-B2 maxabs logprob diff = 0, KL = 0 at 16384/32768/65536 (bitwise deterministic; noise floor 0)
dmean = meanKL(B,C)-meanKL(A,C) matched windows; top delta = B-vs-C minus A-vs-C same-top1 (row0/16 rows). Per-window rows: G1_windows.csv
A=Codex v4 lib GEMM off + BUNDLE3 env; B=A+GGML_CUDA_FA_MAXWELL_GEMM=1; C=same lib, -fa off, F32 KV, ub512 (generic F32 mul_mat; verified in log); Ap=A+FA_PARALLEL_BLOCKS_MULT=2; A2/B2 reruns
GEMM engaged only in B (maxwell_gemm log lines; 0 in A). Prod ckpt settings (16, min-step 2048). Docs: codex,codexs,d0,d1,fx1,u0 (B also fx2). 98K not measured (Codex own A-vs-B 98K KL 5.8e-6)
libs sha256: v4 0c7ffbde334686f17f5ed6a1676c0cd966ec931eaa388900befb06af493330d5 (all arms); prod bundle3 94c777dc55489fba4cab5b68c6b54cf8a45810b0a1011589260e5bd23cd05486 (unused). Cmds+rc: steps.log (all rc=0), run_all.sh, COMMANDS.txt
Dir: occamy35-day-20261112/evidence/G1/20261112T171109Z. Lock released, no g1-* containers.
Read: 65K codex outlier is chaotic sensitivity, not GEMM-specific: stock A moved 0.259 by a tile-order change (A-vs-Ap) ~ A-vs-B 0.294; same window shifted (nkv%4096=512): A-vs-B 0.026, A-vs-Ap 0.0037. Elsewhere B-vs-C dmean small (16K +.0021, 32K -.0011).
