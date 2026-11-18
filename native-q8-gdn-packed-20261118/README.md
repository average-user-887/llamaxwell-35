# Exact Q8 GDN weight-layout test on Maxwell

Moving the two-token Q8 projection weights out of shared-memory staging saved **0.716 µs per recurrent layer** in the first pair and cost **0.714 µs per layer** in the reverse pair. Across 30 layers the corresponding changes were +0.0215 ms and −0.0214 ms of savings. Both results missed the 28 µs solo threshold and the 21 µs threshold for combining with a separate attention improvement. The packed layout did not advance to a whole-model run.

| Fresh process order | Layout | Mean GDN kernel time |
|---|---|---:|
| A1 | Original | 170.047 µs |
| B1 | Packed | 169.331 µs |
| B2 | Packed | 170.267 µs |
| A2 | Original | 169.553 µs |

The work preserved the native Q8 two-row arithmetic, FP32 reduction order, full eight-expert routing and two-row MoE execution. Each 1,152-byte weight row became two 128-word groups and one 32-word group. The packed row loop removed three shared vector stores and nine scalar shared loads, issued two vector and one scalar global load, and retained 128 signed XMAD instructions. Both kernels compiled to 80 registers with zero spills. Runtime recorded 72 blocks and three resident CTAs per SM for GDN, 40 paired MoE dispatches and 40 skipped second callbacks. The two new device functions and all 17 original device functions had identical instruction words across the two implementation attempts.

The first attempt completed its control process and aborted at the first candidate lookup. Its single-row prefix bypassed the assumed preparation path, leaving zero prepacked weights. Exit 134 and successful cleanup are retained alongside all 30 control events and the incomplete candidate trace. No paired result was available from that attempt.

The second attempt obtained the 120 distinct native weight pointers after model loading and before context creation. It checked Q4 type, dimensions, alignment and GPU buffer placement, then made 30 private packed copies totaling 426,885,120 bytes (407.11 MiB). Full preparation took 11.430 and 14.424 ms in the two packed processes, outside all q2 kernel timings. Both timing runs used the same binary and the same fixed history. All four processes matched the original Q8 reference on four complete recurrent snapshots, three complete logit rows and the target choices through forced suffix rollback. Both comparisons were checked locally. The completed attempt took 70 seconds including cleanup.

The fixture covers one verification after a 2,000-token prefix. It does not measure 80K decode, 80K VRAM headroom or cold request latency. The earlier [phase attribution](../native-q8-gdn-phases-20261118/README.md) missed its perturbation gate. This experiment directly tested the layout.

Run `python3 -B replay.py` to reproduce timing differences from the 150 recorded layer events and to check all 60 weight-copy records. The per-layer event traces, whole-preparation times and boolean exactness verdicts are in `r0` and `r1`. Reproduction of the physical fixture requires its private reference and the original lab runtime. The CUDA patch applies to the [Q8 GDN source](../gdn-q8-phase-a-20261118/README.md) with the original [two-row MoE object](../native-q2-moe-cohorts-20261118/README.md). Each directory contains the full measured CUDA and host fixture source. `hydrate-source.py` takes the original build mount and a separate output path and verifies every reconstructed source hash. Model, software and build fingerprints are retained. Tensor values, generated token IDs and private output fingerprints stay local.
