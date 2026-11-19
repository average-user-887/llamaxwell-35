# Four-row down projection on Quadro M6000

The four-row kernel saved **7.990–10.856 µs per layer** in four mirrored comparisons of the complete two-token Occamy MoE operator. Every numerical and physical dispatch check passed. All four timings fell below the prespecified 13.1 µs/layer threshold for advancing to an 80K model run. Two mirrors had complete clock observations. The card stopped after the operator. [Preregistered protocol](RELEASE.md) · [Physical result](result.json)

| Routed experts in common | Pair order | Cohort µs | Four-row µs | Saved µs | SM and memory clock median ratios |
| --- | --- | ---: | ---: | ---: | --- |
| 0 | Cohort → four-row | 163.313634 | 155.324129 | 7.989505 | Unknown for four-row |
| 0 | Four-row → cohort | 164.121022 | 154.755707 | 9.365315 | 1.00 and 1.00 |
| 5 | Four-row → cohort | 160.608321 | 149.752384 | 10.855937 | 1.00 and 1.00 |
| 5 | Cohort → four-row | 160.697567 | 150.671354 | 10.026213 | Unknown for four-row |

Each timed interval comprised 32,768 synchronized calls after 4,096 warmups per path per fixture. The operator counted 278,528 paired launches and corresponding second-callback acknowledgments, of which 139,264 used the candidate. The GPU owner completed the card and cleanup in 47 seconds. The two decision thresholds were 13.1 µs saved in every mirrored pair and SM and memory median ratios within 2%. Six of eight timed windows contained clock samples. All six observed 1,151 MHz SM and 3,304 MHz memory. The other two windows contained no observation whose source timestamp lay inside the interval. [Raw timing](perf.log) · [Clock observations](fleet-clock.jsonl) · [Runner status](status.txt)

Each warp now consumes four adjacent output rows from one selected expert. The shared expert uses all 32 lanes. Four outputs take nine warp dots, down from twelve, with the original Q4_0 weight byte count, all eight routed experts, rank weights, FP32 dot arithmetic, ordered reduction and residual. Each fixture checked two ordered rank lists and 4,096 finite output words against the original cohort, then repeated the candidate. Both kernels physically admitted two 512-thread CTAs per SM on all 24 SMs. The new kernel compiled to 62 registers with zero stack or spills. The 17 original device functions retained all 46,032 instruction and control words. The control relink reproduced 369c6b05. The candidate measured here was b71f011f. [Raw diagnostic](check.log) · [Build and complete instruction counts](build.json) · [Portable receipt](portable-receipt.json)

The smallest measured saving across 40 layers projects to **0.31958 ms**. Even full transfer would leave **0.20170 ms** of the merged-draft model's 0.52128 ms gap at its archived acceptance rate. An independent companion would need to cover the remainder and demonstrate that the combined timings transfer to the model. A future combination test can retain both fixture orders, raise each timed window to 98,304 calls (roughly 15 seconds here) and allow eight seconds after the last window for Fleetdash delivery. Associate samples by original observation timestamps and keep uncovered intervals unknown. This repeat was not run.

## Verify the archive

Run the following in this directory.

```sh
sha256sum --check SHA256SUMS
python3 -B verify-layout.py
python3 -B report.py --check
python3 -B report.py --report .
```

The last command rebuilds the saved decision, including all clock windows, from the raw diagnostic and timing logs. The released clock file contains the ten original timestamps and their SM and memory values. Its source and public digests are recorded in observation-pins.json. Frozen local manifests identify the exact observed binary, code and runtime inputs. The public build commands, compiler log and clock sampler use typed placeholders for the private build path and Fleetdash URL. hydrate-source.py checks the portable and restored byte identities given authorized local substitutions. SHA256SUMS covers every released file except itself. [Source pins](source-pins.json) · [Clock data pins](observation-pins.json)
