# Rank-cap serial controls and the Q8_1 combination

On one 80K workload, native one-token MTP improved warm decode by **13.95% at rank-4** and **13.64% at rank-6** against serial on each cap's own mapped runtime. Both comparisons used 192 cold and 3,072 warm output tokens. All warm requests reused 79,996 prompt tokens and evaluated four.

| Same-target comparison | Warm decode, serial → MTP | Warm wall, serial → MTP | Cold wall change with MTP |
| --- | ---: | ---: | ---: |
| Rank-4 | 84.69939 → 96.51669 tok/s | 36.39864 → 32.09114 s | +4.48968 s |
| Rank-6 | 79.41547 → 90.24548 tok/s | 38.80873 → 34.30895 s | +4.51898 s |

Each row uses one cold and one warm request per mode, with the cap fixed and only `--spec-type` changed from `none` to `draft-mtp`. Both pairs had full GPU offload and observed the capped target path. Serial sampled q1 and MTP sampled q1/q2. The first output difference appeared at token 23 for rank-4 and token 51 for rank-6 in both cache conditions. The original rank-4 serial attempt failed phase attribution after log rotation. Its cleanup passed. The repeat retained the logs and passed the original verification. [Per-request measurements and settings](evidence.json)

## Q8_1 combined with rank-4

On a separately composed CUDA library, enabling Q8_1 Phase A raised warm decode from **96.16174 to 96.96568 tok/s (+0.84%)** and reduced warm wall from **32.20681 to 31.94068 seconds**. Decode time per proposal fell **1.06329 ms**, while warm draft acceptance changed from 1,508/1,562 to 1,437/1,634. Cold wall increased **1.03538 seconds**. Both outputs first differed at token eight. The two runs mapped the same composed module. Only `GGML_CUDA_OCCAMY_GDN_T2_Q8A` changed, with diagnostic events off in both.

The measured build kept the ordered rank-4 link inputs and substituted the previously verified Q8_1 GDN translation unit. It reused the same 147-object link contract. The parent rank-4 and GDN device-word checks passed. The composed run is one fresh pair, separate from the rank-4 serial isolation above.

## Unscored output structure

The benchmark forced a long Python continuation to the emission cap and had no semantic score. A structural check of its warm outputs found substantial repetition. For the counts below, trim and drop blank lines, replace each contiguous digit run with the same marker, and count occurrences beyond the first occurrence of each normalized line within the specified slice.

| Warm pair | Trailing consecutive comment-only nonblank lines | Second-half normalized duplicate lines |
| --- | ---: | ---: |
| Rank-4 serial → MTP | 0 → 248 | 5.56% → 91.27% |
| Rank-6 serial → MTP | 0 → 74 | 25.66% → 95.74% |

The Q8_1-off warm output matched the earlier rank-4 MTP token sequence. Q8_1 removed its trailing comment run on this sample, while the last two quarters still had **80.0% and 90.8%** normalized duplicate lines. These structural counts do not score correctness or general quality.

The earlier rank-cap arithmetic and midpoint recall checks used separate scored fixtures. The Q8_1 plus rank-4 comparison has no separate scored outcome in this package. Reuse the individual source patches with their published reconstruction and the typed flags in this receipt. [Rank-6 patch and scored checks](https://github.com/average-user-887/llamaxwell-35/blob/2fa4f548a787df2ff183f5531fa04580aa61196d/moe-rank6-20261118/README.md) · [Rank-4 patch and scored checks](https://github.com/average-user-887/llamaxwell-35/blob/8237d9fc62804332f2f4c660dfad567282b1a35a/moe-rank4-20261118/README.md) · [Q8_1 Phase A patch](https://github.com/average-user-887/llamaxwell-35/blob/7d35f484e55fc4435e5bc99e66b27cbcda0e35e4/gdn-q8-phase-a-20261118/README.md)
