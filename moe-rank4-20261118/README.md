# Approximate rank-4 MoE projection on Maxwell

On one 80K performance fixture, the rank-4 cap raised cached throughput from **89.900 to 96.517 tok/s (+7.36%)** versus rank-6 on the same CUDA build. It saved **2.360 seconds** on the cached request and **1.514 ms per draft proposal**. The cold request took 0.072 seconds longer.

| Measurement | Rank-6 | Rank-4 |
| --- | ---: | ---: |
| Cold wall, 192 output tokens | 268.32966 s | 268.40153 s |
| Cold decode throughput | 81.88590 tok/s | 81.04675 tok/s |
| Cold drafts accepted / proposed | 84 / 106 | 75 / 115 |
| Cached wall, 3,072 output tokens | 34.45134 s | 32.09114 s |
| Cached decode throughput | 89.90007 tok/s | 96.51669 tok/s |
| Cached decode time per proposal | 21.89063 ms | 20.37688 ms |
| Cached drafts accepted / proposed | 1,509 / 1,561 | 1,508 / 1,562 |

Each mode served one cold 80,000-token request and one same-slot cached replay with 79,996 tokens reused and four evaluated. Both processes mapped the same binaries, offloaded 42 of 42 layers and logged target cap activity on q1 and q2 and long-context Sparse R2 activity. The sample establishes dispatch at those shapes. The first output difference was token 23 in both conditions. The performance fixture had no semantic score.

An [earlier uncapped Sparse R2 no-MTP run](../split-warm-3072-20261118/README.md) measured 76.416 tok/s cached. That historical comparison changes both the MoE cap and speculation. The effect of speculation alone with rank-4 is still open.

## Independent exact checks

On a separate 2K smoke fixture, both ranks returned the expected arithmetic result through EOS. Rank-4 matched the dense masked-top-four reference in three live checks of 2,048 elements each, covering q1 row zero and both q2 rows. These checks compared the same capped arithmetic. The rank-4 cold and cached speed-fixture outputs first differed from rank-6 at token four.

A separate rank-4 80K request returned the known fact planted at token 40,000. Its seven output IDs through EOS matched both the [stock-rank split reference](../split-32k-recall-20261118/README.md) and the [rank-6 split reference](../moe-rank6-20261118/README.md). The recall run exercised rank-4 and long-context Sparse R2 with full offload. These checks cover one arithmetic question and one planted fact.

## Reconstruct and run

Follow the [published Sparse R2 and cache-only stack](../stacked-mtp-20261118/README.md), then apply the [rank-6 patch](../moe-rank6-20261118/candidate.patch) before the rank-4 patch. Forward and reverse checks at zero fuzz passed. Both intermediate and final files matched their measured source byte for byte. The compiled stock, rank-6 and dense masked rank-6 device instruction words matched the prior rank-6 build.

Set `source_dir` to the reconstructed stack, `package` to this package, `rank6_package` to the published rank-6 package, `r2_package` to the Sparse R2 package and `build_dir` to a fresh build directory.

```sh
git -C "$source_dir" apply "$rank6_package/candidate.patch"
git -C "$source_dir" apply --check "$package/candidate.patch"
git -C "$source_dir" apply "$package/candidate.patch"
cmake -S "$source_dir" -B "$build_dir" -G "Unix Makefiles" -C "$r2_package/maxwell-release.cmake"
cmake --build "$build_dir" --config Release --parallel 2 --target ggml-cuda
```

The opt-in `GGML_CUDA_MOE_RANK4=1` path uses the existing sm_52 V3 target-trunk route and resident-grid guards. It renormalizes the four retained ranks, physically skips the second routed chunk and retains the shared expert. Rank-6 takes precedence when both flags are set. Set only the measured cap in each arm. Leave `GGML_CUDA_MOE_RANK4_CHECK` and `GGML_CUDA_MOE_RANK6_CHECK` unset for timings. The common split-MTP1 arguments and feature flags, exact source hashes and measured binary hashes are in [evidence.json](evidence.json). A fresh build can produce different binary hashes. The N80 experiment worker produced the CUDA patch. Existing source attribution and licensing apply.
