# Q8_1 projection for the two-token GDN verifier on Maxwell

The Q8_1 Phase A path saved **0.955 ms and 0.985 ms** across the 30 recurrent layers in two order-reversed pairs on a Quadro M6000. These are sums of CUDA event intervals around the entire fused GDN T2 kernel in one two-token target verification, following a 2,000-token prefill and the same four-token prefix in four fresh processes.

| Pair run order | Original mean per layer | Q8_1 mean per layer | Mean saved per layer | Sum saved over 30 layers |
| --- | ---: | ---: | ---: | ---: |
| Original then Q8_1 | 197.049 µs | 165.219 µs | 31.829 µs | 0.955 ms |
| Q8_1 then original | 196.862 µs | 164.025 µs | 32.837 µs | 0.985 ms |

Both pairs cleared the predeclared 26.67 µs mean-per-layer gate. The same measured CUDA library ran both modes with graph capture disabled. Every arm offloaded 42 of 42 model layers and recorded one event per recurrent layer. The observed verifier work used the same 72-block grid and three resident CTAs per SM in both modes.

## Numerical and rollback results

The new path uses stock-format Q8_1 activations and shares each Q4 weight read across the two token rows. The persistent warp reduction can round differently from stock. At layer zero, both 8,192-channel QKV rows came much closer to stock. Their RMS residuals were **1.574 × 10⁻⁷** for the pending row and **1.590 × 10⁻⁷** for the deliberately wrong row. These are 0.00276% and 0.00127% of the corresponding original-versus-stock gaps, below the predeclared 1% ceiling. Both projections differed from stock bitwise.

Both controls matched the archived N34 reference, and the two runs within either mode repeated bitwise for the captured output and state. The candidate preserved the rollback tap order on the one forced rejection. All arms matched the fixture's post-correction top-1 choices. Candidate recurrent state and logits still differed from both the original kernel and stock. The snapshot comparison covers active recurrent planes and excludes attention KV. This result measures one two-token window on one history. End-to-end 80K performance and serial numerical identity remain open.

## Reconstruct the exact CUDA change

Start with the published [Sparse R2 reconstruction](../sparse-r2-20261117/README.md). The recipe applies the FA2 base layers in order and then the cumulative R2 patch. Apply this package's patch after R2. The optional cache-only C++ overlay is independent of this CUDA change. The input CUDA file and headers matched both the measured R2 source and the published reconstruction. The [receipt](evidence.json) records their SHA-256 values and the exact patched-source and measured library hashes.

Set `source_dir` to the reconstructed R2 source tree, `package` to this directory, `r2_package` to the published Sparse R2 package, and `build_dir` to a fresh build directory.

```sh
git -C "$source_dir" apply --check "$package/candidate.patch"
git -C "$source_dir" apply "$package/candidate.patch"
cmake -S "$source_dir" -B "$build_dir" -G "Unix Makefiles" -C "$r2_package/maxwell-release.cmake"
cmake --build "$build_dir" --config Release --parallel 2 --target ggml-cuda
```

The measured build recompiled the CUDA translation unit and relinked the R2 CUDA library using its existing `sm_52` toolchain. The published CMake preset is the source build recipe. Fresh binary hashes can differ from the measured relink. The measured build kept all 16 original device kernels' compiled instructions and scheduling words unchanged.

Set `GGML_CUDA_OCCAMY_GDN_T2_Q8A=1` to request the experimental path. By default it is off. Runtime admission requires Maxwell `sm_52`, 24 SMs, the 72-block grid and matching three-block residency. The separate `GGML_CUDA_OCCAMY_GDN_T2_Q8A_EVENTS=1` flag is for diagnostic captures. The measured source patch was produced by the Astra 6.2 CUDA experiment. Existing source attribution and licensing apply.
