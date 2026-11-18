# Sparse R2 attention for three query rows on Maxwell

On a Quadro M6000 with 81,920 F16 keys, enabling three-row Sparse R2 reduced synchronized attention time from **1,501–1,502 µs to 365–369 µs per call**. Each of the two order-reversed comparisons cleared the predeclared saving of 530.4 µs per call.

| Measurement | Full three-row attention | Sparse R2 three-row attention | Time saved |
| --- | ---: | ---: | ---: |
| Full then sparse | 1,501.33 µs | 368.95 µs | 1,132.38 µs |
| Sparse then full | 1,501.95 µs | 365.47 µs | 1,136.48 µs |
| Separate whole-attention CUDA events | 1,503.277 µs | 366.117 µs | 1,137.159 µs |

The four synchronized benchmark processes used the same CUDA library and case. Only the three-row enable flag changed. The event comparison ran separately with per-call instrumentation. The paired savings project to **11.34 ms across ten attention layers** at this geometry. The experiment measured one operator and loaded no model. Actual decode throughput and the effect on draft acceptance remain open.

## Numerical comparison and limits

For the causal case, an independent invocation of the existing two-row and one-row R2 kernels provided the reference. All **12,288** output values were finite and within tolerance. The maximum absolute difference was **1.118 × 10⁻⁸**. All six row and KV-head selection plans and the three causal visible boundaries matched the reference. The fixture uses 16 query heads, two KV heads and head width 256. The reference uses row views with the original query and mask strides.

The separate distant-needle diagnostics reproduced the existing Sparse R2 policy. The probed key was recovered. A key between probe positions was omitted and the output differed substantially from full attention. The maximum absolute difference against full attention was 0.00813 for the probed case and 0.99092 for the omitted case.

The first wrapper exited 1 after its parser treated a legitimate zero-failure counter as a failure. Cleanup returned 0. The numerical diagnostics completed on the GPU before that exit, while timing had yet to start. The corrected parser independently revalidated the archived diagnostics and rejects nonzero failure counters. The timing continuation completed all six measurements and cleanup with exit 0 using the same CUDA library and benchmark executable.

## Reconstruct and repeat

Start from the published [Sparse R2 reconstruction](../sparse-r2-20261117/README.md) using its `r2` target. Apply [candidate.patch](candidate.patch) directly over that source. It includes the CUDA header change and the benchmark/test registration in `tests/test-backend-ops.cpp`. Both patched files matched the measured source byte for byte. The compile check also found identical device instruction words for the 33 inherited functions and four new three-row functions.

With `source_dir` set to the reconstructed R2 tree, `package` to this directory, `r2_package` to the published R2 package and `build_dir` to a fresh build directory:

```sh
git -C "$source_dir" apply --check "$package/candidate.patch"
git -C "$source_dir" apply "$package/candidate.patch"
cmake -S "$source_dir" -B "$build_dir" -G "Unix Makefiles" -C "$r2_package/maxwell-release.cmake"
cmake --build "$build_dir" --config Release --parallel 2 --target ggml-cuda test-backend-ops
```

Use the built executable and matching libraries. Set the shared environment and the case-specific overrides in [evidence.json](evidence.json). The three-row path defaults to off. It requires `GGML_CUDA_OCCAMY_FA_Q3_R2=1`, Sparse R2, and a contiguous output. Use a fresh process for each case because the code caches the flags on first use. In the synchronized comparison, use `GGML_CUDA_MV1=0` and zero for both correctness-check flags. Apply `GGML_CUDA_MV1=2` in the two separate event cases. Diagnostic cases have their own overrides in the receipt.

```sh
"$test_ops" perf -b CUDA0 -o FLASH_ATTN_EXT -p 'kv=81920,nb=3,.*mask=mrow_causal' -j 1
"$test_ops" test -b CUDA0 -o FLASH_ATTN_EXT -p 'kv=81920,nb=3,.*mask=sparse_needle' -j 1
```

The measured build replaced the affected CUDA and test objects in the original R2 build. Fresh binary hashes can differ. The N98 experiment worker produced the patch and comparison fixture changes. Existing source attribution and licensing apply.
