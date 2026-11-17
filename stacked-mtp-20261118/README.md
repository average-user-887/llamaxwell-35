# Sparse R2 with cache-only MTP catch-up

On the fixed 80,000-token Occamy 35B workload, adding cache-only catch-up to Sparse R2 MTP1 reduced cold request time from **283.971 s to 268.635 s** on a Quadro M6000. Cold time to first token fell from 281.551 s to 266.221 s. The stack generated all 192 output IDs in the same order as the Sparse R2 MTP1 reference in both cold and warm requests.

| Sparse R2 configuration | Cold TTFT (s) | Cold wall (s) | Cold decode (tok/s) | Warm wall (s) | Warm decode (tok/s) |
|---|---:|---:|---:|---:|---:|
| No MTP | 261.656 | 264.171 | 76.0934 | 2.627 | 76.6465 |
| MTP1 | 281.551 | 283.971 | 79.0497 | 2.739 | 77.3165 |
| MTP1 plus cache-only catch-up | 266.221 | 268.635 | 79.2945 | 2.729 | 77.4747 |

The stack cut cold wall time by **15.337 s (5.40%)** versus unstacked Sparse R2 MTP1. Against the same-sparse no-MTP control, the stack decoded 4.21% faster and took 4.463 s longer for the cold request and 0.102 s longer for the warm request. The earlier full-F16 no-MTP ABBA median was 57.6549 tok/s on the same fixture. The stack's 79.2945 tok/s is 37.53% higher than that historical decode reference, across both the sparse-attention and MTP changes. [Measurement receipt](measurements.json)

Each current arm used one cold request from a fresh, fully offloaded server and one same-slot warm replay. All three cold requests reused zero prompt tokens. All three warm requests reused 79,996 of 80,000 prompt tokens. Both MTP1 arms accepted 90/100 drafts cold and 87/103 warm. The no-MTP performance control generated a different token sequence from the MTP1 arms. Runtime mappings verified the Sparse R2 CUDA library and both cache-only C++ libraries in the stack. The stacked arm emitted live cache-only and full MTP markers and exercised the Sparse R2 kernel at query widths 1 and 2. The markers verify execution of those paths. Total dispatch coverage remains unmeasured.

Sparse R2 is an approximate attention path. Its original synthetic distant-needle miss remains a known limitation. Exact output matching here covers the stacked and unstacked Sparse R2 MTP1 runs on this one fixture. [Sparse R2 evidence](../sparse-r2-20261117/README.md)

## Reconstruct the stack

Use the published revision containing both components. Set `release_root` to a checkout of that revision and `source_dir` to a new working directory. Run these commands in the repository. [Published component revision](https://github.com/average-user-887/llamaxwell-35/commit/5f560ce2a8336c7e068723392070a03956c104a7)

```sh
git worktree add --detach "$source_dir" e4fdc23555f6a9103ea03b854ee5c70c6812517c
bash "$release_root/sparse-r2-20261117/reconstruct-source.sh" "$source_dir" r2
cache_patch="$release_root/cache-only-20261118/cache-only.patch"
git -C "$source_dir" apply --index --check "$cache_patch"
git -C "$source_dir" apply --index "$cache_patch"
git -C "$source_dir" apply --index --reverse --check "$cache_patch"
```

The reconstructed checkout matched all 11 modified source files byte for byte against the two experiment checkouts. The Sparse R2 patch supplies the CUDA selector and fixtures. The cache-only patch supplies the outputless draft catch-up graph and its dispatch. The published component packages record their source and build attribution. [Sparse R2 package](../sparse-r2-20261117/README.md) · [Cache-only package](../cache-only-20261118/README.md)

The measured overlay combined previously built libraries. Sparse R2 supplied `libggml-cuda.so`, cache-only catch-up supplied `libllama.so.0.6.0` and `libllama-common.so.0.6.0`, and the executable and server implementation came from the FA2 baseline. A fresh combined build remains unverified. Enable `GGML_CUDA_OCCAMY_FA_SPARSE=1`, `GGML_CUDA_OCCAMY_FA_SPARSE_R2=1` and `LLAMA_MTP_CACHE_ONLY=1` with the recorded MTP1 settings. [Build and runtime provenance](provenance.json)
