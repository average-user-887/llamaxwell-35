# Occamy 35B sparse attention R2

Sparse R2 with MTP1 measured **78.7911–78.9974 tok/s** across two requests (median **78.8942 tok/s**) for 192 output tokens after a fixed 80,000-token prompt on a Quadro M6000. This is an experimental, default-off approximation. Sparse recall remains unqualified.

| Attention and decoding | Requests | Server decode tok/s | Median prefill s | Median request wall s | Draft accepts |
|---|---:|---:|---:|---:|---:|
| Full F16, no MTP | 2 | 57.6549 | 261.393 | 264.730 | — |
| Full F16, MTP1 | 2 | 64.1705 | 281.620 | 284.710 | 91/100 each |
| Sparse R2, no MTP | 1 | 75.9063 | 262.320 | 264.889 | — |
| Sparse R2, MTP1 | 2 | 78.8942 | 281.017 | 283.466 | 90/100 each |

Sparse R2 MTP1 had 3.94% higher median decode throughput than sparse R2 without MTP and 36.84% higher throughput than the earlier full-attention no-MTP median. Its median request took 18.578 s longer than the same-sparse no-MTP request. The full-attention observations are the ABBA reference medians. The sparse no-MTP arm is a single observation. The MTP1 arm includes the matching FA2, GDN and merged re-eval settings for that stack.

The first MTP1 wrapper exited 2 on a shell syntax error during final reporting after an in-place harness edit. The repeat wrapper exited 0. Both captured requests used the same prompt and binary hashes, loaded 42/42 GPU layers, exercised R2 at query widths 1 and 2, accepted 90/100 draft tokens, and produced all 192 output IDs in the same order. Both passed the independent 96-token streaming/detokenization check and completed cleanup. The no-MTP wrapper exited 0. Timings use the server's `predicted_per_second` field. See [allowlisted measurement receipt](measurements.json).

## Approximation and operator evidence

For each query row and KV head the selector scores four probes per 256-key block, retains the top 16 old blocks and keeps the recent causal tail of at least 4,096 keys. R2 combines each block's mask scan and four probe scores in the same CTA. It retains the R1 probe arithmetic and tie order. At 81,920 KV positions the operator ran at 184.23 µs for one query and 248.01 µs for two, versus 622.02 µs and 888.91 µs for full attention. Six comparisons matched R1's plans and old-score bits. The diagnostic covered short-context fallback. Only 81,920-position cases were timed by the selected operator filter.

Omitted blocks receive zero attention mass. One synthetic distant needle between probe positions was lost. Full-F16 MTP1 and Sparse R2 MTP1 both retrieved the planted fact exactly on the same 80,000-token prompt. That comparison covers one paired recall fixture. Selector identity checks compare R2 with R1. Quality qualification for approximate long-context decoding is open.

## Reconstruct the build source

Run these commands from the public repository with `package` pointing to this package and `source_dir` to a new directory.

```sh
git worktree add --detach "$source_dir" e4fdc23555f6a9103ea03b854ee5c70c6812517c
bash "$package/reconstruct-source.sh" "$source_dir" r2
```

The public evidence revision lacks the recorded `424828b` source base. Applying the first existing MV3 patch directly fails in three files. The included 16-file source bridge reproduces the build-source files at `424828bf9855bebfa3b2c86f51cf4a8211918060`. The existing public MV3 patches 01–03 then reproduce the FA2 build-source files at `26f4013d47b959e13f252abfc9372ac963d3bd1c`, also the build-source content of the measured local base `da7ba0e49dd43c69f9ee5a219ea3124f7acda0c8`. That last revision adds the quality harness. The reconstructed index leaves public evidence and data in place and omits the local quality harness. Digests cover tracked paths, modes and blobs outside those directories. [Source provenance](provenance.json)

The script applies the original public patches in their recorded order and stages the combined source without creating commits. Select `base` for the full FA2 source or `r1` for the verified causal R1 overlay. The R2 patch is cumulative and applies directly over the FA2 source. The public MV3 patches carry their source authorship. The bridge is a tree diff, with the endpoint commit IDs recorded separately. The original R1 and R2 overlay diffs had no commit headers. Their CUDA implementation and diagnostics were produced by the sparse experiment worker, configured as GPT-6.1 Sol xhigh. Existing source attribution and repository licensing apply.

## Build and runtime provenance

The measured overlay rebuilt `ggml-cuda` and `test-backend-ops` in a preconfigured CMake build using GCC 14.2.0, CUDA 12.8.93, CMake 3.28.3 and the `52-real` architecture. It reused the FA2 server and its other libraries. The included options describe a fresh source build using that configuration. The build receipt records measured binary hashes, which depend on the original build environment.

```sh
cmake -S "$source_dir" -B "$build_dir" -G "Unix Makefiles" -C "$package/maxwell-release.cmake"
cmake --build "$build_dir" --config Release --parallel 2 --target llama-server test-backend-ops ggml-cuda
```

Enable sparse R2 with `GGML_CUDA_OCCAMY_FA_SPARSE=1 GGML_CUDA_OCCAMY_FA_SPARSE_R2=1`. For diagnostics, `GGML_CUDA_OCCAMY_FA_SPARSE=2` retains full output and `GGML_CUDA_OCCAMY_FA_SPARSE_R2_CHECK=1` compares the two selectors. Disable check and profile flags for throughput. Unset both sparse flags to restore full attention. The receipt records model, fixture and binary fingerprints, request dimensions and MTP settings. The original text fixture and generations are excluded.
