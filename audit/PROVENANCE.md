# Provenance - Occamy 35B MoE build (as of 2026-11-10)

## What runs
Container `llamacpp-m6000` = image `local/llama.cpp:server-cuda-b11552-sm52-normq8` (built 2026-10-12) with **one file bind-mounted over it**: `/app/libggml-cuda.so`.

| Piece | Source | Identity |
|---|---|---|
| `llama-server`, `libllama*`, `libggml-base/cpu` | image, source = commit 2 (`normq8`) | per-file sha256 in `image-app-sha256.txt`; image metadata in `image-inspect-slim.json` |
| `libggml-cuda.so` (deployed) | commits 3+4 (this repo HEAD) | **sha256 `d14a076e89dad41245d6e1b5c6709872930ed655394e775dae120243fec0a460`**, 37,792,760 B, release asset `libggml-cuda-35b-d14a076e.so` |
| model | `occamy-1.0-Q4_0-headq40b.gguf` (not included) | launch flags in `server-args.txt`, mounts in `mounts.txt` |

## Commit ladder (diff each layer)
1. `23b0202` upstream b11552 (unmodified).
2. normq8 - source of the image's server/libllama.
3. gate-copy tree - Occamy kernels (`occamy-fused`, `occamy-gdn`, `fattn-occamy`, `sm52-xmad`, mmq/mmvq/cpy/concat edits).
4. D14 norm-q8-pair - replaces `ggml/src/ggml-cuda/norm.cu` with candidate (sha256 `94383c9f...5291`; also `audit/norm-q8-pair.patch`). `mmvq.cu` stays stock-gate-copy (sha256 `585803e4...2e33`).

## Build recipe
`audit/RUN-RECEIPT.md`, `audit/release-manifest.json`, `audit/build-release-recovery.log`: CUDA 12.8.93, GCC 14.2.0, CMake 3.28.3, `CMAKE_CUDA_ARCHITECTURES=52-real`, builder image `local/m6000-build-tools:20261013` (id `sha256:747c8a87...6bed`). Original configure/build script: `audit/build-gate-copy-lib.sh`. The D14 library was an *incremental* rebuild: cached CMake tree of gate-copy, only `norm.cu.o` and `mmvq.cu.o` recompiled (158/158 objects compared).

## Known discrepancies (read before auditing)
- **Library identity drift.** `prod-libs/libggml-cuda-gate-copy-20261013.so` was first the gate-copy build (sha256 `953bda22...c8e8`, still named in the old `.sha256` file). On 2026-10-15 it was **replaced in place by D14 `d14a076e...`** (see `adoption-receipt.json`; the adoption was a limited warm-cache acceptance, n=2, cold regime did not pass). The file name was not changed. The deployed hash is D14, not 953bda.
- **Not reproduced from scratch.** D14 is a fresh rebuild whose predecessor artifact (`f64481...`) was overwritten before copying; byte-identity with it is not claimed. A clean from-scratch rebuild from this repo has **not** been run, so reproducibility of `d14a076e` is unverified. NVCC embeds temporary source ids, so byte-equality across builds may not hold; compare symbols/objects too.
- Receipts reference local paths under `/media/avg-usr/Storage/m6000-llama-cpp/`; they are evidence, not runnable here.
- The gate-copy tree includes a pre-existing local edit to `.devops/cuda.Dockerfile` carried from the shared base checkout.

## Verify
```
git clone <this repo> && cd llamaxwell-35 && git log --oneline            # 4 source commits + audit
git diff 23b0202 HEAD -- . ':!audit'                                      # total local delta vs upstream b11552
sha256sum ggml/src/ggml-cuda/norm.cu ggml/src/ggml-cuda/mmvq.cu          # 94383c9f..., 585803e4...
sha256sum libggml-cuda-35b-d14a076e.so                                    # download from the release; expect d14a076e...
```
Rebuild (needs Docker, ~tens of GB disk, hours): follow `audit/build-gate-copy-lib.sh` with this tree, then compare against the release asset.
