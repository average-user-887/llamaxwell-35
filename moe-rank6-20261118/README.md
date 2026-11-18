# Rank-6 MoE on the Maxwell V3 trunk

On a matched 80,000-token sparse split-MTP1 fixture, enabling approximate rank-6 trunk routing raised warm emitted decode from **83.78695 to 90.24548 tok/s (+7.71%)**. Warm wall time fell from **36.92382 to 34.30895 s**, saving **2.61487 s**. The generated streams first differed at **token 30**. This fixed throughput fixture has no semantic score.

| 80K result | Rank-6 off | Rank-6 on |
|---|---:|---:|
| Cold request wall (s) | 269.38928 | 268.54829 |
| Cold emitted decode (tok/s) | 83.05120 | 82.13784 |
| Warm request wall (s) | 36.92382 | 34.30895 |
| Warm emitted decode (tok/s) | 83.78695 | 90.24548 |
| Warm server decode time per proposed draft token (ms) | 22.61840 | 21.80685 |
| Warm draft accepts | 1,449/1,621 | 1,509/1,561 |

Both arms used the same server and CUDA library, one 192-output cold seed, and one same-slot cached replay of 3,072 outputs. Warm replay reused 79,996 prompt tokens and evaluated four. Cold wall improved by 0.84099 s in this pair while cold decode was slower. Decode divides emitted tokens by server decode duration. The per-proposal metric divides that total duration by draft proposals. Changed outputs and higher warm acceptance contribute to the observed throughput. [Allowlisted measurements](evidence.json)

## Kernel and guard

The patch adds an opt-in specialization of the existing Maxwell V3 kernel. It renormalizes the first six routed weights and retains the shared expert. The model and graph retain eight routed ranks. In the rank-6 specialization, phase B bounds the current and prefetched gate/up work to the retained rows. Phase C stages 36 of 72 `uint4` entries in the second routed down chunk and evaluates 16 dot lanes there. The stock path remains available.

Enable with `GGML_CUDA_MOE_RANK6=1`. The dispatch is limited to the supported V3 `sm_52` path, target trunk layers 0–39, and the q1 row or two q2 rows when the resident grid matches the stock grid. Incompatible shapes, alternate launch modes and draft layers retain their existing routing. The measured pair sampled q1 and both q2 rows at a 48-block grid. Launch logs sample power-of-two call counts. Total path coverage remains unmeasured.

## Independent checks and limits

The short diagnostic matched an exact arithmetic answer and EOS in both modes. All five emitted startup comparisons against dense **masked top-six** had zero mismatches over 2,048 elements each. They covered one q1 comparison and two checks for each q2 row. The original wrapper expected a second q1 comparison and exited 1 after capture. The recovered observation records the five actual checks and the original status.

The first trial's arithmetic request returned HTTP 400, and the first retry timed out in startup readiness before generation. Both cleaned up. The later short diagnostic used the same CUDA binary.

A separate 80K midpoint run on rank-6 retrieved the planted fact beginning at token 40,000. Its prompt and rendered template matched the established split-MTP1 reference. All seven generated IDs including EOS matched that reference. Rank-6 dispatch was sampled during recall. This checks one planted fact. The earlier Sparse R2 evidence also retains the synthetic distant-needle miss. [Prior midpoint reference](https://github.com/average-user-887/llamaxwell-35/blob/774ab6b708b52367747fadaba9c238847dde7f72/split-32k-recall-20261118/README.md) · [Sparse R2 evidence](https://github.com/average-user-887/llamaxwell-35/blob/774ab6b708b52367747fadaba9c238847dde7f72/sparse-r2-20261117/README.md)

## Reconstruct and build

Start from the published FA2 plus Sparse R2 plus cache-only stack and use the existing Maxwell CMake preset for Release and `52-real`. Set `source_dir` to the reconstructed source, `release_root` to a checkout with the published packages, `package` to this package and `build_dir` to a new build directory. [Published stack recipe](https://github.com/average-user-887/llamaxwell-35/blob/774ab6b708b52367747fadaba9c238847dde7f72/stacked-mtp-20261118/README.md)

```sh
patch --batch --forward --fuzz=0 --dry-run -d "$source_dir" -p1 < "$package/candidate.patch"
patch --batch --forward --fuzz=0 -d "$source_dir" -p1 < "$package/candidate.patch"
cmake -S "$source_dir" -B "$build_dir" -G "Unix Makefiles" \
  -C "$release_root/sparse-r2-20261117/maxwell-release.cmake"
cmake --build "$build_dir" --config Release --parallel 2 --target ggml-cuda
```

The original build rebuilt only `ggml-cuda` in the prepared R2 build tree. Forward and reverse application with zero fuzz passed on the reconstructed stack file, and the patched file matched the measured source byte for byte. The measured CUDA library is identified by SHA-256 because its binary embeds private build paths. A fresh build can produce a different artifact hash.

Use the pinned Sparse R2 split-MTP1 runtime with `LLAMA_MTP_MERGE_REEVAL=0`, `LLAMA_MTP_CACHE_ONLY=1`, one slot, context 84,224, `-b 2048 -ub 1024`, F16 draft K/V, context shift disabled and the common environment in the receipt. Keep `GGML_CUDA_MOE_RANK6_CHECK` unset for throughput. The short diagnostic enabled it to compare the sampled cap calls against dense masked top-six. Both timing arms used the candidate CUDA library. The control left rank-6 unset.
