# Decode-aware mixed-batch candidate

Isolated experiment, not installed in production. Source baseline is BUNDLE3 commit `2436f7554c0c8d70afc9acbcce85b2457193a516`; source was copied without its Git metadata. Build cache was copied from the existing E5 tree, then reconfigured against this source. The build has no GPU device, no network, at most two compile jobs, CPUs 0-3, and a 6 GiB memory limit.

## Runtime switch

`--mixed-batch-size 1024` or `LLAMA_ARG_MIXED_BATCH_SIZE=1024` enables the experiment. Default `0` leaves scheduling unchanged. Negative and non-integer values are rejected.

The cap is `min(requested mixed size, configured logical batch, configured physical microbatch)`. It applies only when actual generation or draft-verification tokens are already in the current batch and the server is adding a splittable text completion prompt. The cap includes the generation tokens. With one generating slot and cap 1024, at most 1023 prompt tokens are added.

Pure prefill, pure generation, allocation sizes, configured `n_batch`/`n_ubatch`, non-splittable requests, decisions and multimodal requests retain their existing path. Prompt-cache/checkpoint, cancellation, and context-shift machinery is not replaced. After generation finishes, remaining prompt processing automatically returns to the original logical batch size.

This is a bounded text-completion latency experiment, not a general fairness scheduler. Do not configure a mixed cap at or below the number of simultaneous generation/verification tokens: prompt processing will then wait for generation work to subside. The intended two-slot workload uses cap 512 or 1024 with one generating slot and no speculation.

## Integration and comparison

Use the same rebuilt runtime for both arms, with flag `0` versus `1024` (and optionally `512`). This isolates the scheduler change from differences between the older production-image server and the rebuilt server. The new field in `common_params` requires the rebuilt `libllama-common` and server implementation to travel together; replacing only the server library is unsafe.

Preserve the approved BUNDLE3 CUDA library, model, GPU placement, routing, context/cache settings and physical microbatch 1024 for both arms. No model load or GPU test is performed by this subtask. The root investigator owns routing and GPU experiments.

Acceptance should include a matching-runtime flag-off baseline; an existing deep-context stream during a fresh prompt; maximum/p95 inter-token gap, prompt time and total wall time; a solo-prefill control; completion/cancellation/reuse; and matching deterministic output or an explicit quality delta. A lower gap alone is not proof of higher aggregate throughput.

Rollback is omission of the switch or `--mixed-batch-size 0`; no production state was modified here.

## Build

Run `bash build.sh` from any directory. It builds `llama-server` and `test-arg-parser` using the existing local CUDA 12.8 tool image. The compiler can emit Maxwell CUDA binaries without access to a GPU. Logs are retained in `build.log`.

## Delivered runtime and validation

Built successfully on 2026-11-12 at approximately 15:35 CET. Mount this directory's `runtime` read-only at `/candidate`, set `LD_LIBRARY_PATH=/candidate`, and invoke `/candidate/llama-server` with the existing approved model/GPU/routing arguments. The runtime already contains the approved BUNDLE3 CUDA library, SHA256 `94c777dc55489fba4cab5b68c6b54cf8a45810b0a1011589260e5bd23cd05486`.

The package's shared-library dependencies resolved inside the production image `sha256:3a0caa9afea12f5ca5d982621cc2d86b1fa6ac614d735b262fa475d7dc6e2a6d`; its help advertises the new switch. The focused no-network parser validation passed defaults, 512/1024/zero values, unchanged logical/physical batch parameters, negative/non-numeric/overflow rejection, environment parsing and CLI precedence. Reproduce with `bash validate-cpu.sh`; receipt is `focused-cpu-validation.log`.

The extended existing `test-arg-parser` passed its argument checks and then failed its unrelated external HTTP download test because networking is disabled. This is retained honestly in `cpu-validation.log`; it is not a full-suite pass.

`scheduler.patch` contains the four source-file changes and passes `git apply --check` against the original BUNDLE3 checkout. `SHA256SUMS` records key source and runtime hashes. The source checkout remained unchanged; all build containers exited. GPU behavior and latency/throughput acceptance remain the root investigator's pending work.
