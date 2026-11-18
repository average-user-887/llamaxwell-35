# Three-row Sparse R2 in native depth-two decoding at 80K

On one matched 80K workload, three-row Sparse R2 raised warm decode from **52.791 to 71.782 tok/s (+35.97%)** in native depth-two MTP. Warm request time fell from **58.465 to 43.068 seconds**, saving **15.397 seconds**. Server decode time per measured speculative round fell **11.093 ms**.

| Measurement | Full three-row verifier | Sparse R2 three-row verifier |
| --- | ---: | ---: |
| Cold wall, 192 outputs | 269.56057 s | 269.31265 s |
| Cold decode rate | 57.48598 tok/s | 71.56771 tok/s |
| Warm wall, 3,072 outputs | 58.46541 s | 43.06810 s |
| Warm decode rate | 52.79136 tok/s | 71.78181 tok/s |
| Warm decode time per round | 46.96637 ms | 35.87289 ms |
| Warm measured rounds | 1,239 | 1,193 |
| Warm trace yield per round | 2.4778 | 2.5742 |
| Warm drafts accepted / proposed | 1,831 / 2,478 | 1,878 / 2,386 |

Trace yield counts accepted draft tokens plus one correction or bonus token per traced round. Dividing the full emitted output count by the traced round count gives 2.4794 and 2.5750. Decode rate divides the full output count by server decode duration. The round timing includes all server decode work. The separate [operator measurement](https://github.com/average-user-887/llamaxwell-35/blob/7f6bd99b57bdf9f4c8e151f28517c83834511e04/q3-r2-attention-20261118/README.md) projected 11.34 ms across ten attention layers at similar cache length.

## What the arms measured

Both arms used the same binary and library mappings, one model MTP head with two recursive predictions, two saved target recurrent planes and 42 of 42 model layers offloaded. Each ran a cold request and one same-slot warm request with the same 80,000-token input. Both warm arms reused 79,996 tokens and evaluated four. The trace showed long-context full q3 attention with the switch off and Sparse R2 q3 with the switch on. Both sampled long-context Sparse R2 q1. The samples confirm those dispatches. Total coverage and per-kernel durations were unmeasured in this whole-request run.

Neither arm logged a checkpoint restore. The cold and warm output sequences both first differed at token 19. The performance fixture had no semantic score. In a separate earlier run on the same input, depth-one MTP measured **82.476 tok/s** and **37.518 seconds** warm, faster than either depth-two arm here. [Earlier depth-one result](../split-warm-3072-20261118/README.md)

## Startup correction and measured configuration

The first control stopped before generation because startup found the stock CUDA library. The server was a symlink and backend discovery searched the executable's physical directory. Cleanup completed. Placing a regular byte-identical server copy beside the q3 CUDA library fixed discovery. A device-free probe confirmed the corrected selection, and both live runs verified the intended mapped library and executable hashes before generation.

Reconstruct the CUDA library using the published [q3 source and build recipe](https://github.com/average-user-887/llamaxwell-35/blob/7f6bd99b57bdf9f4c8e151f28517c83834511e04/q3-r2-attention-20261118/README.md). Use the existing single-slot split-MTP stack, `--spec-type draft-mtp --spec-draft-n-max 2`, `LLAMA_MTP_MERGE_REEVAL=0` and `LLAMA_MTP_CACHE_ONLY=1`. Keep Sparse R2 enabled in both arms. Set `GGML_CUDA_OCCAMY_FA_Q3_R2=0` for the control and `=1` for the candidate. The [allowlisted receipt](evidence.json) records the common model-independent arguments, numeric runtime flags and actual build digests. The scope is one approximate attention comparison with one cold and one warm request per arm.
