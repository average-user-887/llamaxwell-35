# Sparse R2 microbatch 2048 and midpoint recall

Changing `-ub 1024` to `-ub 2048` reduced the cold 80K request for Sparse R2 plus cache-only MTP1 from **268.635 s to 257.001 s**, a saving of **11.634 s (4.33%)**. Server-reported decode fell from **79.2945 to 78.2468 tok/s**. The same-sparse no-MTP control at ubatch 2048 finished cold in **253.015 s**.

| Sparse R2 configuration | Cold request (s) at 1024 → 2048 | Server cold decode (tok/s) at 1024 → 2048 | Warm request (s) at 1024 → 2048 |
|---|---:|---:|---:|
| No MTP | 264.171 → 253.015 | 76.0934 → 76.5199 | 2.6270 → 2.6227 |
| MTP1 plus cache-only catch-up | 268.635 → 257.001 | 79.2945 → 78.2468 | 2.7287 → 2.6874 |

Each mode and setting had one cold request and one same-slot warm replay at 80,000 prompt tokens and 192 outputs. Both settings reused 79,996 prompt tokens warm. Changing ubatch first changed the greedy output at token 27 in the stacked arm and token 52 in the no-MTP arm, for both cold and warm requests. The stacked cold draft count changed from 90/100 accepted to 89/101. This configuration comparison measures wall and throughput on a fixed fixture. [Allowlisted receipt](evidence.json)

## Reproduce the setting

Use the Sparse R2 plus cache-only MTP1 stack and the runtime settings from the published recipe. Change the server argument from `-ub 1024` to `-ub 2048`. Keep the batch size and scheduler settings below. The single `-ub` setting takes effect in both the target and internal draft contexts for this runtime, with the reported `n_ubatch` equal to 2,048 in each. The same-sparse no-MTP control uses the same batch settings and the recipe's no-MTP mode. The existing source and runtime libraries supply both settings. [Published stack recipe](https://github.com/average-user-887/llamaxwell-35/blob/49f5db19e9ec0b06c255b2113bdcb60c6d98c13d/stacked-mtp-20261118/README.md)

```text
-np 1 -c 84224 -b 2048 -ub 2048 --mixed-batch-size 1024 --prefill-single-stream
```

Keep `GGML_CUDA_OCCAMY_FA_SPARSE=1` and `GGML_CUDA_OCCAMY_FA_SPARSE_R2=1` in both modes. For the MTP1 stack, keep `LLAMA_MTP_CACHE_ONLY=1`, plus `--spec-type draft-mtp --spec-draft-n-max 1 --spec-draft-type-k f16 --spec-draft-type-v f16` and the recipe's MTP environment. The measured stack emitted a live 2,048-row cache-only marker and Sparse R2 query-width 1 and 2 markers.

## Midpoint recall at ubatch 2048

The stack returned the exact planted fact beginning at token index 40,000 of the 80,000-token prompt. The prompt IDs and rendered template matched both earlier ubatch-1024 controls. The seven generated IDs, including EOS, and the output digest matched both earlier runs. This run logged live 2,048-row cache-only catch-up and long-context Sparse R2 query-width-2 dispatch. The result covers one planted fact at this midpoint. The synthetic distant-needle miss documented in the sparse evidence still applies. [Earlier midpoint validation](https://github.com/average-user-887/llamaxwell-35/blob/49f5db19e9ec0b06c255b2113bdcb60c6d98c13d/stack-validation-20261118/README.md)

## Separate warm 3,072-token comparison at ubatch 1024

On a separate 80K warm pair using ubatch 1024, the stack completed 3,072 outputs in **38.9458 s**, versus **40.2066 s** for sparse no-MTP, a **3.14%** wall reduction. Emitted tokens over server decode duration were **79.42196 vs 76.65973 tok/s**, a **3.60%** gain. Both arms reused 79,996 prompt tokens with context shifting disabled. Each reproduced its own earlier first 1,024 output IDs, and the two modes diverged at token 27. The stack accepted 1,447/1,623 drafts. This throughput fixture has no semantic score. Its emitted-token rate is separate from the server-reported rate in the microbatch table.
