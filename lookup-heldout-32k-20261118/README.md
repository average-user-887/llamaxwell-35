# Bounded lookup on a second, held-out 32K code revision

On a different public source revision, bounded single-range lookup raised warm decode from **73.073 to 139.354 tok/s (1.907×)** on the same build. Warm request time fell from **14.976 to 7.904 seconds**, saving **7.073 seconds**. Both modes matched all **1,087 output tokens through EOS**, the independently expected source text after trimming outer whitespace, and its Python AST.

| Measurement | Serial | Bounded single range |
| --- | ---: | ---: |
| Cold request wall | 92.03555 s | 85.36267 s |
| Cold decode rate | 73.16092 tok/s | 136.60210 tok/s |
| Warm request wall | 14.97636 s | 7.90357 s |
| Warm decode rate | 73.07294 tok/s | 139.35370 tok/s |
| Warm speculative drafts accepted / proposed | — | 1,040 / 1,174 |

Each arm ran one cold request and one same-slot warm replay. Both warm requests reused 32,764 prompt tokens and evaluated four. The lookup request logged four q256 verifier calls and a confirmed q123 replay after one checkpoint restore in each cache condition. These widths describe logged verifier work. This tests the same bounded lookup implementation on a second code revision, with one input and one pair per mode. [Allowlisted evidence](evidence.json)

## Independent source and exactness

The held-out task repairs a missing-metadata case in a pinned public llama.cpp file. The expected edit was fixed before generation. A CPU check of the relevant function with a missing-field stub reproduced the old exception and verified that the repaired function reached the existing missing-field guard. The generation check then compared the entire result with the predetermined source and AST. [Pinned upstream before-file](https://github.com/ggml-org/llama.cpp/blob/539f24529bdf99e0baefd54fefff1660034bfe7b/gguf-py/gguf/scripts/gguf_set_metadata.py)

The range input was derived from the independently tokenized before-file. The caller located its unique exact match in the original prompt, verified that the terminal LF was a standalone token, and excluded that token from the lookup range. The input and rendered template matched across modes. The expected output was used only for scoring.

## Reuse the published bounded lookup build

Follow the existing single-range build and request contract. The serial arm used `--spec-type none` and omitted the source range. The bounded arm used `ngram-simple` with key length 16, maximum draft 255, one minimum hit, and the verified single-range field. Both used one fully offloaded slot, full F16 KV, an 84,224-token context, `-b 2048 -ub 1024`, disabled context shift, greedy seed 7, EOS stopping and prompt caching. [Published implementation and build recipe](https://github.com/average-user-887/llamaxwell-35/blob/31f46e620ed3648a3af1b9cdd519b7174e720334/bounded-lookup-20261118/README.md)

The first attempted bounded launch stopped during serial receipt qualification. It captured a diagnostic alongside the verified path, so the receipt check failed before the GPU run started. The corrected wrapper routed diagnostics to stderr and emitted the verified path alone. The existing validator and inference runtime were reused.
