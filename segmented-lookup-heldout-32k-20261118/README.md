# Segmented lookup on the held-out 32K revision

On the held-out code revision, two source ranges cut warm request time from **7.910 to 6.806 seconds (13.96%)** versus the median of two same-build single-range controls. Decode rose from **139.331 to 162.139 tok/s (+16.37%)**. All four measured outputs matched all **1,087 tokens through EOS**, the independently expected source text, and its Python AST.

| Same-slot request order | Range selection | Wall time | Decode rate | Drafts accepted / proposed | Checkpoint restores |
| --- | --- | ---: | ---: | ---: | ---: |
| 1. Cold prime | Single | 85.10757 s | 135.99107 tok/s | 1,040 / 1,174 | 1 |
| 2. Warm control | Single | 7.88757 s | 139.61257 tok/s | 1,040 / 1,174 | 1 |
| 3. Warm candidate | Segmented | 6.80575 s | 162.13872 tok/s | 1,035 / 1,035 | 0 |
| 4. Warm control | Single | 7.93161 s | 139.04863 tok/s | 1,040 / 1,174 | 1 |

Each warm request reused **32,764** input tokens and evaluated **four**. The segmented request logged q118, three q256 calls, and q154. The single-range requests each restored a checkpoint and replayed q123. These measurements describe logged verifier work. They cover one 32,768-token source revision, with a cold single-range prime followed by three warm requests.

The earlier held-out serial run measured **73.073 tok/s** warm. It used a different mapped build and serves as an independent exact-output reference. All four current outputs also match the earlier bounded run. [Held-out serial and bounded results](../lookup-heldout-32k-20261118/README.md)

## Reuse the published implementation

The held-out edit asks to repair one missing-field lookup while preserving the rest of a pinned before-file. To derive the two source ranges, independently tokenize the before-file, confirm its unique match in the rendered prompt, and clip the terminal LF. Use live detokenized prefix lengths to exclude every token overlapping the editable line. The model receives the intact input. The expected full source and the archived serial outputs are used for verification after generation. [Pinned upstream before-file](https://github.com/ggml-org/llama.cpp/blob/539f24529bdf99e0baefd54fefff1660034bfe7b/gguf-py/gguf/scripts/gguf_set_metadata.py)

Use the published N90 segmented build with full F16 KV, one 84,224-token slot, `-b 2048 -ub 1024`, context shift disabled, greedy seed 7, prompt caching and EOS stopping. Keep `ngram-simple` with key length 16, maximum draft 255, and one minimum hit. Submit the single-range field for the prime and controls and the segmented range field for the candidate. Each live request logged validation of the intended scope. The earlier N90 run on the same mapped runtime returned HTTP 400 for eight malformed requests. [Published implementation, reconstruction and N90 ingress checks](https://github.com/average-user-887/llamaxwell-35/blob/2c32420080d5d151d34191743041bc1e7a1379cb/segmented-lookup-32k-20261118/README.md)

The [allowlisted receipt](evidence.json) contains the common setup and per-request measurements.
