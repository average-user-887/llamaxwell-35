# Exact segmented prompt lookup on an 80K code revision

On one 80,000-token code revision, segmented prompt lookup reached **133.966 tok/s**, or **2.305×** the warm serial decode rate on the same build. The warm completion took **12.981 seconds** versus **29.753 seconds** serial, a **2.292×** whole-request speed ratio. Every run returned the same 1,722 token IDs through EOS and the expected complete source text and Python AST.

| Warm condition | Decode rate | Decode time | Request wall |
| --- | ---: | ---: | ---: |
| Serial, separate process | 58.13028 tok/s | 29.62311 s | 29.75316 s |
| Single source range, median of two same-slot controls | 118.63492 tok/s | 14.51514 s | 14.64494 s |
| Segmented, one same-slot request | 133.96556 tok/s | 12.85405 s | 12.98105 s |

The lookup process ran a single-range cold prime, then warm single-range, segmented, and single-range requests. The serial process ran one cold and one warm request. Both processes used the same executable and mapped libraries, model, 80K input and template, full F16 KV, one slot and greedy request settings. Every warm request reused **79,996 tokens** and evaluated **four**. Segmentation saved **1.664 seconds (11.36%)** over the median warm single-range control. The cold single-range prime took 276.214 seconds and the separate cold serial request took 292.206 seconds. No cold segmented request was timed.

The segmented request accepted **1,673 of 1,673** draft tokens. It logged q98, six q256 calls and q47, with zero restores. Each single-range control accepted 1,681 of 1,837 and logged one checkpoint restore followed by a q101 replay. Verifier widths describe scheduled target work. The logs do not attribute every width to a specialized CUDA kernel. The serial requests logged no speculative verification. Eight malformed native requests returned HTTP 400. The exactness and speed measurements cover this one revision fixture. [Allowlisted receipt](evidence.json)

## Reuse the published segmented source

Use the earlier segmented lookup patch and its pinned source chain. The range policy verifies the tokenized committed before-file in the original prompt, keeps the terminal-LF clip, and excludes all tokens overlapping the old line identified by the edit request. The two remaining ranges are checked against the native tokenizer and the original prompt. The prompt given to the model stays intact. The expected text and AST and the archived serial output are used for checks after generation. [Published N90 patch, build and derivation recipe](https://github.com/average-user-887/llamaxwell-35/blob/2c32420080d5d151d34191743041bc1e7a1379cb/segmented-lookup-32k-20261118/README.md)

Use the same build, full F16 attention and KV, `-b 2048 -ub 1024`, one 84,224-token slot and disabled context shift. Run the lookup process with `--spec-type ngram-simple --spec-ngram-simple-size-n 16 --spec-ngram-simple-size-m 255 --spec-ngram-simple-min-hits 1`. Run the serial process with `--spec-type none` and omit both lookup fields. Both profiles use the same 3,072-token cap, seed 7, EOS stopping and prompt caching. Use the request order above and the numeric configuration in [evidence.json](evidence.json).
