# Segmented prompt lookup on a 32K code revision

Two verified source ranges reduced warm request wall time from **10.71250 to 9.53325 seconds (11.01%)** and decode time from **10.61100 to 9.43046 seconds (11.13%)** against the median of two single-range controls. The segmented request emitted **182.600 tok/s** and cleared the predeclared 5% request-wall improvement threshold.

| Same-slot order | Scope | Cache reused / evaluated | Wall (s) | Decode (s) | Drafts accepted / proposed | Checkpoint restores |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | Single range, cold | 0 / 32,768 | 89.03281 | 10.81943 | 1,681 / 1,837 | 1 |
| 2 | Single range, warm | 32,764 / 4 | 10.69823 | 10.59802 | 1,681 / 1,837 | 1 |
| 3 | Two ranges, warm | 32,764 / 4 | 9.53325 | 9.43046 | 1,673 / 1,673 | 0 |
| 4 | Single range, warm | 32,764 / 4 | 10.72677 | 10.62398 | 1,681 / 1,837 | 1 |

One process served all four requests on the same build, slot and 32,768-token input. Each response returned all 1,722 EOS-inclusive IDs from the pinned serial reference and matched the expected complete source text and Python AST. The single-range controls each logged seven q256 checks, the q53 tail and a checkpoint restore with confirmed q101 replay. The segmented arm logged q98, six q256 checks and q47, with all drafts accepted and no restore. These widths describe verifier activity. This is one measured revision fixture. A separate 80K comparison remains pending.

## Request contract and input-only range derivation

Native `/completion` accepts optional `lookup_source_spans` with one to eight ranges. Each range has integer half-open bounds `begin` and `end` and a `token_ids` array that matches that slice of the original input exactly. Ranges must be ordered, nonoverlapping, within the original prompt, and at least twice the configured n-gram key length each. The single-range field remains supported. Use one of the two fields or omit both for stock lookup. A scoped request uses a flat integer prompt, one text-only completion on a single slot, context shift disabled and `ngram-simple` as its only active speculation mode.

Search starts with the latest verified range and uses the first range that supplies a complete anchor and enough draft tokens. A proposal stays within that range. When every scoped match fails, the target advances one token. The target still verifies, samples and restores checkpoints with the existing machinery.

The client verifies a unique match of the independently tokenized public before-file in the rendered prompt and confirms the terminal-LF clipping rule. The edit request names one return line and asks to retain all other lines. Decode donor token prefixes with the live tokenizer to find the tokens that overlap that line, exclude all of them from lookup, and retain the two unchanged intervals. The model receives the same intact prompt. The range policy uses only the old source and the request. The expected output and archived serial tape are used afterward for exactness and offline replay checks. [Pinned public donor source](https://github.com/ggml-org/llama.cpp/blob/e2eb39e81cbd43ff95dfc46bddb5155b582ccdfd/examples/model-conversion/scripts/utils/check-nmse.py)

Eight malformed cases returned HTTP 400. They covered simultaneous fields, an empty list, an object in place of a list, more than eight ranges, overlap, reversed ordering, a range exceeding prompt bounds and token IDs that differed from the prompt slice. The linked CPU matcher test covered the stock, single-range, multiple-range, clipped-draft and scoped-miss paths. A separate offline archived replay reproduced the predicted wide-call counts and checkpoint behavior.

## Build and replay

Follow the [published N42 bounded lookup reconstruction](../bounded-lookup-20261118/README.md) through the FA2 base, cache-only patch and cumulative bounded lookup R1 patch. Apply the segmented patch after those layers. The new patch reproduced all four changed files byte for byte against the measured build source. The new common and server implementation libraries were mapped by the measured same-process experiment. Their digests, tested arguments and settings are in [evidence.json](evidence.json).

With `source_dir` set to the reconstructed bounded R1 source tree, `package` to this package and `r2_package` to the published Sparse R2 package containing the preset, run:

```sh
git -C "$source_dir" apply --check "$package/lookup-segments.patch"
git -C "$source_dir" apply "$package/lookup-segments.patch"
cmake -S "$source_dir" -B "$build_dir" -G "Unix Makefiles" -C "$r2_package/maxwell-release.cmake"
cmake --build "$build_dir" --config Release --parallel 2 --target llama-server
```

Use full F16 attention, one 84,224-token slot, no context shifting and `--spec-type ngram-simple --spec-ngram-simple-size-n 16 --spec-ngram-simple-size-m 255 --spec-ngram-simple-min-hits 1`. Use greedy seed 7 and EOS stopping, with a 3,072-token output cap and prompt caching enabled. Replay the exact input in slot zero using the order in the table. The N90 experiment worker produced the C++ patch. Existing source attribution and licensing apply. [Earlier same-build single-range validation](../bounded-lookup-validation-20261118/README.md)
