# Bounded prompt lookup on a committed-code edit

The corrected bounded-lookup build reached **118.3919 tok/s at 80K warm**, or **2.03631×** the matched serial decode rate on the exact revision fixture. Warm request wall was **14.67534 s** versus **29.74661 s** serial and **16.28611 s** with stock q256 lookup. All 1,722 output IDs, including EOS, matched the corresponding frozen controls. The normalized edited text and AST also matched. The whole warm request improved **2.027×** on this fixture.

The cold 32K bounded run measured **159.6610 tok/s**, **10.785352 s** decode and **88.10883 s** request wall. The pinned first serial and stock q256 controls measured **72.9760 tok/s and 100.85616 s**, and **144.8472 tok/s and 89.19070 s**, respectively. The earlier 32K repeats remain separate evidence. [Published 32K controls](https://github.com/average-user-887/llamaxwell-35/blob/d899acb7d3a016382c6bf34fc0ecf04e98ba732b/lookup-wide-20261118/README.md)

| 80K condition | Serial (tok/s) | Stock q256 (tok/s) | Bounded q256 (tok/s) | Serial wall (s) | Stock wall (s) | Bounded wall (s) |
|---|---:|---:|---:|---:|---:|---:|
| Cold | 58.1836 | 105.3325 | 116.8237 | 291.09588 | 277.85968 | 277.29157 |
| Warm | 58.1404 | 106.5618 | 118.3919 | 29.74661 | 16.28611 | 14.67534 |

Bounded warm decode was **2.03631×** the pinned N32 serial rate. It **passed** the predeclared warm gate of at most 14.809 s decode and at least 2.0× the matched serial rate. Each stage ran one request per reported cache condition. At 80K each warm arm reused 79,996 prompt tokens and evaluated four. Both bounded 80K requests matched all 1,722 serial and stock output IDs including EOS and the expected text and AST. The comparison is specific to this revision fixture.

Cold prefill reported by the server rose from 261.4781 s serial to 262.5265 s bounded. The cold wall saving comes from the shorter decode.

Every bounded request logged seven full q256 calls, terminal q53 with 52/52 accepted, and one recurrent checkpoint restore followed by a confirmed q101 replay. The draft accepted count stayed at 1,681 while proposals fell from 2,040 with stock q256 to 1,837. Decode divides the same emitted count including EOS by server decode duration. Request wall is the nonstreaming completion round trip. The logged widths describe verifier activity. Kernel-level coverage remains unmeasured. [Matched stock and serial 80K controls](https://github.com/average-user-887/llamaxwell-35/blob/d899acb7d3a016382c6bf34fc0ecf04e98ba732b/lookup-80k-20261118/README.md) · [Allowlisted measured receipts](evidence.json)

## Request-local source span

The optional native `/completion` field `lookup_source_span` takes exactly three members. Its default is unset, which retains stock behavior.

| Member | Type and constraint |
|---|---|
| `begin` | Integer zero-based inclusive input position. |
| `end` | Integer exclusive input position greater than `begin` and at most the original prompt length. |
| `token_ids` | Integer array of length `end - begin` that exactly matches the specified original prompt slice. |

The request requires one flat integer token prompt, one text-only completion, one slot, context shifting disabled and `ngram-simple` as the only active speculative mode. The span must cover at least twice the configured lookup key size. Native ingress checks each raw prompt ID against the vocabulary before conversion. The bounds constrain both historical anchor search and proposal copying. Native target verification, sampling and checkpoint recovery remain in use.

For this fixture the caller independently tokenizes the committed before-file, verifies a lossless round trip, and locates its full token sequence exactly once in the rendered prompt. It verifies that clipping the file's terminal LF removes exactly one standalone LF token. It then ends the lookup span before that token. The prompt and template remain intact. This input-only policy runs before generation. The expected edited text and AST are used afterward for scoring. [Committed donor source](https://github.com/ggml-org/llama.cpp/blob/e2eb39e81cbd43ff95dfc46bddb5155b582ccdfd/examples/model-conversion/scripts/utils/check-nmse.py)

## Corrected ingress

The initial 32K run stopped on its first invalid-range test, before generation. Cleanup passed. The original HTTP adapter retained the error outcome without an exact status or response. The CPU argument-parser check found that the native CLI retains an inert `none` entry alongside `ngram-simple`. R1 allows those entries with at least one `ngram-simple` and checks raw token IDs before integer narrowing. The corrected 32K runtime returned HTTP 400 for each of four invalid inputs covering an out-of-range span, a mismatched claimed slice, a non-flat prompt and an out-of-vocabulary prompt ID.

## Build and replay

The patch is the cumulative tested R1 source. It applies after the published FA2 `base` reconstruction and the published cache-only patch. Forward and reverse application passed. All six edited files matched the measured source checkout byte for byte. The tested overlay replaces `libllama-common.so.0.6.0` and `libllama-server-impl.so`. The R1 incremental repair relinked the server implementation and retained the common library hash. Both patched hashes and the base executable and library hashes are recorded in the receipt. [Original N1 reconstruction and preset](https://github.com/average-user-887/llamaxwell-35/blob/d899acb7d3a016382c6bf34fc0ecf04e98ba732b/cache-only-20261118/README.md)

In the repository, set `release_root` to a checkout containing the published packages and `package` to this package. Choose a new `source_dir` and `build_dir`.

```sh
git worktree add --detach "$source_dir" e4fdc23555f6a9103ea03b854ee5c70c6812517c
bash "$release_root/sparse-r2-20261117/reconstruct-source.sh" "$source_dir" base
git -C "$source_dir" apply --index "$release_root/cache-only-20261118/cache-only.patch"
git -C "$source_dir" apply --index --check "$package/bounded-lookup-r1.patch"
git -C "$source_dir" apply --index "$package/bounded-lookup-r1.patch"
cmake -S "$source_dir" -B "$build_dir" -G "Unix Makefiles" \
  -C "$release_root/sparse-r2-20261117/maxwell-release.cmake"
cmake --build "$build_dir" --config Release --parallel 2 --target llama-server
```

Use the recorded one-slot full-F16 profile, context 84,224, `-b 2048 -ub 1024`, disabled context shift and `--spec-type ngram-simple --spec-ngram-simple-size-n 16 --spec-ngram-simple-size-m 255 --spec-ngram-simple-min-hits 1`. Requests use greedy seed 7, EOS stopping and a 3,072-token cap. The 32K run was cold with prompt caching disabled. The 80K run enabled caching for the cold request and then replayed the same prompt and source span in the same slot. The harness used cold/warm request timeouts of 390/90 s and a 480 s owned stage cap including cleanup. This is experimental evidence for the tested route, model and revision fixture.
