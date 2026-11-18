# Wide prompt lookup on a committed-code edit

On a fixed 32,768-token task, stock prompt lookup accelerated generation of a complete, copy-heavy Python file edit on a Quadro M6000. The first q256 run measured **144.8472 tok/s** versus **72.9760 tok/s** serial using the same 1,722-output count, including EOS. Request wall fell from **100.8562 s to 89.1907 s**. Both runs produced the same token sequence and exact requested edit.

## Repeated comparison

| Arm | Decode (s) | Same-count decode (tok/s) | Request wall (s) |
|---|---:|---:|---:|
| Serial 1 | 23.596803 | 72.9760 | 100.8562 |
| q256 1 | 11.888389 | 144.8472 | 89.1907 |
| q256 2 | 11.878212 | 144.9713 | 89.0601 |
| Serial 2 | 23.623001 | 72.8951 | 100.9944 |

The ratio of median decode durations was **1.98681×**. Dividing 1,722 outputs by each median duration gives **72.9355 tok/s serial** and **144.9092 tok/s at q256**. Median request wall fell from **100.9253 s to 89.1254 s**, a saving of **11.7998 s (11.69%)**. All four requests produced the same 1,722 IDs including EOS and returned the exact requested edit. The rates use server decode duration. Wall time is the nonstreaming completion round trip. The table shows the order of the included requests on this one task.

## Workload and online recovery

The before version of the Python NMSE utility was present in the prompt before generation. It came from an independent public llama.cpp commit and the copied file matches the pinned Git blob and SHA-256. The task asks for a single zero-variance return-field fix and the complete file. Stock `ngram-simple` looks up a 16-token suffix in the available prompt and generated history, then proposes up to 255 continuation tokens. The target verifies proposals and controls the output. The expected result is used for scoring after generation. The score checks the full AST and exact text after stripping outer whitespace and an optional surrounding code fence. [Pinned upstream before source](https://github.com/ggml-org/llama.cpp/blob/e2eb39e81cbd43ff95dfc46bddb5155b582ccdfd/examples/model-conversion/scripts/utils/check-nmse.py)

Live q64 lookup was slower on this fixture at 67.4728 tok/s and 102.3612 s wall. A separate teacher-forced q64 capacity check measured 73.7208 checked positions/s versus 73.2796 at q1 over 1,536 measured decisions. That capacity test excludes live proposal and rollback costs. At q256 the online logs recorded eight full-width verification calls. Two rejected suffixes restored recurrent checkpoints and replayed the confirmed prefixes with 101 and 54 rows. Each q256 run accepted 1,681 of 2,040 proposed tokens. The first q256 run's restore-to-replay intervals were 1,126.970 ms and 857.568 ms. These are server-observed recovery intervals. [Allowlisted evidence](evidence.json)

All measured proposals came from native prompt lookup over this edit's existing history. The scope is one task with repeated cold requests, and different tasks may offer less reusable context.

A later single q512 arm used `size-m 511` and reproduced all 1,722 IDs and the exact edit. It recorded five q512 calls, two confirmed restore/replays at q101 and q54, and 1,684/2,555 accepted drafts. It measured **135.1416 tok/s** and **89.5034 s** wall, slower than both q256 samples on this fixture. The repeated comparison above uses the four serial/q256 arms.

## Reuse the pinned build

Use the published FA2 reconstruction in `base` mode, then apply the published cache-only patch. The experiment used the resulting two-library overlay and FA2 CUDA library, with the cache-only MTP feature flag unset. Use the linked Maxwell CMake preset for a source rebuild. The recorded hashes identify the tested binaries. This lookup setting adds no native or kernel changes. [Source and build recipes](https://github.com/average-user-887/llamaxwell-35/blob/19c4f88ee7a0d493364c80803eacfd0bd4a789b7/cache-only-20261118/README.md)

```sh
git worktree add --detach "$source_dir" e4fdc23555f6a9103ea03b854ee5c70c6812517c
bash "$release_root/sparse-r2-20261117/reconstruct-source.sh" "$source_dir" base
cache_patch="$release_root/cache-only-20261118/cache-only.patch"
git -C "$source_dir" apply --index --check "$cache_patch"
git -C "$source_dir" apply --index "$cache_patch"
```

In Bash, set `LLAMA_SERVER`, `MODEL_FILE`, and `RUNTIME_LIB_DIR` for that build and its pinned model. The typed profiles contain the recorded common flags and the serial, q64, q256, and q512 deltas. Select the desired profile. Supply the recorded request fields and a caller-provided prompt. The measurement used a fixed tokenized prompt, one slot, cold cache, greedy seed 7, a 3,072-token cap, and EOS stopping. [Typed server and request profiles](profiles.json)

```bash
profile=q256
mapfile -t server_args < <(jq -er --arg mode "$profile" \
  '(.server_args + .profiles[$mode].server_args)[]' profiles.json)
mapfile -t environment < <(jq -er --arg mode "$profile" \
  '(.environment + .profiles[$mode].environment) | to_entries[] | "\(.key)=\(.value)"' profiles.json)
env -u LLAMA_TRACE -u LLAMA_MTP_CACHE_ONLY "LD_LIBRARY_PATH=$RUNTIME_LIB_DIR" "${environment[@]}" "$LLAMA_SERVER" \
  -m "$MODEL_FILE" "${server_args[@]}"
```

The q256 profile sets `--spec-type ngram-simple --spec-ngram-simple-size-n 16 --spec-ngram-simple-size-m 255 --spec-ngram-simple-min-hits 1`. q64 uses `size-m 63` and q512 uses `size-m 511`. The serial profile uses `--spec-type none`. The lookup profiles enable trace logging for recovery evidence; the serial profile leaves that setting unset.
