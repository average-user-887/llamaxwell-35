# Exact q256 prompt lookup at 80K

On the same committed-code edit used in the earlier 32K tests, extending the prompt to 80,000 tokens yielded **106.56184 tok/s** for q256 cached replay versus **58.14036 tok/s** serial. Warm request wall fell from **29.74661 s to 16.28611 s (−45.25%)**. The cold request also finished sooner, at **277.85968 s** versus **291.09588 s**.

| 80K condition | Serial wall (s) | q256 wall (s) | Serial decode (tok/s) | q256 decode (tok/s) | Cached tokens per arm |
|---|---:|---:|---:|---:|---:|
| Cold | 291.09588 | 277.85968 | 58.18359 | 105.33252 | 0 |
| Same-slot warm | 29.74661 | 16.28611 | 58.14036 | 106.56184 | 79,996 |

Each arm ran one cold request and one warm replay on the same server slot. Both cold requests evaluated all 80,000 tokens. Each warm request evaluated four tokens after reusing 79,996. All four completions produced the same 1,722 IDs including EOS. They matched the earlier 32K output, and the complete edited Python file matched the expected text after stripping outer whitespace and any surrounding code fence. The AST also matched.

The q256 arm had eight full-width verification calls in each request and accepted 1,681 of 2,040 proposed tokens. Two rejected suffixes caused recurrent checkpoint restores followed by confirmed replays at widths 101 and 54. This proves the observed verification schedule from runtime logs. Kernel-level attribution remains unmeasured. Both arms used full F16 attention and offloaded 42/42 layers.

Decode uses emitted IDs including EOS divided by the server decode duration. Wall time is the nonstreaming completion round trip. This is a one-fixture cold and warm comparison, distinct from the repeated 32K cold experiment. The donor is the same committed before-file, embedded in the prompt before generation, with a longer context prefix at 80K. [32K lookup receipt and source reconstruction](https://github.com/average-user-887/llamaxwell-35/blob/5ec57de6c5b988b83f3cfe7fd5ac9e476b3e8880/lookup-wide-20261118/README.md)

## Reproduce the 80K stage

Use the existing FA2 and cache-only source reconstruction and the pinned runtime hashes from the 32K package. The live lookup mode uses native `ngram-simple` with a 16-token lookup key, maximum proposal length 255, and one minimum hit. The serial mode uses `--spec-type none`. The 80K stage keeps the previous server argument profile, including one slot, 84,224 context positions, `-b 2048 -ub 1024`, F16 K/V, and context shift disabled. The recorded environment enables `LLAMA_TRACE=1` in both arms. [Typed configuration and allowlisted measurements](evidence.json)

Set `LLAMA_SERVER`, `MODEL_FILE`, and `RUNTIME_LIB_DIR` for the pinned build and model. Select a mode. Start a fresh server for each arm, send the fixed 80,000-token prompt once to populate the cache, then send the same prompt in the same slot. Both requests use `cache_prompt=true`, greedy seed 7, a 3,072-token cap and EOS stopping. The harness timeouts were 390 s for cold and 90 s for warm, under a 480 s per-arm cap including cleanup.

```bash
mode=q256
mapfile -t server_args < <(jq -er --arg mode "$mode" \
  '(.configuration.server_args + .configuration.mode_argv[$mode])[]' evidence.json)
mapfile -t settings < <(jq -er \
  '.configuration.environment | to_entries[] | "\(.key)=\(.value)"' evidence.json)
env -u LLAMA_MTP_CACHE_ONLY "LD_LIBRARY_PATH=$RUNTIME_LIB_DIR" "${settings[@]}" \
  "$LLAMA_SERVER" -m "$MODEL_FILE" "${server_args[@]}"
```
