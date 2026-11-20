# Native R2 BFCL cached paired evaluation (full, 160 cases)

This package records the paired BFCL multi-turn evaluation of `native` (`draft-mtp`, 1 draft token, f16 draft K/V) against `serial` (`--spec-type none`) under the frozen within-episode prompt-cache reuse policy (`slot=0`, `cache_ram_mib=0`, `n_cache_reuse=0`, `first_request_cache_prompt=false`, `later_requests_cache_prompt=true`) on `qwen36-moe`.

## Headline paired outcome

| Measure | Serial (off) | Native (on) |
| --- | ---: | ---: |
| Passes | 71/160 (160 completed, 0 infra) | 1/160 (5 completed, 155 infra) |
| Model-cap cases | 26 | 1 |
| Infra-error cases | 0 | 155 |
| Accepted model requests | 1,931 | 59 |
| Newly evaluated prompt tokens (`prompt_n`) | 1,318,407 | 40,728 |
| Reused prompt tokens (`cache_n`) | 10,971,559 | 325,640 |
| Generated tokens (`predicted_n`) | 321,720 | 10,532 |
| Prompt evaluation time (`prompt_ms`) | 3503.592 s | 112.385 s |
| Generation time (`predicted_ms`) | 3764.183 s | 97.163 s |
| Case wall time sum | 7333.087 s | 397.145 s |
| Max case wall time | 90.867 s | 64.160 s |
| Evaluator wall time | 7346 s | 406 s |

Case wall time sum (incomplete pair; serial completed 160/160, native completed 5/160): serial 7333.087 s, native 397.145 s.

## Paired comparison and non-inferiority

- Planned pairs: 160 (resolved: 5, unresolved: 155)
- Overall native vs serial: 0 wins, 0 losses, 5 ties (1 both-pass, 4 both-fail)
- Paired delta (`on - off`): null (unresolved pairs present; predeclared 5 pp gate inconclusive)
- One-sided 95% conservative binomial lower bound: -0.9876 (margin: -0.0500, alpha: 0.05)
- Non-inferiority result (`bfcl_pairs.py`): `unresolved` (lower bound > -0.0500: false)

| Stratum | Pairs | Both pass | Both fail | Native wins | Native losses | Unresolved |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `multi_turn_base` | 40 | 1 | 1 | 0 | 0 | 38 |
| `multi_turn_long_context` | 40 | 0 | 1 | 0 | 0 | 39 |
| `multi_turn_miss_func` | 40 | 0 | 1 | 0 | 0 | 39 |
| `multi_turn_miss_param` | 40 | 0 | 1 | 0 | 0 | 39 |

## Physical compact and down4 counters

| Counter | Serial | Native |
| --- | ---: | ---: |
| `down4_launches` | 14,313 | N/A (aborted) |
| `observed_fallback_proposals` | 0 | N/A (aborted) |
| `compact.requested` | 0 | N/A (aborted) |
| `compact.launched` | 0 | N/A (aborted) |
| `compact.compact_only` | 0 | N/A (aborted) |
| `physical_counters.q2_sparse` | 520 | N/A (aborted) |
| `physical_counters.skip` | 14,313 | N/A (aborted) |

## Same-build identity checks

All verified build, weight, effective chat-template, and benchmark-freeze identity fields matched across arms (`true`).

- `freeze_sha256`: `95ec72f53fce030abfb629795bfa9cd29c831784297a5966d4199ff8c4acc082`
- `program_sha256`: `ccb4627d51bfa7253d52b40d9f3290b01a1d5ac557732fbf44b990a43b1d9d44`
- `serial` `observations.sha256`: `c1ccb4e90c4816036c8785abf47c3172c7b15d882cbdfb8478c5428cdc691c29`
- `native` `observations.sha256`: `b4eb78a677ec86b3f4c5063e2d3941f8ba59d922180fd0728758976b3ae8fc13`

## Launch outcomes and superseded attempts

- `serial`: `evaluator_exit=0`, `cleanup_exit=0`, `runner_exit=1`, `observation_status=revalidated_after_log_rotation`
- `native`: `evaluator_exit=0`, `cleanup_exit=0`, `runner_exit=1`, `observation_status=aborted_mid_run_batch_overflow`

### Log-rotation observer revalidation

In `serial`, the evaluator completed all 160 cases (`inspect_status=success`, `evaluator_exit=0`, `cleanup_exit=0`), while the wrapper exited `1` because `physical.private.json` recorded `owned server log rotated or lost startup prefix` after Docker `json-file` log rotation. Hardware counters (`down4`, `combined`, `skip`, `q2_sparse`, `q2_stride`, `row0`, `row1`) were reconstructed from `witness/start.json` and `witness/cold.json` (matching `witness/child.pid`), and 409 `compact_guard:` and 409 `compact_readback:` rows in surviving `server.log` (47,639,818 bytes) were verified (serial --spec-type none disables speculative draft-mtp verification, so compact_guard and compact_readback counters remain zero across all surviving log rows).

### Mid-run native abort and root cause

In `native`, the server container aborted mid-run (`runner_exit=1`, `evaluator_exit=0`, `cleanup_exit=0`; `witness/cold.json` and `physical.private.json` absent). Before aborting, 5/160 cases completed cleanly (`multi_turn_base_41`, `multi_turn_long_context_35`, `multi_turn_miss_func_168`, `multi_turn_miss_param_175`, `multi_turn_base_189`) and 59 wire requests succeeded (including turns 1-5 of `multi_turn_long_context_74`; all remaining 155 cases recorded `status=infra`). Matched subset result: 1 serial vs 1 native (0 wins, 0 losses, 5 ties), leaving the predeclared 5 pp non-inferiority gate inconclusive.

- Crash diagnostic (`common_speculative_impl_draft_mtp::process`): `common.cpp:2245: get_sub_batch: failed to add token 22715 at index 2048 (error -1, n = 2050)`
- Last surviving `common_specu: statistics draft-mtp`: `#calls(b,g,a)=59/5432/5432`, `#gen drafts=5432`, `#acc drafts=5057`, `#mean acc len=1.93`
- Last surviving `compact_guard`: `sampler=5432` (all other guards 0); `compact_readback`: all 0 across 59 rows
- Root cause: under `LLAMA_MTP_MERGE_REEVAL=1` in the frozen R2 binary, `common_speculative_impl_draft_mtp::process()` defers 2 trailing verify rows in `dq[0]` at the end of a turn, then on a subsequent cached turn with `prompt_n >= n_batch (2048)` emits the deferred rows plus 2048 prompt rows (`batch.size() = 2050 > llama_n_batch(ctx_dft) = 2048`) and calls unchunked `batch.get()`, triggering `SIGABRT` (exit 134).
- Cohort exposure from the completed serial wire ledger: 30 cached requests (`cache_prompt=true, cache_n > 0, prompt_n >= 2047`) plus 160 cold first-turn requests with `prompt_n >= 2047` (190 total across 160 cases), vs 0 cached requests with `prompt_n >= 2047` in the 4-case smoke cohort.

Prior superseded full-run attempts are explicitly disclosed in `evidence/launch-outcomes.json`:

- 2026-11-19T14:03:55Z (serial, full): `interrupted_partial` (35/160 completed, 406 wire requests, `runner_exit=143`, `cleanup_exit=0`, reason: `Terminated`)
- 2026-11-20T08:53:48Z (serial, full): `startup_timeout` (0/160 completed, 0 wire requests, `runner_exit=2`, `cleanup_exit=0`, reason: `owned server failed to become healthy within 75s`)

## Cached smoke reference

The prior 4-case cached smoke pair (`passes`: serial 3/4, native 3/4; `case_wall_s`: serial 150.104 s, native 145.579 s) is also recorded in `evidence/launch-outcomes.json`.

## Files and verification

- `evidence/bfcl-full-score.json` - per-case outcomes, paired overall and stratum statistics, and physical counters.
- `evidence/bfcl-full-timing.json` - per-arm and per-stratum prompt/cache/decode token and timing totals.
- `evidence/bfcl-full-requests.json` - per-case anonymized wire request sequences (no prompts or transcripts).
- `evidence/launch-outcomes.json` - completed arm outcomes and superseded attempt disclosures.
- `verify-evidence.py` - standalone zero-dependency verification script.

Verify checksums and all invariants from this directory:

```sh
sha256sum -c SHA256SUMS
python3 -B verify-evidence.py
```
