# Native R2 IFEval paired evaluation (full, 541 prompts, 834 instructions)

This package records the paired IFEval evaluation of `native` (`draft-mtp`, 1 draft token, f16 draft K/V) against `serial` (`--spec-type none`) on `qwen36-moe` using `google/IFEval` (revision `966cd89545d6b6acfd7638bc708b98261ca58e84`).

## Headline paired outcome

| Metric | Serial (off) | Native (on) | Native wins / losses / ties |
| --- | ---: | ---: | ---: |
| Prompts (strict) | 441/541 (81.52%) | 431/541 (79.67%) | 18 / 28 / 495 |
| Prompts (loose) | 462/541 (85.40%) | 451/541 (83.36%) | 18 / 29 / 494 |
| Instructions (strict) | 728/834 (87.29%) | 718/834 (86.09%) | 22 / 32 / 780 |
| Instructions (loose) | 751/834 (90.05%) | 739/834 (88.61%) | 21 / 33 / 780 |

## Timing and token totals

| Measure | Serial | Native |
| --- | ---: | ---: |
| Prompt tokens sum | 31,737 | 31,737 |
| Completion tokens sum | 175,436 | 175,890 |
| Case wall time sum | 2316.317 s | 2090.230 s |
| Max case wall time | 12.306 s | 11.206 s |
| Evaluator wall time | 2320 s | 2095 s |
| Model-cap (`length`) cases | 28 | 33 |

## Physical compact and down4 counters

| Counter | Serial | Native |
| --- | ---: | ---: |
| `down4_launches` | 0 | 4,079,080 |
| `observed_fallback_proposals` | 0 | 101,977 |
| `compact.requested` | 0 | 0 |
| `compact.launched` | 0 | 0 |
| `compact.compact_only` | 0 | 0 |
| `physical_counters.q2_sparse` | 0 | 0 |
| `physical_counters.skip` | 0 | 4,079,080 |

## Same-build identity checks

All verified build, weight, effective chat-template, and benchmark-freeze identity fields matched across arms (`true`).

- `freeze_sha256`: `f3f8d9980e3f12a05ab0a6a9de52ab2a6eae559744e768116e28f7c7c194a7cd`
- `program_sha256`: `82a80352ac146de661d67b6f353b7f57d3da4e3e52a8fe9a3da58e5125a6a738`
- `serial` `observations.sha256`: `c918d808198e0e2ea7a78e3372710e4afb277996382637fd58fecde5e30a7d4e`
- `native` `observations.sha256`: `5aaddf9764e557a7be37d8b3ea62a22c4e1fb7f1b9391888bc30f892fef38873`

## Launch outcomes and revalidation

- `serial`: `evaluator_exit=0`, `cleanup_exit=0`, `runner_exit=1`, `observation_status=revalidated_after_log_rotation`
- `native`: `evaluator_exit=0`, `cleanup_exit=0`, `runner_exit=0`, `observation_status=nominal`

### Log-rotation observer revalidation

In `serial`, the evaluator finished all 541 prompts (`evaluator_exit=0`, `cleanup_exit=0`), while the wrapper exited `1` because `physical.private.json` recorded `server log lost prefix or rotated` after Docker `json-file` log rotation. Hardware counters were reconstructed from `witness/start.json` and `witness/cold.json` (matching `witness/child.pid`), and 42 `compact_guard:` and 42 `compact_readback:` rows in surviving `server.log` (9,231,238 bytes) were verified (serial --spec-type none disables speculative draft-mtp verification, so compact_guard and compact_readback counters remain zero across all surviving log rows).

## Smoke reference

The prior 6-prompt IFEval smoke pair (strict & loose prompts: serial 5/6, native 4/6; `evaluator_wall_s`: serial 25 s, native 23 s) is recorded in `evidence/launch-outcomes.json`.

## Files and verification

- `evidence/ifeval-full-score.json` - per-prompt strict/loose outcomes, paired tallies, instruction breakdown, and physical counters.
- `evidence/ifeval-full-timing.json` - per-prompt and aggregate token and timing totals.
- `evidence/launch-outcomes.json` - completed arm outcomes and any superseded attempt disclosures.
- `verify-evidence.py` - standalone zero-dependency verification script.

Verify checksums and all invariants from this directory:

```sh
sha256sum -c SHA256SUMS
python3 -B verify-evidence.py
```
