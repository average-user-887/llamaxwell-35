# Native R2 local workflow episode-cache evaluation (smoke, 2 cases)

This package records the paired local mock-workflow evaluation of `native` (`draft-mtp`, 1 draft token, f16 draft K/V) against `serial` (`--spec-type none`) under the frozen within-episode prefix cache policy (`episode_local_exact_prefix_v1`) on `qwen36-moe`.

## Headline paired outcome

| Metric | Serial (off) | Native (on) | Native wins / losses / ties |
| --- | ---: | ---: | ---: |
| Final state | 1/2 | 1/2 | 0 / 0 / 2 |
| Joint outcome | 0/2 | 1/2 | 1 / 0 / 1 |

| Cache and timing measure | Serial | Native |
| --- | ---: | ---: |
| Chat requests | 12 | 12 |
| Requests with positive reuse (`cache_n > 0`) | 10 | 10 |
| Reused prompt tokens (`cache_n`) | 51,300 | 51,295 |
| Newly evaluated prompt tokens (`prompt_n`) | 11,733 | 11,717 |
| Generated tokens (`predicted_n`) | 1,320 | 1,428 |
| Prompt evaluation time (`prompt_ms`) | 29.012 s | 30.171 s |
| Generation time (`predicted_ms`) | 16.840 s | 12.765 s |
| Episode wall sum | 49.097 s | 45.724 s |
| Max episode wall | 27.968 s | 26.468 s |
| Evaluator wall | 51 s | 48 s |

Episode wall sum: serial 49.097 s -> native 45.724 s (6.87% reduction).

## Physical compact and down4 counters

| Counter | Serial | Native |
| --- | ---: | ---: |
| `down4_launches` | 0 | 29,120 |
| `sampler_guard_fallbacks` | 0 | 728 |
| `compact_only` | 0 | 0 |
| `physical_counters.q2_sparse` | 0 | 370 |

## Same-build identity checks

All verified build, weight, effective chat-template, budget, and workflow manifest identity checks passed (`true`).

- `manifest_sha256`: `724ca7e5694c1182c06fab2b6bb9e525e81ba562060e86636aaca246608918e7`
- `serial` `observations.sha256`: `e947ef76ba65bbb0921dc967192f5357c728a5eb9b930e0a9094edfe56b6b6c2`
- `native` `observations.sha256`: `9d67b45ad0b9659d310bef2d04a4fb767142f9acdda1a87193a1bf29d84c675f`

## Launch outcomes

- `serial`: `evaluator_exit=0`, `cleanup_exit=0`, `runner_exit=0`, `observation_status=nominal`
- `native`: `evaluator_exit=0`, `cleanup_exit=0`, `runner_exit=0`, `observation_status=nominal`

## Smoke per-case summary

| Case | State/joint (S) | State/joint (N) | Reuse reqs (S/N) | Wall s (S/N) |
| --- | ---: | ---: | ---: | ---: |
| `agentdojo-travel-u1` | 1/0 | 1/1 | 3/3 | 21.129 / 19.256 |
| `tau2-retail-23` | 0/0 | 0/0 | 7/7 | 27.968 / 26.468 |

## Files and verification

- `evidence/agentic-local-cache-pair.json` - per-case outcomes, token/cache totals, scores, and physical counters.
- `evidence/agentic-local-cache-requests.json` - anonymized per-case wire request sequences (no prompts or transcripts).
- `evidence/launch-outcomes.json` - completed arm outcomes and any superseded attempt disclosures.
- `verify-evidence.py` - standalone zero-dependency verification script.

Verify checksums and all invariants from this directory:

```sh
sha256sum -c SHA256SUMS
python3 -B verify-evidence.py
```
