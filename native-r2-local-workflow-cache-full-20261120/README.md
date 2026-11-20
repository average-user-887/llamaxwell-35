# Native R2 local workflow episode-cache evaluation (full, 80 cases)

This package records the paired local mock-workflow evaluation of `native` (`draft-mtp`, 1 draft token, f16 draft K/V) against `serial` (`--spec-type none`) under the frozen within-episode prefix cache policy (`episode_local_exact_prefix_v1`) on `qwen36-moe`.

## Headline paired outcome

| Metric | Serial (off) | Native (on) | Native wins / losses / ties |
| --- | ---: | ---: | ---: |
| Final state | 41/80 | 39/80 | 3 / 5 / 72 |
| Joint outcome | 40/80 | 39/80 | 4 / 5 / 71 |

| Cache and timing measure | Serial | Native |
| --- | ---: | ---: |
| Chat requests | 425 | 411 |
| Requests with positive reuse (`cache_n > 0`) | 345 | 331 |
| Reused prompt tokens (`cache_n`) | 1,538,851 | 1,472,825 |
| Newly evaluated prompt tokens (`prompt_n`) | 368,621 | 363,950 |
| Generated tokens (`predicted_n`) | 45,833 | 41,826 |
| Prompt evaluation time (`prompt_ms`) | 908.358 s | 941.237 s |
| Generation time (`predicted_ms`) | 526.781 s | 371.106 s |
| Episode wall sum | 1475.834 s | 1351.896 s |
| Max episode wall | 38.482 s | 29.955 s |
| Evaluator wall | 1477 s | 1353 s |

Episode wall sum: serial 1475.834 s -> native 1351.896 s (8.40% reduction).

## Physical compact and down4 counters

| Counter | Serial | Native |
| --- | ---: | ---: |
| `down4_launches` | 234 | 859,320 |
| `sampler_guard_fallbacks` | 0 | 21,480 |
| `compact_only` | 0 | 0 |
| `physical_counters.q2_sparse` | 0 | 370 |

## Same-build identity checks

All verified build, weight, effective chat-template, budget, and workflow manifest identity checks passed (`true`).

- `manifest_sha256`: `724ca7e5694c1182c06fab2b6bb9e525e81ba562060e86636aaca246608918e7`
- `serial` `observations.sha256`: `7326f4d32c92d8d13dbf709fe420159d0b4b3cfd23d836ac4e1b101c1b2753ee`
- `native` `observations.sha256`: `5f0099744d5e6f447c99247e0526470278e9bb8211434c1cc4af0320087e40c6`

## Launch outcomes

- `serial`: `evaluator_exit=0`, `cleanup_exit=0`, `runner_exit=0`, `observation_status=nominal`
- `native`: `evaluator_exit=0`, `cleanup_exit=0`, `runner_exit=0`, `observation_status=nominal`

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
