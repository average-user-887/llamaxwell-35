# LlamaMaxwell-35 R3 Card 2: Frozen IFEval (541) and BFCL (160) Paired Evaluations

Paired agentic and instruction-following evaluations on `combined-r3-card1` (with `T1-T3+A2` fixes and 80K-verified Card 1 runtime flags) for `occamy-1.0-Q4_0-headq40b-mtp`.

## 1. Frozen `google/IFEval` (541 prompts, 834 instructions, `--cache-ram 12288`)

| Arm | Wall | Scored / Cap | Request / Scoring Err | Strict Prompt | Strict Inst | Loose Prompt | Loose Inst | Agg tok/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `R3-lossless serial` | 2314.87 s | 513 / 28 | 0 / 0 | 81.52% (441/541) | 87.29% | 85.40% | 90.05% | 75.93 |
| `R3-lossless native` | 2095.32 s | 508 / 33 | 0 / 0 | 79.67% (431/541) | 86.09% | 83.36% | 88.61% | 84.29 |
| `R3-Q8C native` | 2025.93 s | 509 / 32 | 0 / 0 | 80.96% (438/541) | 86.69% | 84.66% | 89.45% | 88.67 |

- `R3-lossless native` vs `serial`: `-1.85 pp` strict prompt (`18` on-only, `28` off-only), `compact_only=101,975/101,975` (`fb=0`), `r3_cuda_graphs` `143,742/206,138` replays (`inc=0`).
- `R3-Q8C native` passed the predeclared `-2.5 pp` IFEval stop rule (`+1.29 pp` vs `R3-lossless native`, `-0.555 pp` vs `R3-lossless serial`).

## 2. Frozen BFCL Multi-Turn (160 cases, `--cache-ram 0`)

- Verified `T2` (`common_speculative_impl_draft_mtp::process` sub-batch overflow) resolved: `R3-lossless native` completed **160/160 cases with `0` infra errors** (vs R2's 5/160 crash).

| Arm | Wall | Pass (/160) | Scored / Cap / Infra | Prompt eval s | Gen s | Gen tok/s | Draft Accept / Compact |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `R3-lossless serial` | 7353.26 s | 44.38% (71/160) | 134 / 26 / 0 | 3510.17 s | 3762.15 s | 85.51 | serial (`cg rep=313450/325398`) |
| `R3-lossless native` | 6621.23 s | 41.25% (66/160) | 130 / 30 / 0 | 3674.25 s | 2867.54 s | 111.06 | `151,371/166,005` (`fb=0`) |
| `R3-Q8C native` (partial) | 2309.13 s | 20/49 completed | 39 / 10 / 111 | 1234.81 s | 883.52 s | 112.78 | stopped by host 80C thermal SIGKILL (exit 137) |

- Official `R3-lossless` paired report (`bfcl_pairs.py`): `both_pass=54`, `both_fail=77`, `off_only=17`, `on_only=12`, `paired_delta=-3.125 pp` (`lower95=-15.51 pp`, `noninferiority=unresolved` on `160` cases). Native saved `732.03 s` wall time and reached **`111.06 tok/s`** generation speed (`+29.9%` vs serial's `85.51 tok/s`).
- Optional `R3-Q8C native` BFCL note: completed `49/160` cases (`20` pass, `39` scored, `10` model_cap, `0` CUDA errors, `restores=0`) before the host thermal watchdog killed the container at `80C >= 80C` during sustained prefill (`exit_code=137`); remaining `111` cases recorded `status=infra` and are retained honestly as a partial receipt.

## Verification
```sh
sha256sum -c SHA256SUMS
python3 -B verify-evidence.py
```
