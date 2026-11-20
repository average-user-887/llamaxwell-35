# Quality80 Early History - Token 170 Trajectory Pair

Sanitized, self-verifying evidence packet for the `quality80-early-history` cold 80k trajectory diagnostic (`serial` vs `native`, 192 output tokens, private decision trace retained for ordinals `1..170`).

## Scope and Privacy

- **Raw prompt text, generated text, token IDs, container IDs, loopback ports, local paths, and GPU UUIDs are redacted.**
- Only anonymous 1-based decision ordinals, row indices, rounded top-8 log-probabilities, equality booleans, timing counters, and hardware witness deltas are published.
- Identical prompt digest (`7c1dcdc2f609928922953fc626d8b90e5b5288092036f21b57f6ffda0a128dcc`), prompt token count (`80000`), `seed=7`, `temperature=0`, and `n_out=192`.

## Headline Trajectory Findings

- **Token-level equality prefix (`192` cold tokens)**: `169` / `192` tokens match before the first emitted divergence at ordinal `170`.
- **First actual MTP draft rejection (`native`)**: offer `30` at decision ordinal `60` (`rejected_proposal_differs_from_raw_top1=True`).
- **Strictly before first rejection (`ordinals 1..59`)**: `first_rounded_score_or_rank_difference=1`, `first_order_difference=1`, `first_emitted_difference=None`.
- **Through ordinal 170 (`1..170`)**: `first_rounded_score_or_rank_difference=1`, `first_order_difference=1`, `first_emitted_difference=170`.
- **Native decision at ordinal 170**: `row_zero_based=1`, `top8_order_equal=False`, `serial top1-top2=0.66038`, `native top1-top2=0.53683`.
- **Divergence counts (`1..170`)**: score diffs in `170/170` steps, top-8 order diffs in `166/170` steps, top-1 flips in `1/170` steps.

## Prefill and Trajectory Mechanism

- Ordinal `1` (`n_dec=0`, end of the 80,000-token prompt prefill before any draft verify batch runs) already shows small raw log-probability differences and a top-8 tail order swap because `draft-mtp` enables `slot.need_embd()` across the prompt, evaluating Layer 39 over `M=1024` uncropped ubatch positions instead of `M=1` final position in serial `--spec-type none`.
- From decision `2` (`n_dec=1`) onward, speculative 2-row verification (`M=2` kernels) vs 1-row serial decode adds sub-nat floating-point drift across pre-rejection rounds before the first draft rejection occurs.

## Arm Summary

| Metric | Serial (`--spec-type none`) | Native (MTP cold) |
| --- | ---: | ---: |
| Observation status | nominal (`runner_exit=0`) | revalidated_terminal_destructor_histogram (`runner_exit=1`) |
| Prompt tokens | 80000 | 80000 |
| Output tokens (`n_out`) | 192 | 192 |
| Verifier rounds (`1..170`) | 0 | 88 |
| Draft offered / accepted | 0 / 0 | 101 / 89 |
| Prompt ms | 260897.913 | 265445.616 |
| Decode ms (`predicted_ms`) | 2790.894 | 2175.663 |
| Decode tok/s (`predicted_per_second`) | 68.437 | 87.789 |
| Wall seconds (cold HTTP) | 263.71 | 267.643 |
| Combined kernel delta | 0 | 4040 |
| `q2_sparse` / `q2_stride` delta | 0 / 0 | 1010 / 1010 |
| `row0` / `row1` delta | 0 / 0 | 0 / 0 |

## Native Histogram Revalidation

Frozen verifier ref_hist matched ARCHIVED server-pre-warm.log at periodic checkpoints rounds=25,50,75,100 [(0,25),(3,47),(7,68),(11,89)]; live container shutdown via docker stop also flushed the destructor line rounds=101 restores=0 acc_hist=[12,89,0,0] matching timings.draft_n=101 and draft_n_accepted=89.

## Verification

```sh
sha256sum -c SHA256SUMS
python3 -B verify-evidence.py
```
