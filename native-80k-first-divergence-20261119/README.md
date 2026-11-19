# First output divergence on the 80K Occamy replay

At output token 170, serial and native MTP each emitted its own target model’s highest-scoring token. The two choices swapped first and second place. The serial target preferred its choice by **0.66038 nat**. The native target preferred its choice by **0.53683 nat**, a **1.19721 nat** change in the relative log odds. The cold and cached phases produced the same scores to the five decimal places recorded.

| Target decision | Emitted before this round | Target row, zero-based | Serial choice log probability | Native choice log probability |
|---|---:|---:|---:|---:|
| Serial, one row | 169 | 0 | −0.52341 | −1.18379 |
| Native MTP1, two rows | 168 | 1 | −1.11982 | −0.58299 |

Both arms generated identical first 169 output tokens. In the native round, the verifier accepted the proposal as output 169. Its second target row then selected output 170. The preceding verifier round also accepted its proposal. The trace brackets the difference at the target scores and shows the sampler followed their ordering in both arms. Its margins exceed the rounding precision by several orders of magnitude. Earlier recurrence, sparse attention, batching and previous rollbacks remain possible contributors. The experiment collected zero recurrent-state tensor comparisons.

Both runs used one Quadro M6000, one slot, all eight experts and 42/42 GPU-offloaded layers. The five mapped executable/library identities matched. Private same-mode archive checks covered every output ID and both complete texts, with 192 outputs after the cold 80,000-token prefill and 3,072 after the cached request reused 79,996 tokens and evaluated four. The source fixture was a forced continuation without semantic grading.

The serial collector first rejected a timestamped snapshot counter line after inference finished. The original runner returned 1 and cleanup returned 0. Revalidation verified the sealed files, reproduced that rejection, and accepted the observed numeric timestamp with or without its information-level character. Counter occurrences, values, phase boundaries, cache counts, output IDs, maps and numerical score checks passed against the original files. The native runner and cleanup each returned 0. The arms occupied separate time intervals.
