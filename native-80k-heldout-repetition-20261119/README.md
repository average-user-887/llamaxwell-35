# Native MTP1 on two scored source edits

On one Quadro M6000, native MTP1 and serial returned identical text and token IDs through EOS on both source edits. Each answer matched the independently expected complete source text and Python AST. Both arms produced the expected source's repetition counts.

| Fixture and outcome | 80,000-token edit | 32,768-token edit |
|---|---:|---:|
| Output IDs through EOS in each arm | 1,722 | 1,087 |
| Normalized repeated lines overall in each arm | 20 of 149 | 8 of 81 |
| Normalized repeated lines in the final quarter in each arm | 5 of 38 | 0 of 21 |
| Repeated 24-word windows in each arm | 0 | 0 |
| Accepted native drafts | 861 of 861 | 543 of 543 |
| Serial to native server decode | 22.598 to 16.652 s | 13.866 to 10.619 s |
| Serial to native request wall time | 284.303 to 283.495 s | 92.704 to 91.069 s |

The same CUDA 369c6b, server 4a0ff8, and N40 llama libraries ran with eight experts, all 42 layers offloaded and one slot in an 84,224-position context. Each mode used a fresh process, ran the larger edit first and evaluated all input tokens in both requests. The receipts record the actual sampling settings, prefill and decode durations, launch deadlines, exit statuses and physical witness counts. The native 80K request showed the merged draft graph and 8,610 fixed-stride calls. The 32K request logged 5,430 sparse Q2 calls and zero fixed-stride calls. [Scores](paired-scores.json) · [Runtime and witnesses](runtime.json)

The original serial launch scored the larger edit successfully and cleaned up. It ended before the second request because the sampler waited for `n32.json` and the supervisor had written `cold.json`. A revised launcher and observer map `start.json`, `cold.json` and `warm.json` to startup, the 80K edit and the 32K edit. The receipt for the first run passed checksum verification. [Original failure and recovery](original-serial-failure.json)

These two copy-heavy edits share reference material and are independent of the earlier forced throughput continuation. The extra late repetition observed in that continuation did not recur on either scored edit in this pair. Broader repetition, general prose and naturally low-acceptance requests remain unmeasured here. [Reproduction](REPRODUCE.md)
