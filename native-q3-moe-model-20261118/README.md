# Native depth two with three-row MoE at 80K

With both targets using the same cohort setting, native depth two produced **78.790 tokens/s** against **76.609 tokens/s** for serial decoding on the warm 80K request. That is **2.85% higher throughput** and **0.979 seconds less request time** over 3,072 output tokens. The native cold request took 3.878 seconds longer. The strict warm comparison falls below 2×. These are single-run comparisons on one fixed prompt and one Quadro M6000 24GB.

| Physical arm | Cold decode, tokens/s | Warm decode, tokens/s | Warm request, seconds | Original runner / cleanup |
|---|---:|---:|---:|---:|
| Short depth two, generic Q3 | 78.992 | 79.144 | 2.526 | 0 / 0 |
| Short depth two, cohort Q3 | 86.025 | 85.574 | 2.342 | 0 / 0 |
| Long serial, cohort flag off | 76.777 | 76.728 | 40.165 | 0 / 0 |
| Long depth two, generic Q3 | 71.986 | 71.920 | 42.977 | 1 / 0, independently recovered |
| Long depth two, cohort Q3 | 79.609 | 78.790 | 39.253 | 0 / 0 |
| Long serial, cohort flag on | 76.350 | 76.609 | 40.232 | 0 / 0 |

The two native 80K arms isolate the verifier change. The cohort arm gained **9.55%** warm throughput over generic Q3, reducing the mean native round from 35.804 to 33.183 ms. Mean tokens yielded per round increased from 2.574 to 2.614. The accepted-prefix histograms for lengths zero, one and two were 172/164/857 with generic Q3 and 156/142/877 with cohorts. The first output difference occurred at position 61 for that pair and position 74 for the strict serial/native pair. The two serial arms produced identical token sequences in both measured requests.

The observer counted every intended target window in the native cohort arm. Cold had 71 width-three verifications and 2,840 physical launches. Warm had 1,174 width-three verifications and 46,960 launches. All forty layers matched those counts. One additional warm round offered one proposal and used the existing two-row path. The generic control executed zero cohorts. The net warm gain of 2.621 ms per native round represents 43.3–48.4% of the earlier standalone projection of 5.415–6.053 ms across forty windows. The two output trajectories diverged, so the difference between those figures cannot be assigned to an individual kernel.

Over 1,025 matched-span warm rounds with complete process-counter anchors, the reported target-decode component fell from 30.112 to 27.363 ms per round. Draft work changed from 4.527 to 4.553 ms. The recorded component snapshots, incomplete records, residual time and precise anchor offsets accompany this report. Direct rollback time was unavailable. There were zero checkpoint restorations. Speculative rejection markers identify the native trim branch; this run did not measure the trim's standalone cost.

The first long native generic run completed both requests and exited 1 during verification. Its observer resolved the correct provider and counted one legacy Q1 dispatch and 78 Q2 row launches during startup. All 74 cold and 1,193 warm proposals had width two, producing width-three target batches. Neither phase needed another legacy Q1/Q2 target MoE launch. The recovery rechecked the frozen raw archive, confirmed phase-local sparse Q3 attention after acceptance and reconciled all 1,267 rounds with zero checkpoint restores. The published original status and trace retain exit 1; the recovered values are identified separately.

Both short independent arithmetic witnesses returned the required single-key JSON, had the correct integer, and stopped at EOS. The 2K performance requests contained 192 output tokens each. At 80K the cold and warm requests produced 192 and 3,072 tokens, with EOS ignored and an explicit length-limit stop. All twelve performance streams had zero observed EOS or EOG tokens and no truncation. Repetition is summarized per arm as identical-line occurrences, repeated four-token sequences and longest same-token run. The long prompt has no independent answer key, and these diagnostics do not establish equivalence of the divergent continuations.
