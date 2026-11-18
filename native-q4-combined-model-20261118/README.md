# Four-row native MTP at 80K on Quadro M6000

Four rows with three proposals per target check reached **79.59669 tokens/s**, versus **76.50948 tokens/s** for serial generation with the same software, sparse-attention and MoE flags. The gain was **4.035%**. The cached 3,072-token request fell from 40.26263 to 38.84101 seconds. With a cold 80,000-token prompt and 192 generated tokens, the combined arm took 5.13705 seconds longer than serial. The output streams first diverged at token 60 on both requests. This sample missed the 2× target.

| Mode | Cached tokens/s | Cached wall, s | Native rounds | Output per round | ms/round |
|---|---:|---:|---:|---:|---:|
| Serial, both options enabled | 76.50948 | 40.26263 | — | — | — |
| Depth 3, both disabled | 60.42233 | 51.10663 | 887 | 3.4611 | 57.3192 |
| Depth 3, four-row sparse attention | 71.81655 | 43.03741 | 969 | 3.1692 | 44.1441 |
| Depth 3, sparse attention and four-row MoE cohorts | 79.59669 | 38.84101 | 949 | 3.2350 | 40.6687 |

Sparse attention saved 13.1751 ms per observed warm round against the same-depth disabled arm. Adding MoE cohorts saved a further 3.47545 ms per round. The streams first diverged at tokens 19 and 59 respectively. Cumulative host counters over 825 matched rounds attributed 13.02755 ms/round to the target-decode scope for attention. Over 875 matched rounds the MoE comparison showed 3.48647 ms/round in that scope. The counters leave direct rollback duration unknown. Full cycle, incomplete-counter, output-boundary and stream-clock records are in the six chronologies.

In the combined warm generation, all 949 width-four calls launched the full-eight-expert MoE operator in all 40 target layers. The extra four-row prompt evaluation launched 40 times, for 38,000 physical launches in total. Serial had 39 prompt launches and zero generation launches. The combined arm accepted 2,121 of 2,847 draft tokens. Prefix lengths zero through three occurred 134, 106, 112 and 597 times. Warm repeated-line occurrences were 64 of 253 nonempty lines in the combined stream and 92 of 293 in serial. These counts describe the output structure. The 80K workload has no independent semantic score. Both separate 2K smoke arms passed the independently scored JSON arithmetic request through EOS. All twelve performance requests ended at their requested limit with zero emitted EOS/EOG entries.

The separate 2K rejection experiment used the same combined CUDA library. Both independently prefilled width-four processes deliberately supplied three wrong suffix tokens, fired 40 physical MoE launches and 30 cache-writing GDN kernels, and trimmed position 2007 back to 2004 before applying the actual target correction. The wrong-tail states differed before trimming. The retained and corrected logical KV and recurrent payloads were bit-identical across the two wide runs. The independently re-prefilled width-one replay chose the same continuations while 15,593 retained KV bytes and 15,518,964 retained recurrent float words differed. The full anonymous per-layer differences are in rollback/observed.json. This separate 67-second check was below the long sparse-attention threshold.

All six model arms and the separate rollback run returned status zero with successful cleanup. Both the single-slot checkpointed 80K workload and short arithmetic are exploratory. The separate integer Phase B operator experiment used a different candidate library.
