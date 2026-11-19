# Ordinary long-context reading on one Quadro M6000

This local InfiniteBench adaptation measures one 80,000-token cold reading request, with one slot, all eight routed experts, and the same mapped R2 build throughout. The order is serial, native MTP1, native MTP1, serial. Each process starts fresh with zero cached prompt tokens. Each native arm can offer one draft token per verifier round.

The shortest complete four-question book by character count was selected before generation. The model receives the complete 70,375-token book, a separately labeled contiguous excerpt from another work, and four questions. It generates original rationales and an answer choice for each question. The sampler is greedy with seed 7, the five EOG IDs masked, a declared text delimiter, and a 1,536-token limit. The predeclared sustained-length floor is 512 visible re-tokenized content tokens. Four repeated executions examine one task. The source-grounding review of each rationale is pending. BFCL160 and local80 agentic quality are separate evaluations.

The earlier forced-continuation stress fixture measured 101.29395 native server decode tokens/s and 101.32693 delivered-count tokens/s on its cached 3,072-token request. The two native arms on this adapted ordinary cold reading sample gave lower decode rates. Whole-request time includes the new 80,000-token prefill in each arm. [Published stress fixture and physical witnesses](https://github.com/average-user-887/llamaxwell-35/tree/ef02d241/native-80k-compact-down4-r2-20261119)


## Measured results

| Arm | Raw server tok/s | Delivered-count tok/s | Stream tok/s | Cold total s | Visible tokens | Choices |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 · serial | 75.35759 | 75.47229 | 75.36396 | 269.818 | 653 | 4/4 |
| 2 · native | 93.71159 | 93.84566 | 93.72493 | 272.390 | 695 | 4/4 |
| 3 · native | 93.99862 | 94.13310 | 94.01028 | 272.860 | 695 | 4/4 |
| 4 · serial | 75.62120 | 75.73630 | 75.62999 | 269.768 | 653 | 4/4 |

Pair 1 measured +24.36% native server decode throughput and +2.573 seconds of native whole-request time against serial.
Pair 2 measured +24.30% native server decode throughput and +3.092 seconds of native whole-request time against serial.

| Arm | Accepted/offered drafts | Compact-only launches | Down4 launches | Word stop verified in IDs | Repeated / book-matched / total 24-word windows |
| --- | ---: | ---: | ---: | --- | ---: |
| 1 · serial | 0/0 | 0 | 0 | true | 0 / 0 / 491 |
| 2 · native | 298/402 | 402 | 16080 | true | 0 / 0 / 526 |
| 3 · native | 298/402 | 402 | 16080 | true | 0 / 0 / 526 |
| 4 · serial | 0/0 | 0 | 0 | true | 0 / 0 / 491 |

## Reproduction

After the final serial arm closes, the owner runs the frozen four-receipt comparison against the source that recorded it. The exporter requires the four private terminal receipts in the prescribed order. It projects only typed numeric counts, booleans, runtime labels, and the observed termination category. Source pins identify the exact installed owner scripts and runtime. The underlying data shard and revision are pinned in the generator. Four owner scripts in `source/` are byte-exact. The shell runner relocates one absolute tokenizer interpreter path. Run `python3 -B hydrate.py ABSOLUTE_ORIGINAL_TOKENIZER_PYTHON OUTPUT_FILE` to reconstruct the byte-exact runner. Hydration enforces its original SHA-256 before writing. The owner scripts use relative campaign dependencies. [Source dataset](https://huggingface.co/datasets/xinrongzhang2022/InfiniteBench/tree/90f0394333616266d9fe85824ceaf505093cbaa5)

Once the packet is sealed, run `python3 -B replay.py` in this directory, using the Python standard library. It verifies the file inventory and hashes, chronological arm separation, process and cleanup statuses, prompt and cache counts, generation clocks, real dispatch counts, word termination, choice score counts and pair ratios. The public replay verifies the arithmetic and consistency of the exported observations. Independent per-item grading against model answers and source-grounding review require private evidence.

Server decode rate uses (generated IDs minus one) divided by server decode seconds. The first ID comes from the final prompt batch. Delivered-count rate divides all generated IDs, including delimiter IDs, by the same server interval. Observed client stream rate uses IDs after the first token-bearing packet divided by first-to-last token-bearing event time. Visible whole-request rate uses visible re-tokenized content and includes cold prefill. Draft acceptance counts accepted proposals per offered proposal. Repetition and copying count matching 24-word windows in the completed answer.
