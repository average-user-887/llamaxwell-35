# Same-build validation of bounded prompt lookup

On the fixed 80,000-token code-edit fixture, the bounded lookup median was **118.495484 tok/s** warm versus **58.044483 tok/s** serial and **106.387711 tok/s** stock q256 on the same patched build. The ratio of decode durations was **2.04146×** versus serial and **1.11381×** versus stock q256. Median bounded warm wall time was **14.665312 s** versus **29.793249 s** serial, a **2.03155×** whole-request speed ratio.

| 80K request | Serial decode (tok/s) | Stock q256 decode (tok/s) | Bounded decode (tok/s) | Serial wall (s) | Stock wall (s) | Bounded wall (s) |
|---|---:|---:|---:|---:|---:|---:|
| Cold | 58.004801 | 102.792543 | 116.882934 | 290.924297 | 278.622904 | 277.220544 |
| Same-slot warm | 58.044483 | 106.387711 | 118.495484 | 29.793249 | 16.315408 | 14.665312 |

The serial and stock controls each ran once per cache condition. The bounded results are medians of two runs per condition. The emitted rate uses all 1,722 IDs including EOS over server decode duration. The speed ratios use decode or request durations as labeled. Cold bounded wall saved 13.703754 s, a **1.04943×** whole-request speed ratio versus serial. Server prefill for the cold bounded median was 262.462837 s. [Allowlisted measurements](evidence.json)

## Matched work

All arms used identical patched runtime library hashes, the same 80,000 prompt IDs and rendered template, and 42/42 GPU offload. Both controls omitted the span. The bounded requests supplied the caller-verified input span with the standalone terminal LF excluded. All warm requests reused 79,996 prompt tokens and evaluated four.

All eight completions matched the same 1,722 output IDs, including EOS, and passed the expected edit's normalized text and AST checks. Stock q256 logged eight full-width calls with 1,681/2,040 accepted drafts and two confirmed checkpoint restores and replays, q101 and q54. Bounded lookup retained 1,681 accepts over 1,837 proposals with seven full q256 calls, terminal q53, and one confirmed q101 restore and replay. These are observed verifier shapes on one copy-heavy source revision.

## Same-slot field removal at 32K

A separate run issued stock cold, bounded warm, then stock warm on the same server and slot. All three distinct tasks produced the same 1,722 IDs through EOS as the earlier serial run and matched the expected normalized text and AST. The stock cold request took 89.060722 s. Both warm requests reused 32,764 tokens and evaluated four.

Bounded warm took **10.686900 s** (162.6785 tok/s). The subsequent stock warm request took **11.818175 s** (146.9808 tok/s). The wall difference was **1.131275 s** and the decode difference was **1.130522 s**. Only the bounded request supplied the span and showed terminal q53. After the field was omitted, the final stock request again logged eight q256 calls, 2,040 draft proposals, and both q101 and q54 restores and replays. This is one ordered same-slot check of request-local state.

## Reuse the published configuration

Reconstruct the published corrected R1 source and use its recorded binary and runtime fingerprints. For serial, use `--spec-type none`. For stock and bounded, use the same `ngram-simple` settings with lookup key 16, maximum proposal 255 and one minimum hit. Add `lookup_source_span` only for the bounded request. Run each 80K arm with one slot, the recorded 84,224-token context, full F16 K/V, ubatch 1,024, and disabled context shift. Set `cache_prompt=true`, seed 7, temperature zero, EOS stopping and output cap 3,072. Replay the same prompt in the same slot. [Corrected source, field contract, and build instructions](https://github.com/average-user-887/llamaxwell-35/blob/31f46e620ed3648a3af1b9cdd519b7174e720334/bounded-lookup-20261118/README.md)
