# R2 compact readback and down4 at 80K

Native MTP1 delivered **101.32693 tok/s** on the warm request versus **75.93112 tok/s** with speculation disabled, a **33.44585%** same-build gain. The raw native server rate was **101.29395 tok/s**. At 3,072 outputs over 1,642 offers, the warm round took **18.46389 ms**, **0.24500 ms below** the budget for 100 delivered tok/s. All three arms exited and cleaned up with status zero.

| Phase and measure | Serial | Native MTP1 |
| --- | ---: | ---: |
| Warm delivered decode, tok/s | 75.93112 | 101.32693 |
| Warm raw server rate, tok/s | 75.90640 | 101.29395 |
| Warm request wall, s | 40.58710 | 30.59133 |
| Cold delivered decode, tok/s | 72.45198 | 100.57285 |
| Cold request wall, s | 264.67187 | 268.17477 |

Each arm independently matched all 3,264 token IDs and both texts from its corresponding archive, including cache and proposal totals. Both timed arms mapped the same software and used identical feature flags on one Quadro M6000 with all eight experts and one slot. Shadow retained full logits and checked 3,486 live target choices with zero mismatches. Active native used compact results for all 1,743 proposals and read **27,888 bytes** of choices, with zero full target-logit bytes and zero fallback. Native and shadow physically executed forty down4 calls per offer. Serial had zero measured down4 calls. Startup-only dispatch is accounted separately.

R2 combines the already-qualified R1 compact path with the down4 flag enabled, using the same mapped software as R1. The [first composite shadow refusal](https://github.com/average-user-887/llamaxwell-35/tree/a04b3c9e60cda23dce4bf3025366aacdbf3936d3/native-80k-compact-shadow-refusal-20261119) remains separate. Its typed guard kept compact dormant. Its output trajectory changed and a later UTF-8 stream-buffering event independently omitted one token ID. R2 independently verifies compact execution, emitted lengths and archive parity. See the separately measured [R1 compact-only pair](https://github.com/average-user-887/llamaxwell-35/tree/ef15661db0598e4ed233ed4ca0940a93e2d1b2cb/native-80k-compact-r1-pair-20261119) and [b71 down4-only pair](https://github.com/average-user-887/llamaxwell-35/tree/32be2864551de6fc0b230cdfeda1fb57ffae618d/native-80k-down4-20261119).

The timed modes first differed at warm output ordinal 170. Native repeated 127 of 304 normalized nonblank lines overall and 39 of 76 in the final quarter. Serial repeated 73 of 283 overall and 16 of 71 in the final quarter. This is an unscored forced continuation. Archive parity and repetition document its trajectory. A representative task-success comparison is separate. [Reproduction](REPRODUCE.md)
