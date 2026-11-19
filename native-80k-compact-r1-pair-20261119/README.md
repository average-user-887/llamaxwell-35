# Compact-only R1 timed pair at 80K

Native MTP1 delivered **99.75087 tok/s** on the cached continuation versus **76.34312 tok/s** with speculation disabled, a **30.66125%** gain in one matched pair. The cached request fell from **40.36693 s to 31.05679 s**. The native arm remained **0.04673 ms per offer** short of 100 delivered tok/s at its measured yield. Both arms exited and cleaned up with status zero. They used the same mapped software and feature flags, eight experts, one Quadro M6000 and one slot. Down4 remained disabled.

| Phase and measure | Serial | Native MTP1 |
| --- | ---: | ---: |
| Warm delivered decode, tok/s | 76.34312 | 99.75087 |
| Warm server-reported post-first rate, tok/s | 76.31827 | 99.71840 |
| Warm request wall, s | 40.36693 | 31.05679 |
| Cold delivered decode, tok/s | 73.99591 | 99.15302 |
| Cold request wall, s | 264.20512 | 269.39259 |

Each mode independently matched all 3,264 of its archived output token IDs, both texts, and the cache and acceptance counts. The warm continuation reused 79,996 prompt tokens and evaluated four in both arms. Native accepted 89 of 101 cold proposals and 1,429 of 1,642 warm proposals. The physical compact output carried two 8-byte records per offer, for **27,888 bytes** across both phases. The typed counters observed zero full target-logit bytes, compact fallbacks and down4 launches. The paired verdict is scoped to this measured request and its matched control.

The modes first differed at output ordinal 170 in both phases. Warm native had 127 duplicate normalized lines out of 304, including 39 out of 76 in the final quarter. Serial had 73 out of 283 overall and 16 out of 71 in the final quarter. This fixture was an unscored continuation. The package retains integer repetition and repeated-window counts for each phase and mode in [`reconciliation.json`](reconciliation.json).

The [published R1 source and exact shadow qualification](https://github.com/average-user-887/llamaxwell-35/tree/68caf44b5f6e7ca331da88354111a6d89221484b/native-80k-compact-r1-20261119) remain the source and verifier-choice evidence. This packet adds the two timed arms and their aggregate replay.
