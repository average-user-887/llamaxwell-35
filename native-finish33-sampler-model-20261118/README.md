# Native MTP1 reaches the +25% cached decode target

One Quadro M6000, Qwen3.6-35B-A3B, one slot, eight active experts and an 80,000-token prompt. Native depth one reached **96.35163 tok/s** on the cached continuation versus **76.59611 tok/s** on the fresh same-build no-spec run. The gain was **25.79181%**. Both arms used the same mapped executable and libraries and the same feature flags. Speculation arguments were the only server argument difference.

| Actual measurement | Fresh no-spec | Native MTP1 |
| --- | ---: | ---: |
| Cached decode | 76.59611 tok/s | 96.35163 tok/s |
| Cached request wall | 40.23386 s | 32.15160 s |
| Cold decode | 74.33609 tok/s | 95.70981 tok/s |
| Cold request wall | 265.54201 s | 268.38011 s |

Both cached requests reused 79,996 input tokens, evaluated four and returned exactly 3,072 tokens. Both cold requests evaluated all 80,000 input tokens and returned 192. Native accepted 89 of 101 cold drafts and 1,429 of 1,642 cached drafts. The measured cached cost was 19.41731 ms per offered round versus 19.54030 ms allowed at that yield, a margin of 0.12300 ms. The previous native selector cost 19.87165 ms per round on the identical cached output. See the [independent formula audit](formula-audit.json), [typed comparison](comparison.json) and [observed runtime profiles](runtime-pair.json).

## Physical and output checks

The first fixed-stride verifier used a 1,024-byte K/V token pitch at 80,128 padded keys. The owned process counters recorded 1,010 cold and 16,420 cached stride launches, matching the ten attention layers per offered round. The native sampler snapshot was elided for all 101 cold and 1,642 cached rounds, including the 12 and 213 rejected drafts. Both measured requests had zero saved fallback snapshots and zero full restores. No-spec executed neither treatment. The [native and serial](chronology.json) typed timelines preserve every available cumulative observation together with entry, emission and teardown boundaries.

Both modes preserved all 3,264 output IDs and both complete texts against their archived same-mode selector run. Native also retained exact draft offer and accept totals. Native first differed from serial at token 170. The cached continuation had 127 of 304 normalized repeated lines in native and 73 of 283 in serial, unchanged from the respective earlier same-mode outputs. This forced throughput continuation has no independent semantic score. See the [output and repetition audit](repetition.json).

The native server completed both requests. Its original post-run parser returned 1 on a sampler-counter line with a numeric logging prefix. Cleanup returned 0. The original validator rejection reproduced from the unchanged receipt, and the timestamp-only correction then passed the same runtime, counter and output checks on those same files. The fresh serial guard and cleanup both returned 0. See the [literal counter and refusal evidence](native.guard-recovery.json) and [parser diff](parser-r1.patch).

## Timing scope

In 1,618 matched native interior rounds with identical cumulative acceptance histograms, the host target phase improved by 0.31313 ms per offered round. The four observed host scopes together improved by 0.26984 ms per round. The complete decode saving was 0.45435 ms per offered round. The 0.18451 ms arithmetic difference compares those two windows and carries no exclusive attribution to cloning, sampling, or a CUDA kernel. The CPU clone screen measures a synthetic object using the exact runtime's clone and free functions. The standalone [retained-attention operator](../native-q2-r2-stride-20261118) timed the 2,048-byte pitch and verified the actual model's 1,024-byte pitch by exact diagnostic replay. This packet records one representative matched session with the combined stack. See the [host timing audit](host-native-comparison.json), [CPU measurements](cpu-clone-screen.txt) and [reproduction notes](REPRODUCE.md).
