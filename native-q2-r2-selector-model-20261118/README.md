# Exact Q2 sparse selector in 80K Occamy native verification

With all eight experts on one Quadro M6000, the selector build delivered **94.14864 cached decode tokens/s in native depth one** and **76.58202 tokens/s in same-build serial**. The measured gain was **22.93831%**, below the 25% warm gate. Native cached request wall fell from 40.25440 to 32.89173 seconds. Compared with the previous 19607daf native arm, native cached decode gained 2.14190 tokens/s and saved 0.46261 ms per offered round. This is one fresh serial/native comparison and one comparison against an earlier native run.

| 80K fixture | Same-build serial | Same-build native D1 |
|---|---:|---:|
| Cold decode, all IDs/s | 74.23110 | 93.80455 |
| Cold request wall, s | 264.08102 | 267.74470 |
| Warm decode, all IDs/s | 76.58202 | 94.14864 |
| Warm request wall, s | 40.25440 | 32.89173 |
| Cold aggregate sparse Q2 launches | 0 | 1,010 |
| Warm aggregate sparse Q2 launches | 0 | 16,420 |
| Cold accepted / offered | 0 / 0 | 89 / 101 |
| Warm accepted / offered | 0 / 0 | 1,429 / 1,642 |

Both arms loaded the identical selector, Q2 MoE, Q8A and server build and enabled the same flags. Both verified full GPU offload and eight active experts. Each completed the 192-token cold and 3,072-token warm request and returned successful owned cleanup. Each reproduced all 3,264 IDs and full text from its previous same-mode 19607daf run. Native also matched the earlier Q8A native archive in IDs, full text, and proposal and acceptance totals. The first native versus strict serial difference remained ID 170 in both phases. Warm repetition matched the respective prior runs. Native and serial had 127/304 and 73/283 duplicate normalized lines, 39/76 and 16/71 in the final quarter, and 63/1,336 and 46/1,318 repeated 24-word windows. Neither arm had malformed characters or trailing comment-only lines. These counts describe one forced continuation. No independent semantic score was collected.

The read-only owned-child witness verified ten aggregate sparse Q2 calls per offered native verifier round in both request phases and zero in serial. The first new-branch dispatch was at 80,128 padded keys. The observed periodic width markers covered all 13 aligned widths through 83,200 across the two native phases. There is no per-call fast-selector counter. The independently [screened standalone operator](../native-q2-r2-selector-20261118) saved 42.95 and 43.10 μs at 81,920 keys with bitwise-identical scores, plans and attention output.

The native comparison to 19607daf matched 66 cumulative acceptance histograms over offered rounds 125 through 1,743. Across that shared warm window, target host time fell 0.44392 ms per offered round. This reported phase covers all target work. It does not isolate the selector or another kernel's exclusive time.

The [same-build comparison](comparison.json), [native receipt](native.receipt.json), [serial receipt](serial.receipt.json), [repetition audit](repetition.json), and [matched host timing](host-native-comparison.json) contain safe aggregate evidence. [Reproduction notes](REPRODUCE.md) describe the source and owner-held inputs.

The [complete typed chronologies](chronology.json) retain every recorded cumulative host snapshot, both phases' stream checkpoints and final timing records, offload and sampled attention dispatch, request and cleanup boundaries, and all five observed counters per boundary. The native log contained 70 cumulative snapshots. Individual native rounds have no clock or literal per-round acceptance record in these logs. The serial trace records observed emission positions and does not expose token IDs.
