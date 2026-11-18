# Counterbalanced repeat of the 80K native MTP1 benchmark

The [original benchmark](../native-finish33-sampler-model-20261118/) ran native depth one first, followed by serial. The repeat uses fresh owned workers in the opposite order on one Quadro M6000, with one slot, all eight experts and the same 80,000-token private prompt. Each arm measures a cold 192-token request and a cached 3,072-token request. The original native runner's timestamp-only parser refusal and successful revalidation remain documented in the original packet.

## Results

| Session and order | Cached serial | Cached native | Native decode gain | Cached serial wall | Cached native wall |
| --- | ---: | ---: | ---: | ---: | ---: |
| Original, native then serial | 76.59611 tok/s | 96.35163 tok/s | 25.79181% | 40.23386 s | 32.15160 s |
| Repeat, serial then native | 76.46155 tok/s | 96.44709 tok/s | 26.13802% | 40.30635 s | 32.11944 s |

Native gained 8.18691 s on cached request wall in the repeat and took 4.40768 s longer on cold request wall. The cached request reused 79,996 input tokens and evaluated four in both modes. Native again accepted 1,429 of 1,642 cached drafts. At that yield, reaching 100 tok/s requires 18.70889 ms per offer versus 19.39809 ms measured, a gap of 0.68920 ms per offer.

Each repeat mode matched all 3,264 output IDs and both full texts from its original run. Native and serial first differed at token 170 in both sessions. The cached native output contained 127 duplicate normalized lines out of 304 and serial had 73 out of 283. Both sets of counts reproduced. The [comparison and parity report](repeat.report.json) includes the four phase comparisons, normalized repetition statistics and the 1,618-round matched host window.

## Evidence

The [attempt ledger](attempts.json) includes exit and cleanup results and links each arm to its typed trace. The initial serial repeat attempt reached 42/42 GPU layer offload at 68.771913 seconds on the server clock. The server did not pass its health check within 75 seconds. It produced no cold or cached response and cleanup returned 0. The next serial launch used the same configured runtime and settings.

The [exporter](export-repeat.py) uses the [existing receipt validator and typed trace exporter](../native-finish33-sampler-model-20261118/REPRODUCE.md). For a successful arm, each trace includes every available recorded host snapshot, sampler completion counter, physical counter boundary and client stream checkpoint. Serial emitted positions and native emitted positions are indexed by response ordinal. Native timings represent cumulative observed host phases and recorded stream checkpoints. The receipt does not time every native offer or assign each emitted position to an offer. Server clock values and client stream clocks remain under their own fields.

Exact output parity compares private emitted token IDs, full text and offer/accept totals in each mode. The report includes equality results and normalized repetition counts. Reproducing those comparisons requires the owner-held prompt and terminal receipts. The throughput continuation has no independent semantic score.

## Local reproduction

The [original build and runner instructions](../native-finish33-sampler-model-20261118/REPRODUCE.md) specify the corrected release wrapper. The [retained-attention operator packet](../native-q2-r2-stride-20261118/) contains the exact CUDA build and both physical pitch checks. The exact server implementation is pinned in the [original build identity](../native-finish33-sampler-model-20261118/build.json). The repeat adds no compiled component or runner changes.

Run the CPU-only exporter in the campaign layout with the four owner-held terminal receipt folders. Keep the original native R0 receipt in the original native position even though it was validated with the corrected parser.

    python3 -B export-repeat.py --failed-serial OWNER_FAILED_SERIAL --repeat \
      OWNER_ORIGINAL_SERIAL OWNER_ORIGINAL_NATIVE OWNER_REPEAT_SERIAL OWNER_REPEAT_NATIVE

For a failed arm, supply the terminal folder with --failed-serial or --failed-native without --repeat. Both completed repeat arms must pass their original terminal validators with runner and cleanup exit 0 before the exporter writes a paired comparison.

Run `sha256sum --check SHA256SUMS` to verify the package.
