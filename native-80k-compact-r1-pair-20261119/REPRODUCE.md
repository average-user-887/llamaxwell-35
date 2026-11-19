# Reproduce the terminal pair accounting

The [R1 source and shadow packet](https://github.com/average-user-887/llamaxwell-35/tree/68caf44b5f6e7ca331da88354111a6d89221484b/native-80k-compact-r1-20261119) pins the runtime, model adapter, source patch, launcher, and exact verifier shadow. The timed pair uses the same R1 source freeze and software profile. The owner replay independently reproduced the controller's exact paired result before exporting only typed evidence.

Run `sha256sum --check --strict SHA256SUMS` and `python3 -B pair-evidence.py` to verify the portable artifacts and rederive rate, phase counter and client interval arithmetic. The already-published R1 `replay.py` also validates `pair-results.json` when passed as `--results`. The replay reads no model files or private requests. Use the R1 source packet to reconstruct the pinned owner build.

For owner-only evidence regeneration, run `python3 -B pair-evidence.py --r1-source VERIFIED_R1_SHADOW_PACKET --owner-card OWNER_R1_CARD --serial SEALED_SERIAL --native SEALED_NATIVE --controller-pair CONTROLLER_PAIR`. The script binds to the published upstream manifest, checks both terminal owner receipts with the exact R1 validator and verifies its independent comparison equals the controller record. It exports public artifact digests, numerical fields and equality booleans. The two source receipts stay private.

## Rate and counter conventions

- The strict delivered decode rate is the number of emitted token IDs multiplied by 1,000 and divided by the server's `predicted_ms`. `predicted_ms` is the server generation window measured from the prompt completion marker. In this implementation the first token comes from the last prompt batch. The delivered convention gives that emitted token credit.
- The raw server field `predicted_per_second` uses the same time denominator and `predicted_n - 1` decode steps. `predicted_per_token_ms` also uses that step count. The two conventions reconcile from the raw server fields in the packet.
- The client wall measurement covers the complete request. The optional warm split rates use the difference in emitted token ordinals divided by the time between token-bearing SSE events. These describe client delivery over the recorded segments. Cold has no matching SSE segment series in this protocol.
- The round accounting divides the same server generation interval by offered native drafts. The round budget for 100 delivered tok/s at measured yield is ten milliseconds times delivered tokens per offer. Raw sampler snapshot witnesses are recorded both cumulatively and as phase deltas.
- Repetition counts normalize each nonblank line by whitespace and replace digit runs with `#`. Duplicate counts are occurrences after the first, both over the complete response and within its last quarter. The 24-word windows use the same digit normalization and count repeated window occurrences after the first.
