# Same-build native MTP width comparison

This release uses the exact N100 r1 server and CUDA module `c973955667f1bb7e21a33447da488f182aa6fee0437d1e275180ea5b9d3cb9c5`. Every arm uses the pinned 80,000 prompt IDs, an 84,224-token single slot, microbatch 1,024, fixed Q3 R2 enablement, disabled context shifting, seed 7, greedy sampling and ignored EOS. Each arm runs one cold 192-token request and one warm 3,072-token replay. Only the typed native draft arguments differ across the four modes. Depth three is recursive use of the single learned head. Full-depth width-four target verification uses the existing mrow kernel; the Q3 R2 switch applies to width three.

## Root release

Run serially under the shared lock with the admitted M6000 UUID from `LAB_FACTS.md`. Each attempt checks fresh idle Fleet8090 at 250 W. The 430-second relative cap includes owned cleanup. The prior overnight cutoff is absent from this reauthorized launcher. Every invocation preserves its own receipt, including preflight, admission, timed or technical failures. The current release retains the N100 r1 45-second readiness window.

```bash
C="$CAMPAIGN_DIR"
L="$C/resume-native-width/release.sh"
"$L" --check
N3_GPU="$N3_GPU" "$L" --run no-mtp
N3_GPU="$N3_GPU" "$L" --run depth1
N3_GPU="$N3_GPU" "$L" --run depth2
N3_GPU="$N3_GPU" "$L" --run depth3
"$L" --compare "$SERIAL_RECEIPT" "$DEPTH1_RECEIPT" "$DEPTH2_RECEIPT" "$DEPTH3_RECEIPT"
```

Use the printed `NATIVE_WIDTH_RECEIPT=` paths. Depth three can be omitted. Compare requires a successful uninstrumented serial receipt first, followed by successful native receipts. It reports same-prompt ID divergence and descriptive text structure alongside cold and warm times, emitted counts, draft yield and observed restores. The throughput fixture has no semantic scoring key.

## Performance and optional phase diagnostic

The ordinary arms keep the existing N100 r1 chronological and host-count tracing so the HTTP offer/accept totals reconcile against observed native cycles. Exact console logging overhead is part of the measured configuration. Per-CUDA-event phase and node timers stay off for those timings. The runtime verifies actual PID 1 args, selected experimental flags, 42/42 GPU offload, exact mapped hashes, recurrent snapshot depths, same-slot 79,996/4 warm cache, actual required long dispatch and complete output. A mismatched verifier survives as a failed receipt with the available raw data.

The no-build phase diagnostic is separately selectable for any typed arm. One representative invocation is below.

```bash
N3_GPU="$N3_GPU" "$L" --run depth2 --profile-phase
```

Its existing event timer emits a cumulative table by context and batch width. Those events include startup, cold and warm work. They do not provide synchronized costs for individual speculative cycles, and the comparator excludes this arm from throughput comparisons.

## Private and sanitized receipts

Each run creates `private/` with original logs, requests, generated IDs and text, actual runtime readback, status and hashes. It also creates `sanitized/` with allowlisted receipt JSON and typed cycle rows. All receipt directories are owner-private before any separate publication review.

After a run completes, even if it fails, the independent exporter adds typed trace events and separate checksums without touching the original files.

```bash
taskset -c 0,1 nice -n 19 python3 -B "$C/resume-native-width/trace_export.py" "$NATIVE_WIDTH_RECEIPT"
```

The full typed trace retains chronological accepts and offers, reconciled emitted counts when possible, explicit checkpoint and shift markers, cumulative host timing with invocation counts, stream milestones and final stop/truncation. Fields with unknown per-cycle costs remain null. A native trim classification follows the observed accept marker and known source branch. It does not establish byte-level state equality. An unattributable residual serial token remains explicitly unassigned. Unrecognized server lines and original text and IDs remain in the private raw logs. Include failed attempts in any sanitized cohort package.

## Offline checks

```bash
cd "$C/resume-native-width"
taskset -c 0,1 nice -n 19 bash -n release.sh
taskset -c 0,1 nice -n 19 shellcheck -x release.sh
taskset -c 0,1 nice -n 19 ./release.sh --check
taskset -c 0,1 nice -n 19 python3 -B offline_check.py
```

The N100 r1 measured depth-two receipt, serial contract, staged native depth-one/two/three geometry, historical phase-table parser, explicit trace chronology and failed-receipt export passed local checks. No GPU, runtime API, build or network operations were made by preparation.
