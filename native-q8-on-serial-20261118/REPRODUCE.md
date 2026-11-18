# Reproducing the paired Q8A profiles

The [original Q8A native benchmark](../gdn-q8-warm80k-20261118) records the measured native depth-one arm and separate scored exercises. Rebuild the original [Q8A source](../gdn-q8-phase-a-20261118) or retain its exact mapped CUDA artifact, then use the pinned N1 server implementation, N1 target and draft libraries, Sparse R2 attention, full eight experts and ub1024. The [lineage file](lineage.json) pins the original artifact and the added serial wrapper and guard.

The new serial profile uses the same model, environment values and common argv as the archived native Q8A-on arm. Its only argument change is the spec group. It uses `--spec-type none` with no draft-only settings. The two-row Q8A specialization runs in the native verifier. The serial profile still enables the same flags and its steady target uses one-row sparse attention.

The wrapper reuses the established isolated launcher and campaign helpers. Place the release and guard in the matching campaign layout, supply the owner-approved device and the original private native reference with `N88_PRIVATE_REFERENCE`, and pin the local inputs in the expected `FROZEN.sha256` before release. The private 80K fixture is required for exact stream reproduction.

```bash
sha256sum -c SHA256SUMS
export N88_PRIVATE_REFERENCE
bash release.sh --check
N3_GPU="$N3_GPU" bash release.sh --run serial
bash release.sh --compare "$SERIAL_RECEIPT"
```

The archived native trace reproduces available typed host timing snapshots, observed request boundaries and stream clocks. The published events carry counts and timing values. Per-native-cycle fields remain unavailable where the original logger supplied only cumulative measurements.
