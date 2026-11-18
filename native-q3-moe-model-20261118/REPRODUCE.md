# Reproduce and audit

The evidence package expects publication parent `b9c6598b8f0b04eeff3e8ae4f34beafc30d2c287`. All arms used the same green MoE CUDA DSO, SHA-256 `30ec9a72d6f938656f331b91eb75266e73268bf2015a0d3dd2867eeb427dd58b`, and the model-side host observer. The standalone source and baseline already shipped in the earlier native-MTP resume packet. The runtime held the original Q3 sparse R2 attention fixed. Native used one learned draft head, two proposals, two recurrent rollback slots and the eight-expert target.

```sh
sha256sum -c SHA256SUMS
python3 -B verify-package.py
python3 -B assemble-source.py initial initial-source
python3 -B assemble-source.py r1 r1-source
python3 -B assemble-source.py r2 r2-source
```

Original model inputs, output fingerprints and machine-local archive names are private. The published parser sources and deltas replace those private literals with named placeholders. All other source is retained exactly. Source-pins provides the portable and original software source hashes for the base, initial, R1 and R2 versions. With local originals, pass `--private-inputs OPERATOR_PRIVATE_MAP.json` to assemble-source.py. The expected keys are listed in private-input-keys.json. The assembler verifies that substitution reconstructs the exact original measured source SHA-256 before writing it. Both initial and R2 reconstruction were checked from the original private values. Release.sh and the host observer require no private value substitution.

The model run requires the admitted M6000, its existing benchmark lock, the pinned server/library layout, the campaign's private fixed fixtures, and the release dependencies referenced by the assembled runner. The original source checkout was `da7ba0e49dd43c69f9ee5a219ea3124f7acda0c8` with N1/R2 and the earlier green MoE DSO. In the original card layout, `./build.sh` uses two logical CPUs to compile the additional observer. Each release's `--check` also checks its frozen binary/input inventory. The long cap was 430 seconds per arm, including cleanup; the short cap was 190.

The initial suite ran each of `./release.sh --run smoke-on`, `--run smoke-off`, `--run no-mtp`, and `--run depth2-off`. R2 ran `./release.sh --run depth2-on` and `./release.sh --run no-mtp-on`. R1 introduced the typed serial-on mode, and R2 retained that mode while adding the width-specific recovery witness. R2's `--compare SERIAL_ON_RECEIPT NATIVE_ON_RECEIPT` recomputes the strict ratio. The operator-only ratio uses `--compare ORIGINAL_REFUSED_OFF_RECEIPT NATIVE_ON_RECEIPT`. Checksums and traced exit status remain separate for recovery.

The six *.trace-v2.jsonl files contain chronological typed records for the measured requests and the short arithmetic witnesses. Each *.trace-v2-meta.json verifies its trace and exporter hashes and original statuses. trace_export.py takes an original completed local receipt directory and a portable label. Its exported clocks are process-cumulative backend/host observations or explicit stream events; unavailable per-cycle costs remain null. The equal-length component comparison uses complete cumulative anchors and reports its three-round phase-start offset. The timing units and matched windows are in q3-warm-comparison.json.
