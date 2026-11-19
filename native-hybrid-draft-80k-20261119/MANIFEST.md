# Packet manifest

- `same-build-pair.json` holds the two fresh, fully validated arms. `pair-audit.json` rederives throughput, yield, the 100 tok/s gap and identical actual build from the original terminal receipts.
- `hybrid-vs-split.json` compares the fresh native arm with the existing clean split-native receipt. `repetition.json` records exact same-mode parity, serial-versus-native divergence and text repetition measures.
- `hybrid-native.trace-v2.jsonl` and `hybrid-serial.trace-v2.jsonl` hold complete identified safe typed events. `trace-index.json` pins hashes, event counts, readiness, chronology and terminal status for both arms.
- `last-query.patch` and the draft source/build artifacts contain the two-library overlay change. The typed model adapters and launcher templates bind the overlay to the pinned benchmark. The serial adapter template replaces its two owner receipt paths and records the original source hash. No private prompts, token IDs, output text, unpublished private output fingerprints or host filesystem paths are included.
- `export-hybrid.py` exports and revalidates the original owned receipts. `replay.py` independently checks the published trace hashes, event counts and reported results without private inputs or a device.
