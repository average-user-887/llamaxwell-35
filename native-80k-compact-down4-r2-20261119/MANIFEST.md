# Packet contents

- `model.py`, `release.sh`, their source deltas and `prepare.py` pin the R2 launch and exact validation changes relative to R1. `typed-cli.json` and `profile.json` show the environment delta and exact binary identities.
- `results.json` carries the validated terminal receipts, physical counters, archived output equality, cache, acceptance, compact instrumentation and the timed comparison when present. `reconciliation.json` retains raw server numerical fields, client wall and SSE checkpoints, split timing, sampler snapshots, repetition counts and both token-rate conventions.
- `export-model.py`, `r2-evidence.py`, `replay.py` and `verify-reconciliation.py` revalidate and reconcile the portable evidence. `source-pins.json`, `owner-frozen.sha256`, `owner-inputs.sha256` and `SHA256SUMS` record the dependencies and payload hashes.
