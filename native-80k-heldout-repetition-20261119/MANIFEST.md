# Packet contents

| File | Contents |
|---|---|
| paired-scores.json | Both paired fixture scores, cross-mode and archived output equality, request clocks, effective sampler values, actual timing fields and expected-source repetition. |
| runtime.json | Image and mapped runtime identities, numeric configuration, device admission, context and expert geometry, physical witness snapshots and deltas, actual runtime epochs, bounds and cleanup. |
| original-serial-failure.json | First serial attempt, its completed 80K score, the witness-label failure, terminal status and the mapping correction. |
| model.py.in, release.sh.in | Exact revised source templates with owner-only alias placeholders. |
| rehydrate.py | Private alias substitution and byte-exact owner-seal verification. |
| export.py, replay.py | Owner receipt verification and typed export, plus public CPU evidence replay. |
| privacy-audit.json | Summary of redaction and the local payload audit. |
| README.md, REPRODUCE.md | Results, study scope and reproduction instructions. |
| SHA256SUMS | Checksums for the packet payloads. |
