# Packet contents

- `pair-results.json` carries the private-owner-validated native and serial timed arms. It includes the mapped build, exact same-mode comparisons, physical counter snapshots, guard counts, compact bytes, draft acceptance, repetition and strict same-build comparison.
- `reconciliation.json` preserves the raw numerical server timing fields, client request boundaries, token-bearing warm checkpoint splits, the raw and delivered rate conventions, sampler snapshot counts, native round arithmetic and cross-mode divergence ordinals.
- `pair-evidence.py` independently replays the original owner comparison, exports the allowlisted fields and checks portable arithmetic. The exact R1 code and shadow case are linked from `REPRODUCE.md`.
