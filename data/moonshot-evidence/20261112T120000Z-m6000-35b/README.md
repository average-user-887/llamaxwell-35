# M6000 35B moonshot evidence, session 20261112T120000Z-m6000-35b

Evidence-only package (branch evidence-20261112, not pushed at time of staging). Paths are relative to this directory.

## What ran
- E1/E4: nvprof + plain phase traces of GDN4 (arm A) and BUNDLE3 (arm B), one run per arm (raw/E4).
- E3: mixed-serving harness, arm A partial only; arm B NOT_RUN; dropped (Codex covered it).
- C1-C4: CPU-only reconciliation, quality (KL/top-1 from existing dumps), build provenance, lifecycle (raw/C).
- S1 (GEMM numerics read), Q8 capacity sizing (paper), E5/E8 scripts (unrun).
- Pending, to be added by a later commit: G1 GEMM qualification, E9, E5, E7, E2, E6.

## Key findings
- BUNDLE3 -8.98% deep 2-slot period is PROVISIONAL: one ABBA block (8 reps/arm).
- E4 ms/token A->B: N1 shallow -3.2%, N2 shallow -14.5%, N2 deep -9.3%, prefill +0.3% (indicative, single run).
- All three BUNDLE3 treatments (N2SPLIT, FA_ADDR, N2GDN) engaged (e1_engagement.json).
- VRAM: usable total 24,463 MiB, BUNDLE3 sampled peak 23,108, margin 1,355 MiB. 2x128K at F16 KV does not fit.
- N2SPLIT pair is 3.4x closer to solo than stock pair is (KL .00089 vs .00306).

## Limitations
Single-run traces, one ABBA block, no allocator high-water, no full BUNDLE3-vs-GDN4 2-slot logits, E3 arm B and E10 not run. Missing values are null with a reason.

## Artifact map
manifest.json, summary.json (status per item), runs.jsonl / requests.jsonl (E3 arm A, new_measurement), phases.csv / kernels.csv (E4), e1_engagement.json, memory.json, quality.json, source-manifest.json, checksums.sha256, PUBLICATION_REVIEW.md; raw/ (E4, C, S1, Q8 sources), historical/ (DAY.md, day-20261112 ledger and provenance), harness/, fixtures/fixtures_manifest.json, E5/, build/.

## Reproduce
- Reconcile claims: `python3 -I harness/c1_block_estimator.py` (needs original lab logs, not published).
- Quality: `python3 -I harness/c2_kld_dumps.py` (needs dumps, not published).
- Checksums: `sha256sum -c checksums.sha256` from this directory.

## Next recommendation (bounded)
Run E2 (>=5 ABBA blocks) for BUNDLE3 vs GDN4, then BUNDLE3 vs GDN4 2-slot logits on GPU (command in raw/C/C2.md).
