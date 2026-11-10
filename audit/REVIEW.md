# Independent read-only source review

Astra review found no additional semantic defect in the isolated candidate. It confirms the two-row cache key `{ptr,2048,2,1,1,2048,4096,4096}` matches the MMVQ key construction (`mmvq.cu` 1743–1776); existing pending/commit and overlap invalidation paths are at `common.cuh` 1467–1473 and `ggml-cuda.cu` 3660–3670, 4728–4737. Candidate kernel and gate are in `norm.cu.candidate` around lines 582–600 and 900–916.

Architecture correction: M6000 is Maxwell sm52, not Pascal. The sm52 Q4_0 MMVQ maximum is 6 (`mmvq.cu` 152–178, 287–296), and ordinary MUL_MAT uses the general MMVQ route up to the compiled max batch (`mmvq.cu` 430 onward, call at `ggml-cuda.cu` 1955); therefore 2 rows are eligible. This establishes eligibility, not runtime dispatch or performance.

The review did not verify that the full copied source tree/object set exactly produced the deployed production library. The subsequent isolated build preflight checks that provenance before qualification. Full-model exact QA and clean server A/B are still required before adoption.
