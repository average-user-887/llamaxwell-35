# A2 moe_v3 shexp idle-wave: GPU results (run <W>/A2/run-20261116T014007Z)

| item | value | accept | verdict |
|---|---|---|---|
| microbench op speedup | 0.9548x, 0.9554x (2 runs) | >= 1.05x | FAIL |
| microbench bitwise diff base vs shidle | 0 / 131072 floats both runs | 0 | PASS |
| server CHECK (SHIDLE vs v2 / vs v3) | 64 lines, 0 nonzero | 0 / 0 | PASS |
| N3 short ABBA x1 (makespan total s) | A 159.10, 158.96 (mean 159.03); B 159.03, 159.22 (mean 159.13); delta +0.06% (noise) | no loss | NEUTRAL, no gain |
| N3 common-window decode tok/s | A 19.49/19.51; B 19.51/19.49 | - | no change |
| engagement counter (OCCAMY_DEBUG) | yes: "MOE_V3_SHIDLE launches 65536 grid 48 nsa 508" in B arms, absent in A | seen | PASS |
| overall | SHIDLE is bit-exact and engaged but ~4.5% slower in microbench; no server gain | | DO NOT ADOPT |

Method: A2/a2_gpu.sh under gpu-bench.lock (POST unload only), microbench x2 then N3 short ABBA x1 (A,B,B,A), B adds GGML_CUDA_MOE_V3_SHIDLE=1 (first B also CHECK=64). Caveat: single ABBA, n=1 per slot, so delta is within noise. Files: <W>/A2/run-20261116T014007Z/steps.log, mb/run*.log, check.log, {A_1,B_2,B_3,A_4}/analyze.log.

---
# A2 moe_v3 shexp gate/up in the phase-A idle wave (build done, GPU pending)

| item | value |
|---|---|
| branch | a2-moe-shidle (worktree A2/src, off bundle9 52c9e84) |
| commit | e40da12 |
| flag | GGML_CUDA_MOE_V3_SHIDLE=1 (default off; needs GGML_CUDA_MOE_V3=1, uses LOP3 path moe_v3<true,true>) |
| counter | OCCAMY_DEBUG: "MOE_V3_SHIDLE variant K blocks/SM -> grid G", "MOE_V3_SHIDLE launches N grid G nsa K" (N = powers of 2) |
| CHECK | GGML_CUDA_MOE_V3_CHECK=N with SHIDLE: "shidle vs v2 bitwise-diff a / 2048 ...; vs v3 bitwise-diff b / 2048 ..." (expect 0 / 0) |
| libggml-cuda.so sha256 | 44de8cf166a3304211b0d88828b6b972ca4c31e40d2d3cf25804e8c9622537c7 |
| llama-server sha256 | 592908f59ff8496ed5800493cbe1ca175a5d7aa447ce84abff632874196813c5 (unchanged vs R0) |
| ptxas | moe_v3<true,true>: 64 regs, 0 spill, 0 stack (same as base); occupancy query in dev_state |
| base path identity | moe_v3<true,false> SASS md5 b600f46cd3ee = R0 bundle9 moe_v3<true> (prod path byte-identical) |
| microbench | tools/moe3_mb (SASS of both kernels md5-identical to the lib object) - PENDING GPU |
| op speedup (gate >= 1.05x) | PENDING: a2_gpu.sh step 1 ("TIMING ... speedup X") |
| bit-equal vs moe_v3 | PENDING: microbench "BITWISE ... diff 0" (4 layers x 8 h x 2 passes) and server CHECK (step 2) |
| N3 short ABBA x1 | PENDING: step 2 (A = A2 libs bundle9 env, B = + SHIDLE) |

Run (orchestrator; takes gpu-bench.lock with flock -w 43200, POST unload only, ~20 min hold):
`N3_GPU=<id> N3_UNLOAD_URL=<.../unload/qwen36-35b-moe> bash A2/a2_gpu.sh` -> A2/LATEST, steps.log has the
microbench BITWISE/speedup lines, CHECK summary and counters.

## What changed
Phase A has NTA = 1028 router tasks over W = 48 x 16 = 768 warps, so the second wave leaves 508 warp slots idle.
With SHIDLE those slots compute shexp gate/up rows 0..507 (gate_sh row prefetched during the router loop, same
moe3_gu_row dot/silu code as phase B), lane 0 writes h_sh[j]. Phase B starts at task nsa = min(FFS, ceil(NTA/W)W - NTA)
and deals the rest warp-major; base B loop left untouched in the else branch.

## Risks
- Gain ceiling: phase B bytes drop by 508 x 2 x 1152 B ~ 1.17 MB of ~18 MB per launch (~6.5% upper bound, only if phase A hides it); the
  1.05x gate may not be reached if phase A becomes bandwidth-bound with the extra loads.
- Phase B loses its pre-barrier prefetch for routed tasks (first B tasks need sel after the barrier).
- Bit-equality relies on identical FMA contraction at the new call site; SASS differs (more FFMA, expected from
  the extra row code) so only the GPU bitwise checks settle it.
- nsa depends on grid; a grid other than 48 changes the split (still exact, just different).

## B5 notes
moe3_gu_row is row-generic (stg/lane/cv/x pointers, prefetch lambda), so an n-row persistent moe_v3 can loop rows
inside one launch with per-row Scratch slices; the SHA idle-wave split generalises to n*NTA router tasks.
Follow-up idea (out of scope): shexp down partials could also move into phase B.
