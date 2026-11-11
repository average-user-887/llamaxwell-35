# T1 candidate gates (M6000, lock held throughout, llamacpp-m6000 left stopped)

Base env = C04 arms env incl. GGML_CUDA_MOE_DOWNX=1. Arm A = same lib, flag unset; arm B = flag set. Libs: C10 80e1dc3f1a8f.., C11 45c9da630c57.., C07 305b6bb36010...

| Candidate | G3 | G1 (64 steps logits + state @2112 tok) | G4 warm_decode (5 cycles+warmup, 22/22 valid) |
|---|---|---|---|
| C10 QKREUSE | PASS (grid 48, qk_reuse 1, launches 1) | PASS bit-equal | +2.44% (+1.947 tps), CI95 [2.43,2.45]%, noise floor 0.0083 tps; clears 1% |
| C11 PCSHARE | PASS (pc_share 1, launches 1) | PASS 3x bit-equal + loaded run bit-equal (A repeatable) | +1.47% (+1.172 tps), CI95 [1.43,1.51]%, noise floor 0.030 tps; clears 1% |
| C07 DOWNH=2 | PASS (variant grid 48, launches 1) | PASS bit-equal | +1.26% (+1.005 tps), CI95 [1.23,1.28]%, floor 0.019 tps; clears 1% |
| C07 DOWNH=1 | PASS | PASS bit-equal | +0.36% (+0.291 tps), CI95 [0.34,0.39]%; below 1% |

Notes:
- State compare is byte compare of the llama_state_seq blob (steps 64, 2112): diffbytes=0 everywhere. Logits via compare_report.py (bit_mismatch=0, no NaN/Inf).
- C11 loaded run: first attempt failed (load container tried a 19 GB model alloc, OOM); retry with a small spin kernel as concurrent GPU load passed.
- G4 results are correctness-qualified only because G1 passed for each.
- C07 launch counters print 1 (only first-launch line); UB counter in debug code not relied upon.
- Raw: raw/ (cfg-*, g4-*.analysis.*, *.cmp.txt, dumps/ 3.4G, 10 root-owned files from first docker runs).
