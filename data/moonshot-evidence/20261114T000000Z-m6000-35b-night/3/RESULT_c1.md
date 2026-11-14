# Card 3d: mixed-batch-size sweep (short profile, 2 slots, 3 alternating blocks base/A/B)

| metric (mean of 3; per-run in brackets) | base: mbs1024 ub1024 | A: mbs512 ub1024 | B: mbs2048 ub2048 |
|---|---|---|---|
| makespan total s | 161.32 (161.29/161.34/161.34) | 165.23 (165.32/165.20/165.16) | 158.65 (158.54/158.48/158.92) |
| slot0 (agent0) completion s | 155.42 | 159.27 | 152.79 |
| slot1 (agent1) completion s | 136.32 | 140.23 | 133.65 |
| TTFT mean s | 17.88 | 18.36 | 16.73 |
| max token gap s | 5.60 | 3.03 | 10.78 |
| peak VRAM MiB | 23459 | 23459 | 23867 |
| vs base makespan | - | +3.91 s (+2.4%) | -2.67 s (-1.65%) |

Reading: B is faster in every block (makespan -1.65%, TTFT -6.4%) but max token gap roughly doubles (10.8 s vs 5.6 s) and VRAM +408 MiB. A is slower (+2.4%) but has the smallest gap (3.0 s). Spread within arm <=0.45 s, so all differences are well outside noise. Gap/makespan trade-off: mbs is a latency-smoothing knob. Note B changes both mbs and ub (as ordered), so its gain is partly the ub2048 effect seen in 3c (-0.58%); mbs2048 itself adds roughly -1.1%.

Method: 3c harness copy (container prefix n3d-, only change: N3_CAP env overrides the --mixed-batch-size value), stack bin $N2/7/bin, GEMM=1 FUSED=1 LLAMA_PREFILL_SINGLE_STREAM=1, short profile, no sweep/ctxfull, order base,A,B x3. flock gpu-bench.lock (-w 14400), unload by POST to qwen36-35b-moe, MAXWELL_GRAPHS unset, GPU idx 1 from nvidia-smi. No allocation failures. Containers n3d-* cleaned.
Caveats: unload endpoint path /api/models/unload/qwen36-35b-moe on the M6000 swap was inferred (run proceeded, GPU free). Mixed-batch flag semantics: caps total batch when adding splittable prompts to generation; limited by -b/-ub, so mbs2048 needs ub2048 to take effect.
Files: $N2/3/c1/run-20261114T145046Z (steps.log, {base,A,B}_N/analyze.log, run.log), $N2/3/c1/card3d_gpu.sh, $N2/3/c1/h
