# Card 19 RESULT: ctxfull 2x96K alloc/VRAM, S vs M

| run | rc | alloc/OOM lines | ctxfull | VRAM peak MiB | makespan s |
|---|---|---|---|---|---|
| M_2 | 0 | 0 | ran status=400 health_after=200 probe_slot1=200 | 23867.00 | 159.72 |
| M_3 | 0 | 0 | ran status=400 health_after=200 probe_slot1=200 | 23867.00 | 159.46 |
| S_1 | 0 | 0 | ran status=400 health_after=200 probe_slot1=200 | 23459.00 | 161.41 |
| S_4 | 0 | 0 | ran status=400 health_after=200 probe_slot1=200 | 23459.00 | 161.61 |

Verdict: **ALLOC-OK** (M alloc failures=0; VRAM peak max M=23867.0 S=23459.0 MiB; rule M<=S+1024)


## Method
Card 16 gpu_ab.sh pattern; n3_replay.py prod arm with --no-sweep (ctxfull stage ON in last/only block), order S,M,M,S. S: 7/bin cap1024 ub1024 GEMM+FUSED+PREFILL_SINGLE_STREAM. M: 9/bin cap1024 ub2048 + MOE_V3 + OCCAMY_DEBUG. MAXWELL_GRAPHS unset. Lock gpu-bench.lock; prod unloaded via POST.

## Caveats
VRAM peak is the 1 Hz nvidia-smi max for the whole run (N3_TABLE peak VRAM); alloc grep over server.log (OOM, cudaMalloc, failed to allocate, CUDA error). A ctxfull request rejection by the server is expected behaviour, not an alloc failure.

## Files
<LAB>/occamy35-night-20261114/19/run-20261115T025129Z, <LAB>/occamy35-night-20261114/19/gpu_ab.sh
