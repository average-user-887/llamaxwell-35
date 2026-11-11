# superseded: runs completed, see profile.md and analysis files
# R1 status (11 Nov, runner)
Tasks 1 (G4 ABBA) and 2 (profile) NOT run: GPU-684f... holds 22.9 GB from docker `llamacpp-m6000` (prod, direct container,
not swap-managed; llama-swap /running is empty so no per-model unload exists). Harness aborts on compute apps; stopping prod is
forbidden by run rules. Ready to run once the GPU is free:
  flock /media/avg-usr/Storage/m6000-llama-cpp/gpu-bench.lock bash g4.sh cfg-r1h2h.json warm_decode   (then d4096)
  prof: run.sh/an.py (nvprof in cuda:12.8.1-devel container, bench -fa on -n 6 -d 0/16384; edit paths/lib/env first).
Arm A = prod-libs call35 (sha c3d0ecd0...), arm B = C24 6f239c9b (sha 8a2a37d4...); both with DOWNX+QKREUSE+PCSHARE+DOWNH=2 in common env.
