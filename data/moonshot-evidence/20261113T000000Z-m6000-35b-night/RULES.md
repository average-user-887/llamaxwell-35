# Night 12->13 Nov card rules (lead: Opus session; advisor ruling). Cut 08:00 CET.
Objective: end-to-end completion time of two agentic workflows near 98,304 tok each (tool-result appends), not tok/s.
- M=<LAB> ; E=$M/occamy35-day-20261112/evidence ; N=$M/occamy35-night-20261113 (your output dir N/<card>).
- Prod container llamacpp-m6000 (BUNDLE3, lib $M/prod-libs/libggml-cuda-bundle3-20261112.so, launcher $M/run-35b-bundle3-20261112.sh) is on call via llama-swap. Do NOT edit, rename, remove or adopt prod; no main pushes; GEMM off and cap 0 stay prod.
- GPU work: hold flock $M/gpu-bench.lock for the whole run (exec 9>$M/gpu-bench.lock; flock 9). One GPU job at a time. Before GPU use: POST <lab-endpoint>/api/models/unload/qwen36-35b-moe (NEVER GET unload/unload-all), then confirm zero compute apps on the M6000 (nvidia-smi -i <gpu>). Abort if busy.
- 250 W, no power/clock changes. Do NOT touch perf_event_paranoid. GGML_CUDA_MAXWELL_GRAPHS must be ABSENT from env (any value crashes BUNDLE3).
- Use your own uniquely named test containers (prefix n<card>-), docker rm -f them at the end. Never stop shared services (m6000-llama-swap, hermes, honcho, zabbix, llamacpp-m6000*).
- Never delete evidence or others' files; move to backup if needed. Large binaries (>50 MB) stay out of text evidence.
- No sleep/poll loops of more than the run itself; run scripts in foreground or background with a completion notice.
- Templates: two-slot server harness E/G1b (launch-arm.sh, g1b_run.py, run-g1b.sh, RUN.md); multi-row logits harness E/G1c (g1c_harness.py, run_all.sh, g1c_rows.cpp, RUN.md, TAIL.md); slot files $M/occamy35-day-20261112/rounds/D0/slots/{s0_98k_off0.bin,s1_90k_off200000.bin}; GEMM source audit E/S1/S1_GEMM_NUMERICS.md; C1 ABBA E/C.
- Output: N/<card>/RESULT.md with a table first, then method, caveats, file list. Keep return message <= 25 lines.
- Publication later by lead: no IPs, GPU UUIDs, hostnames, user paths, prompts in RESULT.md text you expect to be published (use <LAB>, <gpu>, <lab-endpoint>).

## GPU order (advisor, ~00:58 13 Nov)
Q8S -> N1 arm B -> Q8N exactness -> Q8N ABBA (two-slot deep, single 98K, 90K+4K append) -> N2 -> N3 (arms BUNDLE3, BUNDLE3+cap1024, Q8N if passed, GDN4; GEMM only if time) -> N4 -> N5.
- Q8S passes only K or only V: same order, that instance only.
- Q8S fails all: skip Q8N GPU gates; N2 -> N3; report to owner.
- Before Q8N ABBA: explicit q8 slot-restore test (restore, decode 16, compare vs no-restore). Re-prefill => in-process prefill fixtures, log defect, not blocker.

## Revised order (advisor ~01:10): N1 B -> N2 -> Q8N -> N3 -> push. N4, N5 tomorrow.
- Q8S strict FAIL all arms (not adoptable). Q8N measured as measurement-only.
- N2: Ap long (G1c docs + Q8S docs 0-3) + shallow Ap + shallow A-repeat; paired Ap/GEMM/q8KV/q8K vs A.
- Q8N cap 75 min hard: exactness <1,1>,<1,0> at 32K/98K, 4 docs; slot restore once; two-slot deep ABBA x3 K+V and K-only vs BUNDLE3 F16; peak VRAM; nvprof 1 step each. Single-98K/append only inside cap.
- N3: short profile; arms BUNDLE3, +cap1024, GDN4, 5 blocks; faster Q8N as 2-block arm only if ABBA CI excludes 0. Past 08:00 => 4 blocks, not fewer arms.
- Report: N1 FA tile 78% of deep append => native q8/packed tile read is phase-2 prefill lever; NC1 => objective rests on intended, not observed, workload.
