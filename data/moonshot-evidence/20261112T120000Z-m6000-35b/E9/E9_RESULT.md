# E9 result: log verbosity cost and decode stack sampling (measurement only)

Setup: BUNDLE3 prod env + prod CUDA lib (ro), G1b bundle3 server build, single slot (-np 1, -c 98304 = prod per-slot context), ~98.1K-token context restored from a slot file, 64 tokens/unit, greedy, cache-ram 0. A = default verbosity (3), B = -lv 4 (prod setting). ABBA x 2 cycles = 5 containers (adjacent same-arm blocks share a container); 8 units per block after one unlogged warm unit. 64 units, identical output hash in both arms.

## Part 1 (timed)
| metric | A (default) | B (-lv 4) | A-B |
|---|---|---|---|
| decode ms/token | 19.025 (sd 0.029) | 19.073 (sd 0.028) | -0.048 (-0.25%) |
| unit wall s (restore + 4-token tail prefill + 64 tok) | 1.895 (sd 0.011) | 1.900 (sd 0.011) | -0.005 (-0.26%) |
| generate-call wall s | 1.348 | 1.353 | -0.005 (-0.38%) |

Per-cycle A-B ms/token: -0.037, -0.059 (same sign both cycles). Per-block means in e9_part1_stats.txt. Verdict: -lv 4 costs about 0.05 ms/token (0.25%), i.e. ~1 ms per 64-token unit; below the 1% gate.

## Part 2 (untimed): gdb sampling, not perf
100 samples of 'thread apply all bt' on a default-verbosity server decoding at ~98K (restore + 240-token decodes in a loop; 5 requests completed during sampling). Main thread = decode thread. 80 samples caught the decode path, 20 caught restore/idle (excluded below).
Top 5 by inclusive presence, main thread, decode-phase samples (n=80):
1. server_context_impl::update_slots / server_context_impl::decode: 100%
2. llama_context::synchronize -> ggml_backend_sched_synchronize: 97.5%
3. ggml_backend_cuda_synchronize: 97.5%
4. cudaStreamSynchronize: 97.5%
5. cuStreamSynchronize (driver wait): 97.5%
Finding: the host spends ~97.5% of decode time blocked waiting on the GPU in cudaStreamSynchronize. Host graph build/set_inputs (llm_graph_result::set_inputs, kv mask) 2.5%, graph launch 1.2%, sampling 0 samples. No other thread had ggml/llama/cuda frames. Host-side overhead (including logging) is not the limiter; decode is GPU-bound.

## Caveats
1. gdb pauses the process; the sample is a stack snapshot, not a time profile. A sync-bound host looks identical whether GPU kernels or GPU idle gaps cause the wait.
2. Only ~250 free tokens above the 98K context, so decode ran as repeated 240-token requests; restore/idle samples were excluded from percentages.
3. Old 2-stream slot files cannot load into -np 1 ("n_stream mismatch"); a ~98.1K context was prefilled once and saved as an -np 1 file (e9_np1_98k.bin in slots/), restored identically in both arms. Context is 98.1K, not 96K.
4. Server build is the G1b bundle3 build (prod-image server binary cannot read the existing slot files); CUDA lib is the prod bundle3 lib. -lv 4 effect measured on that build; difference is 2 sd-sized steps over 64 units, one machine, 2 cycles.
5. Frames below main() are mis-symbolized (stripped libs); only named llama/ggml/cuda frames were used.
