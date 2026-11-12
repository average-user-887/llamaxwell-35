# G1b runner spec: mixed-batch cap 0 vs 1024 on BUNDLE3 (GPU, ~70 min)
Placeholders: <LAB>=<LAB>, G=<LAB>/occamy35-day-20261112/evidence/G1b. Cut 21:00 CET.

## Arms (only difference: launch flag)
- A = `--mixed-batch-size 0` (off, identical to prod batching); B = `--mixed-batch-size 1024`.
- Both use the same new build (`G/build/bin`, source G/src = 2436f75 + cherry-pick 57076fe of 4041db9, server-only), prod BUNDLE3 libggml-cuda overlaid, prod image/args/GGML_* env (see launch-arm.sh). `--cache-ram 0`.
- Maxwell GEMM/GRAPHS not used: GGML_CUDA_MAXWELL_GRAPHS and GGML_CUDA_FA_MAXWELL_GEMM must be UNSET (even value 0 enables graphs). launch-arm.sh aborts otherwise.

## Preconditions (run-g1b.sh enforces 1-4)
1. `flock -n` on <LAB>/gpu-bench.lock held for the whole run.
2. POST `$G1B_UNLOAD_URL` (.../api/models/unload/qwen36-35b-moe). Never GET unload-all.
3. `nvidia-smi -i $G1B_GPU --query-compute-apps` must be empty, else abort (re-checked before each server launch).
4. Never touch `llamacpp-m6000*`; own containers only `g1b-*`; no power/clock changes.
5. build.rc == 0 and G/SHA256SUMS present (build prep); smoke first.

## Commands
    export G1B_GPU=<GPU> G1B_UNLOAD_URL=<lab-endpoint>/api/models/unload/qwen36-35b-moe
    G/run-g1b.sh --smoke            # 1 server, 1 unit a+b, ~6 min; check run-*/units.jsonl valid:true
    G/run-g1b.sh                    # full; ORDER env overrides schedule
Cleanup (automatic at end): `docker rm -f` only g1b-* containers; lock released on exit. If aborted, run `docker ps -a | grep ^g1b-` and remove only those.

## Schedule
5 counterbalanced cycles, ORDER=ABBA,BAAB,ABBA,BAAB,ABBA (A=cap0, B=cap1024), 20 units; adjacent same-arm units share one server session (~13 launches; flag is launch-time). Per session: warm-up request, then per unit workload a then b. First unit of each arm also gets one non-timed `stream:false` identity pass per workload (exact IDs).

## Workloads (driver g1b_run.py, own code; Codex files not edited)
- a (Codex shape): restore `s1_90k_off200000.bin` (90000 tok) into slots 0 and 1; bg = fixture_tokens[200000:290000]+[198], n_predict 512, slot 0; after >=8 bg events, fg = fixture_tokens[200000:294096] (+4096 new), n_predict 32, slot 1. Valid if fg prompt_n >= 4090, not truncated.
- b (e3 shape): restore `d0.bin`, erase slot 1; bg = d0[:98047]+[198] n_predict 256; after >=8 events, 16K uncached segment uNN (rotating 16 segments), n_predict 32, slot 1.
- All requests: temperature 0, ignore_eos, cache_prompt, `return_tokens:true`.

## Metrics (per unit, units.jsonl)
fg TTFT (ttft_s), max visible bg gap from arrival to first fg token (max_gap_s), wall_s, fg prefill tok/s (timings.prompt_per_second), peak VRAM (nvidia-smi 0.5 s, vram_peak_mib; prod BUNDLE3 ref ~23108 MiB), telemetry.csv 1 Hz (clocks, power). Report per-arm mean/min/max over units and paired (same cycle) deltas; gate: TTFT and wall within ~1%, gap down, VRAM not above ref, no invalid units.

## Output-count receipt (the 510/512 issue)
Cause: server withholds text of tokens that end mid UTF-8 char (server-context.cpp process_token, validate_utf8 ~1983, send block skipped); bytes are merged into the next event's content but only the later token's ID is emitted (send_partial_response tokens.assign(1,...) ~2199), and the final stream chunk carries no tokens (~2240). So streamed ID count < n when output has multi-byte chars built from byte-fallback tokens. Content dependent; the cap does not touch process_token.
Receipt rules:
1. Authoritative count = final event tokens_predicted (== timings.predicted_n) == n_predict; tokens_cached == prompt_n_total + n - 1.
2. Visible IDs are a lower bound: id_gap = tokens_predicted - len(ids) must be explained: gap <= number of events whose content has non-ASCII chars (nonascii_events); else FAIL.
3. Output identity across arms: content_sha256 + tokens_predicted of the same-position unit; greedy text may legitimately differ between repeats, so also compare the repeat-to-repeat variation within one arm before reading an arm difference.
4. Exact IDs: identity pass (stream:false, returns full tokens) per arm/workload; compare ID hash there.
Do not patch the server to flush IDs (would change the tested build).

## Abort conditions
lock busy; compute apps on GPU; unload POST failure; server not ready in 600 s; bg not decoding at arrival; OOM/CUDA error in server log; a unit invalid twice; time past cut.
