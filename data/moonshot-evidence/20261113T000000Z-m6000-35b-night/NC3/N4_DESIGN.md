# N4: GEMM-prefill FA numerics bisect at fixed activations (design, CPU only)
Card NC3, 13 Nov 2026. Nothing here touched the GPU. M=<LAB> ; E=$M/occamy35-day-20261112/evidence ; NC3=$M/occamy35-night-20261113/NC3.

## 1. What is being tested (source located)
- GEMM source: Codex v4, `$M/codex-investigation-20261112/prefill/gemm/integration-v4/src/ggml/src/ggml-cuda/fattn-maxwell-gemm.cuh` (279 lines, sha256 7036c3c8...), dispatched first in `ggml_cuda_flash_attn_ext` (fattn.cu:788). Patch: `integration-v4/integration.patch`. Built lib (the "cand" lib used by G1/G1b/G1c arm B and, by file, arm A too): `integration-v4/src/build/bin/libggml-cuda.so` (sha256 0c7ffbde...; `g1c_harness.py:11` LIB_V4). Arms differ ONLY by env `GGML_CUDA_FA_MAXWELL_GEMM=1` (B) vs absent (A); same lib, same BASE_ENV/FA_ENV. So "stock" = the tile kernel (`fattn-tile.cuh`, sm52 FP32 math) in that lib.
- Engagement (header lines 154-191): cc 5.2, Q f32/K,V f16, D=256, 16 q heads, 2 kv heads, 64 <= nq <= 1024, 4096 <= n_kv <= 98304, F16 mask, scale finite, no bias/softcap/sinks.
- KEY SCOPING FACT from E/G1c/TAIL.md + S1: G1c scored rows are Q=1 teacher-forced decodes (below the 64-query guard), so BOTH arms run the stock/Occamy decode kernel on every scored row. GEMM only runs in the prefill ubatches. Any B-vs-A logit gap is therefore propagated prefill drift (KV + GDN state), not a decode-kernel difference. N4 tests the prefill operator directly, in situ activations, which G1c cannot do.

## 2. Dump method (arm A, stock, one pass)
Tool: `$NC3/n4_dump.cpp` (syntax-checked with g++ against G1b/src headers; not linked/run). Built from g1c_rows.cpp scheduling, adds `cparams.cb_eval` (the `ggml_backend_sched_eval_callback`, same hook `examples/eval-callback` and `common_debug_cb_eval` use; llama.h:393).
- Hook point: graph node op `GGML_OP_FLASH_ATTN_EXT`, name `kqv_out-<il>` (llama-graph.cpp:3000/3108/3199 `cb(cur,"kqv_out",il)`; cb() appends `-il`). Qwen3.5-MoE: full attention at il = 3,7,...,39 (qwen35moe.cpp:24, interval 4) -> 10 attention layers of 40. Default dump layers 3 (shallow), 19 (mid), 39 (last).
- Callback protocol: ask phase returns true ONLY for armed FA nodes of the chosen layers (everything else false, so the graph stays batched, no per-node sync); observe phase reads tensors with `ggml_backend_tensor_get`.
- Tensors taken from the node itself (they are exactly what the kernel receives): `src[0]` Q f32 permuted view {256,nq,16}, `src[1]` K f16 cache view {256,n_kv,2}, `src[2]` V f16 cache view (not transposed with FA), `src[3]` mask f16 {n_kv, nq_pad} (shared by all layers, dumped once per stop), `op_params` scale/max_bias/softcap + op_params[4] (n_kv_max), and the node output `t` (stock O, in-situ reference for the replay: must match the replay bit-for-bit, otherwise the replay is not the same computation).
- Armed schedule (n_ubatch 1024, n_batch 2048, FA on, KV f16/f16, cache layout as G1c arm A): prefill the same token file as G1c (`$E/slots`/doc token files; any real 98K doc), chunks ending at each stop are <=1024 tokens = exactly one ubatch, only that ubatch is dumped. Stops (`stop:Q`, shape n_kv = PAD256(stop) per TAIL.md):

| stop:Q | n_kv | n_kv mod 4096 | role |
|---|---|---|---|
| 4096:1024 | 4096 | 0 | 4096 boundary, first engaged size, only batched-PV path (full_splits=1, tail 0) |
| 4160:64 | 4352 | 256 | 4096 boundary +1 tile, min Q=64, tail-SGEMM path (S1 operator tests covered only this shape) |
| 32768:1024 | 32768 | 0 | 32K, tail 0 (G1 16K/65K outliers had tail 0) |
| 33024:256 | 33024 | 256 | 32K, tail 256 (G1 row 1-15 shape) |
| 98048:1024 | 98048 | 3840 | 98K, long tail (S1 operator shape K=98048) |
| 98304:256 | 98304 | 0 | 98K cap, tail 0, split_count 24 = capacity |
Negative control (optional extra stop 3840:1024 -> n_kv 3840): below guard, both arms stock, outputs must be bit-identical; confirms the harness.
- Run command (arm-A env exactly as `g1c_harness.py` BASE_ENV+FA_ENV, no GEMM flag, no MAXWELL_GRAPHS):
  `n4_dump <model> $NC3/dump <n_ctx 98500> <doc tokens.i32> 3,19,39 4096:1024,4160:64,32768:1024,33024:256,98048:1024,98304:256`
  Note the stops are consecutive in one prefill (pos only increases), so one 98K prefill yields all dumps.
- Output layout `dump/stop<S>/`: `meta.json` {stop,nq,nq_pad,nk,scale,...}, `mask.f16` [nq_pad,nk], `L<il>_Q.f32` [nq,16,256], `L<il>_K.f16` and `L<il>_V.f16` [2,nk,256] compact, `L<il>_Ostock_insitu.f32` [nq,16,256].
- Sizes (3 layers): 4096 134 MB, 4160 34 MB, 32768 369 MB, 33024 245 MB, 98048 904 MB, 98304 679 MB; total ~2.4 GB (largest per-layer 234 MB, mask 201 MB). Disk free on Storage 76 GB; large binaries stay out of text evidence (path-only in RESULT.md).

## 3. Feeding identical tensors to GEMM and stock
Tool: `$NC3/n4_drive.cu` (COMPILED AND LINKED on CPU, `$NC3/build/n4_drive`, 1.1 MB; not executed). It reads one stop/layer, builds `ggml_tensor` descriptors with the graph's layouts (as Codex `integrated-attention-probe.cu` does), and calls the real `ggml_cuda_flash_attn_ext` of the linked lib. The GEMM flag is read once per process, so each arm is its own process:
- stock: `n4_drive dump/stop98048 19 out_stock` ; GEMM: `GGML_CUDA_FA_MAXWELL_GEMM=1 n4_drive dump/stop98048 19 out_gemm`.
- Stage mode (GEMM arm only): `... n4_drive dump/stop98048 19 out_gemm stages 0,1,511,1022,1023`. Replays the v4 pipeline using the header's own static kernels (pack, convert, SGEMM, block reduce, PV pointers, reduce) with a softmax copy that also emits (m,l); copies S (post-SGEMM, scale in alpha), P, (m,l) for the chosen rows; asserts the replayed final O is bit-equal to the library O (`replica_bit_equal_to_library=1`, rc 6 otherwise). Output O is NaN-prefilled so unwritten rows show up.
- Stock has no exposed intermediates (tile kernel keeps them in registers). Stock is compared at O only; stage-level stock behaviour is covered by the numpy FP32 proxy `emu_stock_f32` (tile order, FATTN_KQ_MAX_OFFSET) as a labelled proxy. If the O-level result points at stock being the outlier, an instrumented copy of `fattn-tile.cuh` writing KQ_max/KQ_sum would be the next (Opus) step; not needed up front.
- Build (CPU container, no GPU, same as integration-v4/build.sh probe step; ~5 s):
  ```
  V4=$M/codex-investigation-20261112/prefill/gemm/integration-v4
  docker run --rm --name nnc3-build --network none --cpuset-cpus 6-8 --cpus 2 --memory 6g --memory-swap 6g --user $(id -u):$(id -g) -e TMPDIR=/tmp \
    -v $V4/src:/src:ro -v $NC3:/w -w /w local/m6000-build-tools:20261013 sh -c '/usr/local/cuda/bin/nvcc -O3 -std=c++17 -arch=sm_52 -use_fast_math -extended-lambda -DGGML_CUDA_USE_GRAPHS -I/src -I/src/ggml/src -I/src/ggml/include -L/src/build/bin -lggml-cuda -lggml-base -lcublas -Xlinker=--allow-shlib-undefined /w/n4_drive.cu -o /w/build/n4_drive'
  ```
  `n4_dump` needs the full llama.cpp CMake tree: add it as `tools/n4-dump/` (copy of eval-callback CMakeLists, one target) in a copy of `$E/G1b/src` and build `llama` + this target in `local/m6000-build-tools` as `$E/G1b/build-g1b.sh` does; or run `g1c_rows`-style inside the existing arm-A container after compiling against the already built `$E/G1b/build` libs (`g++ -std=c++17 n4_dump.cpp -I src/include -I src/ggml/include -L build/bin -lllama -lggml -lggml-base`). Runtime lib path for the driver: `LD_LIBRARY_PATH=<v4 build/bin>` (v4 lib).

## 4. FP64 reference and metrics
`$NC3/ref_fa64.py` (numpy 1.26, `python3 -I ref_fa64.py selftest` -> PASS, 0.1 s; checks vs an independent scalar-loop implementation, softmax sums, fully-masked row = zeros, padding-key invariance, ULP helper, FP32 proxies within tolerance incl. multi-split + tail).
- Stages per (query,row,head): S=scale*QK^T, S2=S+mask, m, l, P=exp(S2-m)/l, O=P@V; all FP64 from the f16/f32 dump values (no rounding of inputs: K,V are the exact f16 values both kernels read).
- Metrics per stage (`stage_metrics`): max abs, max rel (floor 1e-5), RMSE, ULP-ish = |x-ref| / spacing(float32(ref)) with max, p50, p90, p99, p99.9, bit-equal fraction (to float32(ref)), non-finite count.
- Error distance (`distance_vs_ref`) on O: RMSE gemm and stock vs FP64, ratio, fraction of elements where GEMM is closer, max abs, ULP p99 each, plus RMSE gemm-vs-stock. S1's synthetic result to compare against: RMSE 7.2e-9 (GEMM) vs 2.8e-8 (stock).
- Stages available from the GEMM replay: S, S2 (host f32 add; ref computes the same in FP64), m, l, P, O. Stock: O only (+ proxy).
- Row sampling: all 16 heads x rows {0, 1, nq/2, nq-2, nq-1} plus, for tail checks, the rows whose causal limit lies in the last 256-key tile and the last 4096-split. CPU cost ~30 ms per (row,head) at 98K: ~3 s per layer/stop, ~2 min for the whole matrix.
- Run: `python3 -I ref_fa64.py compare --dir dump/stop98048 --layer 19 --gemm out_gemm/L19_O.f32 --stock out_stock/L19_O.f32 --stages out_gemm` (JSON out). Checks to add in the wrapper: stock driver O == `L*_Ostock_insitu.f32` bit-equal (validates the replay harness), key-count sweep summary by n_kv mod 4096.
- Decision rules (set before running): defect candidate if (a) any non-finite or GEMM O error > 4x stock RMSE at any stop, (b) error jumps only at tail=0 stops (4096, 32768, 98304) vs tail!=0 stops, (c) S or P stage error beyond FP32 dot bound (~sqrt(256)*6e-8*|q||k|*scale relative) or P row sum != 1 beyond 1e-5, (d) rows near the PAD boundary (last tile, padded keys) differ from interior rows. Otherwise GEMM is closer or equal to FP64 and the G1/G1c outliers are propagated sensitivity.

## 5. VRAM / time estimates (all from existing numbers, none measured)
- Dump run (GPU): same container as G1c arm A (model + KV for ~98.5K single seq, i.e. the already proven arm-A footprint) plus host-side buffers only; callback reads via ~0.4M small `tensor_get` calls per 98K layer (est. 20-60 s each x 3 layers x 2 big stops) -> prefill of one 98K doc ~6.5 min (G1 A) + ~1 min load + ~3 min dump = ~11 min exclusive M6000 (flock, unload first per RULES). Per-FA-layer sync cuts graph batching on the 3 chosen layers only.
- Driver (GPU): per call VRAM ~ inputs 0.42 GB (98K) + out 17 MB + GEMM slab 412 MiB (432 MB) + replay buffers ~0.45 GB (kf,vf 200 MB, scores 201 MB) = ~1.3-1.5 GB; needs the model unloaded or not, it does not need the model at all. Runtime ~1 s init + <1 s kernels + 0.2 s upload + stage copies (~5 rows x 16 heads x 400 KB, negligible). 6 stops x 3 layers x 3 invocations (stock, gemm, gemm+stages) = 54 invocations ~ 3-4 min, one flock hold. Total GPU ~15 min.
- CPU: ref ~2 min, can run while the GPU phase is going.

## 6. What needs what
| piece | GPU? | status |
|---|---|---|
| ref_fa64.py + selftest | no | done, PASS |
| n4_drive.cu build | no (CPU container) | done, binary built |
| n4_dump.cpp build (cmake tree) | no | syntax-checked only, needs the CMake target |
| arm-A dump pass (n4_dump) | YES, ~11 min flock | not run |
| driver runs stock/gemm/stages | YES, ~4 min flock | not run |
| compare / metrics / table | no | script ready, wrapper for the sweep table to write after dump exists |
| Instrumented stock tile kernel (KQ_max/KQ_sum out) | build CPU, run GPU | only if O-level points at stock |
| Kernel fix | Opus (CUDA) | ONLY if a rule in section 4 fires; candidates to inspect: tail=0 branch (full_splits only), padded-key rows (GEMM never skips masked tiles), cuBLAS SgemmBatched pointer/ld for split 24 at 98304 |
No GPU test is needed to claim anything about the code beyond S1; the nonzero-risk item is only that `n4_dump`'s observe-phase reads assume the scheduler keeps FA inputs valid at the callback (standard for eval-callback; verified by the in-situ O vs driver stock O bit-equality check).

## 7. Caveats
- Dump covers 3 of 10 attention layers and 6 stops of one doc; doc/layer generality is a sample, not a proof.
- cuBLAS summation order is unknown, so the numpy GEMM emulation is structural only; GPU stage dumps are the truth.
- Stock uses an exp offset (+2.08) and per-tile rescale; FP64 reference has neither, so the stock max stage cannot be compared (only O).
- The harness uses the v4 lib, whose base commit 2436f75 is BUNDLE3; if the lead wants the dump on the exact prod-libs bundle3 binary, the driver must be linked against that .so, which lacks the GEMM symbol (no GEMM there by design).
