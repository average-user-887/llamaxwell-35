# E8 GPU run: k=1,2,3 decode-call timing, one sequence at ~96-98K depth, under nvprof (NOT yet run). ~5 min GPU.
Tool: llama-k-timing (E5/build/bin). Slot file: $M/occamy35-day-20261112/rounds/D0/slots/s0_98k_off0.bin
 (saved by llama-server via llama_state_seq_save_file; same loader. Needs same model/kv type/n_rs layout; if load fails the tool prints "slot load ... FAILED"
 and falls back to prefilling --prefill-file d0.txt --len 98048 and prints the PREFILL time (est 6-9 min at ~250 t/s: then drop nvprof runs to k=1 only or accept ~15 min).)
Restore after each call: llama_memory_seq_rm(seq,P,-1); if it returns false (rollback > n_rs_seq=2, expected for k=3) the tool counts rm_fail and
 reloads the slot file (or the host snapshot in prefill mode) -- reload is outside the timed region.
Timed region = llama_process + llama_synchronize. Prints per-call ms, SUMMARY per k, V(2)/T and V(3)/T = wall(k)/wall(1), and device VRAM min-free/peak.

```bash
M=<LAB>; E=$M/occamy35-day-20261112/evidence/E5
GPU=<gpu>
SLOT=$M/occamy35-day-20261112/rounds/D0/slots
mkdir -p $E/out8
curl -s -m60 -X POST http://<lab-endpoint>/api/models/unload/qwen36-35b-moe -o /dev/null -w "unload %{http_code}\n"
docker rm -f ev8 2>/dev/null
run() { # $1 = tag, $2.. = extra args (nvprof prefix handled below)
  TAG=$1; shift
  docker run --rm --name ev8 --device nvidia.com/gpu=$GPU \
    -v <MODELS>:/models:ro -v $E/build/bin:/e5:ro \
    -v $M/prod-libs/libggml-cuda-bundle3-20261112.so:/e5/libggml-cuda.so:ro \
    -v $SLOT:/slots:ro -v $M/occamy35-day-20261112/evidence/fixtures/text:/fx:ro -v $E/out8:/out \
    -v <media>/Models/prof-27b-20261107/cuda:/prof/cuda:ro \
    -e LD_LIBRARY_PATH=/prof/cuda/lib:/e5 $(cat $E/env-bundle3.txt) \
    --entrypoint /prof/cuda/bin/nvprof sha256:3a0caa9afea12f5ca5d982621cc2d86b1fa6ac614d735b262fa475d7dc6e2a6d \
    --print-gpu-summary --log-file /out/$TAG.nvprof.txt \
    /e5/llama-k-timing -m /models/occamy-1.0-Q4_0-headq40b.gguf --override-kv tokenizer.ggml.pre=str:qwen35 \
    -ngl 999 -c 98560 -b 1024 -ub 1024 -fa on --slot /slots/s0_98k_off0.bin \
    --prefill-file /fx/d0.txt --len 98048 --rs 2 --calls 20 --warm 3 "$@" > $E/out8/$TAG.out 2> $E/out8/$TAG.err
}
# 1) unprofiled timing, all k (the V(k)/T numbers): drop the nvprof wrapper by using --entrypoint /e5/llama-k-timing if you want zero profiler overhead
flock $M/gpu-bench.lock bash -c "$(declare -f run); M=$M E=$E GPU=$GPU SLOT=$SLOT; run all; run base --only-k 0; run k1 --only-k 1; run k2 --only-k 2; run k3 --only-k 3"
cat $E/out8/all.out | grep -E "SUMMARY|V\(2\)|VRAM"
# 2) per-family table (CPU)
cd /tmp && python3 -I $E/classify_nvprof.py $E/out8/base.nvprof.txt $E/out8/k1.nvprof.txt $E/out8/k2.nvprof.txt $E/out8/k3.nvprof.txt --calls 23
```
Notes: the "all" run is under nvprof too (timings inflated slightly by profiling); for clean V(k)/T also run once with
 --entrypoint /e5/llama-k-timing and no nvprof args (same mounts, drop /prof). 5 loads x ~1-1.5 min load + slot load; ~7-8 min total; trim by skipping "all".
VRAM figure is device-wide (ggml_backend_dev_memory), sampled before/after every call; peak used = total - min free.

## Fast-path guards on day-bundle3 (re-checked in E5/src, line numbers moved vs. your hints)
- fattn-occamy.cu:621  `if (Q->ne[1] != 1) return false;`  -> OFA FA decode path excludes k>1 rows of one sequence (falls to stock FA). EXCLUDED.
- occamy-gdn.cu:965    `if (t->ne[0] != NQKV || t->ne[1] != 1 || t->ne[2] != ns) { return rej("ssm_conv output shape"); }` (ne[1]=n_seq_tokens) and
  :1023 `v->ne[2] != 1 ... v->ne[3] != ns`, :948 `n_rs != n_seqs`, :1052 `ggml_nrows(out) != ns`  -> GDN fast path requires 1 token per sequence. EXCLUDED for k>1 of one seq.
- occamy-fused.cu:541  `!(nt == 1 || (nt == 2 && ggml_cuda_occamy_n2split()))` (nt = ggml_nrows(h)); the :569 check (`t->ne[2] != nt ... "n_tokens != 1"`) is only a shape consistency check inside the window.
  -> MoE fused window is NOT excluded for k=2 of ONE sequence when GGML_CUDA_OCCAMY_N2SPLIT=1 (it only sees 2 rows, not whether they are 2 seqs); k=3 excluded (nt==3). Caveat: with N2SPLIT=1 the k=2 single-seq call may use the N2 fused MoE path (should be numerically fine, rows are independent in MoE).
