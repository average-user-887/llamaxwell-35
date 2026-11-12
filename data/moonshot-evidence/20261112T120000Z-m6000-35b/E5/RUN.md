# E5 GPU run: expert-routing capture (NOT yet run). ~10 min GPU.
Build: bash build-e5b.sh  -> E5/build/bin/{llama-route-capture,llama-k-timing,libggml*.so}
Fixtures: $M/occamy35-day-20261112/evidence/fixtures/text/d0.txt and d1.txt (disjoint public text, ~175K tokens each; E3 dir had no fixtures yet).
Time estimate (ASSUMED 250-400 pp t/s on M6000 at depth, unverified; tool prints real figures):
 N2 part, LEN=32768: 2x32K prefill = 3-4.5 min, 64 decode steps with per-node sync (cb_eval) ~ 1-2 min, model load ~1 min.
 16K prefill part: ~1 min + load 1 min.  Total ~8-10 min.  LEN=98048 option: prefill alone ~11-18 min, exceeds budget.

```bash
M=<LAB>; E=$M/occamy35-day-20261112/evidence/E5
GPU=<gpu>          # GPU UUID (M6000)
FX=$M/occamy35-day-20261112/evidence/fixtures/text
LEN=${LEN:-32768}; CTX=$(( 2*(LEN+64) ))               # -c >= 2*(len+64)
mkdir -p $E/out
curl -s -m60 -X POST http://<lab-endpoint>/api/models/unload/qwen36-35b-moe -o /dev/null -w "unload %{http_code}\n"   # POST only, never GET unload-all
docker rm -f ev5 2>/dev/null
flock $M/gpu-bench.lock docker run --rm --name ev5 --device nvidia.com/gpu=$GPU \
  -v <MODELS>:/models:ro -v $E/build/bin:/e5:ro \
  -v $M/prod-libs/libggml-cuda-bundle3-20261112.so:/e5/libggml-cuda.so:ro \
  -v $FX:/fx:ro -v $E/out:/out \
  -e LD_LIBRARY_PATH=/e5 $(cat $E/env-bundle3.txt) \
  --entrypoint /bin/bash sha256:3a0caa9afea12f5ca5d982621cc2d86b1fa6ac614d735b262fa475d7dc6e2a6d -c '
set -e
COMMON="-m /models/occamy-1.0-Q4_0-headq40b.gguf --override-kv tokenizer.ggml.pre=str:qwen35 -ngl 999 -b 1024 -ub 1024 -fa on"
# (1) N2 decode at depth, 64 steps, both seqs
OCCAMY_ROUTE_DUMP=/out/n2 /e5/llama-route-capture $COMMON -c '$CTX' --route-len '$LEN' --route-steps 64 \
   --route-seq /fx/d0.txt --route-seq /fx/d1.txt
# (2) one 16K uncached prefill, rows-per-expert histogram per ubatch/layer
OCCAMY_ROUTE_DUMP=/out/pf16k /e5/llama-route-capture $COMMON -c 16640 --route-len 16384 --route-prefill /fx/u00.txt,/fx/u01.txt,/fx/u02.txt
' 2>&1 | tee $E/out/run.log
# analysis (host, CPU):
cd /tmp && python3 -I $E/analyze.py --gguf <MODELS>/occamy-1.0-Q4_0-headq40b.gguf \
   --decode $E/out/n2.decode.csv --prefill $E/out/pf16k.prefill.csv
```
Notes: cb_eval makes the scheduler split the graph at every ffn_moe_topk node (sync per layer) and disables CUDA graphs/fusion across that boundary,
so timings of this run are NOT valid speed data; routing is. Fused topk-MoE kernels (if the fork fuses argsort+select) still produce the topk tensor as output.
u00..u02 prefill files = ~220K+78K+85K chars, check that they reach 16384 tokens (tool errors out otherwise; add u03).
Seed token: decode steps start from "\n" at pos LEN for both seqs, then greedy (so seqs diverge only by context).
Stop-check afterwards: docker ps | grep ev5 (should be empty); the swap proxy reloads models on demand.
