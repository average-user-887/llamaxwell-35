# Card 20 swap check (15 Nov 2026)
| Check | Result |
|---|---|
| health after docker start | ok (<5 min) |
| short prompt HTTP | 200 |
| 4641-token prompt HTTP | 200, pp 589.4 t/s, tg 85.8 t/s (16 tok) (stack7 card 7d: 549 / 80) |
| occamy_fa_gemm | engaged=10 q_tiled=10 fallback=370 (q_lt_64:350, keys_lt_4096:20) |
| fused | calls=10 softmax_launches=640 |
| moe_v3 | no log line without OCCAMY_DEBUG (not added to prod); env set in container |
Method: requests direct to the backend port under gpu-bench.lock (as card 7d), container stopped afterwards (on-demand via llama-swap), lock released.
Prod = container llamacpp-m6000 (batch8), previous = llamacpp-m6000-stack7-prev. Script run-35b-batch8-20261115.sh, libs prod-libs/batch8-20261115.
