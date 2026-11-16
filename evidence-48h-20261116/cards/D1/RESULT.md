# D1-pre: CUDA graphs in decode (investigation only; STOPPED after step 2)

| question | answer |
|---|---|
| graphs active in decode today, 1-row step? | NO |
| graphs active, 2-row step? | NO |
| reason | single gate: ggml-cuda.cu ggml_cuda_graph_set_enabled(): cc < VOLTA and getenv("GGML_CUDA_MAXWELL_GRAPHS")==nullptr -> graph->disable_due_to_gpu_arch=true -> is_enabled() false. Applies to every graph key, every row count. |
| upstream batch>1 disable rule present in fork? | NO. Only compat check is mul_mat_id needs_sync (fallback path syncs stream). Re-capture is keyed by cgraph->nodes[0] + node-property memcmp (shape/data ptr/src ptr/ne/nb) + 1 warmup call; a graph is kept per key, so 1-row and 2-row topologies would each cache their own exec (cudaGraphExecUpdate on change). |
| other gates | build has GGML_CUDA_GRAPHS=ON (compiled in); GGML_CUDA_DISABLE_GRAPHS env unused |
| only way to get graphs without code change | the banned env GGML_CUDA_MAXWELL_GRAPHS. A source patch that bypasses the arch gate would be a workaround of the rule -> not built. |
| prior measurement | night-20261111 LEDGER GR1 (env on main+FA_PIPE=3, 1 cycle warm+d4k): bitexact, no per-token recapture, warm -0.15%, d4k +0.17%, period median 11.970->11.986 ms: no gain, REJECT. Night 1113/1114 RULES: "any value crashes BUNDLE3". |
| expected saving | none shown; the one measured test says ~0 (est 1-3% N3 in DECISIONS is unsupported). Decode 1-row 18 ms / 2-row 31.6 ms is GPU-time dominated, so launch-gap removal is small. |
| design (not built) | n/a: per-row-count caching already exists in the fork's graph map; nothing to add. Counters would only count "disabled: arch" for every step. |

Action: STOP per card. No branch d1-graphcount, no build, no d1_gpu.sh, no GPU. Owner decision needed to ever enable (also re-test BUNDLE3 crash cause first).
Source read: A5/src (branch a5-sched 97c4ede on prod batch8 d4c13eb), ggml/src/ggml-cuda/ggml-cuda.cu lines ~2836-2960, 5237-5330; common.cuh is_enabled().
