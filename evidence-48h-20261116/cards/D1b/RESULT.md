# D1b: CUDA graphs (GGML_CUDA_MAXWELL_GRAPHS) on N3 decode

| item | status |
|---|---|
| N3 makespan gain, arm gr vs prod (n=6 each) | PENDING (script queued on GPU lock; summary in run-*/summary.md) |
| token identity per slot vs prod | PENDING |
| peak VRAM (gate <= prod + 100 MiB) | PENDING |
| engagement (graph_launches > 0 in gr, 0 in prod) | PENDING (zero in gr = void arm) |
| accept gate | >=1% N3 at >=3 sd, token-identical, VRAM ok |
| arm gr crash | if it happens: run-*/CRASH.txt (sanitized stderr), run stops |

## Why graphs are off, and the likely BUNDLE3 crash cause (read-only, source on prod base d4c13eb)
- Gate: ggml_cuda_graph_set_enabled(): cc < VOLTA and env absent -> disable_due_to_gpu_arch (any value of the env passes, even "0").
- The exact BUNDLE3 crash text was NOT found in the night 1113/1114 logs or notes (only the RULES line "any value crashes BUNDLE3"); the cause below is inferred from code, to be confirmed by arm gr stderr.
- Capture-illegal host syncs inside the compute path that runs during cudaStreamBeginCapture (relaxed mode):
  1. occamy-gdn.cu ggml_cuda_occamy_gdn_split2 (2-row decode GDN path): cudaMemcpyAsync D2H of s_copy + cudaStreamSynchronize(st) once per graph (cached by g_graph_gen), plus a one-time cudaMalloc + cudaMemsetAsync for dzero. Stream sync on a capturing stream = cudaErrorStreamCaptureUnsupported -> CUDA_CHECK abort.
  2. The same s_copy value is read on the host and selects the kernel/gather topology (sv[0]==head && sv[1]==head+1), so even if capture succeeded, replay would bake in the first row mapping (silent wrong state routing) unless nodes[0]-key/property check changes; s_copy data content is not part of the node-property memcmp.
  3. Other host-side readbacks in the same file (mul_mat_id ids D2H at ggml-cuda.cu ~2241 is gated by ggml_cuda_mul_mat_id_needs_sync and disables graphs; occamy debug check path ~3872 only with check budget).
- If arm gr passes with token identity, point 2 is not triggered in N3 (rows constant). If it crashes, expect a capture-unsupported error at the first 2-row decode step.
- Engagement counter: branch d1b-graphcount a37a3c2 (base d4c13eb), GGML_CUDA_D1B_COUNT=1 prints "d1b_graphs[run|exit]: compute_calls arch_off captures graph_launches". Both arms use this lib.
