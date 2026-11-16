# L2 prefill kernel share - NOT RUN (GPU script ready)

| class | % prefill wall | % N3 wall | roofline | achieved |
|---|---|---|---|---|
| pending GPU run: bash L2/l2_gpu.sh, then L2/run-*/summary.md | - | - | - | - |

Method: GGML_CUDA_NODE_TIMING=<file> (branch l2-nodetime, default off) puts a cudaEvent at each node boundary on non-captured graphs, one sync per graph. Graph phase by max matmul rows: >=1024 prefill, 2..1023 small/mixed, 1 decode. Per key: ms, FLOPs (matmul 2*K*N*rows, MoE with used experts, FA 4*D*nq*nkv*heads), bytes (touched expert weights, srcs+dst). summarize.py maps to % of phase, % of N3 wall (phase share x 40/60/49 s of 149 s), TFLOPS and GB/s vs 6.8 TFLOPS / 317 GB/s.
Caveats: per-graph sync perturbs timing slightly; ops fused into a node are attributed to the first node; nsys/ncu/nvprof not installed.
Files: l2_gpu.sh, summarize.py, n3_replay_l2.py, build-l2.sh, RECEIPT.txt, src/ (worktree), build/.
