# L1 stage split of the post-prefill host gap (build only; GPU run pending)

| stage (gap after prefill-only decode) | sum s per replay | % of gap |
|---|---|---|
| tokenise, post_tasks, task_proc, cache_match_ckpt, ckpt_create, post_decode, remainder | pending | pending |

Hypothesis: ckpt_create dominates the ~0.9 s gap. Idle >= 2 s is classed client vs server via http_recv.
Run: L1/l1_gpu.sh; summary: <run>/summary.md via L1/summarize.py. Details in RECEIPT.txt.
