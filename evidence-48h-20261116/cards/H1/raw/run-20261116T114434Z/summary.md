| arm | n | makespan mean s | sd s | ckpt_create sum/replay s | ckpt_wait_gpu s | ckpt_copy s | peak VRAM max MiB | token-identical to prod ref | checkpoints via new path / created |
|---|---|---|---|---|---|---|---|---|---|
| prod | 6 | 160.52 | 0.56 | 7.730 | 6.843 | 0.860 | 23868 | 1/6 | 0 / 120 |
| h1 | 6 | 160.69 | 0.50 | 7.718 | 6.598 | 1.119 | 23868 | 1/6 | 120 / 120 |

ACCEPT ckpt_copy cut >= 3x (prod 0.860 s -> h1 1.119 s, 0.77x): FAIL
ACCEPT peak VRAM h1 23868 <= prod 23868 + 100 MiB: PASS
ACCEPT token identity h1 == prod ref: FAIL
ACCEPT engagement > 0 (new-path checkpoints in h1, 0 in prod): PASS

Per session:

| session | arm | makespan s | ckpt_create s | wait_gpu s | copy s | n ckpt | peak MiB | identity vs prod ref | engaged |
|---|---|---|---|---|---|---|---|---|---|
| b0_p0_prod | prod | 159.75 | 7.653 | 6.839 | 0.813 | 20 | 23867 | identical | 0 |
| b0_p1_h1 | h1 | 160.32 | 7.896 | 6.858 | 1.037 | 20 | 23867 | (0, 1, 349, 16236, 5878, 'token differs') | 20 |
| b1_p0_h1 | h1 | 160.27 | 7.880 | 6.852 | 1.028 | 20 | 23867 | (0, 3, 339, 12756, 1756, 'token differs') | 20 |
| b1_p1_prod | prod | 161.15 | 7.841 | 6.864 | 0.819 | 20 | 23867 | (0, 3, 339, 12756, 1756, 'token differs') | 0 |
| b2_p0_h1 | h1 | 160.89 | 8.085 | 6.824 | 1.259 | 20 | 23868 | (0, 3, 339, 12756, 1756, 'token differs') | 20 |
| b2_p1_prod | prod | 160.55 | 7.669 | 6.826 | 0.842 | 20 | 23868 | (0, 1, 349, 16236, 5878, 'token differs') | 0 |
| b3_p0_prod | prod | 160.13 | 7.683 | 6.842 | 0.841 | 20 | 23868 | (0, 3, 339, 12756, 1756, 'token differs') | 0 |
| b3_p1_h1 | h1 | 160.69 | 7.906 | 6.845 | 1.060 | 20 | 23867 | (0, 1, 349, 16236, 5878, 'token differs') | 20 |
| b4_p0_prod | prod | 160.37 | 7.790 | 6.846 | 0.942 | 20 | 23867 | (0, 1, 349, 16236, 5878, 'token differs') | 0 |
| b4_p1_h1 | h1 | 160.37 | 7.989 | 6.838 | 1.150 | 20 | 23867 | (1, 1, 494, 364, 434, 'token differs') | 20 |
| b5_p0_h1 | h1 | 161.59 | 6.550 | 5.371 | 1.178 | 20 | 23867 | identical | 20 |
| b5_p1_prod | prod | 161.16 | 7.746 | 6.839 | 0.905 | 20 | 23867 | (0, 1, 349, 16236, 5878, 'token differs') | 0 |

makespan = last request end - t_run0; ckpt_create = server stage (wait + copy + bookkeeping); peak = max whole-GPU 5 s sample in the session window.
