# B1 N3R spec-type matrix (batch8 libs, MTP GGUF, ub1024, 2x84224, q8 draft KV, temp 0, n_predict 800)

Run: <W>/B1/run-20261115T213543Z (00:47-02:23 CET 16 Nov). Table = summary.md; sha columns = turns differing out of 12.

| arm | kind | drafted | accepted | acc/gen | zero-accept turns | decode s | makespan s | VRAM MiB | sha vs offA | sha vs offA2 | void |
|---|---|---|---|---|---|---|---|---|---|---|---|
| offA | edit/copy-heavy | 0 | 0 | 0.000 | 4/4 | 78.1 |  |  |  |  |  |
| offA | tool-call | 0 | 0 | 0.000 | 4/4 | 96.6 |  |  |  |  |  |
| offA | prose | 0 | 0 | 0.000 | 4/4 | 103.3 |  |  |  |  |  |
| offA | ALL | 0 | 0 | 0.000 | 12/12 | 278.0 | 307.7 | 22882 |  | 0/12 differ | ok |
| offA2 | edit/copy-heavy | 0 | 0 | 0.000 | 4/4 | 78.0 |  |  |  |  |  |
| offA2 | tool-call | 0 | 0 | 0.000 | 4/4 | 96.5 |  |  |  |  |  |
| offA2 | prose | 0 | 0 | 0.000 | 4/4 | 103.1 |  |  |  |  |  |
| offA2 | ALL | 0 | 0 | 0.000 | 12/12 | 277.7 | 306.7 | 22882 | 0/12 differ |  | ok |
| mtp2 | edit/copy-heavy | 2146 | 2121 | 0.663 | 0/4 | 109.8 |  |  |  |  |  |
| mtp2 | tool-call | 2129 | 2128 | 0.665 | 0/4 | 144.1 |  |  |  |  |  |
| mtp2 | prose | 2141 | 2123 | 0.663 | 0/4 | 149.0 |  |  |  |  |  |
| mtp2 | ALL | 6416 | 6372 | 0.664 | 0/12 | 402.9 | 377.5 | 24313 | 8/12 differ | 8/12 differ | ok |
| mtp3 | edit/copy-heavy | 1252 | 1222 | 0.743 | 0/4 | 34.3 |  |  |  |  |  |
| mtp3 | tool-call | 1201 | 1198 | 0.747 | 2/4 | 88.0 |  |  |  |  |  |
| mtp3 | prose | 1222 | 1208 | 0.746 | 0/4 | 33.0 |  |  |  |  |  |
| mtp3 | ALL | 3675 | 3628 | 0.745 | 2/12 | 155.3 | 265.6 | 24439 | 8/12 differ | 8/12 differ | ok |
| mtp1 | edit/copy-heavy | 1603 | 1591 | 0.497 | 0/4 | 120.1 |  |  |  |  |  |
| mtp1 | tool-call | 1596 | 1596 | 0.499 | 0/4 | 154.9 |  |  |  |  |  |
| mtp1 | prose | 1599 | 1595 | 0.498 | 0/4 | 160.2 |  |  |  |  |  |
| mtp1 | ALL | 4798 | 4782 | 0.498 | 0/12 | 435.2 | 396.6 | 24187 | 6/12 differ | 6/12 differ | ok |
| ngmod | edit/copy-heavy | 3319 | 2873 | 0.898 | 0/4 | 57.4 |  |  |  |  |  |
| ngmod | tool-call | 3170 | 3092 | 0.966 | 0/4 | 57.5 |  |  |  |  |  |
| ngmod | prose | 2783 | 2418 | 0.815 | 0/4 | 152.5 |  |  |  |  |  |
| ngmod | ALL | 9272 | 8383 | 0.895 | 0/12 | 267.4 | 302.8 | 24409 | 4/12 differ | 4/12 differ | ok |
| ngmapk | edit/copy-heavy | 2292 | 2280 | 0.713 | 0/4 | 100.5 |  |  |  |  |  |
| ngmapk | tool-call | 2250 | 2250 | 0.703 | 0/4 | 169.7 |  |  |  |  |  |
| ngmapk | prose | 2011 | 1977 | 0.618 | 0/4 | 161.6 |  |  |  |  |  |
| ngmapk | ALL | 6553 | 6507 | 0.678 | 0/12 | 431.8 | 347.5 | 24409 | 6/12 differ | 6/12 differ | ok |
| ngmod_mtp2 | edit/copy-heavy | 0 | 0 | 0.000 | 1/1 | 0.0 |  |  |  |  |  |
| ngmod_mtp2 | tool-call | 0 | 0 | 0.000 | 2/2 | 0.0 |  |  |  |  |  |
| ngmod_mtp2 | ALL | 0 | 0 | 0.000 | 3/3 | 0.0 | 25.0 | 24313 | 12/12 differ | 12/12 differ | VOID |

## Findings
- A vs A (offA vs offA2): 0/12 turns differ, so the reorder floor is 0 on spec-off; every spec-arm difference is real (8/12 mtp2, 8/12 mtp3, 6/12 mtp1, 4/12 ngram-mod, 6/12 ngram-map-k). No spec arm is token-identical to spec off.
- Void rule (drafted>0 on both slots): met for all single spec arms (drafted>0 on all arms, slot 0 only marginally; see caveat). ngmod_mtp2 (ngram-mod,draft-mtp comma list is accepted by --spec-type) is VOID: CUDA out of memory at load (24313 MiB peak), it never served; drafted 0. Not retried.
- Best makespan: mtp3 265.6 s vs off 307.7 s (-13.7%); ngram-mod 302.8 (-1.6%); mtp2 377.5 (+23%), mtp1 396.6 (+29%), ngmapk 347.5 (+13%) are SLOWER than spec off.

## Caveats (important)
- The fixture output is degenerate at temp 0: slot 0 hits EOS after 1-23 tokens on every turn (e.g. mtp3 slot 0 generated 23,2,16,21,1,4); slot 1 generates 800 tokens per turn and drafted 599 / accepted 599 per turn in mtp3 (100% accepted, likely repetition loop). So drafted/accepted/accuracy are dominated by slot 1 and tell little about real agent text; zero-accept turns are mostly slot 0 EOS turns. Check turns.md in each arm's run/ before drawing conclusions.
- Decode seconds vary a lot between arms for the same slot-1 work (mtp3 edit 34 s vs mtp2 110 s vs off 78 s) with ~100% acceptance: suspicious, likely MTP/ngram verification cost or slot-0/slot-1 overlap effects. Single run per arm.
- mtp2/mtp1/ngmapk slower than off despite ~66%/50%/68% acc/gen.
- ngram-map-k used --spec-draft-n-max 16 and --spec-ngram-map-k-size-m 16; ngram-mod defaults (n-max 64).

## Method / files
Script b1.sh (lock flock -w 21600, POST unload qwen36-35b-moe only via <IP>:11445, containers nb1-*, removed; no 250 W change; MAXWELL_GRAPHS unset). Files in $B: b1.sh, summarize.py, run dir with steps.log, per-arm {run/run.json,turns.md,server.log,vram.csv,client.out}, spec_type_help.txt.
