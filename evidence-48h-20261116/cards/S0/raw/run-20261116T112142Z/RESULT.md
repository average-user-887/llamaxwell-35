# S0 service soak result (batch8 prod build, 2 slots ~96K)

| metric | value |
|---|---|
| verdict | PASS |
| soak length (min, configured) | 60 |
| turns ok / total | 61 / 61 |
| errors (HTTP != 200, bad receipt, container death, fatal server-log lines, driver error) | 0 (0, 0, 0, 0, 0) |
| token identity after restore (greedy 64) | 10 / 10 identical (mode slot_api) |
| mid-prefill cancels honoured (slot idle < 120 s) / total | 4 / 6; idle-after median 2.5 s |
| peak VRAM used (MiB) | 23867 |
| min free VRAM (MiB) | 709 (first half 709, second half 709) |
| decode tok/s 1K turns, first third -> last third | 32.10 -> 32.02 (-0.2%, n=21) |
| decode tok/s 4K turns, first third -> last third | 31.98 -> 31.22 (-2.4%, n=20) |
| prefill tok/s (fresh ~84K turn 0) first -> last third | 161.88 -> 155.16 (-4.2%, n=11) |
| prefill tok/s (append turns) first -> last third | 109.14 -> 109.01 (-0.1%, n=30) |
| GEMM engaged (server.log) | 5120 |
| mean GPU power (W) / max temp | 234 / see vram.csv |

## Method
One fresh container ns0-* from the L0 prod arm (libs prod-libs/batch8-20261115, env of run-35b-batch8-20261115.sh, cap 1024, ub 2048, b 2048, 2 slots, ctx 196608, PREFILL_SINGLE_STREAM, cache-ram 0, ckpt 16/2048). Each slot loops episodes: fresh ~84K base prefill, then 4 N3-short tool-append turns (1K/4K) to ~96K. Every episode: after turn 1 the other slot is drained, slot saved via the server API, greedy 64 tokens (pre), erase, restore, greedy 64 tokens (post), compare token ids. Odd episodes start with a streaming request cancelled 20-50 s into prefill, then time to slot idle is logged. VRAM sampled every 5 s (vram.csv). Per-turn timings in turns.jsonl.

## errors.log (first 15)
```
1794831734.1 cancel s0_e1_cancel idle_after_s=None health=200
1794831734.7 cancel s1_e1_cancel idle_after_s=None health=200
```

## Files
turns.jsonl, vram.csv, errors.log, soak.json, server.log, libs.sha256, steps.log in the run dir.
