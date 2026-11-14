# Card 1 S1 single-stream prefill — RESULT

| pair | s1 makespan s | base makespan s | delta | s1 TTFT mean | base TTFT mean | s1 max gap | base max gap |
|---|---|---|---|---|---|---|---|
| 1 | 170.28 | 178.03 | -4.4% | 19.45 | 22.24 | 11.84 | 11.82 |
| 2 | 169.18 | 177.51 | -4.7% | 19.35 | 22.19 | 11.79 | 11.80 |
| 3 | 169.56 | 177.47 | -4.5% | 19.43 | 22.19 | 11.85 | 11.80 |
| mean | 169.67 | 177.67 | **-4.5%** | 19.41 | 22.21 | 11.83 | 11.81 |

- Residual prefill_work_share_lost_to_ne3 = 0.00% (was 20.19%); deferred batches counter = 2.
- Branch s1 145c0ad, flag --prefill-single-stream (default off). Base arm = K0 lib gemm (not the s1 server binary).
- Not yet measured: per-slot completion/slot-B TTFT split, round-trip identical-token check. Run: <LAB>/occamy35-night-20261114/1/run-20261113T184653Z
- Passes the 1% gain line. Candidate for the card 7 stack.
