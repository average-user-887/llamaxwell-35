# L1 stage split: run-20261116T084807Z

### L_1/ledger/nl0-b0-p0-prod-18887.csv

| after decode kind | windows | sum gap s | tokenise s | post_tasks s | task_proc s | cache_match_ckpt s | ckpt_create s | post_decode s | slot_launch s (nested) | remainder (batch build etc) s | sse_result events |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| prefill | 11 | 7.071 | 0.000 | 0.000 | 0.000 | 0.000 | 7.014 | 0.006 | 0.000 | 0.050 | 2 |
| decode | 1897 | 3.508 | 0.004 | 0.000 | 0.001 | 0.000 | 0.049 | 1.341 | 0.001 | 2.112 | 1816 |
| mixed | 18 | 0.478 | 0.000 | 0.000 | 0.000 | 0.000 | 0.416 | 0.014 | 0.000 | 0.048 | 23 |

prefill-only windows > 50 ms: 6, dominant stage counts: {'ckpt_create': 6}

idle windows (>= 2.0 s between decodes): 1, total 18.4 s
idle windows containing a request arrival: 1; client-side wait (decode end -> http_recv) 18.3 s; server-side (http_recv -> next decode start) 0.1 s
request arrival vs previous final response on the same slot (client think time): n=7, mean 13.28 s, min 1.05, max 37.58, sum 93.0 s
stage tokenise: n=9 sum 0.018 s mean 1.95 ms max 2.5 ms
stage post_tasks: n=9 sum 0.000 s mean 0.02 ms max 0.0 ms
stage task_proc: n=3872 sum 16.293 s mean 4.21 ms max 12710.2 ms
stage slot_launch: n=9 sum 0.001 s mean 0.11 ms max 0.1 ms
stage cache_match_ckpt: n=9 sum 0.000 s mean 0.05 ms max 0.1 ms
stage ckpt_create: n=20 sum 7.533 s mean 376.66 ms max 1997.3 ms
stage post_decode: n=1928 sum 1.365 s mean 0.71 ms max 12.8 ms

### L_2/ledger/nl0-b0-p0-prod-19140.csv

| after decode kind | windows | sum gap s | tokenise s | post_tasks s | task_proc s | cache_match_ckpt s | ckpt_create s | post_decode s | slot_launch s (nested) | remainder (batch build etc) s | sse_result events |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| prefill | 10 | 5.079 | 0.000 | 0.000 | 0.000 | 0.000 | 5.037 | 0.005 | 0.000 | 0.037 | 2 |
| decode | 1898 | 3.315 | 0.005 | 0.000 | 0.001 | 0.000 | 0.043 | 1.175 | 0.001 | 2.091 | 1794 |
| mixed | 18 | 0.476 | 0.000 | 0.000 | 0.000 | 0.000 | 0.420 | 0.012 | 0.000 | 0.044 | 21 |

prefill-only windows > 50 ms: 4, dominant stage counts: {'ckpt_create': 4}

idle windows (>= 2.0 s between decodes): 2, total 11.2 s
idle windows containing a request arrival: 1; client-side wait (decode end -> http_recv) 9.1 s; server-side (http_recv -> next decode start) 0.1 s
request arrival vs previous final response on the same slot (client think time): n=7, mean 13.30 s, min 1.04, max 37.65, sum 93.1 s
stage tokenise: n=9 sum 0.019 s mean 2.12 ms max 2.6 ms
stage post_tasks: n=9 sum 0.000 s mean 0.05 ms max 0.2 ms
stage task_proc: n=3874 sum 7.083 s mean 1.83 ms max 3562.2 ms
stage slot_launch: n=9 sum 0.001 s mean 0.11 ms max 0.2 ms
stage cache_match_ckpt: n=9 sum 0.000 s mean 0.04 ms max 0.1 ms
stage ckpt_create: n=20 sum 7.556 s mean 377.81 ms max 2002.4 ms
stage post_decode: n=1929 sum 1.195 s mean 0.62 ms max 9.8 ms

### L_3/ledger/nl0-b0-p0-prod-19363.csv

| after decode kind | windows | sum gap s | tokenise s | post_tasks s | task_proc s | cache_match_ckpt s | ckpt_create s | post_decode s | slot_launch s (nested) | remainder (batch build etc) s | sse_result events |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| prefill | 10 | 5.060 | 0.000 | 0.000 | 0.000 | 0.000 | 5.019 | 0.005 | 0.000 | 0.036 | 2 |
| decode | 1898 | 2.880 | 0.003 | 0.000 | 0.001 | 0.000 | 0.041 | 0.801 | 0.000 | 2.034 | 1603 |
| mixed | 18 | 0.409 | 0.000 | 0.000 | 0.000 | 0.000 | 0.361 | 0.010 | 0.000 | 0.039 | 22 |

prefill-only windows > 50 ms: 3, dominant stage counts: {'ckpt_create': 3}

idle windows (>= 2.0 s between decodes): 2, total 13.6 s
idle windows containing a request arrival: 1; client-side wait (decode end -> http_recv) 11.6 s; server-side (http_recv -> next decode start) 0.0 s
request arrival vs previous final response on the same slot (client think time): n=7, mean 13.26 s, min 1.03, max 37.52, sum 92.8 s
stage tokenise: n=9 sum 0.011 s mean 1.25 ms max 2.3 ms
stage post_tasks: n=9 sum 0.000 s mean 0.04 ms max 0.2 ms
stage task_proc: n=3874 sum 9.536 s mean 2.46 ms max 4992.4 ms
stage slot_launch: n=9 sum 0.001 s mean 0.08 ms max 0.1 ms
stage cache_match_ckpt: n=9 sum 0.000 s mean 0.03 ms max 0.1 ms
stage ckpt_create: n=20 sum 7.477 s mean 373.84 ms max 2009.6 ms
stage post_decode: n=1929 sum 0.818 s mean 0.42 ms max 9.4 ms

## Totals over all ledger files (gap after prefill-only decodes, per replay sums shown above)

| stage | sum s after prefill-only | sum s after decode-only | sum s after mixed |
|---|---|---|---|
| gap | 17.210 | 9.703 | 1.363 |
| tokenise | 0.000 | 0.012 | 0.000 |
| post_tasks | 0.000 | 0.000 | 0.000 |
| task_proc | 0.001 | 0.004 | 0.000 |
| cache_match_ckpt | 0.000 | 0.001 | 0.000 |
| ckpt_create | 17.070 | 0.133 | 1.197 |
| post_decode | 0.016 | 3.317 | 0.036 |
| slot_launch | 0.000 | 0.002 | 0.000 |
| remainder | 0.123 | 6.237 | 0.131 |
