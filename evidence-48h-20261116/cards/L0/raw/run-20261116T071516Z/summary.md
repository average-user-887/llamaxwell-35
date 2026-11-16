# L0 ledger summary: run-20261116T071516Z

### L_1/ledger/nl0-b0-p0-prod-13316.csv

decodes 1929, span 168.5 s, sum call wall 149.0 s, sum attributed 54.6 s, sum host gap (<2s) 10.2 s, idle (>= 2s) 9.2 s

| kind | decodes | tokens | sum call wall s | mean call wall ms | sum attributed s | mean attributed ms | mean gap ms | sum gap s |
|---|---|---|---|---|---|---|---|---|
| prefill | 11 | 9226 | 40.15 | 3650.35 | 2.29 | 458.26 | 990.59 | 8.92 |
| decode | 1900 | 3049 | 48.88 | 25.73 | 51.63 | 27.20 | 0.40 | 0.76 |
| mixed | 18 | 11292 | 59.99 | 3332.56 | 0.64 | 91.34 | 29.42 | 0.53 |

mixed-batch share: 18 / 1929 decodes (0.93%), 11292 / 23567 tokens (47.91%)
decode rows co-batched with a prefill chunk: YES (18 batches; max decode rows in such a batch 1)

prefill throughput by active slots (tok / attributed time; call-wall variant in brackets):

| active slots | batches | tokens | tok/s attributed | tok/s call-wall |
|---|---|---|---|---|
| 1 | 4 | 14 | 6.5 | 6.8 |
| 2 | 1 | 6 | 41.1 | 56.3 |

row split: slot0 11789, slot1 11778, other 0; decode-kind rows per batch mean 1.60

| tasks | prompt tok total | reused from cache (n_past) | prefilled | appended (new vs slot cache) | prefilled but not new (recompute) | ckpt restores | full resets |
|---|---|---|---|---|---|---|---|
| 9 | 721172 | 700672 | 20500 | 20500 | 0 | 0 | 0 |

### L_2/ledger/nl0-b0-p0-prod-13523.csv

decodes 1929, span 168.3 s, sum call wall 148.9 s, sum attributed 54.6 s, sum host gap (<2s) 8.2 s, idle (>= 2s) 11.1 s

| kind | decodes | tokens | sum call wall s | mean call wall ms | sum attributed s | mean attributed ms | mean gap ms | sum gap s |
|---|---|---|---|---|---|---|---|---|
| prefill | 11 | 9226 | 40.03 | 3638.86 | 2.24 | 447.67 | 863.83 | 6.91 |
| decode | 1900 | 3049 | 48.92 | 25.75 | 51.69 | 27.23 | 0.42 | 0.80 |
| mixed | 18 | 11292 | 59.98 | 3332.27 | 0.64 | 91.06 | 28.62 | 0.52 |

mixed-batch share: 18 / 1929 decodes (0.93%), 11292 / 23567 tokens (47.91%)
decode rows co-batched with a prefill chunk: YES (18 batches; max decode rows in such a batch 1)

prefill throughput by active slots (tok / attributed time; call-wall variant in brackets):

| active slots | batches | tokens | tok/s attributed | tok/s call-wall |
|---|---|---|---|---|
| 1 | 4 | 14 | 6.7 | 7.0 |
| 2 | 1 | 6 | 41.4 | 56.4 |

row split: slot0 11789, slot1 11778, other 0; decode-kind rows per batch mean 1.60

| tasks | prompt tok total | reused from cache (n_past) | prefilled | appended (new vs slot cache) | prefilled but not new (recompute) | ckpt restores | full resets |
|---|---|---|---|---|---|---|---|
| 9 | 721172 | 700672 | 20500 | 20500 | 0 | 0 | 0 |

### L_3/ledger/nl0-b0-p0-prod-13713.csv

decodes 1929, span 168.2 s, sum call wall 149.3 s, sum attributed 54.5 s, sum host gap (<2s) 8.2 s, idle (>= 2s) 10.8 s

| kind | decodes | tokens | sum call wall s | mean call wall ms | sum attributed s | mean attributed ms | mean gap ms | sum gap s |
|---|---|---|---|---|---|---|---|---|
| prefill | 11 | 9226 | 40.28 | 3661.94 | 2.30 | 459.93 | 868.18 | 6.95 |
| decode | 1900 | 3049 | 48.89 | 25.73 | 51.60 | 27.19 | 0.39 | 0.74 |
| mixed | 18 | 11292 | 60.11 | 3339.30 | 0.64 | 91.26 | 29.55 | 0.53 |

mixed-batch share: 18 / 1929 decodes (0.93%), 11292 / 23567 tokens (47.91%)
decode rows co-batched with a prefill chunk: YES (18 batches; max decode rows in such a batch 1)

prefill throughput by active slots (tok / attributed time; call-wall variant in brackets):

| active slots | batches | tokens | tok/s attributed | tok/s call-wall |
|---|---|---|---|---|
| 1 | 4 | 14 | 6.5 | 6.8 |
| 2 | 1 | 6 | 40.6 | 56.6 |

row split: slot0 11789, slot1 11778, other 0; decode-kind rows per batch mean 1.60

| tasks | prompt tok total | reused from cache (n_past) | prefilled | appended (new vs slot cache) | prefilled but not new (recompute) | ckpt restores | full resets |
|---|---|---|---|---|---|---|---|
| 9 | 721172 | 700672 | 20500 | 20500 | 0 | 0 | 0 |

### ALL (decodes pooled across servers)

decodes 5787, span 552.5 s, sum call wall 447.2 s, sum attributed 163.7 s, sum host gap (<2s) 26.6 s, idle (>= 2s) 31.1 s

| kind | decodes | tokens | sum call wall s | mean call wall ms | sum attributed s | mean attributed ms | mean gap ms | sum gap s |
|---|---|---|---|---|---|---|---|---|
| prefill | 33 | 27678 | 120.46 | 3650.38 | 6.83 | 455.29 | 910.85 | 22.77 |
| decode | 5700 | 9147 | 146.68 | 25.73 | 154.92 | 27.21 | 0.40 | 2.30 |
| mixed | 54 | 33876 | 180.07 | 3334.71 | 1.92 | 91.22 | 29.20 | 1.58 |

mixed-batch share: 54 / 5787 decodes (0.93%), 33876 / 70701 tokens (47.91%)
decode rows co-batched with a prefill chunk: YES (54 batches; max decode rows in such a batch 1)

prefill throughput by active slots (tok / attributed time; call-wall variant in brackets):

| active slots | batches | tokens | tok/s attributed | tok/s call-wall |
|---|---|---|---|---|
| 1 | 12 | 42 | 6.6 | 6.8 |
| 2 | 3 | 18 | 41.0 | 56.4 |

row split: slot0 35367, slot1 35334, other 0; decode-kind rows per batch mean 1.60

| tasks | prompt tok total | reused from cache (n_past) | prefilled | appended (new vs slot cache) | prefilled but not new (recompute) | ckpt restores | full resets |
|---|---|---|---|---|---|---|---|
| 27 | 2163516 | 2102016 | 61500 | 61500 | 0 | 0 | 0 |

