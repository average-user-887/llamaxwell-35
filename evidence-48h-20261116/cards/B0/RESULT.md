# B0 N3R fixture: result (CPU only, 15 Nov 2026)
| check | result |
|---|---|
| syntax (py_compile x4, bash -n) | pass |
| --dry-run via run_n3r.sh | pass (schedule: start 78000, 6 turns, worst final depth 98165 at ctx 98304) |
| mock server run, A1 vs A2 token sha | 12/12 turns EQ, run_sha equal (mock only; real A vs A is the GPU check) |
| mock --ctx 81920 --turns 3 | start 61616, runs |
| fixture | 2 prefixes x 420K chars, 6 turns/slot, kinds: slot0 edit,tool,prose,edit,tool,prose; slot1 rotated |
Per turn record: prompt/gen tokens, token sha, stop_type, draft_n/draft_n_accepted/accept_rate (llama-server timings), max token gap, decode ms, metrics before/after, occamy_spec lines from --server-log.
Run: `B0/run_n3r.sh http://127.0.0.1:PORT ARM [--ctx 98304] --server-log LOG`; compare: `n3r_compare.py runs/A runs/A2`.
Risks: server must return timings.draft_n for draft stats (else only log lines); real token count of 78K prefix is checked at run time (fails loudly if fixture too short).
