# L0 N3 wall ledger: instrumentation ready, GPU run pending (queued by lead)

| item | value |
|---|---|
| build | d4c13eb + l0-ledger 4b0fcfb; only libllama-server-impl.so differs from prod batch8 (sha in RECEIPT.txt) |
| flag | LLAMA_N3_LEDGER=<file in container>, default off; per-decode CSV + <file>.tasks |
| run | L0/l0_gpu.sh (N3 short replay x3, ledger on); summary at L0/run-*/summary.md via L0/summarize.py |
| measured results | none yet (no GPU run done by this card) |

Per-decode columns: t_start (epoch s), wall_ms (inside llama_process call; async for prefill without outputs), gap_ms (since previous decode call returned), period_ms (start to previous start), n_tokens, slot0, slot1, slot_other rows, n_prompt, n_gen, n_active slots, kind (prefill/decode/mixed), has_output, sync, ret.
Per-task columns: n_prompt_tokens, n_cache_before (slot cache length at task start), n_common (prefix match), n_past (after checkpoint logic = tokens reused), n_prefill, n_appended (new beyond slot cache), ckpt_restore, full_reset.
Caveat: prefill-only batches return before the GPU finishes, so wall_ms understates them; the summarizer therefore also reports start-to-start attributed time (gaps >= 2 s counted as idle). Tokenisation happens in the HTTP path, not the slot loop, so only full-prompt reprocessing (full_reset) is logged.
