# NC1 workload census, 35B server logs (CPU only, read-only)

Window (UTC): 2026-11-10 15:27 to 2026-11-12 14:23 (about 47 h, with gaps between containers). Requests: 150 (all with timing lines). No truncation, context-shift or context-full event in any log (truncated=0 on all 150; zero matching lines).

## 1. Share of accounted time by event type (slot-seconds; two slots can overlap)

| Event type | Tasks | Seconds | Share |
|---|---|---|---|
| Decode (eval time) | 150 | 328.1 | 56.6% |
| Shallow prefill (processed tokens, end position <=64K, not a full reprocess) | 141 | 149.5 | 25.8% |
| Full reprocess (>4096 tokens processed from depth <1024) | 2 | 71.9 | 12.4% |
| Slot-file restore (load_slot, timed via llama-swap; 7 of 18 restores could be timed) | 18 (7 timed) | 30.0 | 5.2% |
| Deep prefill (end position >64K) | 7 | 0.7 | 0.13% |
| Total accounted | | 580.2 | 100% |

Accounted time is 580 s against 47 h of wall: the server was about 0.3% busy in this history, and it was benchmark/harness traffic (see caveats), not a sustained agentic load. Deep prefill is tiny because the 7 requests ending at >=90K tokens reused 90000 to 97996 cached tokens and processed 1 to 5 new tokens each. The 12.4% full-reprocess share is two 16384-token cold prompts (about 455 tok/s). Restore share is a lower bound: 11 restores had no matching llama-swap duration.

Prefill throughput (tasks with >1000 processed tokens): start depth 0-4K 462 tok/s (n=6, 36864 tok); start depth 4K-32K 401 tok/s (n=12). No measured prefill of >1000 tokens started past 32K depth.

## 2. Distributions (completed requests, n=150)

Prefill size (tokens actually processed):

| Tokens | 1-15 | 16-63 | 64-255 | 256-1023 | 1024-4095 | 4096-16383 | 16384-32767 | >=32768 |
|---|---|---|---|---|---|---|---|---|
| Requests | 11 | 57 | 0 | 64 | 16 | 0 | 2 | 0 |

Depth at append (cached tokens kept, i.e. where new tokens start):

| Depth | 0 | 1-1023 | 1024-4095 | 4096-16383 | 16384-89999 | >=90000 |
|---|---|---|---|---|---|---|
| Requests | 66 | 2 | 15 | 60 | 0 | 7 |

Prompt end position: 1-1023: 63; 1024-4095: 15; 4096-16383: 60; 16384-32767: 5; >=90000: 7; nothing between 32K and 90K.

Totals: prompt tokens 1,352,016; reused from cache 1,268,780 (93.8%); processed 83,236 (6.2%). Reprocess events >4K processed tokens: 2 (both full, 16384 tokens each, 71.9 s). Partial reprocess >4K: 0.

Other events: context checkpoints created 228; erased as invalidated 69; "too close" erasures 112; in-task checkpoint restores 1; slot-file restores 18; slot-file saves 49; RAM prompt-cache lookups 130, of which 80 had lcp=0 (no reusable prefix); generated tokens 7,796.

Per container: prod llamacpp-m6000 92 requests (12 Nov 14:08 to 14:23); gdn4-prev 1; night-prev 7; call-prev 33; downx-prev 16; ub1024-prev 1.

## Method
census.py reads `docker logs -t` of llamacpp-m6000 and the -prev rollback containers (all run the same 35B head-quant gguf) plus `journalctl --user -u m6000-llama-swap`. Per task (slot, task id) it takes: new-prompt token count; first "cached n_tokens" as depth at append; print_timing prompt eval / eval / total; release line (truncated flag); checkpoint create/erase/restore lines. Slot-file restores (load_slot_ch lines) are matched to llama-swap POST /slots/N entries within 1.5 s and the same slot to get their duration. Output: census.json. Counts and timings only; no prompt text is read or stored.

## Caveats
- The history is essentially lab benchmark traffic (the 12 Nov prod container log is the 98K/90K two-slot append harness plus 16K cold probes; llama-swap clients are urllib/curl harnesses), not real agentic client load. It confirms the append case (all 7 deep appends resumed from 90000-97996 cached tokens and processed <=5 tokens) but says nothing about organic mix.
- Docker logs only cover each container's lifetime; llamacpp-m6000 ran 14 min before it exited; older traffic is in the -prev containers. 27B containers excluded.
- Wall-share denominator is accounted slot-time, not wall clock; slot overlap not removed.
- Slot-file restore time covers only matched llama-swap calls; save time is not included.
- Prefill class "full reprocess" = >4096 processed tokens with start depth <1024; "deep" = prompt end >64K otherwise; rest shallow.

## Files
census.py, census.json, RESULT.md in this directory.
