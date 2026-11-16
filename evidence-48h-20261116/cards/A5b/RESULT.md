# A5b co-decode hold (LLAMA_N3_SCHED=4): NOT RUN (GPU pending)

| item | value |
|---|---|
| status | built + CPU checks only (bash -n, --plan, synthetic summarize); no GPU run |
| branch / base | a5b-codecode / 97c4ede (a5-sched) |
| default | off (policy 0 = prod behaviour; policy 4 only with LLAMA_N3_SCHED=4) |
| run | `a5b_gpu.sh` (N3_GPU, N3_UNLOAD_URL env), 12 sessions p0/p4 ABBA, ~45 min; A5B_P3=1 adds p3 (18 sessions) |
| accept | peak VRAM <= p0 peak + 100 MiB; worst visible gap <= 10 s (both printed by summarize.py); token identity vs p0 reported |

Design: when exactly one slot generates (no speculation) and the other slot is in prompt processing with <= REACH (4096) tokens left, skip the generating slot's decode row so both slots decode together
(2-row steps). Episode bounded by HOLD ms (150), HOLDB held batches (4), then disabled until the condition clears. If the batch would be empty, the held row is added anyway (fallback counter).
Exit lines: `n3_sched_hold: holds= hold_ms_total= steps_merged= empty_batch_fallbacks= decode_steps_1row= decode_steps_2row=` (holds = held batches; steps_merged = 2-row steps right after a hold).
Env: LLAMA_N3_SCHED_HOLD, _HOLDB, _REACH. Note: held rows change batch composition so tokens may differ from p0 (not a numerics change; identity is reported, not gated).
