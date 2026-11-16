# M1d RESULT: lever ladder for np 2 draft-mtp (n-max 2), 300-token gen on both slots at once

Only passing arms (gen ok, both slots drafted>0), per-slot ctx 84224 (tokens):
| arm | levers (cumulative) | slot0 drafted/accepted | slot1 drafted/accepted | decode t/s s0 / s1 | VRAM peak MiB |
|---|---|---|---|---|---|
| L3_84224 | q8 draft KV + ckpt 2 + **ub1024** + cache-ram 0 | 266/165 (0.62) | 257/170 (0.66) | 37.5 / 29.4 | 24313 |
| L4_84224 | L3 + cache-ram 12288 | 266/165 | 257/170 | 38.2 / 29.5 | 24313 |

Best rung: 84224 per slot (2 x 84224 = 168448). Only with ub1024 (prefill speed changes, label it; ub2048 is prod). VRAM peak 24313 MiB is <= 24576-250 = 24326 MiB but > 23750 MiB (the strict 23.75 GB figure), so the stricter bound is NOT met; peak is 1 Hz nvidia-smi.

Fail map (gen result; ctxfull passed wherever the arm loaded):
| lever | 98304 | 96000 | 91904 | 88064 | 84224 | 80128 |
|---|---|---|---|---|---|---|
| L1 q8 draft KV (ckpt16 ub2048) | - | - | alloc fail | alloc fail | gen OOM | gen OOM |
| L2a +ckpt 4 | - | - | alloc fail | alloc fail | gen OOM | gen OOM |
| L2b +ckpt 2 | - | - | alloc fail | alloc fail | gen OOM | gen OOM |
| L3 +ub1024 | alloc fail | gen OOM | gen OOM | gen OOM | PASS | not run (stopped at pass) |
| L4 +cache-ram 12288 | alloc fail | gen OOM | gen OOM | gen OOM | PASS | not run |
(L1/L2 at 91904 and 88064 died at load with peak 18986 or less: alloc OOM; earlier f16 runs M1 showed 88064 loads but gen OOM.)

Findings: q8 draft KV and ctx-checkpoints (host memory) do not help; ub1024 is the lever that matters. ckpt count and cache-ram have no effect on VRAM or pass/fail.
L5: no MTP-ctx batch flag in --help (only spec-draft ngl/device/cpu flags); not tried. L6: ggml-cuda.cu:585 is the VMM pool grow in the stock pool; no NO_VMM or pre-reserve env in the source (only GGML_CUDA_ENABLE_UNIFIED_MEMORY, not tried); not tried, no code edits.
Not tried: ub512; smaller cap with ub2048; lower target KV; 80128 with ub1024 (unnecessary, 84224 passes).
Acceptance at 84224 is ~0.62-0.66 (drafted 257-266 of 300 tokens), above the 0.25 D1 bar.

## Method
m1d.sh (detached, holds gpu-bench.lock, trap cleanup, containers nm1d-*, all removed, POST unload only, 250 W untouched). Gen = two concurrent /completion, 300 tokens, ignore_eos, temp 0, 3.5K-token real text prompts (PLAN/HANDBOOK). Pass = both slots 300 tokens, drafted>0, peak <= 23750 (strict rule would have recorded none; gen-ok arms listed anyway).
## Files
M1/rund-*/arms.txt, steps.log, per-arm dirs with server.log, vram.csv, gen.json, gen.out, ctxfull.json.
