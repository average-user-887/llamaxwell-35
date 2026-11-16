# M1 RESULT: MTP (-mtp GGUF, draft-mtp n-max 2, np 2, ub2048 cap1024, F16 KV, batch8 libs)

| per-slot ctx (tokens) | variant | load+alloc | ctxfull (400, health, slot1) | 300-tok dual-slot gen | VRAM peak MiB (1 Hz) |
|---|---|---|---|---|---|
| 98304 | draft KV f16, n-max 2 | FAIL (compute buf 544 MiB OOM) | - | - | 24021 |
| 98304 | draft KV q8_0 | FAIL (alloc OOM) | - | - | n/a |
| 98304 | n-max 1 | FAIL (alloc OOM) | - | - | n/a |
| 96000 | f16, n-max 2 | FAIL (alloc OOM) | - | - | 23922 |
| 91904 | f16, n-max 2 | FAIL (alloc OOM) | - | - | 23746 |
| 88064 | f16, n-max 2 | ok | pass | FAIL: CUDA OOM (cuMemCreate) during prefill/decode, server abort | 24444 |
| 84224 | f16, n-max 2 | ok | pass | FAIL: same | 24417 |
| 80128 | f16, n-max 2 | ok | pass | FAIL: same (OOM at ~3.8K tokens into prefill) | 24209 |

Per-slot drafted/accepted: NOT MEASURED, 300-token dual-slot run never completed at any rung (server aborts). Both slots drafted>0: NO DATA (void).
Only drafting seen: 8-token probe on slot 1 (88064): 7 drafted, 2 accepted, acc/pos 0.50, 0.00. No slot-0 data.

Pass/fail: alloc ladder: ctxfull passes from 88064 down, but real generation OOMs even at 80K (floor), so MTP does NOT fit at >=80K with these levers (D1 Stop condition on memory, before extra levers).
Per-buffer MiB at 88064 (target ctx): model CUDA0 18866.77 (+272.81 CPU), KV 3440.00, RS 376.88, compute 776.08 (+408.09 host);
MTP ctx ("creating MTP draft context against the target model"): KV 344.00, compute 504.06 (+376.07 host).
Failing alloc at 98304: 544 MiB compute buffer (570491008 B). Runtime OOM site: ggml-cuda.cu:585 cuMemCreate (VMM pool grow) during process_ubatch.

## Method
m1.sh (ladder; K = tokens: 98304, 96000, 91904, 88064 ...; first-fail rung 98304 repeated with q8 draft KV and n-max 1), m1b.sh (88064 gen retry, error capture), m1c.sh (84224, 80128). Each holds gpu-bench.lock (exec 9, flock, trap cleanup), POST unload qwen36-35b-moe (prod already unloaded), containers nm1-*, all removed; lock free. GPU via N3_GPU env. 250 W untouched, MAXWELL_GRAPHS unset. ctxfull = 434K-token prompt on slot 0 -> expect 400, then /health and slot-1 probe. Gen = two concurrent /completion, 300 tokens, ignore_eos, temp 0, ids a/b from PLAN/HANDBOOK text.
## Caveats
Ladder rungs are 1000-based approximations rounded to 256 (98304 = prod "98K"). --cache-ram 0 (prod 12288), ctx-checkpoints 16 kept (70 MiB each, device? not checked). Untried levers: fewer checkpoints, ub1024, cap smaller, q8 draft KV at 88064 and below, lower target KV (Q8). VRAM peak is 1 Hz nvidia-smi incl. crash.
## Files
run-20261115T194706Z (ladder, 98304 variants), runb-20261115T212017Z, runc-* ; each pNNN/server.log, vram.csv, ctxfull.json, gen.out.
