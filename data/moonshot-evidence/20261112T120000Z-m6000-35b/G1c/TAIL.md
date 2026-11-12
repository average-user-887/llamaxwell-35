# G1c TAIL: n_kv mod 4096 tag, verified from source (CPU only)
Source tree: `evidence/G1b/src` (same llama.cpp as the G1 server; G1 A `server.log` shows `n_rs_seq = 0`).

## Formula
1. Server prompt batching, `tools/server/server-context.cpp:3952-4016`: batches of up to n_batch=2048 tokens; `checkpoint_offsets[] = {4 + n_ubatch, 4}` (`:4006`) force a batch end at N-1028 and N-4. A prompt cached c -> N is therefore 2048-batches, a break at N-1028, a break at N-4, a final 4-token batch [N-4, N). Mixed-batch cap `:3502-3504` (only with `--mixed-batch-size`, off in G1).
2. Batch -> ubatch: `src/llama-memory-hybrid.cpp:89` `split_equal(n_ubatch, ..., n_rs_seq>0 ? n_rs_seq+1 : 0)`. With `n_rs_seq=0` (default; clamp `src/llama-context.cpp:108-110`) there is NO tail-keep rule: plain 1024-token splits.
   G1's `nkv()` rule `if n<64: e-=n` is not source-backed (harmless in the G1 windows, see below). Confirmed against G1 A log: checkpoints at 15356 and 16380 for N=16384.
3. n_kv, `src/llama-kv-cache.cpp:1280-1292`: `n_kv = GGML_PAD(used_max_p1, 256)`, evaluated after the ubatch is applied (`llama_kv_cache_context::apply`, ~2855-2867), so it includes the ubatch's own tokens: **n_kv = PAD256(max_pos_in_ubatch + 1)**.
4. FA kernel dispatch uses `ntiles_KV = ceil(n_kv / nbatch_fa)` (`ggml/src/ggml-cuda/fattn-common.cuh:1182`; parallel_blocks heuristic `:1213-1265`, MULT env `:39`). Stride `FATTN_KQ_STRIDE=256` (`:11`). Tag = n_kv mod 4096.

## Corrected tags per G1 window (G1 rows: prompt = tokens[:N+k], one new token at pos N+k-1; `tail_calc.py`)
| N | last staged ubatches (start,end) | row 0 n_kv / tag | rows 1-15 n_kv / tag | G1 label | verdict |
|---|---|---|---|---|---|
| 16384 | (14336,15356) (15356,16380) (16380,16384) | 16384 / 0 | 16640 / 256 | 0 | row 0 right, rows 1-15 mislabelled |
| 32768 | (30720,31740) (31740,32764) (32764,32768) | 32768 / 0 | 33024 / 256 | 0 | same; tail 0 is correct arithmetic for row 0 only |
| 65536 | (63488,64508) (64508,65532) (65532,65536) | 65536 / 0 | 65792 / 256 | 0 | same |
| 65836 (codexs) | (63788,64808) (64808,65832) (65832,65836) | 66048 / 512 | 66048 / 512 | 512 | right for all rows |
So the 32K "tail 0" is not wrong, but it holds only for row 0 (1 of 16 rows); 15 of 16 rows ran with tag 256. The 16 rows of a window thus mix two tags, and "valid KV length mod 4096" (N mod 4096) is a different quantity from the kernel's n_kv tag.

## G1c rows
Teacher-forced single-token decodes at pos B..B+63 (B = 32768, 65536, 98304; ubatch 1024 prefill ends exactly at B). Every row has n_kv = PAD256(pos+1) = B+256, tag 256, so all 192 rows of a doc share one kernel tag per bucket (3 buckets x tag 256). Deliberate: tag is controlled, not mixed. A second tag would need B+192 offsets; not planned.
