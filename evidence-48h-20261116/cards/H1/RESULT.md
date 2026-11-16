| item | value |
|---|---|
| GPU status | NOT RUN (CPU-only card); h1_gpu.sh ready, ABBA n=6 prod vs h1 |
| finding | ckpt_create (n=20/replay, 7.5 s sum, max ~2 s) is mostly an implicit ctx->synchronize() in llama_state_seq_get_data_ext waiting on the async no-output prefill decode; real copy = per-tensor pageable DtoH of ~63 MiB (~60 tensors) |
| fix (flag LLAMA_H1_CKPT=1) | batch tensor_get_async into one static pinned host buffer, one sync, memcpy to the same vector: bytes identical, 0 MiB VRAM |
| ledger | new stages ckpt_wait_gpu and ckpt_copy split the wait from the copy (ledger-on only) |
| expected gain | bounded by real copy time (est. ~0.5% makespan); wait portion is unavoidable GPU work, only re-attributed |
| engagement counter | log line H1_CKPT_ENGAGED per checkpoint |
| CPU checks | build 0 warnings in changed files; bash -n; --plan; python parse |
