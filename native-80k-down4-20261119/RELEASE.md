# Matched 80K two-arm runner for the optional four-row verifier

This staged profile adds one runtime change to the existing merged-draft arm and its serial control. Both use the same pinned CUDA DSO b71f011f and GGML_CUDA_OCCAMY_MOE_Q2_DOWN4=1. The 46c4dd70 libllama, 1088044d libllama-common, 4a0ff82d server library and f70e5e1f server executable retain their prior prespecified hashes. Both arms use the existing 80,000-token input, 192 cold outputs and 3,072 warm outputs, one slot, 42/42 offload, all eight routed experts, existing stride/Q8 kernels and original SSE script. Running either arm requires a separate owner decision. The standalone operator previously missed its 13.1 µs/layer gate. This packet contains no new model measurement.

## Witness and terminal validation

The supervisor reads seven fixed counter words from the CUDA image mapped into its own direct child. Source, mapped image identity, executable, symbol sizes, PT_LOAD geometry and CPU fixture are pinned. The new 64-bit down4 counter is incremented by the dispatch function after issuing the actual kernel. It records launch issues. Synchronization and successful requests provide completion evidence at phase boundaries. The counter does not record elapsed device time or individual layer identifiers. Witness snapshots are taken at ready, after cold, and after warm. Both original per-row counters must remain flat and combined launches must equal skipped callbacks. For native, down4, combined and skip deltas must all equal forty times the offered count, independently in both phases. The serial phase deltas must all be zero. Startup events are reported separately in both modes. The existing sparse, stride, merged graph, bounded rollback, shape, all-eight and sampler checks remain active.

The new runner directly checks raw times, cache and acceptance totals, all 3,264 per-mode token IDs and full responses against the original merged native and same-build serial receipts. It computes repetition metrics from each actual response. The final comparison requires two receipts belonging to this release, bound to the same frozen manifest, clean terminal status and an independent replay of both results. It verifies the actual child library identities, flags and command lines. Only the speculation arguments can differ. The result includes same-build native/serial gain, the 100 tok/s gate and the native round difference from the historical merged model. The forced continuation has no semantic score.

## Owner operations

Use the following commands inside this card. The first four are device-free. The offline pair replay uses temporary copies of the archived requests and explicitly synthetic launch counters and runtime identities. Its result is marked CPU_SYNTHETIC_ONLY.

```sh
bash release.sh --check
python3 -B cpu-replay.py
python3 -B build-probe.py --check
bash release.sh --probe native PATH_TO_OWNED_RECEIPT
```

The per-arm owner invocation is N3_GPU=<admitted device> ROOT_GPU_DEADLINE_EPOCH=<lease end> ./release.sh --run serial or --run native. Each arm independently requires the existing Fleetdash admission, campaign lock and full 430-second envelope. Each emits its own BREAK100_MODEL_RECEIPT. After both finish, run ./release.sh --compare SERIAL_RECEIPT NATIVE_RECEIPT. Archived token and text files stay local; the owner can export a cleaned trace using the current campaign process. The validated output carries raw-derived repetition counts.

## Composite profile

The current profile admits only the four-row CUDA change. A future compact host change needs its own card and immutable profile. Copy the source and replace only the approved host overlay mounts, mapped library hashes and the explicit expected-map transformation for the already pinned control. Rebuild or extend the supervisor only for a new approved phase-bound counter and bind its exact ELF, mapped object and byte width. Add the compact feature flag to both typed environment vectors. Require the candidate-only witness in native generation and its absence from serial generation if that is the approved design. If that change also replaces the CUDA DSO, regenerate and test all seven CUDA offsets, data sizes, PT_LOAD and file-map boundaries against the new DSO, then pin its new digest. Recompute the separate freeze and enforce that manifest on both new terminal receipts. Retain the paired no-MTP publication control and all existing exact output and acceptance checks. A filename or environment override cannot substitute a new host DSO into this profile. The existing standalone microbenchmark result neither grants model release nor proves composability.
