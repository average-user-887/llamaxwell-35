# Native Q8_0 decode attention feasibility microbenchmark

This is an isolated experiment, not a model-serving integration or production qualification. The worker only compiles and runs the CPU-only layout check. The root owns GPU scheduling and all GPU runs.

## Provenance and scope

- Source: BUNDLE3 commit `2436f7554c0c8d70afc9acbcce85b2457193a516`, `ggml/src/ggml-cuda/fattn-occamy.cu`.
- Baseline source SHA256: `527cb02b6b8935015ff33bef14bf21d7154befa25701d70d95396657d507baef`.
- `prepare.py` checks that SHA before extracting the address-reduced head-split kernel and its combine kernel.
- F16 and Q8_0 use the same query, split, warp, softmax, reduction and output-combine algorithms. Only the K/V loader and associated stride differ.
- Q8_0 is the native 34-byte layout: one FP16 scale followed by 32 signed bytes. Each lane directly loads its eight quantized values and the scale; unpacking is fused into the attention kernel. There is no separate dequantization kernel or expanded floating cache.
- Input K/V are seeded normal FP16 data. Q8_0 is quantized from those exact FP16 values with the baseline reference rule: max-absolute/127 scale, nearest-away-from-zero integer rounding, scale stored as FP16. Queries are seeded normal FP32 values, optionally scaled to stress concentrated softmax.
- Dimensions: 256, 16 query heads, two KV heads, eight GQA heads per KV head, one query token per stream, one or two streams. All cache positions are visible. This does not exercise padding holes, causal multi-query prefill, sinks or nonzero attention bias.

## Build and CPU check

```sh
bash build.sh
./q8-attention-micro --cpu-check
```

The build uses the existing `local/m6000-build-tools:20261013` image, no network, no GPU device, CPUs 9–11 with a two-CPU limit, CUDA 12.8, SM52, and the fast-math setting used by the existing microbenchmarks. Build logs, source hashes and generated source remain in this directory. One initial build failed because the inherited image TMPDIR pointed outside the mount; the corrected build explicitly uses `/tmp`.

## Root-owned GPU runs

Only run after the root's ownership/admission checks. The command itself does not stop or evict anything. It refuses non-M6000/non-SM52 devices and requires the estimated allocation plus 256 MiB free. Select the approved GPU explicitly.

```sh
CUDA_VISIBLE_DEVICES=<GPU_UUID> timeout 180 ./q8-attention-micro 4096 1 3 20 1
CUDA_VISIBLE_DEVICES=<GPU_UUID> timeout 180 ./q8-attention-micro 16384 1 5 50 1
CUDA_VISIBLE_DEVICES=<GPU_UUID> timeout 180 ./q8-attention-micro 65536 1 5 50 1
CUDA_VISIBLE_DEVICES=<GPU_UUID> timeout 180 ./q8-attention-micro 98304 1 5 50 1
CUDA_VISIBLE_DEVICES=<GPU_UUID> timeout 180 ./q8-attention-micro 98304 2 5 50 4
```

Arguments are context length, stream count, ABBA blocks, repetitions per arm, query scale, optional seed, optional split count (`0` means the current two-resident-block Maxwell rule). All results are JSON lines. Capture the full output plus contemporaneous clock, temperature, power, memory and other-workload state.

## Interpretation

Each format must agree with a CPU-double softmax reference computed from that format's own cache values at 128 output positions per stream. The maximum allowed reference error is `1e-4`. The program reports F16-versus-Q8 full-output max error, RMSE, relative L2 and cosine separately; these are quantization observations, not model-quality acceptance criteria.

CUDA-event measurements include both decode and combine kernels and exclude dataset creation, transfers and CPU validation. Five ABBA blocks are interleaved in one process, not five independent deployment trials. Treat clock drift, warm caches, synthetic input distributions and isolated kernel compilation as limitations. The useful payload does not include duplicated head-group loads or count measured DRAM transactions.

A win here supports a narrow integrated experiment. It does not establish whole-model tokens/s, prompt-processing speed, KV write/quantization cost, long-context retrieval quality, agent correctness or memory-safe production operation. A loss only rejects this specific loader and layout implementation, not compressed KV as a whole.
