# Portable reproduction

The evidence-parcel parent is `8fae40f4c93d743087269425a5bab0ee040d860a`. The underlying Occamy checkout was `da7ba0e49dd43c69f9ee5a219ea3124f7acda0c8` with the pinned N1/R2 runtime and three-row R2 attention. Each receipt gives hashes for the five actual runtime objects. The shared sources are exact copies of the two measured harnesses. They expect the CUDA backend, model libraries and executable-relative imports to resolve through the container mount `/candidate`.

First check the package and correction.

```sh
sha256sum -c SHA256SUMS
python3 -B verify-public.py
patch --dry-run --silent -o /dev/null state.original.cpp serializer-fix.patch
```

Use the pinned CPU-only builder image `sha256:747c8a878fc2a36aca96e10da800f6cfe80d1e813c7e4af49ded30588be16bed` with the checkout's llama headers at `/llama`, ggml headers at `/ggml`, and the pinned library directory at `/candidate`. Run from a writable copy of the package.

```sh
export LD_LIBRARY_PATH=/candidate
for revision in original corrected; do
  cp "state.$revision.cpp" state.cpp
  g++-14 -std=c++17 -O2 -Wall -Wextra -Werror -I/llama -I/ggml state.cpp \
    -L/candidate -Wl,-rpath-link,/candidate \
    -Wl,--export-dynamic-symbol=ggml_backend_sched_graph_compute_async \
    '-Wl,--export-dynamic-symbol=_Z23ggml_cuda_occamy_gdn_t2*' \
    '-Wl,--export-dynamic-symbol=_Z28ggml_cuda_op_gated_delta_net*' \
    '-Wl,--export-dynamic-symbol=_Z40ggml_cuda_op_gated_delta_net_fused_cache*' \
    -lllama -lggml -lggml-base -ldl -o "state.$revision"
  "./state.$revision" --self-test
done
```

For a physical replay use the already-admitted exclusive GPU slot, a 190-second all-in bound for each revision, the experiment runtime image from the receipt and its exact mapped libraries. Load `runtime.env` as a container environment file. Provide a private Occamy 35B Q4_0 headq40b MTP GGUF, 2,000 input IDs, 192 continuation IDs, and the three prefill boundaries 972, 1996 and 2000. Each ID file is whitespace-separated decimal integers. A different local fixture can exercise the rejection mechanism, but it can produce different results and must cause the two artificial rejected first proposals to disagree with the target. A fresh writable private output directory is required for each run.

Inside the admitted container, with `MODEL`, `PROMPT_IDS`, `ORACLE_IDS`, `PREFILL_ENDS` and `OUT` set to the operator's own mounted paths, the original attempt runs the first wide arm. Its recorded outcome was exit 2 during the first full/partial comparison. Keep raw stdout and stderr private.

```sh
mkdir -p "$OUT"
./state.original wide_a "$MODEL" "$PROMPT_IDS" "$ORACLE_IDS" "$PREFILL_ENDS" "$OUT" \
  > "$OUT/wide_a.rows" 2> "$OUT/wide_a.log"
```

For the corrected revision, set `OUT` to a fresh directory. The first wide arm writes the correction used by the other two. Each executable invocation reloads and re-prefills the model.

```sh
set -eu
mkdir -p "$OUT"
for arm in wide_a wide_b serial; do
  ./state.corrected "$arm" "$MODEL" "$PROMPT_IDS" "$ORACLE_IDS" "$PREFILL_ENDS" "$OUT" \
    > "$OUT/$arm.rows" 2> "$OUT/$arm.log"
done
python3 -B compare-private.py "$OUT" > anonymous-comparisons.json
```

With the matching private inputs and raw outputs, `python3 -B compare-private.py "$OUT" corrected.receipt.json` recomputes all 240 per-layer summaries, four attention-KV comparisons, four logit-row comparisons and the subsequent-choice comparisons. This was checked against the actual archived raw arrays. The numerical comparator requires NumPy and Python 3.11 or newer. Private inputs and outputs are supplied by the reproducer.
