# Portable reproduction

The source checkout used `da7ba0e49dd43c69f9ee5a219ea3124f7acda0c8` with the pinned Sparse R2 compile tree, N98 attention object and the [preceding Q3 MoE build](../native-mtp-resume-first-20261118). The Q3 control and exact unmodified relink share SHA-256 `30ec9a72d6f938656f331b91eb75266e73268bf2015a0d3dd2867eeb427dd58b`. The benchmark used the new Q4 MoE standalone CUDA DSO `a68cf344d6c810cd8f9ca8bf46a7fcf8a0b2f466d660daa21ed576ef96da44b2`.

```bash
sha256sum -c SHA256SUMS
PACKAGE="$(pwd -P)"
WORK="$(mktemp -d)"
cp occamy-fused.base.cu "$WORK/occamy-fused.cu"
cp operator.base.cpp "$WORK/operator.cpp"
(cd "$WORK" && git apply --check "$PACKAGE/candidate.patch" "$PACKAGE/operator.patch")
(cd "$WORK" && git apply "$PACKAGE/candidate.patch" "$PACKAGE/operator.patch")
cmp occamy-fused.cu "$WORK/occamy-fused.cu"
cmp operator.cpp "$WORK/operator.cpp"
python3 -B report.py --check
```

Rebuild only the patched `occamy-fused.cu` with the pinned Q3 build's exact sm_52 NVCC invocation, and relink against that build's original object list and dependencies. First relink the untouched Q3 object to check the frozen control hash. Inspect both CUDA modules with `cuobjdump -sass` to verify the existing device instructions. The new symbol is `_ZN6occamy17moe_v3_q4_cohortsENS_6ParamsE`.

Compile the host operator against the pinned GGML headers and companion libraries, and verify the fixture before booking a GPU run.

```bash
g++ -std=c++17 -O3 -Wall -Wextra -Werror \
  -I"$LLAMA_SOURCE/ggml/include" operator.cpp -L"$RUNTIME" \
  -lggml -lggml-base -ldl -o operator
LD_LIBRARY_PATH="$RUNTIME" ./operator --fixture
```

Set `RUNTIME` to the exact Q4 MoE module and its companion GGML libraries. Use the admitted device, pinned image and two allowed CPU cores. The harness sets the candidate flag dynamically for the relevant graph and validates the physical counters.

```bash
for MODE in check perf; do
  DEBUG=0
  CHECKS=0
  if [[ "$MODE" = check ]]; then DEBUG=1; CHECKS=8; fi
  docker run --rm --pull never --network none --cpuset-cpus "$CPU_SET" \
    --device "$CUDA_DEVICE" -v "$RUNTIME:/runtime:ro" \
    -v "$PWD/operator:/operator:ro" -e LD_LIBRARY_PATH=/runtime \
    -e GGML_CUDA_DISABLE_GRAPHS=1 -e GGML_CUDA_MOE_DOWNX=1 \
    -e GGML_CUDA_MOE_DOWNH=2 -e GGML_CUDA_MMVQ_SM52=1 \
    -e GGML_CUDA_OCCAMY_FUSED=1 -e GGML_CUDA_MOE_LOP3=1 \
    -e GGML_CUDA_MOE_TAILBAL=1 -e GGML_CUDA_OCCAMY_HOSTGAP=1 \
    -e GGML_CUDA_OCCAMY_SMALLFUSE=2 -e GGML_CUDA_OCCAMY_N2SPLIT=1 \
    -e GGML_CUDA_MOE_V3=1 -e GGML_CUDA_OCCAMY_DEBUG="$DEBUG" \
    -e GGML_CUDA_MOE_V3_CHECK="$CHECKS" --entrypoint /bin/sh "$IMAGE" -ec '
      sha256sum /runtime/libggml.so.0.26.0 /runtime/libggml-base.so.0.26.0 /runtime/libggml-cuda.so /operator
      exec nice -n 19 /operator "$1"
    ' operator "--$MODE" > "$MODE.log" 2>&1
done
```

The parser expects those two log basenames plus `RUNTIME.sha256` and `FROZEN.sha256` inventories for the three mapped libraries and executable. The names and measured digests are in [lineage.json](lineage.json). Run the checks with `python3 -B report.py --numeric MEASUREMENT_DIRECTORY` and `python3 -B report.py --report MEASUREMENT_DIRECTORY`. These files reproduce the synthesized operator fixtures and protocol.
