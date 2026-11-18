# Portable reproduction

The publication parent is `f3cd901e828d73f2d52bed4c23e7b8b2cfcd4747`. The source checkout used `da7ba0e49dd43c69f9ee5a219ea3124f7acda0c8` with the pinned Sparse R2 build and N98 attention object. The two patches apply to the included actual preimages. The c973 control and exact unmodified relink share SHA-256 `c973955667f1bb7e21a33447da488f182aa6fee0437d1e275180ea5b9d3cb9c5`. The compile and runtime hashes are in the operator receipt.

```bash
sha256sum -c SHA256SUMS
PACKAGE="$(pwd -P)"
WORK="$(mktemp -d)"
mkdir -p "$WORK/ggml/src/ggml-cuda"
cp occamy-fused.base.cu "$WORK/ggml/src/ggml-cuda/occamy-fused.cu"
cp operator-base.cpp "$WORK/operator.cpp"
(cd "$WORK" && git apply --check "$PACKAGE/candidate.patch" "$PACKAGE/operator.patch")
(cd "$WORK" && git apply "$PACKAGE/candidate.patch" "$PACKAGE/operator.patch")
cmp operator.cpp "$WORK/operator.cpp"
```

Recompile only the patched `occamy-fused.cu` in a scratch copy of the pinned c973 source and build. Use its exact NVCC compile invocation for sm_52 and relink its original object list, including the pinned N98 attention object. Re-linking the original MoE object verifies the c973 SHA-256 before comparing the device instructions of the replacement with `cuobjdump -sass`.

```bash
g++ -std=c++17 -O3 -Wall -Wextra -Werror \
  -I"$LLAMA_SOURCE/ggml/include" operator.cpp -L"$C973_LIBS" \
  -lggml -lggml-base -ldl -o operator
LD_LIBRARY_PATH="$C973_LIBS" ./operator --fixture
```

Set `RUNTIME` to the rebuilt CUDA module and its c973 companion libraries. Use the admitted device, isolated container image and two allowed CPU cores. The host operator checks its exact container mappings and actual device occupancy. Run its two phases independently.

```bash
for MODE in --check --perf; do
  docker run --rm --pull never --network none --cpuset-cpus "$CPU_SET" \
    --device "$CUDA_DEVICE" -v "$RUNTIME:/runtime:ro" \
    -v "$PWD/operator:/operator:ro" -e LD_LIBRARY_PATH=/runtime \
    -e GGML_CUDA_OCCAMY_MOE_Q3_COHORTS=1 \
    -e GGML_CUDA_MOE_DOWNX=1 -e GGML_CUDA_MOE_DOWNH=2 \
    -e GGML_CUDA_MMVQ_SM52=1 -e GGML_CUDA_OCCAMY_FUSED=1 \
    -e GGML_CUDA_MOE_LOP3=1 -e GGML_CUDA_MOE_TAILBAL=1 \
    -e GGML_CUDA_OCCAMY_HOSTGAP=1 -e GGML_CUDA_OCCAMY_SMALLFUSE=2 \
    -e GGML_CUDA_OCCAMY_N2SPLIT=1 -e GGML_CUDA_MOE_V3=1 \
    --entrypoint /operator "$IMAGE" "$MODE"
done
```

The serial receipts preserve the runtime, no-spec profile, timing mode, cache reuse and typed request evidence. Byte-identical replay of the serial benchmark requires its private input.
