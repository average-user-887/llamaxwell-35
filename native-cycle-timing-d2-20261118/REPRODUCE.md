# Portable reproduction

This uses the exact N1 server translation unit and CMake build artifacts from source checkout `da7ba0e49dd43c69f9ee5a219ea3124f7acda0c8`. The regular runtime is the pinned N100 c973 CUDA backend, executable, llama and llama-common, with only the instrumented server implementation library substituted. The build preserved all 3,081 exported dynamic symbol names and types, runtime dependencies, SONAME and RUNPATH. The byte-identical baseline relink and instrumented outputs are pinned in [lineage.json](lineage.json) and [abi.json](abi.json).

Apply the patch to its included preimage in a scratch directory, and copy the patched file to the NVCC-independent host compile location.

```bash
sha256sum -c SHA256SUMS
PACKAGE="$(pwd -P)"
SCRATCH="$(mktemp -d)"
mkdir -p "$SCRATCH/tools/server"
cp server-context.base.cpp "$SCRATCH/tools/server/server-context.cpp"
(cd "$SCRATCH" && git apply --check "$PACKAGE/server-context.patch")
(cd "$SCRATCH" && git apply "$PACKAGE/server-context.patch")
cp "$SCRATCH/tools/server/server-context.cpp" "$SCRATCH/server-context.cpp"
cp rebuild-in-container.sh "$SCRATCH"
sha256sum "$SCRATCH/server-context.cpp"
```

Use the pinned build image and a matching preexisting N1 build tree mounted read-only at `/b`. Mount its full pinned source tree at the source location recorded in its CMake include flags. Set `PINNED_SOURCE` and `PINNED_BUILD` to those host locations, `CMAKE_SOURCE_ROOT` to the configured container path, `BUILD_IMAGE` to the build digest in the lineage file, and `CPU_SET` to two permitted CPU cores. The script reproduces the original relink first, rebuilds one C++ object with the original GCC 14 CMake flags and relinks the same object list against the replaced archive.

```bash
docker run --rm --pull never --runtime runc --network none --cpuset-cpus "$CPU_SET" \
  -u "$(id -u):$(id -g)" -e NVIDIA_VISIBLE_DEVICES=void \
  -v "$PINNED_SOURCE:$CMAKE_SOURCE_ROOT:ro" -v "$PINNED_BUILD:/b:ro" \
  -v "$SCRATCH:/work" --entrypoint /bin/bash "$BUILD_IMAGE" /work/rebuild-in-container.sh
sha256sum "$SCRATCH/original-relink.so" "$SCRATCH/server-context.cpp.o" "$SCRATCH/libllama-server-impl.so"
```

The approved regular launcher supplies the pinned server argv and experimental environment in the runtime receipt. Substitute the rebuilt server implementation at the original library destination and mount a writable timing file at `/native-cycle.tsv`. The opt-in `LLAMA_NATIVE_CYCLE_TIMING=/native-cycle.tsv` emits host-clock events for the one-slot, 80K, depth-two run. A byte-identical replay and exact 3,264-ID check require the original private benchmark input. The release includes public timings and exact source without that input.

Each timing row contains the attempted verifier span and distinct stage intervals in monotonic host microseconds, with begin and end offsets and observed call counts. The gap between attempts is tracked separately. The trace-v2 events retain literal verifier acceptance and state events and mark derived emission reconciliation. The host API timings cannot locate deferred device rollback.
