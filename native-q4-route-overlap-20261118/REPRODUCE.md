# Portable reproduction

The [combined model package](../native-q4-combined-model-20261118) records the unchanged abe7 target runtime and the original attention-only 80K reference. Use the matching N1 llama and GGML headers. The actual observer hashes, source pins and build flags are in [lineage.json](lineage.json).

```bash
sha256sum -c SHA256SUMS
PACKAGE="$(pwd -P)"
WORK="$(mktemp -d)"
cp audit.base.cpp "$WORK/audit.cpp"
(cd "$WORK" && git apply --check "$PACKAGE/audit-r0.patch")
(cd "$WORK" && git apply "$PACKAGE/audit-r0.patch")
cmp audit-r0.cpp "$WORK/audit.cpp"
(cd "$WORK" && git apply --check "$PACKAGE/audit-r1.patch")
(cd "$WORK" && git apply "$PACKAGE/audit-r1.patch")
cmp audit-r1.cpp "$WORK/audit.cpp"
g++ -O2 -shared -fPIC -std=c++17 -Wall -Wextra -Werror -I "$LLAMA_SOURCE/include" -I "$LLAMA_SOURCE/ggml/include" -I "$LLAMA_SOURCE/src" audit-r1.cpp -ldl -o audit.so
LLAMA_SOURCE="$LLAMA_SOURCE" bash offline-check.sh
```

The device-free fixture exercises a temporary fit context, a distinct main target and a separate draft context. It verifies actual dynamic interposition, padded I32 ID strides, temporary callback placement and removal, and count-only collection. Set `Q4_COMBINED_CARD` to the reconstructed combined card and `Q4_ROUTE_PRIVATE_REFERENCE` to the private directory of the original attention-only control for the typed guard. These two input locations are caller-supplied in the portable guard.

The actual route arm used the original owner's bounded one-slot launcher, mapped R1 observer, exact attention-only settings, and an owner-private counter file at `/native-route-counters`. The diagnostic compared against its immutable cold and warm reference. For an independently retained private receipt, use the exact parity check below. Only the pass/fail fields, checked counts and first divergent position are written to standard output.

```bash
python3 -B -c 'import json, sys, pathlib, guard; print(json.dumps(guard.parity(pathlib.Path(sys.argv[1])), indent=2))' "$PRIVATE_OBSERVATION"
```

The two complete public traces provide chronology without the private prompt, generated output or route payload. Both original technical failures are retained.
