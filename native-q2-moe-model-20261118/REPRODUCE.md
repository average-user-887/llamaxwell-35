# Reproducing the 80K experiment

The [two-row standalone operator](../native-q2-moe-cohorts-20261118) contains the complete pinned CUDA implementation, compiler commands and synthetic fixture receipts. Use its exact candidate and the [original full-eight Q8A native benchmark](../gdn-q8-warm80k-20261118) with the [matching serial collector](../native-q8-on-serial-20261118). Both model arms use the same admitted single-slot Maxwell runtime, an 80,000-token prompt, 192 cold output tokens and 3,072 cached output tokens. Warm requests evaluate four uncached prompt tokens. The original private prompt, reference ID streams and reference texts are required to reproduce the parity gate.

Run the model packet check on any host with Python. It checks every listed file, the immutable runtime and release source fingerprints, the original early refusal, and all available physical model records.

```sh
sha256sum -c SHA256SUMS
python3 -B verify-package.py
```

To recreate the local launcher, copy the listed model collector, ELF verifier and supervisor source into an owner-approved campaign checkout with the pinned routing, stream collector and benchmark inputs verified against INPUTS.sha256. Build the CPU supervisor using the original compiler command, retaining its exact binary identity where bit-identical reproduction is required. The initial launcher and one-line health budget recovery are both retained. Initial and recovery frozen manifests include the private-layout benchmark dependencies. The portable offline regression swaps its single private reference location for the D1_SERIAL_REPLAY environment variable. Some original campaign components bind additional private inputs; these must be supplied by their authorized holder. The public package excludes these inputs and all captured model output.

```sh
python3 -B hydrate-private-source.py "$CAMPAIGN" "$N88_PRIVATE_REFERENCE" "$D1_SERIAL_REPLAY" "$HYDRATED_DIR"
# Install hydrated offline.py in the card and serial_probe.py in the pinned serial-collector directory.
gcc -std=c11 -O2 -Wall -Wextra -Werror supervise.c -o supervise
cp initial.release.sh release.sh
patch release.sh < recovery.patch
chmod 700 release.sh
cp recovery.FROZEN.sha256 FROZEN.sha256
# In the original pinned layout use the original offline.py for the frozen hash check.
# The distributed offline.py can be run separately with D1_SERIAL_REPLAY set.
./release.sh --check
./release.sh --run native
./release.sh --run serial
./release.sh --compare "$SERIAL_RECEIPT" "$NATIVE_RECEIPT"
```

Only the benchmark owner launches runs through the existing admission/cleanup wrapper. In this experiment the serial release depends on the native arm's terminal condition. Individual arms have the same 430-second lifetime and 24 MiB bounded source log. The public collector records only typed events, quarter checkpoints, cumulative host measurements when available, shape/counter evidence and numerical summaries.

```sh
python3 -B export-outcome.py "$INITIAL_RECEIPT" native-r0 "$OUT"
python3 -B export-outcome.py "$NATIVE_RECEIPT" native-r1 "$OUT"
# When physically run
python3 -B export-outcome.py "$SERIAL_RECEIPT" serial-r1 "$OUT"
```
