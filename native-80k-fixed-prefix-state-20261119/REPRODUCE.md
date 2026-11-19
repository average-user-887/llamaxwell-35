# Reproduce the portable check

The measured result and witness can be checked offline with Python 3. These commands do not load the model.

```sh
sha256sum --check SHA256SUMS
python3 -B replay.py
python3 -B check-report.py
```

The result replay uses the exact original summarizer. It reads the anonymous typed witness, restores the announced numerical results and verifies graph coverage. The synthetic check exercises deliberate graph, rollback and cleanup failures. Replaying the portable witness performs no model inference.

To build the observer offline, use the compiler image and exact link library builds identified in `runtime.json`. Provide the pinned N40 llama.cpp and GGML include directories and compatible Llama/GGML link library directory to `build.sh`. The build uses g++-14 and forwards the existing graph submit, sparse attention and GDN dispatch entries. The CPU-only self-test exercises the byte reader and row comparison without opening a model.

A physical replay requires the compatible Quadro M6000, the same-runtime GGUF with all eight experts, the pinned CUDA and core libraries, and the exact private fixture. The observer relies on the runtime’s fixed `/candidate` library mount, with any internal links resolving to the pinned GGML and N40 builds. The native common-library build was verified as provenance before execution. The observer’s loaded maps covered the four target core and CUDA libraries listed in `runtime.json`.

The owner supplies the approved isolated single-device lease and runs the compiled helper in the pinned runtime image with `environment.args`. The helper takes the local model filename and an owner-provided private tape filename as two arguments. The tape has 80,171 integer lines. It contains the 80,000 prompt IDs, the first 169 shared emitted IDs, then the original serial and native choices at decision 170. Before staging it, verify the prompt identity, the shared emitted prefix and the choice-to-row attribution against the private archive. The observer implements the prefill boundaries and uses one host-memory state snapshot, restoring it into each of four sequential fresh target contexts. The reference release used a 430-second cap including cleanup. Physical witnesses and numerical outcomes must be measured anew on a new run.

The exact private fixture stays with its operator. A different tape exercises the harness and yields evidence only about that tape.
