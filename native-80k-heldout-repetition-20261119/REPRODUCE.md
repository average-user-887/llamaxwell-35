# Reproduction

The public replay checks both terminal results, per-request timing fields, both independent source scores, EOS, normalized repetition, observed witness deltas and the original serial witness failure. It runs on CPU from this directory.

```sh
sha256sum --check --strict SHA256SUMS
python3 -B replay.py
```

The two runner templates preserve the measured source once the owner supplies the private values and sealed manifests. Use a private JSON file containing `N32_FIXTURE`, `N72_FIXTURE`, and `HYBRID_SERIAL` as campaign-relative receipt directories, plus `MODEL_DIRECTORY`, `MODEL_BASENAME` and `MODEL_SHA256` for the pinned model. The rehydrator verifies both reconstructed files against the owner's frozen manifest before creating the new card. It also verifies the sealed input manifest. Both manifests remain owner-local. Place the new card directly under the existing campaign directory with its pinned runtime, source-edit scorers and archived references.

```sh
python3 -B rehydrate.py "$PRIVATE_ALIASES" "$SEALED_INPUTS" "$SEALED_FROZEN" "$NEW_PRIVATE_CARD"
"$NEW_PRIVATE_CARD/release.sh" --check
```

The GPU owner can run the serial and native modes separately under the campaign's M6000 reservation. The launcher needs `N3_GPU` and `ROOT_GPU_DEADLINE_EPOCH` and reserves at most 600 seconds per arm including cleanup. Both replayed edits are open-ended greedy requests with natural EOS and a 3,072-ID cap. Prompt reuse is disabled and the completion receipts must account for every input token. The response text is checked against live detokenization, then the local scorers check expected AST, expected text and EOS. The repetition algorithm collapses whitespace on each nonblank line and replaces digit runs before counting repeats within the full output and final quarter. Repeated 24-word windows provide a second measure.

Use the original owner launcher and sealed receipt directories to re-export the public numerical evidence. The exporter independently reruns the owner comparison and checks all three observation manifests before writing its three typed JSON files.

```sh
python3 -B export.py "$OWNER_LAUNCHER" "$ORIGINAL_FAILURE" "$SERIAL_RECEIPT" "$NATIVE_RECEIPT" "$OWNER_PAIRED_JSON" "$OUTPUT_DIR"
```

The local run needs the original fixture material. The public packet carries its results, numeric diagnostics and stable equality checks.
