# Reproduction

The public CPU replay checks interval ordering, physical software equality, output counts, timestamp correction, the 46 ranked target rows, decision mapping and both signs of the score reversal. Run the following in this packet.

```sh
sha256sum --check --strict SHA256SUMS
python3 -B replay.py
```

The owner replay also needs the sealed 80K fixture, original validators, frozen manifests, pinned runtime and complete private receipts. The adjacent native-hybrid-draft-80k-20261119 package documents the parent runtime and model launch. The four templates in this packet reproduce the actual two launchers and validators. Supply a private JSON alias file with six string fields. HYBRID_NATIVE and HYBRID_SERIAL are the original hybrid receipt basenames. SERIAL_SCORE is the timestamp-refused serial score receipt basename. MODEL_DIRECTORY is the model directory basename, MODEL_BASENAME the model filename and MODEL_SHA256 its sealed digest. The rehydrator validates the reconstructed byte streams against the original source digests and creates a new directory.

```sh
python3 -B rehydrate.py "$PRIVATE_ALIASES" "$NEW_PRIVATE_TREE"
```

In the prepared owner campaign layout, quality80-divergence/release.sh --run serial writes the serial score receipt. The R1 collector binds the archived serial receipt from the completed inference and its original timestamp-parser refusal. Use quality80-divergence-r1/release.sh --run native for native MTP1. Both launchers require the owner’s single-device lease and cutoff, and check the original hardware and runtime contract. Numeric pre-increment indices 160 through 170 select the bounded observer window. The logger writes log-softmax values rounded to five decimals and records the IDs actually returned by the sampler. The private cold-end byte boundary splits the append-only dump into phases. R0 and R1 check each committed diagnostic ID against the actual response, then check the entire response against its same-mode archive. A fresh private fixture requires new references.

Run the exporter on copies of the packet scripts in a private output workspace to regenerate the public four-file numerical evidence from the owner receipts. It writes alongside its source and prints no tokens or prompt material.

```sh
python3 -B export.py "$CAMPAIGN" "$ORIGINAL_SERIAL_SCORE_RECEIPT" "$NATIVE_SCORE_RECEIPT"
python3 -B replay.py
```

The observable window contained 11 serial and six native rounds per phase. Each serial round provided one target row and each native round provided two. Every published row reports its ordered top eight. Only the two choices at ordinal 170 receive stable labels. Each other candidate uses the label other independently at every rank. An uncommitted row has no emitted ordinal. An absent choice in the top eight has an unknown exact rank and margin. The displayed top-score differences use the original rendering precision.
