# Four-row native verifier at 80K

The isolated four-row MoE down projection produced 3,072 warm outputs in 31,232.535 ms on one Quadro M6000 slot with all eight routed experts. The server reported **98.32695 tok/s** using its 3,071 timed intervals. The campaign's delivered-count rate was **98.35897 tok/s**. Both measures were below 100. This is one observed warm continuation of an 80,000-token input with 79,996 cached tokens and four newly evaluated tokens.

The native cold and warm streams matched their prior native archive on all 192 and 3,072 token IDs, the full text and draft acceptance. The device launch counters observed 4,040 and 65,680 four-row launches. Each equals 40 times the respective 101 and 1,642 proposals. The draft accepted 89 and 1,429. All 42 layers were offloaded. The physical counter records launch issues. Repetition counts come from the actual response. Semantic quality was unscored.

Run `python3 -B replay.py` and `sha256sum -c SHA256SUMS` from this directory. The trace provides numeric host snapshots and stream checkpoints. Native output events enumerate the emitted ordinals. The model responses and tokens remain private. Each trace index records the terminal runner and cleanup statuses.

The [CUDA source and build](../native-q2-down4-20261119/README.md), [clock repeat](../native-q2-down4-clock-20261119/README.md) and [merged-draft base](../native-hybrid-draft-80k-20261119/REPRODUCE.md) supply the other implementation layers. `profile.json` identifies the mapped server and four DSOs. `build-probe.py` checks the CUDA symbols and ELF layout used by the owned-process counter reader.

The source templates restore the private model/receipt references from an owner-supplied JSON mapping. Run `python3 -B hydrate-source.py OUTPUT --private-inputs PRIVATE.json` to reconstruct and hash-check the original code in the campaign layout. See `source-pins.json` for required keys and `FROZEN.sha256` for build pins. `RELEASE.md` is the original pre-run runner document. To regenerate the numeric exports with access to the original campaign and sealed receipts, use `python3 -B export.py CAMPAIGN NATIVE_RECEIPT [--serial SERIAL_RECEIPT]`. The paired export checks the original validators, mapped libraries, active flags and exact argv difference before writing the comparison.

## Same-build serial comparison

One matched pair measured **98.359 native versus 76.277 serial delivered tok/s, a 28.949% warm gain**. The server interval rates were 98.327 and 76.253 tok/s. Warm decode took 31.233 and 40.274 seconds. Both arms passed their respective archived outputs and repetition checks. The first native-versus-serial ID difference occurred at ordinal 170 in both phases. Serial issued zero down4 launches during both measured phases. The five mapped software identities and child flags agreed. The speculation argv was the sole command-line difference.

The [separate compact-overlay shadow refusal](../native-80k-compact-shadow-refusal-20261119/README.md) used a different stack. This isolated pair used the pinned b71f011f CUDA DSO with the original host libraries.
