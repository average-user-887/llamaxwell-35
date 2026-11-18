# Two-row router weight reuse on Quadro M6000

This opt-in Phase A kernel slowed the full MoE operator in all four comparisons. The independent routes took 3.784 and 2.717 µs longer per call. The routes sharing five of eight experts took 9.824 and 4.973 µs longer. The model escalation threshold was a saving of at least 9 µs in every pair. The experiment ends at the operator.

On each synthetic fixture, the two native rows selected the expected experts in the expected order. Every one of the 4,096 finite output words matched the reference and the repeated candidate. The runtime reported two resident 512-thread blocks per SM across the M6000's 24 SMs. The library preserves the previously measured sparse selector.

Run `python3 -B report.py --report observed` to check both traces against the recorded software identities, synthetic route IDs, dispatch and residency observations, numeric counts, ordered timings and materiality decision. The original traced GPU runs completed in two seconds under a 180-second cap and cleanup succeeded. RELEASE.md contains the frozen protocol. The two .sh.in templates substitute @CUDA_BUILD_SOURCE_MOUNT@ for the original absolute source location. INPUTS.sha256 and FROZEN.sha256 pin the measured build and its dependencies. The launch and build scripts expect the campaign directory structure.
