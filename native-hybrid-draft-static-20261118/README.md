# One-graph native draft overlay, CPU validation

The two historical N40 draft libraries passed the static compatibility and CPU validator checks against the exact current split run. **Hold the GPU trial.** The clean current run reached 96.44709 tok/s over 3,072 cached output tokens. It needs 0.68920 ms less per offered draft to reach 100 tok/s at the same yield. The separate historical N40 comparison saved 0.16052 ms per offer on a 192-token cached request and added 0.41705 s to cold request wall. Transferring that entire saving would give about 97.25 tok/s. The historical and current outputs first differed at token 27 on the same prompt.

## Source and checks

[Stage code](stage.py) verifies the historical N40 source parent, build products and static ELF interface, and derives an isolated runner from the [corrected finish33 release](../native-finish33-sampler-model-20261118/REPRODUCE.md). It changes the two draft library bindings and the two coordinated draft settings. The CUDA library, server implementation, executable, supervisor, target route and 430-second cap retain their existing pins.

[Model binder](model.py) extends the original validation to require the alternate mapped libraries, observed merge enablement and actual graph marker. The original route and counter checks still run. A future measured arm must reproduce the clean split baseline's 3,264 output IDs, both full texts, and offer, acceptance and cache totals.

[CPU evidence](evidence.json) records safe source and library pins, preserved dependency and symbol metadata, and the executed checks. The [offline check](offline-check.py) runs on a temporary copy of the verified split receipt with synthetic alternate binding fields. It accepted the expected maps and graph and refused an old mapped library and a wrong query width. The [counterbalanced split baseline](../native-finish33-repeat-20261118/) is the current measurement. The older GPU timing is a separate comparison. Live hybrid qualification is unavailable.

## Reproduce the CPU checks

Use a complete owner-held campaign containing the corrected finish33 build and the archived N1 and N40 inputs. The original N40 libraries, raw measurements, and private model and prompt remain with the owner. Copy the Python sources into a new sibling directory named hundred-hybrid-draft. Supply OWNER_NATIVE_TERMINAL_RECEIPT as the clean split native receipt from the counterbalanced repeat.

    sha256sum --check --strict SHA256SUMS
    mkdir OWNER_CAMPAIGN/hundred-hybrid-draft
    cp stage.py model.py offline-check.py OWNER_CAMPAIGN/hundred-hybrid-draft/
    cd OWNER_CAMPAIGN/hundred-hybrid-draft
    python3 -B stage.py OWNER_NATIVE_TERMINAL_RECEIPT
    sha256sum --check --strict FROZEN.sha256
    sha256sum --check --strict INPUTS.sha256
    python3 -B check-layout.py
    python3 -B offline-check.py

The generated local binding follows the [binding schema](binding.schema.json). The staging check validates the exact archive before producing the runner.
