# Compact verifier output on a Quadro M6000

The standalone physical screen passed the numerical and route checks and saved **431.103–432.290 µs per offer** in four order and mask comparisons weighted by the historical acceptance split. The standalone 550 µs gate missed. The smallest saving exceeds the 202 µs residual projected from the independent four-output Phase C operator. A native model run is required to measure their joint effect.

Both arms used the actual pinned N40 common sampling code and synthetic device rows. The raw arm copied 1,986,560 bytes. The compact arm copied 16 bytes in eligible rounds. Seven edge and off-path checks exercised the exact live-row fallback. The working set rotated through 127,139,840 bytes. The runner and cleanup each exited zero. The build includes the failed compiler invocations and subsequent successful compile. This assay measured the device transfer and host acceptance path without a model or an 80K prompt.

Verify the public payload with `sha256sum -c SHA256SUMS`. Recreate the result with `python3 -B report.py operator.log` and compare it to `result.json`. `FROZEN.sha256` and `INPUTS.sha256` identify the original local executable and its upstream dependency pins.
