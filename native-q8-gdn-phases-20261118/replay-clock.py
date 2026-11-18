"""Recompute the published per-layer timings and bias verdict without a GPU or model."""
import gzip
import hashlib
import json
import re
from pathlib import Path

import numpy as np

from analyze import LAYERS, describe, fractions

base = Path(__file__).resolve().parent
receipt = json.loads((base / "summary.json").read_text())
pins = json.loads((base / "provenance.json").read_text())
raw = gzip.decompress((base / "clock-marks.u32.gz").read_bytes())
assert hashlib.sha256(raw).hexdigest() == pins["raw_little_endian_u32_clock_sha256"]
f = fractions(np.frombuffer(raw, dtype="<u4").reshape(30,72,8,6))
events = {}
for arm in ("control", "clock"):
    text = (base / (arm + "-events.txt")).read_text()
    rows = re.findall(rf"^Q8CLK_EVENT mode={arm} layer_slot=(\d+) tensor=blk\.(\d+)\.attn_qkv\.weight us=([\d.]+)$", text, re.M)
    assert [(int(i), int(l)) for i, l, _ in rows] == list(enumerate(LAYERS))
    assert text.count("Q8CLK_COHORT fallback0=0 fallback1=0 paired=40 skipped=40") == 1
    events[arm] = np.array([float(v) for _, _, v in rows])
    for metric, array in f.items():
        assert describe(array*events[arm]) == receipt["phase_us_estimates"][metric][arm]
a, b = events["control"], events["clock"]
mean = float(b.mean()/a.mean()-1)
p90 = float(np.percentile(abs(b-a)/a,90))
assert mean == receipt["mean_event_bias_fraction"] and p90 == receipt["p90_paired_absolute_relative_bias"]
for i, record in enumerate(receipt["per_layer"]):
    assert record["layer"] == LAYERS[i] and (record["control_event_us"],record["clock_event_us"]) == (a[i],b[i])
    assert all(record["clock_fractions"][metric] == values[i] for metric,values in f.items())
assert (abs(mean)<=.05 and p90<=.10) == receipt["bias_within_predeclared_limits"]
assert receipt["exact_paired_states_logits_rows"] and receipt["exact_vs_archived_Q8_states_logits_rows"]
assert receipt["archived_Q8_checked_tensors"]==6
print(f"Verified {len(raw)//4:,} clock readings, 60 layer events and all phase summaries. Mean bias {mean:.3%}, p90 absolute paired difference {p90:.3%}.")
