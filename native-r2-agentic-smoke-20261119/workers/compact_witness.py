"""Exact R2 compact counter rows, including the observed timestamp-only log prefix."""
import re

# fprintf() witness output can be bare or share the numeric logger prefix.
PREFIX = r"^(?:\d+\.\d{2}\.\d{3}\.\d{3} (?:I )?)?"


def marker(label, keys):
    if label not in {"compact_readback", "compact_guard"}:
        raise ValueError("Unknown R2 compact witness")
    shape = " ".join(re.escape(key) + r"=(\d+)" for key in keys)
    return re.compile(PREFIX + label + ": " + shape + "$", re.M)
