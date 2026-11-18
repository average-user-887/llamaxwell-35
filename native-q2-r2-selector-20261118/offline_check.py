"""Host simulation of the warp selection network and its reference semantics."""
import math
import random
import struct

BLOCK, KEEP, RECENT, TILE = 256, 16, 4096, 4

def key(value, index):
    if math.isnan(value) or value == -math.inf:
        return 0
    bits, = struct.unpack("<I", struct.pack("<f", 0.0 if value == 0.0 else value))
    ordered = bits ^ (0xFFFFFFFF if bits >> 31 else 0x80000000)
    return (ordered << 32) | (0xFFFFFFFF - index)

def network(items, worst):
    out = items[:]
    width = 2
    while width <= len(out):
        stride = width // 2
        while stride:
            out = [out[lane ^ stride] if worst(out[lane], out[lane ^ stride]) ==
                   ((lane & width == 0) == (lane & stride == 0)) else out[lane]
                   for lane in range(len(out))]
            stride //= 2
        width *= 2
    return out

def merge(left, right):
    out = [max(a, b) for a, b in zip(left, reversed(right), strict=True)]
    for stride in (8, 4, 2, 1):
        out = [(max if lane & stride == 0 else min)(out[lane], out[lane ^ stride])
               for lane in range(KEEP)]
    return out

def rank_network(scores):
    groups = []
    for first in range(0, 512, 32):
        warp = [key(scores[i], i) if i < len(scores) else 0 for i in range(first, first + 32)]
        groups.append(network(warp, int.__lt__)[:KEEP])
    while len(groups) > 1:
        groups = [merge(groups[i], groups[i + 1]) for i in range(0, len(groups), 2)]
    ordered = groups[0]
    if ordered[0] >> 32 == 0xFF800000:
        return []
    ids = [0xFFFFFFFF - (item & 0xFFFFFFFF) if item else 0x7FFFFFFF for item in ordered]
    return [i for i in network(ids, int.__gt__) if i != 0x7FFFFFFF]

def rank_reference(scores):
    kept, scores = [], list(scores)
    for _ in range(KEEP):
        value, index = -math.inf, 0x7FFFFFFF
        for i, v in enumerate(scores):
            if v > value or (v == value and i < index):
                value, index = v, i
        if math.isfinite(value):
            kept.append(index)
            scores[index] = -math.inf
    return sorted(kept)

def plan(rank, scores, visible, mask_bad, score_bad):
    old = max(0, visible - RECENT) // BLOCK
    invalid = any(mask_bad) or any(score_bad[:old])
    ids = rank(scores[:old])
    recent_begin = old * BLOCK
    recent_keys = visible - recent_begin
    return (not invalid and visible > 0 and (not old or bool(ids)),
            invalid, ids, recent_begin, visible, len(ids)*(BLOCK//TILE) + (recent_keys+TILE-1)//TILE,
            len(ids)*BLOCK + recent_keys, visible - len(ids)*BLOCK - recent_keys)

def compare(scores, visible, mask_bad=(), score_bad=()):
    actual = plan(rank_network, scores, visible, mask_bad, score_bad)
    expected = plan(rank_reference, scores, visible, mask_bad, score_bad)
    assert actual == expected, (actual, expected)
    return actual

rng = random.Random(0x2a5127c3)
trials = 0
for nkv in range(80128, 83201, 256):
    n = nkv // BLOCK
    for end in (0, 1, 4095, 4096, 4097, 4351, 4352, 4353, nkv - 261, nkv - 260, nkv - 1, nkv):
        old = max(0, end - RECENT) // BLOCK
        values = [rng.choice((0.0, -0.0, -3.0, 3.0, -math.inf, math.nan, 1.0, 2.0))
                  for _ in range(n)]
        for curse in (-1, max(0, old - 1), old):
            bad = tuple(i == curse for i in range(n))
            result = compare(values, end, score_bad=bad)
            assert result[1] == (0 <= curse < old)
            trials += 1
        assert compare(values, end, mask_bad=(True,))[1]
        trials += 1
        result = compare([0.0, -0.0] * (n // 2 + 1), end)
        assert result[2] == list(range(min(KEEP, old)))
        trials += 1
    end = nkv - 260
    trial = [rng.getrandbits(24) - (1 << 23) for _ in range(n)]
    assert compare(trial, end)[0]
    trials += 1
    trial[0] = math.inf
    result = compare(trial, end, score_bad=(True,))
    assert not result[0] and result[1] and not result[2]
    trials += 1
    result = compare([-math.inf] * n, end)
    assert not result[0] and not result[1] and not result[2]
    trials += 1

# Network boundaries and unrelated-width routing excluded by the host wrapper.
for size in (0, 1, 15, 16, 17, 31, 32, 33, 127, 128, 129, 303, 304, 305, 511, 512):
    trial = [rng.getrandbits(25) - (1 << 24) for _ in range(size)]
    assert rank_network(trial) == rank_reference(trial)
    trials += 1
print(f"offline plan parity passed {trials} fixtures across 13 aligned widths and causal, invalid, nonfinite, tie and short cases")
