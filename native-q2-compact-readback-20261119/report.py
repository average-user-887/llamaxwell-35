#!/usr/bin/env python3
"""Validate the fixed physical A/B packet and report standalone materiality."""
import itertools
import json
import math
from pathlib import Path
import sys


def need(value, text):
    if not value:
        raise ValueError(text)


def summarize(log):
    strata = {}
    weights = {}
    final = None
    for line in log.splitlines():
        if line.startswith('edge_checks='):
            need(line == 'edge_checks=7 all_masked_fixture_uses_forced_device_mask=1', 'edge count or mode differs')
        elif line.startswith(('mirror=', 'technical_pass=')):
            row = dict(item.split('=', 1) for item in line.split())
            if 'technical_pass' in row:
                need(final is None, 'duplicate terminal result')
                final = {k: int(v) for k, v in row.items()}
                continue
            key = (int(row.pop('mirror')), int(row.pop('mask_case')))
            if 'weighted_net_us' in row:
                need(key not in weights, 'duplicate weighted result')
                weights[key] = {k: float(v) for k, v in row.items()}
            else:
                key += (int(row.pop('rejected')),)
                need(key not in strata, 'duplicate measurement stratum')
                strata[key] = {k: float(v) for k, v in row.items()}
    need(log.count('edge_checks=') == 1, 'missing or duplicate edge pass')
    need(set(weights) == set(itertools.product((1, 2), (0, 1))), 'missing A/B mirror or mask stratum')
    need(set(strata) == set(itertools.product((1, 2), (0, 1), (0, 1))), 'missing accept or reject stratum')
    for row in strata.values():
        need(all(math.isfinite(v) for v in row.values()), 'nonfinite timing')
        need(row['samples'] == 64, 'unexpected sample count')
        for phase in ('raw', 'compact'):
            need(row[f'{phase}_us'] > 0 and row[f'{phase}_copy_us'] > 0 and row[f'{phase}_cpu_us'] > 0, 'nonpositive timing')
        need(abs(row['raw_us'] - row['compact_us'] - row['net_us']) < 0.003, 'timing subtraction mismatch')
    for key, row in weights.items():
        weighted = (1429 * strata[(*key, 0)]['net_us'] + 213 * strata[(*key, 1)]['net_us']) / 1642
        need(abs(row['weighted_net_us'] - weighted) < 0.003 and row['gate_us'] == 550, 'bad acceptance weighting or gate')
        need(row['pass'] in (0, 1) and (abs(weighted - 550) < 0.003 or row['pass'] == (weighted >= 550)), 'gate outcome mismatch')
    need(final is not None and final['technical_pass'] == 1, 'missing technical pass')
    need(final['normal_raw_bytes'] == 1986560 and final['normal_compact_bytes'] == 16, 'wrong output byte accounting')
    need(final['rolling_device_bytes'] == 64 * 1986560 and 0 < final['l2_bytes'] < final['rolling_device_bytes'], 'bad working set')
    need(final['standalone_gate_pass'] == all(row['pass'] for row in weights.values()), 'wrong terminal speed gate')
    return {
        'scope': 'standalone synthetic physical readback and pinned CPU sampler; no model inference',
        **final,
        'edge_checks': 7,
        'weighted': [dict(mirror=m, mask_case=k, **v) for (m, k), v in sorted(weights.items())],
        'strata': [dict(mirror=m, mask_case=k, rejected=r, **v) for (m, k, r), v in sorted(strata.items())],
    }


if __name__ == '__main__':
    print(json.dumps(summarize(Path(sys.argv[1]).read_text()), indent=2))
