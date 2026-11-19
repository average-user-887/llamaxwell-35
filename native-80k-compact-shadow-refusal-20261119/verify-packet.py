#!/usr/bin/env python3
"""Verify portable payload hashes and independently reconcile public shadow evidence."""
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parent


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(relative):
    return json.loads((ROOT / relative).read_text())


def verify_sources():
    sums = {}
    for row in (ROOT / 'SHA256SUMS').read_text().splitlines():
        digest, relative = row.split(maxsplit=1)
        path = ROOT / relative
        require(path.is_file() and path.resolve().is_relative_to(ROOT), 'invalid checksum target')
        require(path not in sums and sha(path) == digest, 'payload checksum differs')
        sums[path] = digest
    require(set(sums) == {f for f in ROOT.rglob('*') if f.is_file() and f.name != 'SHA256SUMS'}, 'payload coverage differs')
    pins = load('source/source-pins.json')
    require(sha(ROOT / 'source/owner-frozen.sha256') == pins['owner_source_freeze_sha256'], 'original source freeze differs')
    for name, digest in pins['exact_composite_sources'].items():
        require(sha(ROOT / 'source' / name) == digest, 'exact source pin differs')
    for item in pins['rehydration']:
        template = (ROOT / 'source' / item['template']).read_text()
        require(all(alias in template for alias in item['aliases']), 'missing reconstruction alias')
    return pins, len(sums)


def verify_observations(pins):
    r = load('shadow-observations.json')
    require(set(r) == {'schema', 'arm', 'source_freeze_sha256', 'sealed_observation_files', 'terminal', 'validation',
                       'validator_message', 'mapped_software_sha256', 'child_feature_environment',
                       'child_speculation_arguments', 'down4_residency', 'physical_cumulative_snapshots', 'phases'},
            'unexpected export fields')
    require((r['schema'], r['arm'], r['validation'], r['sealed_observation_files']) == (1, 'shadow', 'refused', 22), 'unexpected arm disposition')
    require(r['source_freeze_sha256'] == pins['owner_source_freeze_sha256'], 'receipt freeze differs')
    require(r['terminal'] == {'runner_exit': 1, 'cleanup_exit': 0, 'bounded_wall_s': 374, 'cap_s': 430}, 'terminal observation differs')
    require(r['mapped_software_sha256'] == load('source/profile.json') == load('source/build.json')['candidate_software'], 'mapped software differs')
    require(r['child_speculation_arguments'] == ['--spec-type', 'draft-mtp', '--spec-draft-n-max', '1',
                                                 '--spec-draft-type-k', 'f16', '--spec-draft-type-v', 'f16'], 'child speculation differs')
    env = r['child_feature_environment']
    require(env['LLAMA_OCCAMY_COMPACT_READBACK'] == 'shadow' and env['GGML_CUDA_OCCAMY_MOE_Q2_DOWN4'] == '1', 'child feature flags differ')
    require(all(re.fullmatch(r'(?:LLAMA|GGML_CUDA)_[A-Z0-9_]+', k) and
                (re.fullmatch(r'\d+(?:,\d+)*', v) or (k, v) == ('LLAMA_OCCAMY_COMPACT_READBACK', 'shadow'))
                for k, v in env.items()), 'environment export includes non-feature data')
    require(r['validator_message'] == ('warm prompt, slot, stop or exact output length differs from the pinned protocol; '
                                      'token-bearing SSE checkpoints incomplete'), 'recorded validator refusal differs')
    counts = r['physical_cumulative_snapshots']
    counter_keys = {'row0', 'row1', 'combined', 'skip', 'q2_sparse', 'q2_stride', 'down4'}
    require(counts.keys() == {'start', 'cold', 'warm'} and all(s.keys() == counter_keys for s in counts.values()), 'unexpected physical counter fields')
    require(counts['start']['down4'] == 39 and counts['warm']['down4'] == 69959, 'physical boundary differs')
    for earlier, phase, expected in [('start', 'cold', (192, 192, 192, 102, 88, 101, 89, 0, 80000, None, True)),
                                     ('cold', 'warm', (3072, 3071, 3072, 1646, 1424, 1642, 1429, 79996, 4, 1603, False))]:
        p = r['phases'][phase]
        server = p['server']
        ref = p['archived_scorer_control']
        expected_fields = {'requested_prompt_tokens', 'requested_generation_tokens', 'observed_response_tokens', 'wall_s',
                           'ttft_s', 'server', 'archived_scorer_control', 'compact_counts', 'physical_delta', 'finish_reason', 'truncated'}
        require(p.keys() == expected_fields | ({'sse_checkpoints_observed'} if phase == 'warm' else set()), 'unexpected phase fields')
        numbers = {'cache_n', 'prompt_n', 'predicted_n', 'draft_n', 'draft_n_accepted'}
        require(server.keys() == numbers and ref.keys() == numbers | {'same_request', 'same_slot', 'exact_full_text',
                'exact_token_sequence', 'first_token_mismatch_ordinal', 'observed_response_tokens'}, 'unexpected request details')
        require(tuple([p['requested_generation_tokens'], p['observed_response_tokens'], server['predicted_n'],
                       server['draft_n'], server['draft_n_accepted'], ref['draft_n'], ref['draft_n_accepted'], server['cache_n'],
                       server['prompt_n'], ref['first_token_mismatch_ordinal'], ref['exact_token_sequence']]) == expected, 'phase evidence differs')
        require(p['requested_prompt_tokens'] == server['cache_n'] + server['prompt_n'] == 80000, 'prompt accounting differs')
        require(p['finish_reason'] == 'limit' and p['truncated'] is False and 0 < p['ttft_s'] <= p['wall_s'], 'invalid request terminal/timing')
        require(ref['same_request'] and ref['same_slot'] and ref['exact_full_text'] == ref['exact_token_sequence'], 'reference identity differs')
        compact = p['compact_counts']
        require(compact.keys() == {'requested', 'launched', 'compact_only', 'ambiguous', 'unsupported', 'raw_bytes',
                                   'compact_bytes', 'shadow_rows', 'shadow_mismatches'} and all(type(v) is int and v == 0 for v in compact.values()),
                'compact counters differ')
        delta = {k: counts[phase][k] - counts[earlier][k] for k in counter_keys}
        require(delta == p['physical_delta'] and delta['down4'] == delta['skip'] == delta['combined'] == 40 * server['draft_n'],
                'down4 counters fail to reconcile')
        require(delta['q2_sparse'] == delta['q2_stride'] == 10 * server['draft_n'] and delta['row0'] == delta['row1'] == 0,
                'q2 counters fail to reconcile')
    require(r['phases']['warm']['sse_checkpoints_observed'] == [1, 512, 1024, 1536, 2048, 2560], 'warm checkpoint evidence differs')


if __name__ == '__main__':
    pins, count = verify_sources()
    verify_observations(pins)
    print(f'{count} packet payload files verified; one sealed shadow refusal reconciled')
