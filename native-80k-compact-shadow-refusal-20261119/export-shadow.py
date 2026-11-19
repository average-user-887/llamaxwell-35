#!/usr/bin/env python3
"""Export only allowlisted aggregate observations from the sealed owner receipts."""
import argparse
import hashlib
import json
from pathlib import Path
import re

HERE = Path(__file__).resolve().parent
PHASES = ('cold', 'warm')
COUNTERS = ('row0', 'row1', 'combined', 'skip', 'q2_sparse', 'q2_stride', 'down4')
COMPACT = ('requested', 'launched', 'compact_only', 'ambiguous', 'unsupported',
           'raw_bytes', 'compact_bytes', 'shadow_rows', 'shadow_mismatches')
PHASE_NUMBERS = ('cache_n', 'prompt_n', 'predicted_n', 'draft_n', 'draft_n_accepted')
SETTINGS = ('prompt_n', 'n_predict', 'seed', 'temperature', 'cache_prompt', 'ignore_eos',
            'finish_reason', 'truncated', 'error')


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_manifest(root, name, complete=False):
    entries = {}
    for line in (root / name).read_text().splitlines():
        digest, relative = line.split(maxsplit=1)
        path = (root / relative).resolve()
        if complete and not path.is_relative_to(root.resolve()):
            raise ValueError('observation manifest has an external path')
        if path in entries or sha(path) != digest:
            raise ValueError('sealed file checksum differs')
        entries[path] = digest
    if complete:
        actual = {path.resolve() for path in root.rglob('*')
                  if path.is_file() and path != root / name}
        if actual != entries.keys():
            raise ValueError('observation manifest coverage differs')
    return entries


def mapped_hashes(receipt):
    actual = {}
    for row in (receipt / 'runtime-library-hashes.txt').read_text().splitlines():
        digest, path = row.split()
        name = Path(path).name
        if name in actual or not re.fullmatch('[0-9a-f]{64}', digest):
            raise ValueError('invalid runtime inventory')
        actual[name] = digest
    if actual != read(HERE / 'source/profile.json'):
        raise ValueError('mapped runtime differs from source profile')
    return actual


def first_mismatch(candidate, control):
    return next((i for i, (a, b) in enumerate(zip(candidate, control), 1) if a != b), None)


def comparison(actual, reference):
    return {
        'same_request': all(actual[k] == reference[k] for k in SETTINGS)
                        and actual['prompt_sha256'] == reference['prompt_sha256'],
        'same_slot': actual['id_slot'] == reference['id_slot'],
        'first_token_mismatch_ordinal': first_mismatch(actual['token_ids'], reference['token_ids']),
        'exact_token_sequence': actual['token_ids'] == reference['token_ids'],
        'exact_full_text': actual['text'] == reference['text'],
        **{k: reference['timings'][k] for k in PHASE_NUMBERS},
        'observed_response_tokens': len(reference['token_ids']),
    }


def export(card, receipt, reference):
    sealed = check_manifest(receipt, 'observations.sha256', complete=True)
    check_manifest(reference, 'observations.sha256')
    check_manifest(card, 'FROZEN.sha256')
    freeze = sha(card / 'FROZEN.sha256')
    if freeze != read(HERE / 'source/source-pins.json')['owner_source_freeze_sha256']:
        raise ValueError('wrong source freeze')
    manifest = (receipt / 'manifest.txt').read_text().splitlines()
    if manifest[0] != 'stage=shadow' or f'{freeze}  {card}/FROZEN.sha256' not in manifest:
        raise ValueError('receipt has a different stage or source freeze')
    if receipt.parent != card / 'receipts' or not receipt.name.startswith('shadow-'):
        raise ValueError('wrong receipt ownership')
    sanitized = read(receipt / 'sanitized/model-receipt.json')
    if sanitized['validated'] is not None or (sanitized['runner_exit'], sanitized['cleanup_exit']) != (1, 0):
        raise ValueError('expected a terminal refusal with successful cleanup')
    statuses = dict(line.split('=', 1) for line in (receipt / 'status.txt').read_text().splitlines())
    terminal = {'runner_exit': int(statuses['runner_exit']), 'cleanup_exit': int(statuses['cleanup_exit']),
                'bounded_wall_s': sanitized['bounded_wall_s'], 'cap_s': 430}
    if any(terminal[k] != sanitized[k] for k in ('runner_exit', 'cleanup_exit')):
        raise ValueError('terminal witness differs')
    env = {}
    for row in (receipt / 'child-selected-environment.txt').read_text().splitlines():
        key, value = row.split('=', 1)
        if key == 'LLAMA_OCCAMY_COMPACT_READBACK' and value == 'shadow':
            env[key] = value
        elif re.fullmatch(r'(?:LLAMA|GGML_CUDA)_[A-Z0-9_]+', key) and re.fullmatch(r'\d+(?:,\d+)*', value):
            env[key] = value
    if (env.get('LLAMA_OCCAMY_COMPACT_READBACK'), env.get('GGML_CUDA_OCCAMY_MOE_Q2_DOWN4')) != ('shadow', '1'):
        raise ValueError('expected child features missing')
    args = (receipt / 'child-server.args').read_text().splitlines()
    spec_arg = args.index('--spec-type')
    spec_setup = args[spec_arg:]
    if spec_setup != ['--spec-type', 'draft-mtp', '--spec-draft-n-max', '1',
                      '--spec-draft-type-k', 'f16', '--spec-draft-type-v', 'f16']:
        raise ValueError('unexpected child speculative arguments')
    mapped = mapped_hashes(receipt)
    logs = (receipt / 'server.log').read_text()
    down4_residency = re.findall(r'^Q2_DOWN4 residency device=\d+ sm=(\d+) active=(\d+) regs=(\d+) static=(\d+) dynamic=(\d+) local=(\d+) ready=(\d+)$', logs, re.M)
    if len(down4_residency) != 1:
        raise ValueError('expected one down4 residency witness')
    compact_pattern = r'^compact_readback: ' + ' '.join(key + r'=(\d+)' for key in COMPACT) + r'$'
    raw_compact = re.findall(compact_pattern, logs, re.M)
    prewarm = re.findall(compact_pattern, (receipt / 'server-pre-warm.log').read_text(), re.M)
    startup = re.findall(compact_pattern, (receipt / 'server-start.log').read_text(), re.M)
    if startup or len(raw_compact) != 2 or prewarm != raw_compact[:1]:
        raise ValueError('compact request boundary witnesses differ')
    snapshots = {name: {key: read(receipt / f'witness/{name}.json')[key] for key in COUNTERS}
                 for name in ('start', *PHASES)}
    if snapshots != sanitized['snapshots']:
        raise ValueError('terminal and physical witness snapshots differ')
    phases = {}
    for i, phase in enumerate(PHASES):
        candidate = read(receipt / f'{phase}.json')
        control = read(reference / f'{phase}.json')
        old, new = snapshots[('start', *PHASES)[i]], snapshots[phase]
        compact_before = tuple(map(int, raw_compact[i-1])) if i else (0,) * len(COMPACT)
        compact = {key: int(new) - old for key, new, old in zip(COMPACT, raw_compact[i], compact_before)}
        observation = sanitized['observations'][phase]
        if any(candidate[k] != v for k, v in observation.items()) or len(candidate['token_ids']) != observation['n_out']:
            raise ValueError('sealed client observations differ')
        if candidate['finish_reason'] != 'limit' or candidate['truncated'] is not False:
            raise ValueError('expected terminal limit without truncation')
        phases[phase] = {
            'requested_prompt_tokens': candidate['prompt_n'],
            'requested_generation_tokens': candidate['n_predict'],
            'observed_response_tokens': observation['n_out'],
            'ttft_s': observation['ttft_s'],
            'wall_s': observation['wall_s'],
            'finish_reason': 'limit',
            'truncated': False,
            'server': {k: candidate['timings'][k] for k in PHASE_NUMBERS},
            'compact_counts': compact,
            'physical_delta': {k: new[k] - old[k] for k in COUNTERS},
            'archived_scorer_control': comparison(candidate, control),
        }
        if phase == 'warm':
            phases[phase]['sse_checkpoints_observed'] = [row['output_token'] for row in candidate['observed_sse_checkpoints']]
    expected_refusal = ('warm prompt, slot, stop or exact output length differs from the pinned protocol; '
                        'token-bearing SSE checkpoints incomplete')
    if sanitized['refusal'] != expected_refusal or expected_refusal not in (receipt / 'run.log').read_text():
        raise ValueError('expected validator refusal missing')
    return {
        'schema': 1, 'arm': 'shadow', 'source_freeze_sha256': freeze,
        'sealed_observation_files': len(sealed), 'terminal': terminal,
        'validation': 'refused', 'validator_message': expected_refusal,
        'mapped_software_sha256': mapped, 'child_feature_environment': env,
        'child_speculation_arguments': spec_setup,
        'down4_residency': dict(zip(('sms', 'active_blocks_per_sm', 'registers_per_thread', 'static_bytes',
                                    'dynamic_bytes', 'local_bytes', 'ready'), map(int, down4_residency[0]))),
        'physical_cumulative_snapshots': snapshots, 'phases': phases,
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--owner-card', type=Path, required=True)
    parser.add_argument('--receipt', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    args = parser.parse_args()
    result = export(*(p.resolve() for p in (args.owner_card, args.receipt, args.reference)))
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
