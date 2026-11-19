#!/usr/bin/env python3
"""Validate model provenance, output parity and compact bytes on the frozen down4 runtime."""
from contextlib import contextmanager
import importlib.util
import json
from pathlib import Path
import re
import sys
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
C = HERE.parent
spec = importlib.util.spec_from_file_location('down4_model', C/'break100-model-base/model.py')
parent = importlib.util.module_from_spec(spec)
spec.loader.exec_module(parent)
hybrid, control, core, q2 = parent.hybrid, parent.control, parent.core, parent.q2
need, read, phases, ARCHIVE = parent.need, parent.read, parent.phases, parent.ARCHIVE
profile = read(HERE/'profile.json')
STAGES = ('serial', 'native', 'shadow')
COUNTERS = ('requested', 'launched', 'compact_only', 'ambiguous', 'unsupported', 'raw_bytes', 'compact_bytes', 'shadow_rows', 'shadow_mismatches')
MARKER = re.compile(r'^(?:\d+\.\d{2}\.\d{3}\.\d{3} I )?compact_readback: '+ ' '.join(x+r'=(\d+)' for x in COUNTERS)+r'$', re.M)
GUARDS = ('ready','observer','types','rollback','shape','model','draft','sampler','mask')
GUARD_MARKER = re.compile(r'^(?:\d+\.\d{2}\.\d{3}\.\d{3} I )?compact_guard: '+ ' '.join(x+r'=(\d+)' for x in GUARDS)+r'$', re.M)


def mode(stage):
    need(stage in STAGES, 'unknown composite stage')
    return 'native' if stage == 'shadow' else stage


def settings(stage):
    env, args = parent.settings(mode(stage))
    need(env.count(parent.flag)==1, 'original profile lacks the explicit down4 flag')
    flag = 'shadow' if stage == 'shadow' else '1'
    need(not any('LLAMA_OCCAMY_COMPACT_READBACK' in one for one in env), 'unexpected inherited compact setting')
    return [*env, '-e', f'LLAMA_OCCAMY_COMPACT_READBACK={flag}'], args


@contextmanager
def bindings(stage):
    actual_mode = mode(stage)
    inherited = hybrid.candidate_bindings if actual_mode == 'native' else control.serial_bindings
    with inherited():
        original_maps = q2.base.maps

        def expected_maps(path):
            observed = original_maps(path)
            if path == q2.REF/'runtime-library-hashes.txt':
                name = 'libllama.so.0.6.0'
                key = str(hybrid.libs/name)
                need(observed[key] == parent.profile[name], 'original core identity differs')
                observed[key] = profile[name]
            return observed

        with patch.object(q2.base, 'maps', expected_maps), patch.object(q2, 'settings', lambda _mode: settings(stage)), \
                patch.object(q2, 'DSO', profile['libggml-cuda.so']), patch.object(q2, 'SUP', HERE/'supervise'), \
                patch.object(q2, 'SERVER_IMPL_SHA', profile['libllama-server-impl.so']):
            yield


def ready(stage, out):
    actual_mode = mode(stage)
    with bindings(stage):
        startup, actual = q2.ready(actual_mode, out)
    if actual_mode == 'native':
        hybrid.confirm_merge(startup)
    else:
        need(hybrid.MERGE not in startup and 'mtp_last_graph' not in startup, 'unexpected serial draft graph')
    observed = {'exe' if k.startswith('/proc/') else Path(k).name: v for k,v in actual.items()}
    need(observed == profile, 'actual mapped model software differs from frozen compact profile')
    return startup, observed


def compact_probe(stage, out, metrics):
    logs = [(out/f).read_text() for f in ('server-start.log', 'server-pre-warm.log', 'server.log')]
    rows = [MARKER.findall(text) for text in logs]
    need(all(len(found) == text.count('compact_readback:') for found, text in zip(rows, logs)), 'malformed compact witness')
    need(not rows[0] and len(rows[1]) == 1 and len(rows[2]) == 2 and rows[1][0] == rows[2][0], 'missing request-boundary compact witnesses')
    snapshots = [(0,)*len(COUNTERS), *[tuple(map(int, x)) for x in rows[2]]]
    phase_stats = {}
    for phase, old, new in zip(phases, snapshots, snapshots[1:]):
        d = dict(zip(COUNTERS, (b-a for a,b in zip(old,new))))
        expected = int(metrics[phase].get('draft_n') or 0)
        need(min(d.values()) >= 0 and d['requested'] == expected and d['launched'] == expected, 'physical compact launch and offer counts differ')
        need(d['unsupported'] == 0 and d['compact_bytes'] == 16*expected and d['shadow_mismatches'] == 0, 'compact fallback, byte or shadow witness differs')
        good = expected - d['ambiguous']
        need(0 <= good <= expected and (expected == 0 or good > 0), 'missing valid compact choices')
        if stage == 'shadow':
            need(d['compact_only'] == 0 and d['raw_bytes'] == 1986560*expected and d['shadow_rows'] == 2*good,
                 'shadow must preserve full logits and verify both eligible rows')
        else:
            need(d['compact_only'] == good and d['raw_bytes'] == 1986560*d['ambiguous'] and d['shadow_rows'] == 0,
                 'compact-only and fallback bytes do not reconcile')
        phase_stats[phase] = d
    return phase_stats


def guard_probe(out):
    logs = [(out/f).read_text() for f in ('server-start.log','server-pre-warm.log','server.log')]
    rows = [GUARD_MARKER.findall(text) for text in logs]
    need(all(len(v) == text.count('compact_guard:') for v,text in zip(rows,logs)), 'malformed typed guard witness')
    need(not rows[0] and len(rows[1]) == 1 and len(rows[2]) == 2 and rows[1][0] == rows[2][0], 'missing guard boundaries')
    snapshots=[(0,)*len(GUARDS), *[tuple(map(int,one)) for one in rows[2]]]
    return {phase:dict(zip(GUARDS,(b-a for a,b in zip(old,new))))
            for phase,old,new in zip(phases,snapshots,snapshots[1:])}


def probe(stage, out, metrics):
    guards=guard_probe(out)
    for phase,detail in guards.items():
        offers=int(metrics[phase].get('draft_n') or 0)
        need(detail['ready']==offers and all(v==0 for k,v in detail.items() if k!='ready'), 'typed compact gate refused a measured offer')
    return {'down4':parent.probe(mode(stage),out,metrics), 'compact':compact_probe(stage,out,metrics), 'compact_guard':guards}


def collect(stage, out):
    actual_mode = mode(stage)
    ready(stage, out)
    with bindings(stage):
        result = core.collect(actual_mode, out)
    if actual_mode == 'native':
        result['draft_graph_witness'] = hybrid.graph_witness(out)
    else:
        for name in ('server-pre-warm.log','server.log'):
            log = (out/name).read_text()
            need(hybrid.MERGE not in log and 'mtp_last_graph' not in log, 'serial draft graph detected')
    result.update(probe(stage,out,result['metrics']))
    result['stage'] = stage
    result['same_mode_archive'] = {}
    for phase, n in zip(phases,(192,3072)):
        current, ref = [read(p/f'{phase}.json') for p in (out, ARCHIVE[actual_mode])]
        need(all(current[k] == ref[k] for k in hybrid.INPUT_FIELDS), 'input identity or decoding settings differ')
        need(current['token_ids'] == ref['token_ids'] and len(current['token_ids']) == n and current['text'] == ref['text'], 'same-mode output differs')
        need(all(current['timings'].get(k) == ref['timings'].get(k) for k in ('draft_n','draft_n_accepted','cache_n','prompt_n')), 'same-mode acceptance or cache counts differ')
        result['same_mode_archive'][phase] = {'exact_ids':n,'exact_full_text':True,'cache_and_draft_totals_equal':True,
                                              'repetition':parent.export.base.repetition(current['text'])}
    return result


def finished(stage, out):
    need(out.parent == HERE/'receipts' and out.name.startswith(stage+'-'), 'wrong compact receipt owner or mode')
    hybrid.validator.sha_check(out,'observations.sha256')
    manifest = (out/'manifest.txt').read_text().splitlines()
    need(manifest[0] == 'stage='+stage and f'{q2.n9.hash_file(HERE/"FROZEN.sha256")}  {HERE}/FROZEN.sha256' in manifest, 'arm used a different compact freeze')
    need({'runner_exit=0','cleanup_exit=0'} <= set((out/'status.txt').read_text().splitlines()), 'terminal run or cleanup failed')
    actual = collect(stage,out)
    official = read(out/'sanitized/model-receipt.json')
    need(official['mode'] == stage and official['runner_exit'] == official['cleanup_exit'] == 0 and official['refusal'] is None and actual == official['validated'] == read(out/'result.json'), 'terminal compact result did not replay')
    return actual


def compare(serial, native):
    actual = {m:finished(m,p) for m,p in [('serial',serial),('native',native)]}
    need(actual['serial']['actual_server_software_sha256'] == actual['native']['actual_server_software_sha256'] == profile, 'same-build identity differs')
    need(sorted((serial/'child-selected-environment.txt').read_text().splitlines()) == sorted((native/'child-selected-environment.txt').read_text().splitlines()), 'different child feature vectors')
    a,b = [(p/'child-server.args').read_text().splitlines() for p in (serial,native)]
    i = b.index('--spec-type')
    need(a == b[:i]+['--spec-type','none'] and b[i:] == ['--spec-type','draft-mtp','--spec-draft-n-max','1','--spec-draft-type-k','f16','--spec-draft-type-v','f16'], 'difference exceeds speculation arguments')
    result = {'same_build':profile,'same_process_feature_flags':True,'semantic_score':None,'arms':{},'phases':{}}
    for m,x in actual.items():
        result['arms'][m] = {k:x[k] for k in ('down4','compact','compact_guard','same_mode_archive','accepted_experts')}
    for phase in phases:
        a,b = [actual[m]['metrics'][phase] for m in ('serial','native')]
        ref = read(ARCHIVE['native']/'result.json')['metrics'][phase]
        result['phases'][phase] = {'serial':a,'native':b,'same_build_gain_pct':100*(b['decode_toks_s']/a['decode_toks_s']-1),
                                   'native_saved_ms_per_round_vs_369c':ref['decode_ms_per_round']-b['decode_ms_per_round'],
                                   'observed_100_tps':b['decode_toks_s']>=100}
    return result


def check():
    parent.check()
    need(settings('native')[0] == settings('serial')[0] and settings('shadow')[1] == settings('native')[1], 'compact stage typed settings differ')
    r1_card=C/'break100-compact-model-r1'
    r1=parent.module(r1_card/'model.py','compact_r1_exact')
    need(profile == r1.profile, 'R2 runtime differs from R1')
    for stage in STAGES:
        old_env,old_args=r1.settings(stage)
        new_env,new_args=settings(stage)
        need(old_env.count('GGML_CUDA_OCCAMY_MOE_Q2_DOWN4=0')==1 and new_args==old_args and
             new_env==['GGML_CUDA_OCCAMY_MOE_Q2_DOWN4=1' if value=='GGML_CUDA_OCCAMY_MOE_Q2_DOWN4=0' else value for value in old_env],
             'R2 stage differs beyond the down4 flag')


if __name__ == '__main__':
    command,*args = sys.argv[1:]
    try:
        if command == 'check':check()
        elif command in ('env','args'):print('\n'.join(settings(args[0])[command=='args']))
        elif command == 'ready':ready(args[0],Path(args[1]))
        elif command == 'verify':core.n9.write_json(Path(args[1])/'result.json',collect(args[0],Path(args[1])))
        elif command == 'finalize':q2.finish(args[0],Path(args[1]),*map(int,args[2:]))
        elif command == 'probe':
            stage,path = args;out = Path(path)
            print(json.dumps(probe(stage,out,{ph:read(out/f'{ph}.json')['timings'] for ph in phases}),indent=2))
        elif command == 'guard':print(json.dumps(guard_probe(Path(args[0])),indent=2))
        elif command == 'compare':print(json.dumps(compare(*[Path(p).resolve() for p in args]),indent=2,allow_nan=False))
        else:raise ValueError('unexpected command')
    except (ValueError,KeyError,OSError) as e:raise SystemExit('Q2 model: '+str(e)) from None
