#!/usr/bin/env python3
"""Check the portable source pins and an optional aggregate model export."""
import argparse
import hashlib
import json
import math
from pathlib import Path

HERE=Path(__file__).resolve().parent
PHASES=('cold','warm')
STAGES=('shadow','native','serial')


def need(ok,text):
    if not ok:
        raise ValueError(text)


def sha(file):
    return hashlib.sha256(file.read_bytes()).hexdigest()


def sources():
    pins=json.loads((HERE/'source-pins.json').read_text())
    need(sha(HERE/'owner-frozen.sha256') == pins['owner_source_freeze_sha256'],'owner source freeze mismatch')
    for file,digest in pins['exact_composite_sources'].items():
        need(sha(HERE/file)==digest,'embedded exact source differs')
    for spec in pins['rehydration']:
        content=(HERE/spec['template']).read_text()
        need(all(k in content for k in spec['aliases']),'required reconstruction alias missing')
    sums={name:digest for digest,name in (row.split(maxsplit=1) for row in (HERE/'SHA256SUMS').read_text().splitlines())}
    for file,digest in sums.items():
        need(Path(file).name==file and sha(HERE/file)==digest,'portable artifact checksum mismatch')
    return pins


def verify_model(results, pins):
    need(results['schema']==1 and results['source_freeze_sha256']==pins['owner_source_freeze_sha256'],'wrong result source freeze')
    expected=json.loads((HERE/'profile.json').read_text())
    rows={}
    for row in results['attempts']:
        alias, stage, model = row['attempt'],row['stage'],row['model']
        need(stage in STAGES and alias.startswith(stage+'-') and alias not in rows,'invalid attempt identity')
        rows[alias]=row
        term=row['terminal']
        need(term['cap_seconds']==430 and row['private_request_observations_excluded'] is True,'terminal scope mismatch')
        if row['disposition']!='validated':
            need(model is None,'refused attempt carries validated model claims')
            continue
        need(term['runner_exit']==term['cleanup_exit']==0 and term['observation_manifest_verified'] is True,'invalid terminal success')
        need(model['mapped_software_sha256']==expected and model['accepted_experts']==8 and model['semantic_score'] is None,'mapped identity or model scope mismatch')
        counts=model['physical_cumulative_snapshots']
        need(len(counts)==3,'missing startup/cold/warm physical boundaries')
        for idx,phase in enumerate(PHASES):
            p=model['phases'][phase]
            offers=p['metrics']['draft_n'] or 0
            need(p['archived_parity']['exact_ids']==p['metrics']['n_out'] and p['archived_parity']['exact_full_text'] and p['archived_parity']['cache_and_draft_totals_equal'],'archive parity does not reconcile')
            need(p['down4']['down4_launches']==p['down4']['expected']==counts[idx]['down4']==counts[idx+1]['down4']==0,'unexpected down4 physical launch')
            need(p['down4']['ordinary_q2_launches']==counts[idx+1]['combined']-counts[idx]['combined']==offers*40,'ordinary Q2 cohort counters differ')
            need(p['guard']['ready']==offers and all(v==0 for k,v in p['guard'].items() if k!='ready'),'typed guard witness differs')
            c=p['compact']
            good=offers-c['ambiguous']
            need(c['requested']==c['launched']==offers and c['compact_bytes']==16*offers and c['unsupported']==c['shadow_mismatches']==0 and 0<=good<=offers,'invalid compact accounting')
            if stage=='shadow':
                need(c['compact_only']==0 and c['raw_bytes']==1986560*offers and c['shadow_rows']==2*good,'invalid shadow accounting')
            else:
                need(c['compact_only']==good and c['raw_bytes']==1986560*c['ambiguous'] and c['shadow_rows']==0,'invalid active compact accounting')
    need(results['unrepresented_stages']==[s for s in STAGES if not any(a['stage']==s for a in rows.values())],'stage coverage differs')
    pair=results['same_build_pair']
    if pair:
        serial,native=[rows[pair[k+'_attempt']] for k in ('serial','native')]
        need(serial['stage']=='serial' and native['stage']=='native' and serial['model'] and native['model'],'invalid strict pair')
        need(pair['same_build']==expected and pair['same_process_feature_flags'] is True,'strict pair identity differs')
        for phase in PHASES:
            a,b=[r['model']['phases'][phase]['metrics']['decode_toks_s'] for r in (serial,native)]
            p=pair['phases'][phase]
            need(math.isclose(p['same_build_gain_pct'],100*(b/a-1),abs_tol=1e-8) and p['observed_100_tps']==(b>=100),'pair arithmetic differs')
    return len(rows)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--results',type=Path)
    args=parser.parse_args()
    pins=sources()
    if args.results:
        print(f'portable source verified; {verify_model(json.loads(args.results.read_text()),pins)} aggregate attempts reconciled')
    else:
        print('portable source, byte pins, and reconstruction template markers verified')
