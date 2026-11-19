#!/usr/bin/env python3
"""Read only aggregate R2 files and rederive the published rate accounting."""
import json
import math
from pathlib import Path

HERE=Path(__file__).resolve().parent


def read(name):
    return json.loads((HERE/name).read_text())


def near(a,b):
    return math.isclose(a,b,rel_tol=1e-11,abs_tol=1e-11)


def yes(ok,reason):
    if not ok:
        raise ValueError(reason)


def check():
    results,recon=read('results.json'),read('reconciliation.json')
    yes(results['source_freeze_sha256']==recon['source_freeze_sha256'],'aggregate source freeze differs')
    attempts={a['stage']:a['model'] for a in results['attempts']}
    yes(set(attempts)==set(recon['arms']),'mode coverage differs')
    for mode,arm in recon['arms'].items():
        for phase,row in arm.items():
            terminal=attempts[mode]['phases'][phase]
            timing=row['server_timing_fields']
            rates=row['decode_rates']
            n,t=row['delivered_tokens'],timing['predicted_ms']
            yes(n==timing['predicted_n']==terminal['metrics']['n_out']==row['requested_output_tokens'] and row['stop_reason_limit'] and not row['truncated'],'request lengths or stop differ')
            yes(all((near(rates['delivered_per_server_decode_s'],1000*n/t),near(rates['delivered_per_server_decode_s'],terminal['metrics']['decode_toks_s']),
                     near(rates['server_reported_post_first_per_s'],1000*(n-1)/t),near(rates['server_reported_post_first_per_s'],timing['predicted_per_second']),
                     near(rates['delivered_per_client_request_wall_s'],n/row['client_wall_s']))),'rate conventions differ')
            offers=timing['draft_n'] or 0
            rejects=offers-(timing['draft_n_accepted'] or 0)
            yes(row['sampler_snapshot_delta']=={'elided':offers,'elided_rejected':rejects,'kept':0,'restored':0},'sampler clone counters differ')
            yes(row['repetition_counts']==terminal['repetition'],'repetition count mismatch')
            down4=terminal['down4']['down4_launches']
            compact=terminal['compact']
            yes(down4==offers*40 and compact['compact_bytes']==16*offers and compact['launched']==offers,'physical compact or down4 launch accounting differs')
            yes(compact['raw_bytes']==(1986560*offers if mode=='shadow' else 0),'full target logit accounting differs')
            if mode=='native':
                r=row['round_accounting']
                yes(near(r['observed_server_ms_per_offer'],t/offers) and near(r['remaining_ms_per_offer_to_100_delivered_tps'],(t-10*n)/offers),'native round rate differs')
            if phase=='warm':
                checkpts=row['client_stream']['token_bearing_checkpoints']
                yes([c['output_token'] for c in checkpts]==[1,512,1024,1536,2048,2560,3072],'warm emitted checkpoints differ')
                yes(near(row['client_ttft_s'],checkpts[0]['elapsed_s']),'warm first-token timing differs')
                segs=row['client_stream']['segments']
                yes(len(segs)==6,'warm split count differs')
                for left,right,split in zip(checkpts,checkpts[1:],segs):
                    dt=right['elapsed_s']-left['elapsed_s']
                    nsplit=right['output_token']-left['output_token']
                    yes(nsplit==split['tokens'] and near(split['elapsed_s'],dt) and near(split['client_stream_tokens_per_s'],nsplit/dt),'warm split arithmetic differs')
    reported=results['same_build_pair']
    yes((reported is None)==(recon['pair'] is None),'unpaired and paired closures disagree')
    if reported:
        for phase in ('cold','warm'):
            s,n=(recon['arms'][m][phase] for m in ('serial','native'))
            rate0,rate1=(r['decode_rates']['delivered_per_server_decode_s'] for r in (s,n))
            yes(near(reported['phases'][phase]['same_build_gain_pct'],100*(rate1/rate0-1)),'same-build gain arithmetic differs')
            yes(near(recon['pair'][phase]['native_minus_serial_client_wall_s'],n['client_wall_s']-s['client_wall_s']),'same-build wall delta differs')
    print(f'{len(attempts)} R2 terminal arms reconciled; strict timed pair '+('verified' if reported else 'pending'))


if __name__=='__main__':
    check()
