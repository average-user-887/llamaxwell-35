#!/usr/bin/env python3
"""C1 corrected estimators (NEW file; originals untouched). CPU only, reads retained logs.
Run: python3 -I c1_block_estimator.py <LAB>   (default LAB below)
Formulas: period_change = period_B/period_A - 1 (negative = faster); throughput_gain = period_A/period_B - 1.
Block = one ABBA cycle; the cycle value uses ALL four runs (mean of both A, mean of both B).
"""
import re, sys, json, math, statistics as st
LAB = sys.argv[1] if len(sys.argv) > 1 else '<LAB>'
DAY = LAB + '/occamy35-day-20261112'
T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571}
def ci(xs):
    m = st.mean(xs); n = len(xs)
    if n < 2: return m, None, None
    h = T975[n-1] * st.stdev(xs) / math.sqrt(n); return m, m-h, m+h
def parse_speed(path, kind):
    runs = {}   # (cycle, arm) -> [values in log order]
    for l in open(path):
        m = re.match(r'sp_%s_c(\d)_([AB]): .*?%s=([\d.]+)' % (kind, 'pair_ms_mean' if kind == 'two' else 'one_ms_mean'), l)
        if m: runs.setdefault((int(m[1]), m[2]), []).append(float(m[3]))
    return runs
def speed_report(label, path, kind):
    r = parse_speed(path, kind)
    if not r: return None
    cyc = sorted({c for c, _ in r})
    out = {'label': label, 'kind': kind, 'n_runs_retained': sum(len(v) for v in r.values()), 'cycles': []}
    ds_block, ds_last, ds_first = [], [], []
    for c in cyc:
        a, b = r[(c, 'A')], r[(c, 'B')]
        assert len(a) == 2 and len(b) == 2, (label, c, a, b)
        blk = 100*(st.mean(b)/st.mean(a) - 1); last = 100*(b[-1]/a[-1] - 1); first = 100*(b[0]/a[0]-1)
        ds_block.append(blk); ds_last.append(last); ds_first.append(first)
        out['cycles'].append({'cycle': c, 'A_runs_ms': a, 'B_runs_ms': b, 'A_mean': st.mean(a), 'B_mean': st.mean(b),
                              'block_period_change_pct': blk, 'throughput_gain_pct': 100*(st.mean(a)/st.mean(b)-1),
                              'old_last_only_pct': last})
    for k, xs in (('block_all_runs', ds_block), ('old_last_A_last_B_only', ds_last), ('first_A_first_B_only', ds_first)):
        m, lo, hi = ci(xs); out[k] = {'mean_period_change_pct': m, 'ci95_t': [lo, hi], 'n_blocks': len(xs)}
    # pooled-run (not independent) ratio of grand means for reference
    A = [x for (c, a), v in r.items() if a == 'A' for x in v]; B = [x for (c, a), v in r.items() if a == 'B' for x in v]
    out['grand_mean_ratio_period_change_pct'] = 100*(st.mean(B)/st.mean(A)-1)
    out['grand_mean_throughput_gain_pct'] = 100*(st.mean(A)/st.mean(B)-1)
    out['position_drift'] = {'A1_mean': st.mean(r[(c,'A')][0] for c in cyc), 'A2_mean': st.mean(r[(c,'A')][1] for c in cyc),
                             'B1_mean': st.mean(r[(c,'B')][0] for c in cyc), 'B2_mean': st.mean(r[(c,'B')][1] for c in cyc)}
    return out
def b3_report(path, label):
    rows = [json.loads(l) for l in open(path) if l.startswith('{')]
    single = 's1' not in rows[0]
    slots = ['s0'] if single else ['s0', 's1']
    o = {'label': label, 'fixture': 'single 98000' if single else '98000+90000 prompt tokens, 1 ABBA block (4 server starts), 4 measured reps + 1 discarded warm per start',
         'n_rows': len(rows), 'servers': {}}
    arms = {}
    for r in rows:
        if r['warm']: continue
        arms.setdefault((r['pos'], r['arm']), []).append(r)
    res = {}
    for arm in 'AB':
        R = [r for (p, a), v in sorted(arms.items()) if a == arm for r in v]
        per = {}
        for s in slots:
            per[s+'_ms_per_tok_div64_script'] = st.mean(r[s]['predicted_ms']/r[s]['predicted_n'] for r in R)
            per[s+'_ms_per_interval_div63'] = st.mean(r[s]['predicted_ms']/(r[s]['predicted_n']-1) for r in R)
            per[s+'_reported_tps_63interval'] = st.mean(r[s]['predicted_per_second'] for r in R)
        per['sum_of_per_slot_reported_tps'] = st.mean(sum(r[s]['predicted_per_second'] for s in slots) for r in R)
        per['period_div64_mean_of_slots'] = st.mean(st.mean(r[s]['predicted_ms']/r[s]['predicted_n'] for s in slots) for r in R)
        per['period_div63_mean_of_slots'] = st.mean(st.mean(r[s]['predicted_ms']/(r[s]['predicted_n']-1) for s in slots) for r in R)
        # common-window style aggregates from the same rows
        per['committed_tokens_per_rep'] = sum(R[0][s]['predicted_n'] for s in slots)
        per['agg_tps_tokens_over_max_slot_predicted_ms'] = st.mean(sum(r[s]['predicted_n'] for s in slots)/(max(r[s]['predicted_ms'] for s in slots)/1000) for r in R)
        per['agg_tps_tokens_over_client_wall_incl_prompt_eval'] = st.mean(sum(r[s]['predicted_n'] for s in slots)/r['wall'] for r in R)
        per['n_measured_reps'] = len(R)
        res[arm] = per
    o['arms'] = res
    A, B = res['A'], res['B']
    o['period_change_pct_div63_equals_div64'] = 100*(B['period_div63_mean_of_slots']/A['period_div63_mean_of_slots']-1)
    o['throughput_gain_pct_from_period'] = 100*(A['period_div63_mean_of_slots']/B['period_div63_mean_of_slots']-1)
    o['sum_of_rates_gain_pct'] = 100*(B['sum_of_per_slot_reported_tps']/A['sum_of_per_slot_reported_tps']-1)
    o['agg_tokens_over_wall_gain_pct'] = 100*(B['agg_tps_tokens_over_client_wall_incl_prompt_eval']/A['agg_tps_tokens_over_client_wall_incl_prompt_eval']-1)
    o['agg_tokens_over_max_slot_gain_pct'] = 100*(B['agg_tps_tokens_over_max_slot_predicted_ms']/A['agg_tps_tokens_over_max_slot_predicted_ms']-1)
    # server-start-level (arm-position) values: pos0 A, pos1 B, pos2 B, pos3 A
    sv = {}
    for (p, a), v in sorted(arms.items()):
        sv[f'pos{p}_{a}'] = st.mean(st.mean(r[s]['predicted_ms']/(r[s]['predicted_n']-1) for s in slots) for r in v)
    o['server_start_period_div63_ms'] = sv
    Am = [sv[k] for k in sv if k.endswith('_A')]; Bm = [sv[k] for k in sv if k.endswith('_B')]
    o['server_start_level'] = {'A_mean': st.mean(Am), 'B_mean': st.mean(Bm), 'n_each': len(Am),
        'welch_like_note': 'n=2 server starts per arm: descriptive only; single ABBA block => no block-level CI'}
    o['block_count'] = 1
    return o
rep = {'device_total_cuda_MiB': 24463, 'nvidia_smi_total_MiB': 24576, 'nvidia_smi_reserved_MiB': 113,
       'usable_total_MiB': 24463,
       'margins_MiB': {k: 24463-v for k, v in {'serve_smoke_peak_B3_23108': 23108, 'dxt_tool_peak_23820': 23820, 'D0_deep_peak_23143': 23143}.items()},
       'old_hardcoded_margins_MiB': {'23108': 24576-23108, '23820': 24576-23820, '23143': 24576-23143}}
rep['N2SPLIT_D1B'] = {'two_slot_dxt': speed_report('N2SPLIT vs stock-GDN4-prod-env (A=N2SPLIT0,B=N2SPLIT1) 98K+90K', DAY+'/rounds/D1B/speed.log', 'two'),
                      'one_slot_dxt': speed_report('N2SPLIT single 98K', DAY+'/rounds/D1B/speed.log', 'one')}
rep['N2GDN_D1BF'] = {'two_slot_dxt': speed_report('N2GDN vs N2SPLIT-only (A=N2SPLIT1+N2GDN0,B=+N2GDN1)', DAY+'/rounds/D1BF/speed.log', 'two')}
rep['B3_server_two_slot'] = b3_report(DAY+'/rounds/B3/b3_deep.log', 'B3 vs GDN4 prod lib, A=gdn4 lib+prod env, B=b3 lib+N2SPLIT+FA_ADDR+N2GDN')
rep['B3_server_single'] = b3_report(DAY+'/rounds/B3/b3_deep1.log', 'B3 single 98K')
rep['ATT1_server_two_slot'] = b3_report(DAY+'/rounds/D1ATT/deep_abba.log', 'ATT1 FA_ADDR 2-slot ~94K, 4 reps/arm positions, 3 measured reps in last start')
json.dump(rep, open(sys.argv[2] if len(sys.argv) > 2 else DAY+'/evidence/C/c1_estimators.json', 'w'), indent=1)
print(json.dumps(rep, indent=1)[:9000])
