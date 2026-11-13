#!/usr/bin/env python3
"""Q8N server-gate summary. usage: python3 -I q8n_srv_analyze.py <OUT>   reads <OUT>/units.jsonl, writes <OUT>/Q8N_SRV.md
Per arm and unit: valid n, median decode tok/s (both_deep: per slot and sum), single_98k decode tok/s, append_4k prefill tok/s, peak VRAM MiB.
Paired ratio Q/F (and G/F) per cycle; release gate for a speed claim stays the owner's >=1% rule."""
import sys, json, statistics as st
def med(x): return st.median(x) if x else float('nan')
def main(out):
    U = [json.loads(l) for l in open(out + '/units.jsonl')]; units = [u for u in U if u.get('unit') in ('both_deep', 'single_98k', 'append_4k')]
    def metric(u):
        r = u['res']
        if u['unit'] == 'both_deep': return r['s0']['predicted_per_second'] + r['s1']['predicted_per_second']
        if u['unit'] == 'single_98k': return r['s0']['predicted_per_second']
        return r['s1']['prompt_per_second']
    md = ['# Q8N server gates (analyzer output)', '', '| unit | arm | valid/n | median | per-cycle | peak VRAM MiB |', '|---|---|---|---|---|---|']
    arms = sorted({u['arm'] for u in units}); by = {}
    for k in ('both_deep', 'single_98k', 'append_4k'):
        for a in arms:
            xs = [u for u in units if u['unit'] == k and u['arm'] == a]; v = [u for u in xs if u['valid']]
            by[k, a] = {}
            for u in v: by[k, a].setdefault(u['cycle'], []).append(metric(u))
            md.append('| %s | %s | %d/%d | %.2f | %s | %s |' % (k, a, len(v), len(xs), med([metric(u) for u in v]),
                      ' '.join('%.2f' % med(by[k, a][c]) for c in sorted(by[k, a])), max([u['vram_peak_mib'] for u in xs], default=0)))
    md += ['', 'Paired per-cycle ratio vs F (median over cycles):']
    for k in ('both_deep', 'single_98k', 'append_4k'):
        for a in arms:
            if a == 'F': continue
            rs = [med(by[k, a][c]) / med(by[k, 'F'][c]) for c in sorted(by.get((k, a), {})) if c in by.get((k, 'F'), {})]
            md.append('- %s %s/F: %s -> median %+.2f%%' % (k, a, ' '.join('%.4f' % r for r in rs), 100 * (med(rs) - 1)) if rs else '- %s %s/F: n/a' % (k, a))
    md += ['', 'Session peak VRAM MiB: ' + ', '.join('%s %d' % (u['arm'], u['session_vram_peak_mib']) for u in U if 'session' in u)]
    b = [u for u in U if u.get('unit') == 'build_q8_slots']
    if b: md.append('q8 slot build (prefill, D0 shape): s0 %.1f tok/s, s1 %.1f tok/s' % (b[0]['s0_prefill']['s0']['prompt_per_second'], b[0]['s1_prefill']['s1']['prompt_per_second']))
    open(out + '/Q8N_SRV.md', 'w').write('\n'.join(md) + '\n'); print('\n'.join(md))
if __name__ == '__main__': main(sys.argv[1])
