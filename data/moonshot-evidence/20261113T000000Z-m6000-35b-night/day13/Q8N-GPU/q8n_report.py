#!/usr/bin/env python3
"""Assemble Q8N/GPU_RESULT.md from a run dir. usage: q8n_report.py <RUN> <out.md>   (tables first; analyzer outputs verbatim)"""
import sys, os, json
R, outp = sys.argv[1], sys.argv[2]
def rd(p, d='(not produced)'):
    try: return open(os.path.join(R, p)).read().strip()
    except Exception: return d
md = ['# Q8N GPU result (measurement only)', '', 'Run dir: `%s`. Steps log at the end.' % R, '']
md += ['## 1. Exactness (native N vs generic G; K-only Nk vs Gk)', '', rd('exact/Q8N_EXACT.md'), '']
rj = rd('restore/restore.json', '')
md += ['## 2. Slot restore test', '']
if rj:
    r = json.loads(rj)
    md += ['| item | value |', '|---|---|', '| n_restored | %s |' % r.get('n_restored'), '| restore decode prompt_n | %s |' % r.get('restore_decode', {}).get('prompt_n'), '| re-prefill (prompt_n>=64) | %s |' % r.get('reprefill'),
           '| restore tokens (16) == no-restore tokens | %s (first diff idx %s) |' % (r.get('tokens_equal'), r.get('first_diff_idx')), '| no-restore source | %s |' % r.get('no_restore_src'),
           '| restore tokens | %s |' % r.get('restore_tokens'), '| no-restore tokens | %s |' % r.get('no_restore_tokens')]
    if r.get('reprefill'): md.append('\nDEFECT: restore of a q8 slot re-prefills; use in-process prefill fixtures (not a blocker).')
else: md.append('(not produced)')
md += ['', '## 3. Two-slot deep ABBA and extra units', '', rd('srv/Q8N_SRV.md'), '']
pj = rd('prof/prof.json', '')
md += ['## 4. nvprof, one decode step at 98K (slot 0)', '']
if pj:
    p = json.loads(pj); md += ['| arm | ofa kernel | ofa ms/step | ofa ms/launch | GPU busy ms/step (window) | grid | regs |', '|---|---|---|---|---|---|---|']
    for a, v in p.items(): md.append('| %s | %s | %s | %s | %s | %s | %s |' % (a, v.get('ofa_kernel', v.get('error')), *['%.3f' % v[k] if k in v else '' for k in ('ofa_ms_per_step', 'ofa_ms_per_launch', 'gpu_busy_ms_per_step_window')], v.get('grid', ''), v.get('regs', '')))
else: md.append('(not produced)')
md += ['', '## 5. Single 98K / 90K+4K append', '', rd('srv5/Q8N_SRV.md'), '', '## Steps log', '', '```', rd('steps.log'), '```', '']
open(outp, 'w').write('\n'.join(md))
