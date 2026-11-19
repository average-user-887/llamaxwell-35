#!/usr/bin/env python3
"""Bind the original R1 guard and quality validation to the typed down4-on profile."""
import difflib
from pathlib import Path

HERE=Path(__file__).resolve().parent
R1=HERE.parent/'break100-compact-model-r1'


def replace_once(text, old, new):
    assert text.count(old)==1,old
    return text.replace(old,new)


def declare(file,old,new):
    (HERE/file).write_text(new)
    (HERE/(file+'.patch')).write_text(''.join(difflib.unified_diff(old.splitlines(True),new.splitlines(True),f'a/{file}',f'b/{file}')))


old=s=(R1/'model.py').read_text()
s=replace_once(s, "    env = [item.replace('GGML_CUDA_OCCAMY_MOE_Q2_DOWN4=1','GGML_CUDA_OCCAMY_MOE_Q2_DOWN4=0') if item==parent.flag else item for item in env]\n", '')
start=s.index('def disabled_down4(')
end=s.index('def probe(',start)
s=s[:start]+s[end:]
s=replace_once(s,"return {'down4':disabled_down4(out,metrics),", "return {'down4':parent.probe(mode(stage),out,metrics),")
s=replace_once(s,'''    need(all(profile[k] == parent.profile[k] for k in ('exe','libllama-common.so.0.6.0')), 'unrelated runtime changed')''','''    r1_card=C/'break100-compact-model-r1'
    r1=parent.module(r1_card/'model.py','compact_r1_exact')
    need(profile == r1.profile, 'R2 runtime differs from R1')
    for stage in STAGES:
        old_env,old_args=r1.settings(stage)
        new_env,new_args=settings(stage)
        need(old_env.count('GGML_CUDA_OCCAMY_MOE_Q2_DOWN4=0')==1 and new_args==old_args and
             new_env==['GGML_CUDA_OCCAMY_MOE_Q2_DOWN4=1' if value=='GGML_CUDA_OCCAMY_MOE_Q2_DOWN4=0' else value for value in old_env],
             'R2 stage differs beyond the down4 flag')''')
declare('model.py',old,s)

old=s=(R1/'release.sh').read_text()
s=replace_once(s,'overlay=$campaign/break100-compact-model\n','overlay=$campaign/break100-compact-model\nr1=$campaign/break100-compact-model-r1\n')
s=replace_once(s,'  "$overlay/release.sh" --check\n  python3 -B "$card/verify-r1.py" --check','  "$r1/release.sh" --check\n  cmp "$r1/profile.json" "$card/profile.json"\n  cmp "$r1/supervise" "$card/supervise"')
s=s.replace('$card/libllama-server-impl.so:$n1/libllama-server-impl.so','$r1/libllama-server-impl.so:$n1/libllama-server-impl.so')
s=s.replace('break100.compact.r1.owner','break100.compact.r2.owner').replace('nmmoon-break100-compact-r1-','nmmoon-break100-compact-r2-').replace('BREAK100_COMPACT_R1_RECEIPT','BREAK100_COMPACT_R2_RECEIPT')
declare('release.sh',old,s)
