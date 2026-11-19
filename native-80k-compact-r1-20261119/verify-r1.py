#!/usr/bin/env python3
from pathlib import Path
import argparse
import hashlib
import json
import re
import subprocess

HERE=Path(__file__).resolve().parent
C=HERE.parent
BASE=C/'break100-compact-model'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check():
    old=json.loads((BASE/'profile.json').read_text())
    new=json.loads((HERE/'profile.json').read_text())
    assert old.keys()==new.keys() and [key for key in old if old[key]!=new[key]]==['libllama-server-impl.so']
    assert sha(HERE/'supervise')==sha(BASE/'supervise')
    for name in ('libggml-cuda.so','libllama.so.0.6.0','libllama-server-impl.so'):
        assert sha(HERE/name)==new[name]
    def dynamic(path):
        text=subprocess.check_output(['readelf','-d',path],text=True)
        return [m.groups() for line in text.splitlines() if (m:=re.search(r'\((NEEDED|SONAME|RPATH|RUNPATH)\).*?\[(.*?)\]',line))]
    parent, server=BASE/'libllama-server-impl.so',HERE/'libllama-server-impl.so'
    assert dynamic(parent)==dynamic(server)
    def exports(path):
        return {tuple(x.split()[:2]) for x in subprocess.check_output(['nm','-D','--defined-only','--format=posix',path],text=True).splitlines()}
    old_exports,new_exports=exports(parent),exports(server)
    removed=old_exports-new_exports
    assert removed == {('_ZN14occamy_compact13fixed_samplerERK22common_params_sampling', 'W')}
    assert 'compiled runtime parser yields NONE+MTP' in (HERE/'parser-check.log').read_text()
    assert 'exact-runtime unresolved symbol check passed' in (HERE/'loader-check.log').read_text()
    return {'base_source_freeze':sha(BASE/'FROZEN.sha256'),'baseline_software':old,'compact_only_software':new,
            'dynamic_needed_soname_searchpath_equal':True,'removed_inline_policy_weak_symbols':len(removed),
            'compiled_parser_has_inert_none':True,'down4_environment':'0','down4_absolute_counter_expected':0}


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--check',action='store_true')
    result=json.dumps(check(),indent=2,sort_keys=True)+'\n'
    if parser.parse_args().check:
        assert (HERE/'build.json').read_text()==result
    else:
        print(result,end='')
