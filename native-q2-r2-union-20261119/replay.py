"""Recompute timing verdict, plan geometry and reported numerical witnesses."""
import json,re,math
from pathlib import Path
root=Path(__file__).resolve().parent
obs=root/'observed'
r=json.loads((obs/'result.json').read_text())
b=json.loads((root/'build.json').read_text())
assert r['build']==b
cases=('a-control','a-candidate','b-candidate','b-control','edge-low','edge-high','padded')
us={}
for name in cases:
    s=(obs/(name+'.txt')).read_text()
    c=not name.endswith('control')
    p=r['performance'][name]; w,pitch=p['width'],p['pitch']
    assert b['candidate_sha256' if c else 'control_sha256']+'  /candidate/libggml-cuda.so' in s
    assert b['harness_sha256']+'  /candidate/test-backend-ops' in s
    m,=re.findall(rf'FLASH_ATTN_EXT\([^\n]+kv={w},nb=2,.*kv_view={int(pitch==2048)},.*tail=260\):\s*(\d+) runs -\s*([.0-9]+) us/run',s)
    assert int(m[0])==p['runs'] and float(m[1])==p['whole_operator_us']
    us[name]=float(m[1])
    kv={}
    for row,h,lo,hi,ids in re.findall(r'^occamy_sparse_plan call=1 n_q=2 row=(\d+) kvh=(\d+) .*?recent_begin=(\d+) visible_end=(\d+).*?block_ids=([0-9,]+)$',s,re.M):
        a=set(range(int(lo),int(hi)))
        for block in map(int,ids.split(',')): a.update(range(256*block,256*(block+1)))
        kv[int(row),int(h)]=a
    assert len(kv)==4 and sum(len(kv[0,h]|kv[1,h]) for h in range(2))==r['unique_logical_head_tokens'][name]
    if c:
        d=r['diagnostics'][name]
        assert f'q2_union_dispatch calls=1 width={w} pitch={pitch} split=24 threads=128 tile=16 fallback_split=12' in s
        assert f'q2_r2_probe4_meta entries={w//64} exact=1' in s
        assert f'q2_r2_top16_check n_kv={w} plans=4 scores={w//64} outputs=8192 plan_diff=0 score_diff=0' in s
        assert f'q2_union_fallback width={w} pitch={pitch} invalid_plans=1 old_head_words=4096 exact=1 all_finite=1' in s
        num=re.search(r'q2_union_check .*?nonfinite=0 outside=0 max_abs=([.eE0-9+-]+) nrmse=([.eE0-9+-]+)',s)
        assert num and float(num[1])==d['max_abs'] and float(num[2])==d['nrmse']<=1e-5
    assert '2/2 backends passed' in s
delta={x:round(us[x+'-control']-us[x+'-candidate'],2) for x in ('a','b')}
assert delta==r['dense_81920_saved_us'] and (min(delta.values())>=70)==r['solo_70us_each'] and (min(delta.values())>=50)==r['compound_50us_operator_screen_only']
assert all(r['local_array_equality'].values()) and 'runner_exit=0\ncleanup_exit=0' in (obs/'status.txt').read_text()
print('Seven physical chronologies validated; mirrored savings in microseconds:',delta)
