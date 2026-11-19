"""Inspect each owned full-operator process and evaluate predeclared gates."""
import hashlib,json,math,re,struct,sys
from pathlib import Path
out=Path(sys.argv[1]).resolve()
build=json.loads((out/'build.json').read_text())
pairs=('a-control','a-candidate','b-candidate','b-control')
cases={**{x:(81920,1024) for x in pairs},'edge-low':(80128,1024),'edge-high':(83200,1024),'padded':(81920,2048)}
times,seen,values,plans={},{},{},{}
for label,(width,pitch) in cases.items():
    cand=not label.endswith('control')
    s=(out/(label+'.log')).read_text()
    assert build['candidate_sha256' if cand else 'control_sha256']+'  /candidate/libggml-cuda.so' in s,label
    assert build['harness_sha256']+'  /candidate/test-backend-ops' in s,label
    assert 'loaded CUDA backend from /candidate/libggml-cuda.so' in s and 'Quadro M6000' in s,label
    assert f'q2_r2_top16_dispatch n_q=2 n_kv={width} nkvh=2 plans=4' in s,label
    assert f'q2_r2_probe4_dispatch width={width} plans=4' in s,label
    if cand:
        marker=f'q2_union_dispatch calls=1 width={width} pitch={pitch} split=24 threads=128 tile=16 fallback_split=12'
        assert s.count(marker)==1 and f'dispatch calls=10 n_q=2 n_kv={width} ' in s,label
        assert f'q2_r2_probe4_meta entries={width//256*4} exact=1' in s,label
        assert f'q2_r2_top16_check n_kv={width} plans=4 scores={width//256*4} outputs=8192 plan_diff=0 score_diff=0 ' in s,label
        assert f'q2_union_fallback width={width} pitch={pitch} invalid_plans=1 old_head_words=4096 exact=1 all_finite=1' in s,label
        assert 'q2_r2_probe4_edge' in s and re.search(r'q2_r2_probe4_edge entries=\d+ mask_bad=[1-9]\d* score_bad=[1-9]\d* exact=1',s),label
        check=re.search(rf'q2_union_check width={width} pitch={pitch} plans=4 scores={width//256*4} outputs=8192 nonfinite=0 outside=0 max_abs=([^ ]+) nrmse=([^\s]+)',s)
        assert check,label
        max_abs,nrmse=map(float,check.groups())
        assert math.isfinite(max_abs) and 0<=nrmse<=1e-5,label
        seen[label]=dict(width=width,pitch=pitch,scores=width//256*4,max_abs=max_abs,nrmse=nrmse,mixed_fallback_exact_words=4096)
    else:
        assert 'q2_union_dispatch' not in s and f'q2_r2_stride_dispatch width={width} k_token_bytes={pitch} v_token_bytes={pitch}' in s,label
    plan_lines=re.findall(r'^occamy_sparse_plan call=1 n_q=2 (.+)$',s,re.M)
    assert len(plan_lines)==4,label
    plans[label]=[]
    for line in plan_lines:
        assert f'n_kv={width} ok=1 invalid=0 fallback=0' in line and 'full_warps=0' in line and 'old_blocks=16' in line
        row=int(re.search(r'row=(\d+)',line)[1])
        assert f'visible_end={width-261+row} ' in line
        ids=list(map(int,line.split('block_ids=')[1].split(',')))
        assert len(ids)==16 and ids==sorted(set(ids))
        plans[label].append(re.sub(r'engaged_warps=\d+ ','',line))
        assert int(re.search(r'engaged_warps=(\d+)',line)[1])>0
    pattern=rf'FLASH_ATTN_EXT\([^\n]+kv={width},nb=2,.*kv_view={int(pitch==2048)},.*mask=mrow_causal,tail=260\):\s*(\d+) runs -\s*([.0-9]+) us/run'
    m=re.findall(pattern,s)
    assert len(m)==1,(label,m)
    runs,us=m[0]; assert int(runs)>0 and float(us)>0,label
    times[label]=dict(width=width,pitch=pitch,runs=int(runs),whole_operator_us=float(us))
    raw=(out/(label+'.bin')).read_bytes()
    assert len(raw)==8192*4,label
    vals=struct.unpack('<8192f',raw)
    assert all(map(math.isfinite,vals)),label
    values[label]=(hashlib.sha256(raw).hexdigest(),vals)
for label in (*pairs,'padded'):
    assert plans[label]==plans['a-control'],label
assert values['a-control'][0]==values['b-control'][0]=='84bc271a57313f79787f30ab571059030514c8948deda0c6b200aa369fda18ef'
assert values['a-candidate'][0]==values['b-candidate'][0]==values['padded'][0]
for x,y in zip(values['a-candidate'][1],values['a-control'][1]):
    assert abs(x-y)<=1e-5+1e-4*abs(y)
savings={x:round(times[f'{x}-control']['whole_operator_us']-times[f'{x}-candidate']['whole_operator_us'],2) for x in ('a','b')}
# Derive unique logical head/token counts from the physically emitted plans.
# These are logical data demands, without DRAM counters or cache assumptions.
union_counts={}
for label,ps in plans.items():
    sets={}
    for line in ps:
        row,head=map(int,re.search(r'row=(\d+) kvh=(\d+)',line).groups())
        recent,end=map(int,re.search(r'recent_begin=(\d+) visible_end=(\d+)',line).groups())
        tokens=set(range(recent,end))
        for page in map(int,line.split('block_ids=')[1].split(',')):
            tokens.update(range(256*page,256*(page+1)))
        sets[row,head]=tokens
    union_counts[label]=sum(len(sets[0,h]|sets[1,h]) for h in range(2))
result=dict(unique_logical_head_tokens=union_counts,technical_pass=True,build=build,performance=times,diagnostics=seen,output_sha256={k:v[0] for k,v in values.items()},
            dense_81920_order=pairs,dense_81920_saved_us=savings,solo_70us_each=min(savings.values())>=70,
            compound_50us_operator_screen_only=min(savings.values())>=50,
            independently_qualified_companion_ms=None,compound_promotion=False)
s=json.dumps(result,indent=2)+'\n'
(out/'result.json').write_text(s)
print(s,end='')
