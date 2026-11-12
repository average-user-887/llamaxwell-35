import json,sys,statistics as S,hashlib
d=sys.argv[1]; U=[json.loads(l) for l in open(d+'/units.jsonl')]
T={1:12.706,2:4.303,3:3.182,4:2.776,5:2.571}
M=[('ttft_s','TTFT s'),('max_gap_s','max gap s'),('wall_s','wall s'),('fg_prefill_tps','prefill tok/s'),('vram_peak_mib','peak VRAM MiB')]
def val(r,k): return r.get(k)
out=['# G1b table: scheduler cap 0 (A) vs --mixed-batch-size 1024 (B)','','Measurement only. Timed units (identity passes excluded). Cell = mean [min, max] (n).','']
tim=[r for r in U if not r['identity_pass']]
ncyc=sorted({r['cycle'] for r in tim}); out.append('Cycles present: %s; timed units: %d; invalid: %d'%(ncyc,len(tim),sum(not r['valid'] for r in tim)));out.append('')
for w in 'ab':
    nm={'a':'(a) restored-90K append (+4096)','b':'(b) e3 deep stream + 16K arrival'}[w]
    out+=['## '+nm,'','| metric | A cap0 | B cap1024 | paired B-A (mean, 95% CI over cycles) |','|---|---|---|---|']
    for k,lab in M:
        cells=[];
        for arm in 'AB':
            v=[val(r,k) for r in tim if r['kind']==w and r['arm']==arm and r['valid'] and val(r,k) is not None]
            cells.append('%.3f [%.3f, %.3f] (%d)'%(S.mean(v),min(v),max(v),len(v)) if v else 'n/a')
        dl=[]
        for c in ncyc:
            a=[val(r,k) for r in tim if r['kind']==w and r['arm']=='A' and r['cycle']==c and r['valid'] and val(r,k) is not None]
            b=[val(r,k) for r in tim if r['kind']==w and r['arm']=='B' and r['cycle']==c and r['valid'] and val(r,k) is not None]
            if a and b: dl.append(S.mean(b)-S.mean(a))
        if len(dl)>=2:
            m=S.mean(dl); h=T.get(len(dl),1.96)*S.stdev(dl)/len(dl)**.5; am=S.mean([val(r,k) for r in tim if r['kind']==w and r['arm']=='A' and r['valid'] and val(r,k) is not None])
            ds='%+.3f +/- %.3f (%+.1f%%, n=%d cycles)'%(m,h,100*m/am,len(dl))
        else: ds='n/a'
        out.append('| %s | %s | %s | %s |'%(lab,cells[0],cells[1],ds))
    out.append('')
out+=['## Output receipt per timed unit','','fg/bg: authoritative tokens_predicted / streamed ID count / non-ASCII events / content sha256[:12]. id_gap must be <= nonascii events.','','| cyc.pos | arm | wl | valid | fg pred/ids/nonascii/sha | bg pred/ids/nonascii/sha | receipt ok |','|---|---|---|---|---|---|---|']
def rc(s,n): return '%s/%d/%d/%s'%(s['n_gen_server'],s['n_ids_visible'],s['nonascii_events'],s['content_sha256'][:12])
for r in tim:
    f,b=r['fg'],r['bg']; ok=all(s['n_gen_server']==n and s['id_gap'] is not None and 0<=s['id_gap']<=s['nonascii_events'] and s['tokens_cached']==s['tokens_evaluated']+n-1 for s,n in ((f,32),(b,512 if r['kind']=='a' else 256)))
    out.append('| %d.%d | %s | %s | %s | %s | %s | %s |'%(r['cycle'],r['pos'],r['arm'],r['kind'],r['valid'],rc(f,32),rc(b,0),'ok' if ok else 'CHECK'))
out+=['','## Identity pass (stream:false, exact IDs sha256[:12] of fg/bg)','']
for r in U:
    if r['identity_pass']: out.append('- arm %s wl %s: fg %s bg %s (n %s/%s)'%(r['arm'],r['kind'],hashlib.sha256(json.dumps(r['fg']['ids']).encode()).hexdigest()[:12],hashlib.sha256(json.dumps(r['bg']['ids']).encode()).hexdigest()[:12],r['fg']['n_gen_server'],r['bg']['n_gen_server']))
open(d+'/G1B_TABLE.md','w').write('\n'.join(out)+'\n'); print('\n'.join(out[:40]))
