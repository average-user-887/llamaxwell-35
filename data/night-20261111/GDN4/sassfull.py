import sys,re
# sassfull.py <old.sass> <new.sass>: compare every function's SASS (addresses stripped).
# gdn_fused gained a 5th template int (RP) in GDN3: old 4-int names map to <...,0>.
def funcs(p):
    d={};cur=None
    for l in open(p,errors='replace'):
        m=re.match(r'\s*Function : (\S+)',l)
        if m: cur=m.group(1); d[cur]=[]; continue
        if cur is None: continue
        s=l.strip()
        if s.startswith('/*') and re.match(r'/\*[0-9a-f]{4}\*/',s):
            d[cur].append(re.sub(r'/\*[0-9a-f]{4}\*/','',s).strip())
    return d
def norm(k):
    if 'gdn_fused' in k: return re.sub(r'ILi(\d+)ELi(\d+)ELi(\d+)ELi(\d+)EEE',r'ILi\1ELi\2ELi\3ELi\4ELi0EEE',k)
    return k
a={norm(k):v for k,v in funcs(sys.argv[1]).items()}; b=funcs(sys.argv[2])
added=[k for k in b if k not in a]; removed=[k for k in a if k not in b]
changed=[k for k in a if k in b and a[k]!=b[k]]
print('old',len(a),'new',len(b)); print('added',added); print('removed',removed); print('changed',changed)
print('VERDICT', 'IDENT_PLUS_ADDED' if not removed and not changed else 'DIFF')
