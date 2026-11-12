import sys,re,subprocess
def funcs(p):
    out=subprocess.run(['cuobjdump','-sass',p],capture_output=True,text=True).stdout
    d={};cur=None
    for l in out.splitlines():
        m=re.match(r'\s*Function : (\S+)',l)
        if m: cur=m.group(1); d[cur]=[]; continue
        if cur and ('occamy_gdn' in cur) :
            l=re.sub(r'/\*[0-9a-f]{4}\*/','',l); d[cur].append(l.strip())
    return {k:v for k,v in d.items() if 'occamy_gdn' in k}
a=funcs(sys.argv[1]); b=funcs(sys.argv[2])
def norm(k): return re.sub(r'ILi(\d)ELi(\d)ELi(\d)ELi(\d)EEE','ILi\\1ELi\\2ELi\\3ELi\\4ELi0EEE',k)
for k,v in sorted(a.items()):
    k2=norm(k) if norm(k) in b else k
    print(k, 'IDENT' if b.get(k2)==v else 'DIFF', len(v))
print('new-only:',[k for k in b if k not in {norm(x) for x in a} and k not in a])
