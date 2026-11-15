import re,glob,json,os,sys
M='<LAB>/occamy35-night-20261114'
V=len(sys.argv)>1
for lg in sorted(glob.glob(M+'/2[45]/run-*/*/runs/*/server.log')):
    t={};order=[]
    for l in open(lg,errors='replace'):
        m=re.search(r'id\s+(\d) \| task (\d+) \| new prompt.*task.n_tokens = (\d+)',l)
        if m: k=(int(m[1]),int(m[2])); t[k]=dict(N=int(m[3]),first=None,ev=None,gen=None,end=None); order.append(k); continue
        m=re.search(r'slot\s+(?:operator\(\)|release|print_timing): id\s+(\d) \| task (\d+) \| (.*)',l)
        if not m or (int(m[1]),int(m[2])) not in t: continue
        x=t[(int(m[1]),int(m[2]))]; r=m[3]
        c=re.match(r'cached n_tokens = (\d+)',r)
        if c and x['first'] is None: x['first']=int(c[1])
        c=re.match(r'stop processing: n_tokens = (\d+)',r)
        if c: x['end']=int(c[1])
        c=re.match(r'prompt eval time =.*?/\s+(\d+) tokens',r)
        if c: x['ev']=int(c[1]); x['evms']=float(re.search(r'=\s+([\d.]+) ms',r)[1])
        c=re.match(r'\s*eval time =.*?/\s+(\d+) tokens',r)
        if c: x['gen']=int(c[1])
    prev={};per={}
    tag=lg.replace(M+'/','').replace('/runs/b0_p0_prod/server.log','')
    for k in order:
        s=k[0];x=t[k]
        if x['ev'] is None or x['N']<1000 or x['first'] is None: continue
        app=x['N']-(prev[s] if s in prev else x['first'])
        a=per.setdefault(s,[0,0,0,0.0]);a[0]+=app;a[1]+=x['ev'];a[2]+=1;a[3]+=x['evms']
        if V: print('  ',tag,s,k[1],'N',x['N'],'first',x['first'],'app',app,'ev',x['ev'])
        prev[s]=x['end'] or x['N']
    print(tag,json.dumps(per))
