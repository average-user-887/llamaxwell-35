#!/usr/bin/env python3
# CPU only. Builds public-text fixtures from llama.cpp source clone (disjoint file sets by sorted index mod 4).
# usage: build_text.py <clone> <outdir>
import sys,os,hashlib,json,subprocess
clone,out=sys.argv[1],sys.argv[2]
ext=('.cpp','.h','.cu','.cuh','.py')
fs=sorted(os.path.relpath(os.path.join(r,f),clone) for r,d,fl in os.walk(clone) if '.git' not in r.split(os.sep) for f in fl
  if f.endswith(ext) and 2000<os.path.getsize(os.path.join(r,f))<300000)
sets={0:[],1:[],2:[],3:[]}
for i,f in enumerate(fs): sets[i%4].append(f)
def cat(files,tag,limit):
    s=''
    for f in files:
        s+='### %s file %s\n'%(tag,f)+open(os.path.join(clone,f),errors='replace').read()+'\n'
        if len(s)>=limit: break
    return s
man={'clone_head':subprocess.run(['git','-C',clone,'rev-parse','HEAD'],capture_output=True,text=True).stdout.strip(),'files':{}}
def w(name,txt,files):
    open(os.path.join(out,name+'.txt'),'w').write(txt)
    man['files'][name]={'chars':len(txt),'sha256':hashlib.sha256(txt.encode()).hexdigest(),'src_files':files}
w('d0',cat(sets[0],'D0',700000),sets[0]); w('d1',cat(sets[1],'D1',700000),sets[1]); w('sentinel',cat(sets[3],'SENT',3000),sets[3][:3])
# pool of 16K prompt segments (uncached), each starts with a unique marker line; drawn from set 2
pool=[];seg=0;cur=''
for f in sets[2]:
    cur+=open(os.path.join(clone,f),errors='replace').read()+'\n'
    if len(cur)>=75000:
        pool.append(cur); cur=''
for k,p in enumerate(pool): w('u%02d'%k,'### UNCACHED SEGMENT %02d of llama.cpp source\n'%k+p,[])
man['n_u']=len(pool)
json.dump(man,open(os.path.join(out,'text_manifest.json'),'w'),indent=1)
print({k:v['chars'] for k,v in man['files'].items()})
