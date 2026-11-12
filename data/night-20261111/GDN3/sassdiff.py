import sys,re,subprocess,difflib
def get(p,name):
    out=subprocess.run(['cuobjdump','-sass','-fun',name,p],capture_output=True,text=True).stdout
    return [re.sub(r'/\*[0-9a-f]{4}\*/','',l).strip() for l in out.splitlines() if l.strip().startswith('/*')]
a=get(sys.argv[1],sys.argv[3]); b=get(sys.argv[2],sys.argv[4])
d=[l for l in difflib.unified_diff(a,b,lineterm='',n=0)]
print(len(a),len(b),len(d)); print('\n'.join(d[:30]))
