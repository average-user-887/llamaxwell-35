"""Apply the portable patches and optionally hydrate private source pins on the operator's machine."""
import argparse,hashlib,json,subprocess,tempfile
from pathlib import Path
here=Path(__file__).resolve().parent
def sha(b):return hashlib.sha256(b).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('revision',choices=['initial','r1','r2']);p.add_argument('output',type=Path)
 p.add_argument('--private-inputs',type=Path);args=p.parse_args()
 subs=json.loads(args.private_inputs.read_text()) if args.private_inputs else {}
 if args.private_inputs and set(subs)!=set(json.loads((here/'private-input-keys.json').read_text())):raise ValueError('private input key set differs')
 pins=json.loads((here/'source-pins.json').read_text())
 args.output.mkdir(exist_ok=False,parents=True)
 for label,name in [('probe','native_probe.py'),('release','release.sh'),('observation','add_observation.py')]:
  text=(here/(label+'.base')).read_bytes()
  revisions=['initial','r1','r2']
  if label=='observation':revisions=revisions[1:]
  with tempfile.TemporaryDirectory() as tmp:
   path=Path(tmp)/name;path.write_bytes(text)
   for rev in revisions:
    if label=='observation' and args.revision=='initial':break
    delta=here/(rev+'.'+label+'.patch')
    if delta.stat().st_size:subprocess.run(['patch','--silent',str(path),str(delta)],check=True)
    if rev==args.revision:break
   text=path.read_bytes()
  key=pins[args.revision+'.'+label]
  if sha(text)!=key['portable_source_sha256']:raise ValueError('portable source digest differs')
  if subs:
   for k,v in subs.items():text=text.replace(('@PRIVATE_'+k+'@').encode(),v.encode())
   if sha(text)!=key['actual_source_sha256']:raise ValueError('hydrated source digest differs')
  dest=args.output/name;dest.write_bytes(text)
  if label=='release':dest.chmod(0o755)
 for name in ('audit.cpp','match.inc','build.sh'):
  (args.output/name).write_bytes((here/name).read_bytes())
 print('Verified assembled source for',args.revision)
if __name__=='__main__':main()
