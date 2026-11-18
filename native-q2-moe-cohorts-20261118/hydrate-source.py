"""Restore typed local literals and verify the original source digests."""
import argparse, hashlib, json
from pathlib import Path
root=Path(__file__).resolve().parent
parser=argparse.ArgumentParser()
parser.add_argument("output",type=Path)
parser.add_argument("--private-inputs",type=Path)
args=parser.parse_args()
pins=json.loads((root/"source-pins.json").read_text())
values=json.loads(args.private_inputs.read_text()) if args.private_inputs else None
keys={k for rec in pins.values() for k in rec["private_input_keys"]}
if values is not None and set(values)!=keys:raise ValueError("incorrect private input names")
args.output.mkdir(parents=True,exist_ok=False)
for name,rec in pins.items():
    b=(root/name).read_bytes()
    assert hashlib.sha256(b).hexdigest()==rec["portable_sha256"]
    if values is not None:
        for k in rec["private_input_keys"]:b=b.replace(("@PRIVATE_"+k+"@").encode(),values[k].encode())
        assert hashlib.sha256(b).hexdigest()==rec["observed_source_sha256"]
    dest=args.output/name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(b)
    if dest.suffix==".sh":dest.chmod(0o755)
print("Original source verified" if values is not None else "Portable source verified; private keys: "+", ".join(sorted(keys)))
