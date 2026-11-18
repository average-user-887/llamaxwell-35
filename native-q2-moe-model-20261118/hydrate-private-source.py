"""Reconstruct the two local source literals from owner-supplied paths and verify exact original SHA256."""
import argparse
import hashlib
import json
from pathlib import Path

def write_verified(path, text, expected):
    actual=hashlib.sha256(text.encode()).hexdigest()
    if actual!=expected:
        raise ValueError('supplied private reference does not reconstruct the pinned original source')
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        if path.read_text()!=text:raise FileExistsError('destination already contains a different source file')
    else:path.write_text(text)

def main():
    here=Path(__file__).resolve().parent
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ['campaign','native_reference','serial_reference','destination']:
        ap.add_argument(name,type=Path)
    args=ap.parse_args(); campaign=args.campaign.resolve()
    native=args.native_reference.resolve().relative_to(campaign).as_posix()
    serial=args.serial_reference.resolve().relative_to(campaign).as_posix()
    pins=json.loads((here/'source-pins.json').read_text())
    raw=(here/'offline.py').read_text().replace('import json,os','import json').replace('Path(os.environ["D1_SERIAL_REPLAY"])',"p.C/"+repr(serial))
    write_verified(args.destination/'offline.py',raw,pins['initial_source_sha256']['offline.py'])
    old=(here.parent/'native-q8-on-serial-20261118/serial_probe.py').read_text().replace('import os\n','').replace('Path(os.environ["N88_PRIVATE_REFERENCE"])','C / '+json.dumps(native))
    digest=next(s.split()[0] for s in (here/'INPUTS.sha256').read_text().splitlines() if s.endswith('/serial_probe.py'))
    write_verified(args.destination/'serial_probe.py',old,digest)
    print('Recovered both original private-layout source fingerprints, without copying stream data.')
if __name__=='__main__':main()
