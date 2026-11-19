"""Write a flat digest manifest for this packet."""
from hashlib import sha256
from pathlib import Path
root=Path(__file__).resolve().parent
files=sorted(p for p in root.iterdir() if p.name!='SHA256SUMS')
assert all(p.is_file() for p in files)
(root/'SHA256SUMS').write_text(''.join(f'{sha256(p.read_bytes()).hexdigest()}  {p.name}\n' for p in files))
