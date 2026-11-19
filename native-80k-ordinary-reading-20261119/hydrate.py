"""Substitute the one typed interpreter path and require the exact owner SHA."""
from hashlib import sha256
from pathlib import Path
import re
import sys

from replay import HERE,load,need,verify_sources


def hydrate(python):
    verify_sources()
    need(Path(python).is_absolute() and re.fullmatch(r'/[A-Za-z0-9_./+@-]+',python) is not None,'invalid interpreter path')
    pins=load('source-pins.json')
    rel=pins['relocation']
    template=(HERE/'source'/rel['template_file']).read_bytes()
    content=template.replace(rel['placeholder'].encode(),python.encode())
    need(sha256(content).hexdigest()==pins['exact_owner_scripts_sha256'][rel['source_name']],'owner source digest differs')
    return content


if __name__=='__main__':
    if len(sys.argv)!=3:raise SystemExit('usage: hydrate.py ABSOLUTE_ORIGINAL_TOKENIZER_PYTHON NEW_OUTPUT_FILE')
    content=hydrate(sys.argv[1])
    with Path(sys.argv[2]).open('xb') as stream:stream.write(content)
