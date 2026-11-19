"""Finish the public packet only after four strictly replayed owner receipts exist."""
from hashlib import sha256
from pathlib import Path
import json
import os

from replay import HERE, load, need, summarize, verify_manifest,verify_sources


def main():
    verify_sources()
    raw=load('arms.json')
    summary=summarize(raw)
    need(summary==load('summary.json'),'run export.py on the four sealed receipts first')
    lines=['## Measured results','',
           '| Arm | Raw server tok/s | Delivered-count tok/s | Stream tok/s | Cold total s | Visible tokens | Choices |',
           '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for a,r in zip(raw['arms'],summary['arms'],strict=True):
        q=a['quality']
        lines.append(f"| {a['order']} · {a['mode']} | {r['server_decode_tps']:.5f} | {r['delivered_count_over_server_decode_s']:.5f} | {r['first_to_last_stream_id_tps']:.5f} | {a['request_wall_s']:.3f} | {a['visible_retokenized_n']} | {q['choice_correct']}/{q['choice_total']} |")
    lines.append('')
    for i,p in enumerate(summary['pairs'],1):
        gain=100*(p['native_over_serial']['server_decode_tps']-1)
        wall=p['native_minus_serial_cold_request_s']
        lines.append(f"Pair {i} measured {gain:+.2f}% native server decode throughput and {wall:+.3f} seconds of native whole-request time against serial.")
    lines+=['','| Arm | Accepted/offered drafts | Compact-only launches | Down4 launches | Word stop verified in IDs | Repeated / book-matched / total 24-word windows |',
            '| --- | ---: | ---: | ---: | --- | ---: |']
    for a in raw['arms']:
        q=a['quality']
        lines.append(f"| {a['order']} · {a['mode']} | {a['accepted']}/{a['offered']} | {a['compact']['compact_only']} | {a['down4']['down4']} | {str(a['word_verified_in_raw_ids']).lower()} | {q['repeated_word_windows_24']} / {q['book_copied_word_windows_24']} / {q['word_windows_24']} |")
    rd=HERE/'README.md'
    doc=rd.read_text()
    initial=doc.split('\n## Measured results\n')[0].split('\n## Reproduction\n')[0]
    reproduce=doc.split('\n## Reproduction\n',1)[1]
    rd.write_text(initial+'\n\n'+'\n'.join(lines)+'\n\n## Reproduction\n'+reproduce)
    allowed={'README.md','source-pins.json','profile.json','export.py','replay.py','seal.py','arms.json','summary.json','SHA256SUMS'}
    allowed.update(str(p.relative_to(HERE)) for p in (HERE/'source').iterdir())
    allowed.add('hydrate.py')
    files=sorted(p for p in HERE.rglob('*') if p.is_file() and p.name!='SHA256SUMS')
    names={str(p.relative_to(HERE)) for p in files}
    need(names|{'SHA256SUMS'}==allowed,'unexpected or missing packet files')
    temp=HERE/'SHA256SUMS.tmp'
    temp.write_text(''.join(f'{sha256(p.read_bytes()).hexdigest()}  {p.relative_to(HERE)}\n' for p in files))
    os.replace(temp,HERE/'SHA256SUMS')
    verify_manifest()
    print('Sealed complete public four-arm ordinary reading packet.')


if __name__=='__main__':main()
