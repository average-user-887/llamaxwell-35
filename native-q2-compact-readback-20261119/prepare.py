#!/usr/bin/env python3
"""Stage model-defined suppression state, without writing prompt or prediction data."""
from pathlib import Path
import re
import sys

card = Path(__file__).resolve().parent
campaign = card.parent
sys.path.insert(0, str(campaign / 'night-one-graph-draft/src/gguf-py'))
from gguf import GGUFReader, GGUFValueType

model_record = (campaign / 'model.sha256').read_text().strip().split(maxsplit=1)
assert model_record[0] == 'd5ff985eea0bcb3a7305a2033582bf63f756c9f4b3700d96b4939f81ee041eae'
model = Path(model_record[1])
assert model.stat().st_size == 20080296992 and int(model.stat().st_mtime) == 1791834438
logs = list((campaign / 'hundred-hybrid-draft/receipts').glob('native-*/server-start.log'))
assert len(logs) == 1
log = logs[0].read_text()
vocab_sizes = set(map(int, re.findall(r'\bn_vocab\s*= (\d+)', log)))
assert vocab_sizes == {248320}
eog = sorted(set(map(int, re.findall(r'\bEOG token\s*= (\d+)', log))))
assert len(eog) == 5 and all(0 <= x < 248320 for x in eog)
reader = GGUFReader(model)
field = reader.get_field('tokenizer.ggml.suppress_tokens')
suppress = []
if field is not None:
    assert field.types == [GGUFValueType.ARRAY, GGUFValueType.INT32]
    suppress = [x for x in field.contents() if 0 <= x < 248320]

def array(name, values):
    return f'inline constexpr std::array<llama_token, {len(values)}> {name} = {{' + ', '.join(map(str, values)) + '};\n'

(card / 'mask.inc').write_text(array('eog_ids', eog) + array('suppression_ids', suppress))
print(f'vocab=248320 model_suppress_count={len(suppress)} model_eog_count={len(eog)} merged_count={len(set(eog + suppress))}')
