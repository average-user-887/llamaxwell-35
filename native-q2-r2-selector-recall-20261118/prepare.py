"""Use the accepted selector model runner for one typed midpoint recall request."""
from pathlib import Path

here = Path(__file__).resolve().parent
prior = here.parent / "last-ms-r2-selector-model"
text = (prior / "release.sh").read_text()

def replace(a, b):
    global text
    assert text.count(a) == 1, a
    text = text.replace(a, b)

replace('stream_probe=$campaign/night-n18-warm3072/n18_probe.py\n', '')
replace('  "$campaign/resume-q2-moe-native-r1/release.sh" --check\n  "$overlay/release.sh" --check',
        '  "$campaign/last-ms-r2-selector-model/release.sh" --check')
replace('''  python3 -B "$stream_probe" cold "$port" "$out"
  sample "$name" "$out" cold
  capture_logs "$name" "$out/server-pre-warm.log"
  python3 -B "$stream_probe" warm "$port" "$out"
  sample "$name" "$out" warm''', '''  python3 -B "$probe" request "$port" "$out"
  sample "$name" "$out" cold''')
text = text.replace('nmmoon-r2-top16-model-', 'nmmoon-r2-top16-recall-')
text = text.replace('r2.top16.model.owner', 'r2.top16.recall.owner')
text = text.replace('Q2_R2_MODEL_RECEIPT', 'Q2_R2_RECALL_RECEIPT')
text = text.replace('model arm', 'recall arm')
replace("usage() { echo 'usage: release.sh --check | --run native|serial | --compare SERIAL_RECEIPT NATIVE_RECEIPT' >&2; exit 2; }",
        "usage() { echo 'usage: release.sh --check | --run recall' >&2; exit 2; }")
replace('    [[ $2 == serial || $2 == native ]] || usage', '    [[ $2 == recall ]] || usage')
replace('  --compare) [[ $# == 3 ]] || usage; check >&2; exec python3 -B "$probe" compare "$2" "$3" ;;\n', '')
replace('  --run) [[ $# == 2 && ( $2 == serial || $2 == native ) ]] || usage; check ;;',
        '  --run) [[ $# == 2 && $2 == recall ]] || usage; check ;;')
(here / "release.sh").write_text(text)
