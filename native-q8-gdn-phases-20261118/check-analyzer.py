import tempfile
from pathlib import Path
import numpy as np
from analyze import main, fractions, LAYERS, NAMES, SIZES, lines_dict

base=Path(__file__).resolve().parent
arch=base.parent/'night-gdn-t2-q8-fast/receipts/nmmoon-n82-2k-20261118T034505Z-2895756-1198/b1'
oldsha=lines_dict(arch/'outputs.sha256')
def must_fail(fn):
    try: fn()
    except ValueError: return
    raise RuntimeError('expected invalid input to fail')
with tempfile.TemporaryDirectory(dir=base) as tmp:
    root=Path(tmp)
    for mode in ('control','clock'):
        out=root/mode
        out.mkdir()
        (out/'t2.rows').write_bytes((arch/'t2.rows').read_bytes())
        log=(arch/'t2.log').read_text()+f'\nQ8CLK_BEGIN mode={mode} cc=52 grid=72 sm=24 standard_blocks=3 diagnostic_blocks=3\n'
        log+='Q2_COHORT residency device=0 sm=24 active=2 regs=63 static=96 dynamic=39424 local=0 ready=1\n'
        log+='Q8CLK_COHORT fallback0=0 fallback1=0 paired=40 skipped=40\n'
        log+='Q8CLK_DSO '+(base/'diagnostic.sha256').read_text().split()[0]+'  /candidate/libggml-cuda.so\n'
        log+=''.join(f'Q8CLK_EVENT mode={mode} layer_slot={i} tensor=blk.{lay}.attn_qkv.weight us=200.00\n' for i,lay in enumerate(LAYERS))
        log+=f'Q8CLK_END mode={mode} distinct_t2_layers=30\n'
        (out/'t2.log').write_text(log)
        (out/'outputs.bytes').write_text(''.join(f'{sz} {n}\n' for n,sz in zip(NAMES,SIZES)))
        (out/'outputs.sha256').write_text(''.join(f'{oldsha[n]}  {n}\n' for n in NAMES))
    # Synthetic warps and CTAs straddle a 32-bit counter rollover.
    shift=np.arange(72,dtype='int64')[None,:,None,None]*7+np.arange(8,dtype='int64')[None,None,:,None]
    raw=np.tile(np.array([0,99000,100100,153000,154100,200000],dtype='int64') + shift + 2**32-70,(30,1,1,1)).astype('<u4')
    raw.tofile(root/'q8.clock.u32')
    result=main(root)
    assert result['exact_paired_states_logits_rows'] and result['bias_within_predeclared_limits']
    assert result['exact_vs_archived_Q8_states_logits_rows'] and result['directional_phase_attribution_eligible']
    modified=raw.copy();modified[2,4,1,2]=modified[2,4,1,1]
    must_fail(lambda: fractions(modified))
    oldlog=(root/'clock/t2.log').read_text()
    (root/'clock/t2.log').write_text(oldlog.replace('tensor=blk.38.', 'tensor=blk.39.'))
    must_fail(lambda: main(root))
    (root/'clock/t2.log').write_text(oldlog.replace('us=200.00','us=234.00'))
    assert not main(root)['bias_within_predeclared_limits']
    (root/'clock/t2.log').write_text(oldlog)
    digest=(root/'clock/outputs.sha256').read_text(); (root/'clock/outputs.sha256').write_text(digest.replace(oldsha['t2.kept.state'],'0'*64))
    assert not main(root)['exact_paired_states_logits_rows']
    (root/'clock/outputs.sha256').write_text(digest)
    (root/'clock/t2.log').write_text(oldlog.replace('paired=40', 'paired=39'))
    must_fail(lambda: main(root))
print('Offline parser checked true Q8 hashes/rows, 30 named layers, clock rollover, perturbation, mistaken route, incomplete events and changed state rejection.')
