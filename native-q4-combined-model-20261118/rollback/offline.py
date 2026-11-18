"""Verify the byte reader on the frozen N34 archive and physical K4 test API pins."""
from pathlib import Path
import subprocess
import analyze as a
n34=a.N34
raw=next((n34/'receipts').iterdir())
s=a.base.State(raw/'stock.kept.state',2004)
assert len(s.payload)==60 and sum(v.size for v in s.payload.values())==16465920
assert a.base.logits(raw/'stock.badpair.logits',2).shape == (2,248320)
assert a.digest(a.COMBINED)==a.COMBINED_SHA
symbols=subprocess.check_output(['nm','-D',a.HERE/'state'],text=True)
for name in ['ggml_backend_sched_graph_compute_async','_Z40ggml_cuda_op_gated_delta_net_fused_cache','_Z28ggml_cuda_op_gated_delta_net','_Z23ggml_cuda_occamy_gdn_t2']:
    assert any(' T '+name in line for line in symbols.splitlines())
src=(a.C/'night-cache-only/src/src/llama-memory-hybrid.cpp').read_text()
section=src.split('void llama_memory_hybrid::state_write(')[1].split('void llama_memory_hybrid::state_read(')[0]
assert section.index('mem_attn->state_write')<section.index('mem_recr->state_write') and 'LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY' in section
runtime_symbols=subprocess.check_output(['nm','-D',a.COMBINED],text=True)
assert ' T ggml_cuda_occamy_q4_cohort_launch_count' in runtime_symbols
envelope=(a.C/'night-cache-only/src/src/llama-context.cpp').read_text()
record=envelope.split('size_t llama_context::state_seq_get_data(')[1].split('size_t llama_context::state_seq_set_data(')[0]
assert record.index('io->write(&io_magic')<record.index('io->write(&seq_id')<record.index('state_seq_write_data')
rw=(a.C/'night-cache-only/src/src/llama-memory-recurrent.cpp').read_text()
assert 'rollback >= 1 && rollback <= (llama_pos) n_rs_seq' in rw
print('Offline target state reader, live logical row dimensions, full-hybrid serialization order, scheduler and three CUDA forwarding symbols verified.')
