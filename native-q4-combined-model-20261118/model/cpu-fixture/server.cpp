#include "llama-batch.h"
llama_batch_ext::llama_batch_ext(llama_context *):n_tokens_max(4),n_embd_inp(0),n_embd_inp_enc(0),n_embd_state(0),n_seq_max(1),mem(nullptr),n_vocab(24),n_pos_per_embd(1){}
extern "C" int fake_server() {
    llama_batch_ext batch(nullptr);
    batch.tokens.resize(4);
    int rc=0;
    for (int stage=0;stage<2;++stage) {
        for(int i=0;i<4;++i) batch.tokens[i].pos[0]=1996+i+4*stage;
        rc+=llama_process(nullptr,LLAMA_PROCESS_TYPE_DECODE,&batch);
    }
    return rc;
}
