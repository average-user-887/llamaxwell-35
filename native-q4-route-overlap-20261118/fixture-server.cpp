#include "llama-batch.h"
#include <cstdint>
#include <fcntl.h>
#include <unistd.h>

llama_batch_ext::llama_batch_ext(llama_context *):n_tokens_max(4),n_embd_inp(0),n_embd_inp_enc(0),n_embd_state(0),n_seq_max(1),mem(nullptr),n_vocab(24),n_pos_per_embd(1){}
extern "C" int fake_server() {
    llama_context_params params{};
    llama_context * fit = llama_init_from_model(reinterpret_cast<llama_model *>(0x111),params);
    llama_context * ctx = llama_init_from_model(nullptr,params);
    params.ctx_type = LLAMA_CONTEXT_TYPE_MTP;
    llama_context * draft = llama_init_from_model(reinterpret_cast<llama_model *>(0x222),params);
    if (fit == ctx || draft == ctx) return 1;
    llama_batch_ext batch(ctx);
    batch.tokens.resize(4);
    for (int i=0;i<4;++i) batch.tokens[i].pos[0]=79996+i;
    if (llama_process(ctx,LLAMA_PROCESS_TYPE_DECODE,&batch)!=0) return 2;
    const int fd=open("/native-route-counters",O_RDWR);
    uint64_t one=1;
    if (fd<0 || pwrite(fd,&one,sizeof(one),16)!=sizeof(one)) return 3;
    close(fd);
    for (int round=0;round<512;++round) {
        for(int i=0;i<4;++i) batch.tokens[i].pos[0]=80000+round*4+i;
        if (llama_process(ctx,LLAMA_PROCESS_TYPE_DECODE,&batch)!=0) return 4;
    }
    return 0;
}
