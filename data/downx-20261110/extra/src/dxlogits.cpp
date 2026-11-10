// C5 harness: teacher-forced batch-1 decode of a fixed public-domain text, dumps full-vocab raw FP32 logits per step.
// usage: dxlogits <model.gguf> <out.bin> [n_tok=64]
#include "llama.h"
#include "ggml-backend.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
// Jane Austen, Pride and Prejudice (1813), ch. 1, public domain.
static const char * TEXT = "It is a truth universally acknowledged, that a single man in possession of a good fortune, must be in want of a wife. "
"However little known the feelings or views of such a man may be on his first entering a neighbourhood, this truth is so well fixed in the minds "
"of the surrounding families, that he is considered the rightful property of some one or other of their daughters.";
int main(int argc, char ** argv) {
    if (argc < 3) { fprintf(stderr, "usage\n"); return 2; }
    const int NT = argc > 3 ? atoi(argv[3]) : 64;
    ggml_backend_load_all();
    llama_backend_init();
    llama_model_params mp = llama_model_default_params(); mp.n_gpu_layers = 999;
    llama_model * model = llama_model_load_from_file(argv[1], mp); if (!model) return 3;
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 4096; cp.n_batch = 2048; cp.n_ubatch = 1024; cp.n_seq_max = 1; cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    llama_context * ctx = llama_init_from_model(model, cp); if (!ctx) return 4;
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int nv = llama_vocab_n_tokens(vocab);
    std::vector<llama_token> tok(512);
    int n = llama_tokenize(vocab, TEXT, strlen(TEXT), tok.data(), tok.size(), true, false);
    if (n < NT) { fprintf(stderr, "only %d tokens\n", n); return 5; }
    tok.resize(NT);
    printf("n_vocab=%d n_tok=%d tokens=", nv, NT); for (int i = 0; i < NT; ++i) printf("%d%c", tok[i], i + 1 < NT ? ',' : '\n');
    llama_memory_clear(llama_get_memory(ctx), true);
    FILE * f = fopen(argv[2], "wb"); if (!f) return 6;
    long nan = 0, inf = 0;
    for (int i = 0; i < NT; ++i) {
        llama_batch b = llama_batch_get_one(&tok[i], 1);
        if (llama_decode(ctx, b) != 0) { fprintf(stderr, "decode failed at %d\n", i); return 7; }
        const float * lg = llama_get_logits_ith(ctx, -1);
        int am = 0; for (int v = 0; v < nv; ++v) { if (std::isnan(lg[v])) nan++; else if (std::isinf(lg[v])) inf++; if (lg[v] > lg[am]) am = v; }
        fwrite(lg, sizeof(float), nv, f);
        printf("step=%d tok=%d argmax=%d\n", i, tok[i], am);
    }
    fclose(f);
    printf("DONE steps=%d values=%ld nan=%ld inf=%ld\n", NT, (long)NT * nv, nan, inf);
    llama_free(ctx); llama_model_free(model); llama_backend_free();
    return 0;
}
