// D1B two-sequence exactness tool. One process runs a schedule of events on a 2-seq context (kv_unified=false, like
// llama-server -np 2). mode=pair batches the active sequences' decode tokens in one llama_decode (seq0 first);
// mode=solo decodes each active sequence alone (one token per llama_decode). Same timeline in both modes, so per-seq
// logits/state must be bit-equal between modes if batch-2 decode is exact.
// usage: dx2seq <model> <text> <outdir> <mode pair|solo> <n_ctx> <events...>
//   P<k>:<a>:<b>   prefill seq k with text tokens [a,b) (chunks of 2048); seq k's stream continues at b
//   L<k>:<file>    restore seq k from a llama-server slot file (tokens inside); stream continues at stored token count offset given by S
//   S<k>:<a>       set seq k teacher-forcing stream start (text token index) after an L event
//   D<n>           n decode steps over the active seqs
//   C              fork seq0 into seq1 (seq_cp)
//   X<k>           cancel seq k (memory seq_rm, inactive)
// outputs: <outdir>/s<k>.logits (FP32 rows, appended per decode of seq k), <outdir>/s<k>.state (seq state at end), stdout log
#include "llama.h"
#include "ggml-backend.h"
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
int main(int argc, char ** argv) {
    if (argc < 7) { fprintf(stderr, "usage\n"); return 2; }
    const std::string out = argv[3], mode = argv[4]; const int NC = atoi(argv[5]);
    const bool pair = mode == "pair";
    ggml_backend_load_all(); llama_backend_init();
    llama_model_params mp = llama_model_default_params(); mp.n_gpu_layers = 999;
    llama_model * model = llama_model_load_from_file(argv[1], mp); if (!model) return 3;
    llama_context_params cp = llama_context_default_params();
    printf("defaults n_ctx=%u n_batch=%u n_ubatch=%u n_seq_max=%u kv_unified=%d\n", cp.n_ctx, cp.n_batch, cp.n_ubatch, cp.n_seq_max, (int) cp.kv_unified);
    cp.n_ctx = NC; cp.n_batch = 2048; cp.n_ubatch = 1024; cp.n_seq_max = 2; cp.kv_unified = false; cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    { auto ty = [](const char * n) { return !n ? GGML_TYPE_F16 : !strcmp(n, "q8_0") ? GGML_TYPE_Q8_0 : GGML_TYPE_F16; }; cp.type_k = ty(getenv("DX_KT")); cp.type_v = ty(getenv("DX_VT")); }   // Q8S
    llama_context * ctx = llama_init_from_model(model, cp); if (!ctx) return 4;
    llama_memory_t mem = llama_get_memory(ctx);
    const llama_vocab * vocab = llama_model_get_vocab(model); const int nv = llama_vocab_n_tokens(vocab);
    FILE * tf = fopen(argv[2], "rb"); if (!tf) return 11;
    std::string text(4000000, '\0'); text.resize(fread(&text[0], 1, text.size(), tf)); fclose(tf);
    std::vector<llama_token> tok(text.size() + 16);
    int n = llama_tokenize(vocab, text.data(), text.size(), tok.data(), tok.size(), true, false); if (n < 0) return 5;
    tok.resize(n);
    printf("n_vocab=%d text_tokens=%d n_ctx=%d mode=%s\n", nv, n, NC, mode.c_str());
    int pos[2] = {0, 0}, nxt[2] = {0, 0}; bool act[2] = {false, false}; long nd[2] = {0, 0};
    FILE * lf[2] = { fopen((out + "/s0.logits").c_str(), "wb"), fopen((out + "/s1.logits").c_str(), "wb") };
    llama_batch b = llama_batch_init(2048, 0, 1);
    auto add = [&](llama_token t, int p, int s, bool lg) { int i = b.n_tokens++; b.token[i] = t; b.pos[i] = p; b.n_seq_id[i] = 1; b.seq_id[i][0] = s; b.logits[i] = lg; };
    double ms_pair = 0; long n_pair = 0;
    for (int e = 6; e < argc; ++e) {
        const char * ev = argv[e]; const char c = ev[0];
        if (c == 'P') {
            int k, a, z; sscanf(ev + 1, "%d:%d:%d", &k, &a, &z);
            for (int i = a; i < z; i += 2048) {
                b.n_tokens = 0; int m = std::min(2048, z - i);
                for (int j = 0; j < m; ++j) add(tok[i + j], pos[k]++, k, j == m - 1);
                if (llama_decode(ctx, b) != 0) { fprintf(stderr, "prefill failed\n"); return 7; }
            }
            nxt[k] = z; act[k] = true; printf("EV %s pos=%d\n", ev, pos[k]);
        } else if (c == 'L') {
            int k; char f[512]; sscanf(ev + 1, "%d:%511s", &k, f);
            std::vector<llama_token> st(200000); size_t cnt = 0;
            size_t r = llama_state_seq_load_file(ctx, f, k, st.data(), st.size(), &cnt);
            if (r == 0) { fprintf(stderr, "load %s failed\n", f); return 12; }
            pos[k] = (int) cnt; act[k] = true; printf("EV L%d bytes=%zu tokens=%zu\n", k, r, cnt);
            // last stored token's logits are not restored; stream continues from S event
        } else if (c == 'S') {
            int k, a; sscanf(ev + 1, "%d:%d", &k, &a); nxt[k] = a; printf("EV %s\n", ev);
        } else if (c == 'C') {   // fork: seq0 -> seq1 (recurrent: both share the source cell until seq1's next step)
            llama_memory_seq_rm(mem, 1, -1, -1); llama_memory_seq_cp(mem, 0, 1, -1, -1); pos[1] = pos[0]; nxt[1] = nxt[0]; act[1] = true; printf("EV C\n");
        } else if (c == 'X') {
            int k = atoi(ev + 1); llama_memory_seq_rm(mem, k, -1, -1); act[k] = false; pos[k] = 0; printf("EV %s\n", ev);
        } else if (c == 'D') {
            const int steps = atoi(ev + 1);
            for (int s = 0; s < steps; ++s) {
                int ks[2], na = 0; for (int k = 0; k < 2; ++k) if (act[k]) ks[na++] = k;
                for (int g = 0; g < (pair ? 1 : na); ++g) {
                    b.n_tokens = 0;
                    const int lo = pair ? 0 : g, hi = pair ? na : g + 1;
                    for (int q = lo; q < hi; ++q) { const int k = ks[q]; add(tok[nxt[k]], pos[k], k, true); }
                    auto t0 = std::chrono::steady_clock::now();
                    if (llama_decode(ctx, b) != 0) { fprintf(stderr, "decode failed\n"); return 7; }
                    const float * l0 = llama_get_logits_ith(ctx, 0); (void) l0;   // syncs
                    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
                    if (hi - lo == 2) { ms_pair += ms; n_pair++; }
                    for (int q = lo; q < hi; ++q) {
                        const int k = ks[q]; const float * lg = llama_get_logits_ith(ctx, q - lo);
                        int am = 0; for (int v = 0; v < nv; ++v) if (lg[v] > lg[am]) am = v;
                        fwrite(lg, sizeof(float), nv, lf[k]);
                        printf("seq=%d step=%ld pos=%d tok=%d argmax=%d ms=%.2f\n", k, nd[k], pos[k], tok[nxt[k]], am, ms);
                        pos[k]++; nxt[k]++; nd[k]++;
                    }
                }
            }
        } else { fprintf(stderr, "bad event %s\n", ev); return 13; }
    }
    for (int k = 0; k < 2; ++k) {
        fclose(lf[k]);
        const size_t sz = llama_state_seq_get_size(ctx, k); std::vector<uint8_t> buf(sz);
        if (sz && llama_state_seq_get_data(ctx, buf.data(), sz, k) != sz) return 8;
        FILE * g = fopen((out + "/s" + std::to_string(k) + ".state").c_str(), "wb"); fwrite(buf.data(), 1, sz, g); fclose(g);
        printf("STATE seq=%d bytes=%zu decodes=%ld\n", k, sz, nd[k]);
    }
    printf("DONE pair_steps=%ld pair_ms_mean=%.3f\n", n_pair, n_pair ? ms_pair / n_pair : 0.0);
    llama_batch_free(b); llama_free(ctx); llama_model_free(model); llama_backend_free();
    return 0;
}
