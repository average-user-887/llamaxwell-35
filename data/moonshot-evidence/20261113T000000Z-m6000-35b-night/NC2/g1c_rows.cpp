// G1c teacher-forced rows tool (single seq). One process = one doc = one arm.
// usage: g1c_rows <model> <outdir> <n_ctx> <fa 0|1> <ktype f16|f32> <n_ubatch> <events...>
//   T<file>   load int32 token file (one doc); K clears KV/recurrent memory, pos=0 (start of next doc); O<name> open <outdir>/<name>.f32 for rows
//   P<a>:<b>  prefill tokens [a,b) in chunks of 2048 (n_batch 2048)
//   R<n>      n teacher-forced single-token decode rows at the current position; full-vocab FP32 logits appended to <outdir>/rows.f32
// Prefill/row schedule is identical in every arm; only env, lib and flags differ.
#include "llama.h"
#include "ggml-backend.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <string>
#include <vector>
#include <algorithm>
int main(int argc, char ** argv) {
    if (argc < 8) { fprintf(stderr, "usage\n"); return 2; }
    const std::string out = argv[2]; const int NC = atoi(argv[3]); const int fa = atoi(argv[4]);
    const std::string kt = argv[5]; const int ub = atoi(argv[6]);
    ggml_backend_load_all(); llama_backend_init();
    llama_model_params mp = llama_model_default_params(); mp.n_gpu_layers = 999;
    llama_model * model = llama_model_load_from_file(argv[1], mp); if (!model) return 3;
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = NC; cp.n_batch = 2048; cp.n_ubatch = ub; cp.n_seq_max = 1; cp.kv_unified = false;
    cp.flash_attn_type = fa ? LLAMA_FLASH_ATTN_TYPE_ENABLED : LLAMA_FLASH_ATTN_TYPE_DISABLED;
    const ggml_type t = kt == "f32" ? GGML_TYPE_F32 : GGML_TYPE_F16; cp.type_k = t; cp.type_v = t;
    llama_context * ctx = llama_init_from_model(model, cp); if (!ctx) return 4;
    const int nv = llama_vocab_n_tokens(llama_model_get_vocab(model));
    llama_memory_t mem = llama_get_memory(ctx);
    std::vector<llama_token> tok; FILE * lf = nullptr;
    printf("n_vocab=%d n_ctx=%d fa=%d kv=%s ub=%d\n", nv, NC, fa, kt.c_str(), ub);
    llama_batch b = llama_batch_init(2048, 0, 1); int pos = 0; long nrow = 0;
    auto add = [&](llama_token x, int p, bool lg) { int i = b.n_tokens++; b.token[i] = x; b.pos[i] = p; b.n_seq_id[i] = 1; b.seq_id[i][0] = 0; b.logits[i] = lg; };
    for (int e = 7; e < argc; ++e) {
        const char * ev = argv[e];
        if (ev[0] == 'T') {
            FILE * tf = fopen(ev + 1, "rb"); if (!tf) return 11;
            fseek(tf, 0, SEEK_END); long sz = ftell(tf); fseek(tf, 0, SEEK_SET);
            tok.assign(sz / 4, 0); if (fread(tok.data(), 4, tok.size(), tf) != tok.size()) return 12; fclose(tf);
            printf("EV T tokens=%zu\n", tok.size());
        } else if (ev[0] == 'K') { llama_memory_clear(mem, true); pos = 0; printf("EV K\n");
        } else if (ev[0] == 'O') { if (lf) fclose(lf); lf = fopen((out + "/" + (ev + 1) + ".f32").c_str(), "wb"); if (!lf) return 15;
        } else if (ev[0] == 'P') {
            int a, z; sscanf(ev + 1, "%d:%d", &a, &z);
            if ((size_t) z > tok.size()) { fprintf(stderr, "P end %d beyond %zu tokens\n", z, tok.size()); return 16; }
            if (a != pos) { fprintf(stderr, "P start %d != pos %d\n", a, pos); return 13; }
            auto t0 = std::chrono::steady_clock::now();
            for (int i = a; i < z; i += 2048) {
                b.n_tokens = 0; const int m = std::min(2048, z - i);
                for (int j = 0; j < m; ++j) add(tok[i + j], pos++, false);
                if (llama_decode(ctx, b) != 0) { fprintf(stderr, "prefill failed\n"); return 7; }
            }
            printf("EV %s pos=%d s=%.1f\n", ev, pos, std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
        } else if (ev[0] == 'R') {
            const int n = atoi(ev + 1);
            for (int s = 0; s < n; ++s) {
                if ((size_t) pos >= tok.size()) { fprintf(stderr, "R pos %d beyond %zu tokens\n", pos, tok.size()); return 17; }
                b.n_tokens = 0; add(tok[pos], pos, true);
                if (llama_decode(ctx, b) != 0) { fprintf(stderr, "row failed\n"); return 7; }
                const float * lg = llama_get_logits_ith(ctx, 0);
                fwrite(lg, sizeof(float), nv, lf); pos++; nrow++;
            }
            fflush(lf); printf("EV %s rows_total=%ld pos=%d\n", ev, nrow, pos);
        } else { fprintf(stderr, "bad event %s\n", ev); return 14; }
    }
    if (lf) fclose(lf); printf("DONE rows=%ld nv=%d\n", nrow, nv);
    llama_batch_free(b); llama_free(ctx); llama_model_free(model); llama_backend_free(); return 0;
}
