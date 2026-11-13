// N4 dump tool: arm-A (stock) prefill with a ggml eval callback that dumps FA inputs/outputs.
// Derived from g1c_rows.cpp scheduling. usage:
//   n4_dump <model> <outdir> <n_ctx> <tokens.i32> <layers csv e.g. 3,19,39> <stop:Q csv e.g. 4096:1024,4160:64,32768:1024,33024:256,98048:1024,98304:256>
// Prefill is 0..stop_i in P-chunks; the chunk ending at each stop is <=1024 tokens (one ubatch, ubatch=1024),
// and only that ubatch is dumped (armed). Env must be the arm-A env (GEMM flag ABSENT). Same lib/flags as G1c arm A.
#include "llama.h"
#include "ggml-backend.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <set>
#include <algorithm>
#include <sys/stat.h>

struct dump_state { std::string dir; std::set<int> layers; bool armed = false; int stop = 0; bool mask_done = false; std::vector<uint8_t> buf; };

static void wr(const std::string & p, const void * d, size_t n) { FILE * f = fopen(p.c_str(), "wb"); if (!f) { perror(p.c_str()); exit(21); } fwrite(d, 1, n, f); fclose(f); }
// copy tensor (any strides, up to 4D) to a compact buffer in logical order, element size es
static void fetch_compact(const ggml_tensor * t, std::vector<uint8_t> & out) {
    const size_t es = ggml_type_size(t->type); const bool contig = ggml_is_contiguous(t);
    if (contig) { out.resize(ggml_nbytes(t)); ggml_backend_tensor_get(t, out.data(), 0, out.size()); return; }
    // strided views (KV cache): read row by row (dim0 contiguous)
    out.resize(es * ggml_nelements(t)); size_t o = 0;
    for (int64_t i3 = 0; i3 < t->ne[3]; ++i3) for (int64_t i2 = 0; i2 < t->ne[2]; ++i2) for (int64_t i1 = 0; i1 < t->ne[1]; ++i1) {
        ggml_backend_tensor_get(t, out.data() + o, i3 * t->nb[3] + i2 * t->nb[2] + i1 * t->nb[1], es * t->ne[0]); o += es * t->ne[0]; }
}
static bool cb(struct ggml_tensor * t, bool ask, void * ud) {
    dump_state & s = *(dump_state *) ud;
    // ask phase: true => scheduler computes up to this node and calls us again with ask=false; false => leave node batched (no sync).
    // observe phase: return value true => continue the graph.
    if (t->op != GGML_OP_FLASH_ATTN_EXT) return ask ? false : true;
    int il = -1; if (sscanf(t->name, "kqv_out-%d", &il) != 1) return ask ? false : true;
    if (!s.armed || !s.layers.count(il)) return ask ? false : true;
    if (ask) return true;                                             // ask: yes, observe this node after it computes (only these nodes sync)
    const ggml_tensor * q = t->src[0], * k = t->src[1], * v = t->src[2], * m = t->src[3];
    if (q->ne[1] < 64 || k->ne[1] < 4096 || k->ne[1] > 98304) { fprintf(stderr, "N4 skip il=%d q=%ld k=%ld (GEMM guard range)\n", il, (long) q->ne[1], (long) k->ne[1]); }
    std::string p = s.dir + "/stop" + std::to_string(s.stop) + "/"; mkdir(p.c_str(), 0755);
    // Q is a permuted view [256, nq, 16, 1]-ish of Qcur; the GEMM kernel sees F32 q with ne={256,nq,16}. Store as [nq,16,256] f32.
    std::vector<uint8_t> b; const int nq = (int) q->ne[1], nh = (int) q->ne[2], nk = (int) k->ne[1];
    { std::vector<float> o((size_t) nq * nh * 256); b.resize(256 * 4);
      for (int h = 0; h < nh; ++h) for (int i = 0; i < nq; ++i) { ggml_backend_tensor_get(q, b.data(), i * q->nb[1] + h * q->nb[2], 1024); memcpy(&o[((size_t) i * nh + h) * 256], b.data(), 1024); }
      wr(p + "L" + std::to_string(il) + "_Q.f32", o.data(), o.size() * 4); }
    { // K, V: [2][nk][256] f16
      for (int which = 0; which < 2; ++which) { const ggml_tensor * x = which ? v : k; std::vector<uint8_t> o((size_t) 2 * nk * 512), r(512);
        for (int g = 0; g < 2; ++g) for (int j = 0; j < nk; ++j) { ggml_backend_tensor_get(x, r.data(), j * x->nb[1] + g * x->nb[2], 512); memcpy(&o[((size_t) g * nk + j) * 512], r.data(), 512); }
        wr(p + "L" + std::to_string(il) + (which ? "_V.f16" : "_K.f16"), o.data(), o.size()); } }
    if (m && !s.mask_done) { fetch_compact(m, s.buf); wr(p + "mask.f16", s.buf.data(), s.buf.size()); s.mask_done = true;
      float sc, mb, cap; memcpy(&sc, (float *) t->op_params + 0, 4); memcpy(&mb, (float *) t->op_params + 1, 4); memcpy(&cap, (float *) t->op_params + 2, 4);
      FILE * f = fopen((p + "meta.json").c_str(), "w"); fprintf(f, "{\"stop\":%d,\"nq\":%d,\"nq_pad\":%d,\"nk\":%d,\"scale\":%.9g,\"max_bias\":%g,\"softcap\":%g,\"mask_type\":%d,\"q_nb\":[%zu,%zu,%zu],\"k_nb\":[%zu,%zu,%zu],\"mask_nb1\":%zu,\"n_kv_max_op4\":%d}\n",
        s.stop, nq, (int) m->ne[1], nk, sc, mb, cap, (int) m->type, q->nb[0], q->nb[1], q->nb[2], k->nb[0], k->nb[1], k->nb[2], m->nb[1], ((int32_t *) t->op_params)[4]); fclose(f); }
    fetch_compact(t, s.buf);                                         // stock output [256,16,nq] f32 == (nq,16,256) row-major
    wr(p + "L" + std::to_string(il) + "_Ostock_insitu.f32", s.buf.data(), s.buf.size());
    fprintf(stderr, "N4 dumped il=%d stop=%d nq=%d nk=%d\n", il, s.stop, nq, nk);
    return true;
}
int main(int argc, char ** argv) {
    if (argc < 7) { fprintf(stderr, "usage\n"); return 2; }
    dump_state st; st.dir = argv[2]; mkdir(argv[2], 0755);
    for (char * c = strtok(argv[5], ","); c; c = strtok(nullptr, ",")) st.layers.insert(atoi(c));
    std::vector<int> stops, qs; for (char * c = strtok(argv[6], ","); c; c = strtok(nullptr, ",")) { int a = 0, b = 1024; sscanf(c, "%d:%d", &a, &b); stops.push_back(a); qs.push_back(b); }
    ggml_backend_load_all(); llama_backend_init();
    llama_model_params mp = llama_model_default_params(); mp.n_gpu_layers = 999;
    llama_model * model = llama_model_load_from_file(argv[1], mp); if (!model) return 3;
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = atoi(argv[3]); cp.n_batch = 2048; cp.n_ubatch = 1024; cp.n_seq_max = 1; cp.kv_unified = false;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED; cp.type_k = GGML_TYPE_F16; cp.type_v = GGML_TYPE_F16;
    cp.cb_eval = cb; cp.cb_eval_user_data = &st;
    llama_context * ctx = llama_init_from_model(model, cp); if (!ctx) return 4;
    FILE * tf = fopen(argv[4], "rb"); if (!tf) return 11; fseek(tf, 0, SEEK_END); long sz = ftell(tf); fseek(tf, 0, SEEK_SET);
    std::vector<llama_token> tok(sz / 4); if (fread(tok.data(), 4, tok.size(), tf) != tok.size()) return 12; fclose(tf);
    llama_batch b = llama_batch_init(2048, 0, 1); int pos = 0;
    auto prefill = [&](int a, int z, bool arm, int stop) {
        for (int i = a; i < z; i += 2048) {                              // non-armed chunks use the normal 2048 batching (like G1c)
            const int m = std::min(2048, z - i); b.n_tokens = 0;
            for (int j = 0; j < m; ++j) { int x = b.n_tokens++; b.token[x] = tok[i + j]; b.pos[x] = pos++; b.n_seq_id[x] = 1; b.seq_id[x][0] = 0; b.logits[x] = false; }
            st.armed = arm; st.stop = stop; st.mask_done = false;
            if (llama_decode(ctx, b) != 0) { fprintf(stderr, "prefill failed\n"); exit(7); }
            st.armed = false; }
    };
    for (size_t i = 0; i < stops.size(); ++i) {                          // stops[i] = stop position, qs[i] = size of the final (armed) chunk, <=1024
        const int stop = stops[i], q = qs[i], a = stop - q;
        if (a < pos) { fprintf(stderr, "stop %d: start %d < pos %d\n", stop, a, pos); return 13; }
        if (a > pos) prefill(pos, a, false, stop);
        prefill(pos, stop, true, stop);
    }
    return 0;
}
