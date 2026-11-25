#!/usr/bin/env python3
"""C3 v1: runtime-selectable tile_queries (TQ) / split_keys (SK) for the non-V5 Maxwell GEMM FA path.
Env GGML_CUDA_FA_MAXWELL_GEMM_TQ in {64,128}, GGML_CUDA_FA_MAXWELL_GEMM_SPLIT in {2048,4096,8192}.
Defaults 64/4096 = card1 behaviour. V5 path pinned to 64/4096."""
import re, sys
p = sys.argv[1]
s = open(p).read()
lines = s.split('\n')
def find(pat, start=0):
    for i in range(start, len(lines)):
        if pat in lines[i]: return i
    raise SystemExit('missing: ' + pat)
k0 = find('static __global__ void maxwell_gemm_pack_queries(')
k1 = find('static __global__ void maxwell_gemm_convert_kv_v5(')
reg = '\n'.join(lines[k0:k1])
reg = reg.replace('static constexpr int maxwell_gemm_fused_per_thread = maxwell_gemm_split_keys / maxwell_gemm_fused_threads;\n', '')
reg = reg.replace('maxwell_gemm_fused_per_thread', '(SK / maxwell_gemm_fused_threads)')
reg = reg.replace('maxwell_gemm_tile_queries', 'TQ').replace('maxwell_gemm_split_keys', 'SK')
for name, tp in (('pack_queries', 'template<int TQ>'), ('masked_softmax', 'template<int TQ>'), ('pv_pointers', 'template<int TQ, int SK>'),
                 ('reduce_products', 'template<int TQ>'), ('reduce_products_fused', 'template<int TQ>')):
    old = 'static __global__ void maxwell_gemm_%s(' % name
    assert reg.count(old) == 1, name
    reg = reg.replace(old, tp + '\n' + old)
old = 'static __global__ void __launch_bounds__(maxwell_gemm_fused_threads) maxwell_gemm_fused_split_softmax('
assert reg.count(old) == 1
reg = reg.replace(old, 'template<int TQ, int SK>\n' + old)
lines[k0:k1] = reg.split('\n')
s = '\n'.join(lines)

# env selection helpers, placed before the entry function
helper = r'''
// C3: runtime tile/split selection for the non-V5 path (read once). Defaults = card1 (64 / 4096).
static int occamy_fa_gemm_env_int(const char * name, int def, std::initializer_list<int> allowed) {
    const char * v = std::getenv(name);
    if (!v || !*v) return def;
    const int x = std::atoi(v);
    for (int a : allowed) if (a == x) return x;
    return def;
}
static int occamy_fa_gemm_tq() {
    static const int v = occamy_fa_gemm_env_int("GGML_CUDA_FA_MAXWELL_GEMM_TQ", 64, {64, 128});
    return v;
}
static int occamy_fa_gemm_split() {
    static const int v = occamy_fa_gemm_env_int("GGML_CUDA_FA_MAXWELL_GEMM_SPLIT", 4096, {2048, 4096, 8192});
    return v;
}

static bool ggml_cuda_flash_attn_ext_maxwell_gemm(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {'''
old = '\nstatic bool ggml_cuda_flash_attn_ext_maxwell_gemm(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {'
assert s.count(old) == 1
s = s.replace(old, helper, 1)

# sizing with tq/sk
rep = [
 ('    const int split_count = (key_count + maxwell_gemm_split_keys - 1) / maxwell_gemm_split_keys;\n'
  '    const int full_splits = key_count / maxwell_gemm_split_keys;\n'
  '    const int tail_keys = key_count % maxwell_gemm_split_keys;\n'
  '    const int split_capacity = key_capacity / maxwell_gemm_split_keys;\n'
  '    const int packed_columns = maxwell_gemm_grouped_heads * maxwell_gemm_tile_queries;\n',
  '    const int tq = v5 ? maxwell_gemm_tile_queries : occamy_fa_gemm_tq();\n'
  '    const int sk = v5 ? maxwell_gemm_split_keys : occamy_fa_gemm_split();\n'
  '    const int split_count = (key_count + sk - 1) / sk;\n'
  '    const int full_splits = key_count / sk;\n'
  '    const int tail_keys = key_count % sk;\n'
  '    const int split_capacity = key_capacity / sk;\n'
  '    const int packed_columns = maxwell_gemm_grouped_heads * tq;\n'),
 ('kv_staging_heads=1 call=%llu', 'kv_staging_heads=1 tq=%d split=%d call=%llu'),
 ('v5 ? 5 : 4, fused ? "fused_split_lse" : "online", (unsigned long long) call,',
  'v5 ? 5 : 4, fused ? "fused_split_lse" : "online", tq, sk, (unsigned long long) call,'),
 ('        maxwell_gemm_pv_pointers<<<1, 256, 0, stream>>>(\n            values_float, scores, partials, pv_left, pv_right, pv_output, split_count);\n        CUDA_CHECK(cudaGetLastError());\n\n        if (use_mask_bounds)',
  '        maxwell_gemm_pv_pointers<maxwell_gemm_tile_queries, maxwell_gemm_split_keys><<<1, 256, 0, stream>>>(\n            values_float, scores, partials, pv_left, pv_right, pv_output, split_count);\n        CUDA_CHECK(cudaGetLastError());\n\n        if (use_mask_bounds)'),
 ('                        maxwell_gemm_fused_split_softmax<<<dim3(maxwell_gemm_grouped_heads, t_splits,',
  '                        maxwell_gemm_fused_split_softmax<maxwell_gemm_tile_queries, maxwell_gemm_split_keys><<<dim3(maxwell_gemm_grouped_heads, t_splits,'),
 ('                        maxwell_gemm_masked_softmax<<<dim3(maxwell_gemm_tile_queries, maxwell_gemm_grouped_heads), 256, 0, stream>>>(\n                            scores, mask_chunk, mask ? mask->nb[1] : 0,\n                            eff_keys,',
  '                        maxwell_gemm_masked_softmax<maxwell_gemm_tile_queries><<<dim3(maxwell_gemm_tile_queries, maxwell_gemm_grouped_heads), 256, 0, stream>>>(\n                            scores, mask_chunk, mask ? mask->nb[1] : 0,\n                            eff_keys,'),
 ('                        maxwell_gemm_reduce_products_fused<<<(product_elements + 255) / 256, 256, 0, stream>>>(\n                            partials, softmax_stats, dst_chunk, query_count, query_start,\n                            group * maxwell_gemm_grouped_heads, t_splits,',
  '                        maxwell_gemm_reduce_products_fused<maxwell_gemm_tile_queries><<<(product_elements + 255) / 256, 256, 0, stream>>>(\n                            partials, softmax_stats, dst_chunk, query_count, query_start,\n                            group * maxwell_gemm_grouped_heads, t_splits,'),
 ('                        maxwell_gemm_reduce_products<<<(product_elements + 255) / 256, 256, 0, stream>>>(\n                            partials, dst_chunk, query_count, query_start, group * maxwell_gemm_grouped_heads, t_splits);',
  '                        maxwell_gemm_reduce_products<maxwell_gemm_tile_queries><<<(product_elements + 255) / 256, 256, 0, stream>>>(\n                            partials, dst_chunk, query_count, query_start, group * maxwell_gemm_grouped_heads, t_splits);'),
]
for a, b in rep:
    assert s.count(a) == 1, a[:80]
    s = s.replace(a, b)

# v4 path -> generic lambda over (TQ, SK)
a0 = s.index('    for (int q_off = 0; q_off < total_queries; q_off += maxwell_gemm_max_chunk_queries) {')
a1 = s.index('    return true;\n}\n\n#else')
body = s[a0:a1]
body = body.replace('maxwell_gemm_tile_queries', 'TQ').replace('maxwell_gemm_split_keys', 'SK')
for k in ('pack_queries', 'masked_softmax', 'reduce_products_fused', 'reduce_products'):
    body = re.sub(r'maxwell_gemm_%s<<<' % k, 'maxwell_gemm_%s<TQ><<<' % k, body)
body = body.replace('maxwell_gemm_pv_pointers<<<', 'maxwell_gemm_pv_pointers<TQ, SK><<<')
body = body.replace('maxwell_gemm_fused_split_softmax<<<', 'maxwell_gemm_fused_split_softmax<TQ, SK><<<')
body = '\n'.join(('    ' + l) if l else l for l in body.split('\n'))
wrapped = ('    const auto run_v4 = [&](auto tq_c, auto sk_c) {\n'
           '        constexpr int TQ = decltype(tq_c)::value;\n'
           '        constexpr int SK = decltype(sk_c)::value;\n'
           + body.rstrip() + '\n    };\n'
           '    using I64 = std::integral_constant<int, 64>;\n'
           '    using I128 = std::integral_constant<int, 128>;\n'
           '    using S2 = std::integral_constant<int, 2048>;\n'
           '    using S4 = std::integral_constant<int, 4096>;\n'
           '    using S8 = std::integral_constant<int, 8192>;\n'
           '    if (tq == 128) {\n'
           '        if (sk == 2048) run_v4(I128{}, S2{}); else if (sk == 8192) run_v4(I128{}, S8{}); else run_v4(I128{}, S4{});\n'
           '    } else {\n'
           '        if (sk == 2048) run_v4(I64{}, S2{}); else if (sk == 8192) run_v4(I64{}, S8{}); else run_v4(I64{}, S4{});\n'
           '    }\n')
s = s[:a0] + wrapped + s[a1:]
if '#include <type_traits>' not in s:
    s = s.replace('#include <cstring>\n', '#include <cstring>\n#include <initializer_list>\n#include <type_traits>\n', 1)
open(p, 'w').write(s)
print('ok')
