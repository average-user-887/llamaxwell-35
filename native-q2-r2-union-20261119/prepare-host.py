from pathlib import Path
import difflib
c = Path(__file__).resolve().parent
base = (c.parent / 'last-ms-r2-selector/test-backend-ops.cpp').read_text()
p = base.replace('int64_t nr3, int64_t tail)\n        : test_flash_attn_ext', 'int64_t nr3, int64_t tail, bool padded = true)\n        : test_flash_attn_ext',1)
anchor = 'GGML_TYPE_F16, GGML_TYPE_F16, {0, 2, 1, 3}),\n          tail(tail)'
assert p.count(anchor) == 1
p = p.replace(anchor, anchor.replace('3}),','3}, padded),'))
p = p.replace('if (nb == 2 && kv == 81920 && std::getenv(', 'if (nb == 2 && kv >= 80128 && kv <= 83200 && std::getenv(')
anchor='        test_cases.emplace_back(new test_flash_attn_ext_mrow_causal(81920, nb, 1, 260));'
assert p.count(anchor)==2
p = p.replace(anchor, anchor+'\n        if (nb==2) for (int w : {80128,81920,83200}) test_cases.emplace_back(new test_flash_attn_ext_mrow_causal(w,2,1,260,false));')
# Dense fixtures skip exactly the same PRNG draws consumed by the gaps in
# the historical padded fixture. Q, logical K/V and mask therefore match.
anchor='                for (auto & value : data) value = table[next()];'
assert p.count(anchor)==1
p=p.replace(anchor,'''                if (kv_view) {
                    for (auto & value : data) value = table[next()];
                } else {
                    GGML_ASSERT(t->nb[1]==1024 && t->nb[2]==512 && data.size()==size_t(kv)*512);
                    for (int64_t token=0;token<kv;++token) {
                        for (int i=0;i<512;++i) data[token*512+i]=table[next()];
                        if (token+1<kv) for (int i=0;i<512;++i) (void) next();
                    }
                }''')
(c/'test-backend-ops.cpp').write_text(p)
(c/'fixture.patch').write_text(''.join(difflib.unified_diff(base.splitlines(True),p.splitlines(True),'a/tests/test-backend-ops.cpp','b/tests/test-backend-ops.cpp')))
