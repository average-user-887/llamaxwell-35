            if (tile4) {
                // Standalone score/meta check on the same GPU Q/K; the measured inputs and plans remain untouched.
                const size_t mask_bytes=size_t(p.nbm1)+size_t(n_kv)*sizeof(uint16_t);
                ggml_cuda_pool_alloc<uint16_t> edge_masks(ctx.pool(),(mask_bytes+1)/2);
                std::vector<uint16_t> masks((mask_bytes+1)/2,0);
                for (int r=0;r<2;++r) {
                    uint16_t * mr=masks.data()+r*p.nbm1/2;
                    for (int i=0;i<n_kv;++i) mr[i]=i<n_kv-260+r ? 0 : 0xfc00;
                    mr[17]=0x7e00; mr[64]=0x7c00; mr[256+192]=0x7e00;
                    mr[n_kv-256-64]=0x7c00;
                }
                CUDA_CHECK(cudaMemcpyAsync(edge_masks.get(),masks.data(),mask_bytes,cudaMemcpyHostToDevice,stream));
                ofa_params edge=p; edge.mask=(const char *)edge_masks.get();
                ofa_sparse_scan_probe_r2<NQ><<<dim3(n_blocks,nplans),128,0,stream>>>(edge,old_metas.get(),old_scores.get(),n_blocks);
                ggml_cuda_pool_alloc<ofa_sparse_meta> edge_meta(ctx.pool(),sa.size());
                ggml_cuda_pool_alloc<float> edge_score(ctx.pool(),sa.size());
                ofa_sparse_probe_tile4<<<dim3((n_blocks+3)/4,nplans),128,0,stream>>>(edge,edge_meta.get(),edge_score.get(),n_blocks);
                CUDA_CHECK(cudaGetLastError());
                copy(actual_meta,edge_meta.get()); copy(reference_meta,old_metas.get());
                copy(sa,edge_score.get()); copy(sb,old_scores.get());
                CUDA_CHECK(cudaStreamSynchronize(stream));
                int mask_bad=0,score_bad=0;
                for (const auto & m : actual_meta) { mask_bad+=m.mask_invalid; score_bad+=m.score_invalid; }
                GGML_ASSERT(mask_bad>0 && score_bad>0 && memcmp(actual_meta.data(),reference_meta.data(),sa.size()*sizeof(ofa_sparse_meta))==0 &&
                            memcmp(sa.data(),sb.data(),sa.size()*sizeof(float))==0);
                fprintf(stderr,"q2_r2_probe4_edge entries=%zu mask_bad=%d score_bad=%d exact=1\n",sa.size(),mask_bad,score_bad);
            }
