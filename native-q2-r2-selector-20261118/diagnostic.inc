    if constexpr (NQ == 2) {
        static const bool check_top16 = [] {
            const char * s = getenv("GGML_CUDA_OCCAMY_FA_Q2_R2_TOP16_CHECK");
            return s && strcmp(s, "1") == 0;
        }();
        if (use_top16 && check_top16 && nc == 1) {
            ggml_cuda_pool_alloc<ofa_sparse_plan> old_plans(ctx.pool(), nplans);
            ggml_cuda_pool_alloc<ofa_sparse_meta> old_metas(ctx.pool(), (size_t) nplans*n_blocks);
            ggml_cuda_pool_alloc<float> old_scores(ctx.pool(), (size_t) nplans*n_blocks);
            ggml_cuda_pool_alloc<float> old_output(ctx.pool(), ggml_nelements(dst));
            GGML_ASSERT(ggml_is_contiguous(dst));
            ofa_sparse_scan_probe_r2<NQ><<<dim3(n_blocks, nplans), 128, 0, stream>>>(p, old_metas.get(), old_scores.get(), n_blocks);
            ofa_sparse_select_r2<<<nplans, 128, 0, stream>>>(old_plans.get(), old_scores.get(), old_metas.get(), n_blocks);
            ofa_sparse_attention<NQ, NU><<<dim3(nsplit, nkvh, 1), NT, 0, stream>>>(p, old_plans.get());
            ofa_mrow_combine_kernel<<<dim3(nkvh*OFA_G, NQ, 1), OFA_D, 0, stream>>>(part_o.get(), part_ml.get(), (char *) old_output.get(), nsplit, nkvh, NQ, dst->nb[1], dst->nb[2], dst->nb[3]);
            CUDA_CHECK(cudaGetLastError());
            std::vector<ofa_sparse_plan> a(nplans), b(nplans);
            std::vector<float> sa((size_t) nplans*n_blocks), sb(sa.size()), va(ggml_nelements(dst)), vb(va.size());
            auto copy = [&](auto & output, const void * device) {
                CUDA_CHECK(cudaMemcpyAsync(output.data(), device, output.size()*sizeof(output[0]), cudaMemcpyDeviceToHost, stream));
            };
            copy(a, plans.get()); copy(b, old_plans.get());
            copy(sa, scores.get()); copy(sb, old_scores.get());
            copy(va, dst->data); copy(vb, old_output.get());
            CUDA_CHECK(cudaStreamSynchronize(stream));
            int plan_diff = 0, score_diff = 0, output_diff = 0, nonfinite = 0;
            for (int i = 0; i < nplans; ++i) {
                const size_t bytes = offsetof(ofa_sparse_plan, blocks) + b[i].n_blocks*sizeof(int);
                plan_diff += memcmp(&a[i], &b[i], bytes) != 0;
            }
            for (size_t i = 0; i < sa.size(); ++i) score_diff += memcmp(&sa[i], &sb[i], sizeof(float)) != 0;
            for (size_t i = 0; i < va.size(); ++i) {
                output_diff += memcmp(&va[i], &vb[i], sizeof(float)) != 0;
                nonfinite += !std::isfinite(va[i]) || !std::isfinite(vb[i]);
            }
            fprintf(stderr, "q2_r2_top16_check n_kv=%d plans=%d scores=%zu outputs=%zu plan_diff=%d score_diff=%d output_diff=%d nonfinite=%d\n",
                    n_kv, nplans, sa.size(), va.size(), plan_diff, score_diff, output_diff, nonfinite);
            GGML_ASSERT(plan_diff == 0 && score_diff == 0 && output_diff == 0 && nonfinite == 0);
        }
    }
