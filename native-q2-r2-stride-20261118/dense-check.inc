            // The standalone padded fixture and normal model use different token
            // pitches. Exercise the model pitch with the same logical vectors.
            if (stride && p.nbk1 == 2048 && p.nbk2 == 512 && p.nbv2 == 512) {
                constexpr unsigned dense_pitch = 1024;
                ggml_cuda_pool_alloc<char> dense_k(ctx.pool(), size_t(n_kv)*dense_pitch);
                ggml_cuda_pool_alloc<char> dense_v(ctx.pool(), size_t(n_kv)*dense_pitch);
                CUDA_CHECK(cudaMemcpy2DAsync(dense_k.get(), dense_pitch, p.K, p.nbk1, dense_pitch, n_kv, cudaMemcpyDeviceToDevice, stream));
                CUDA_CHECK(cudaMemcpy2DAsync(dense_v.get(), dense_pitch, p.V, p.nbv1, dense_pitch, n_kv, cudaMemcpyDeviceToDevice, stream));
                std::vector<float> old_partial(size_t(nsplit)*nkvh*NQ*OFA_G*OFA_D), dense_partial(old_partial.size());
                std::vector<float2> old_ml(size_t(nsplit)*nkvh*NQ*OFA_G), dense_ml(old_ml.size());
                copy(old_partial, part_o.get()); copy(old_ml, part_ml.get());
                ofa_params dense = p;
                dense.K = dense_k.get(); dense.V = dense_v.get();
                dense.nbk1 = dense.nbv1 = dense_pitch;
                ofa_sparse_attention_q2_stride<dense_pitch><<<dim3(nsplit,nkvh,1),NT,0,stream>>>(dense,old_plans.get());
                ofa_mrow_combine_kernel<<<dim3(nkvh*OFA_G,NQ,1),OFA_D,0,stream>>>(part_o.get(),part_ml.get(),(char *)old_output.get(),nsplit,nkvh,NQ,dst->nb[1],dst->nb[2],dst->nb[3]);
                CUDA_CHECK(cudaGetLastError());
                copy(vb,old_output.get());
                copy(dense_partial,part_o.get()); copy(dense_ml,part_ml.get());
                CUDA_CHECK(cudaStreamSynchronize(stream));
                GGML_ASSERT(memcmp(va.data(),vb.data(),va.size()*sizeof(float)) == 0);
                GGML_ASSERT(memcmp(old_partial.data(),dense_partial.data(),old_partial.size()*sizeof(float))==0 &&
                            memcmp(old_ml.data(),dense_ml.data(),old_ml.size()*sizeof(float2))==0);
                fprintf(stderr,"q2_r2_stride_partial_replay values=%zu max_sum_words=%zu exact=1\n",
                        old_partial.size(),old_ml.size()*2);
                fprintf(stderr,"q2_r2_stride_dense_replay key_bytes=%zu value_bytes=%zu src_pitch=2048 dst_pitch=1024 outputs=%zu exact=1\n",
                        size_t(n_kv)*dense_pitch,size_t(n_kv)*dense_pitch,va.size());
                // Exercise the full-key fallback on an explicitly invalid plan
                // and ordinary finite fixture input. The primary output and
                // measured plan buffer stay intact.
                auto bad = b;
                for (auto & plan : bad) {
                    plan.invalid=1; plan.ok=0; plan.fallback=0;
                    plan.sparse_warps=0; plan.full_warps=0;
                }
                std::vector<float> full_original(va.size()), full_stride(va.size());
                std::vector<ofa_sparse_plan> checked(nplans);
                for (int variant = 0; variant < 2; ++variant) {
                    CUDA_CHECK(cudaMemcpyAsync(old_plans.get(),bad.data(),bad.size()*sizeof(ofa_sparse_plan),cudaMemcpyHostToDevice,stream));
                    if (variant == 0) ofa_sparse_attention<NQ,NU><<<dim3(nsplit,nkvh,1),NT,0,stream>>>(p,old_plans.get());
                    else ofa_sparse_attention_q2_stride<2048><<<dim3(nsplit,nkvh,1),NT,0,stream>>>(p,old_plans.get());
                    ofa_mrow_combine_kernel<<<dim3(nkvh*OFA_G,NQ,1),OFA_D,0,stream>>>(part_o.get(),part_ml.get(),(char *)old_output.get(),nsplit,nkvh,NQ,dst->nb[1],dst->nb[2],dst->nb[3]);
                    CUDA_CHECK(cudaGetLastError());
                    copy(variant == 0 ? full_original : full_stride,old_output.get());
                    copy(checked,old_plans.get());
                    CUDA_CHECK(cudaStreamSynchronize(stream));
                    for (const auto & plan : checked)
                        GGML_ASSERT(plan.fallback == 1 && plan.full_warps == nsplit*2*NU && plan.sparse_warps == 0);
                }
                GGML_ASSERT(memcmp(full_original.data(),full_stride.data(),full_stride.size()*sizeof(float))==0);
                for (float value : full_stride) GGML_ASSERT(std::isfinite(value));
                fprintf(stderr,"q2_r2_stride_fallback plans=%d full_warps_each=%d outputs=%zu finite=1 exact=1\n",
                        nplans,nsplit*2*NU,full_stride.size());
            }
