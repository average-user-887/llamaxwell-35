            if (union_mode) {
                double sq=0, norm=0, worst=0;
                int outside=0;
                for (size_t i=0;i<va.size();++i) {
                    const double delta=std::abs(double(va[i])-vb[i]);
                    outside+=delta>1e-5+1e-4*std::abs(double(vb[i]));
                    sq+=delta*delta; norm+=double(vb[i])*vb[i]; worst=std::max(worst,delta);
                }
                const double nrmse=norm>0 ? std::sqrt(sq/norm) : (sq==0 ? 0.0 : INFINITY);
                fprintf(stderr,"q2_union_check width=%d pitch=%zu plans=%d scores=%zu outputs=%zu nonfinite=%d outside=%d max_abs=%.9g nrmse=%.9g\n",
                    n_kv,p.nbk1,nplans,sa.size(),va.size(),nonfinite,outside,worst,nrmse);
                GGML_ASSERT(nonfinite==0 && outside==0 && nrmse<=1e-5);
                // Force exactly one invalid row/head in otherwise valid plans.
                // Its entire KV-head uses the original, preserving the other
                // row's sparse policy. The other KV-head still uses the union.
                auto forced=b;
                forced[0].invalid=1; forced[0].ok=0;
                std::vector<float> ref(va.size()), candidate(va.size());
                for (int arm=0;arm<2;++arm) {
                    for (auto & plan:forced) { plan.full_warps=plan.sparse_warps=plan.fallback=0; }
                    CUDA_CHECK(cudaMemcpyAsync(old_plans.get(),forced.data(),nplans*sizeof(ofa_sparse_plan),cudaMemcpyHostToDevice,stream));
                    if (arm==0) {
                        ofa_sparse_attention<NQ,NU><<<dim3(nsplit,nkvh),NT,0,stream>>>(p,old_plans.get());
                        ofa_mrow_combine_kernel<<<dim3(nkvh*OFA_G,NQ),OFA_D,0,stream>>>(part_o.get(),part_ml.get(),(char *)old_output.get(),nsplit,nkvh,NQ,dst->nb[1],dst->nb[2],dst->nb[3]);
                    } else {
                        ofa_union_launch(p,old_plans.get(),(char *)old_output.get(),dst,stream);
                    }
                    CUDA_CHECK(cudaGetLastError());
                    copy(arm==0 ? ref:candidate,old_output.get());
                    copy(a,old_plans.get());
                    CUDA_CHECK(cudaStreamSynchronize(stream));
                    GGML_ASSERT(a[0].fallback==1 && a[0].full_warps==96 && a[2].fallback==0 && a[2].sparse_warps==96);
                }
                size_t exact=0;
                double mixed_sq=0,mixed_norm=0;
                for (size_t i=0;i<ref.size();++i) {
                    GGML_ASSERT(std::isfinite(ref[i]) && std::isfinite(candidate[i]));
                    const int head=int((i*sizeof(float))%dst->nb[2]/dst->nb[1]);
                    if (head<8) { GGML_ASSERT(memcmp(&ref[i],&candidate[i],sizeof(float))==0); ++exact; }
                    const double e=double(candidate[i])-ref[i];
                    GGML_ASSERT(std::abs(e)<=1e-5+1e-4*std::abs(double(ref[i])));
                    mixed_sq+=e*e; mixed_norm+=double(ref[i])*ref[i];
                }
                GGML_ASSERT(exact==4096 && mixed_sq<=1e-10*mixed_norm);
                fprintf(stderr,"q2_union_fallback width=%d pitch=%zu invalid_plans=1 old_head_words=%zu exact=1 all_finite=1\n",n_kv,p.nbk1,exact);
            }
