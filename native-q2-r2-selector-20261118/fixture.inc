        if (nb == 2 && kv == 81920 && std::getenv("OCCAMY_Q2_R2_FIXED_INPUT")) {
            // Deterministic Q/K/V across processes. The same preexisting causal
            // mask initializer below keeps row 0 from seeing row 1's final key.
            uint32_t state = 0x2a5127c3u;
            const auto next = [&] {
                state ^= state << 13;
                state ^= state >> 17;
                state ^= state << 5;
                return state & 0xFFFFu;
            };
            const auto sample = [](uint32_t index) {
                return (static_cast<int32_t>(index) - 32768) * (1.0f / 32768.0f);
            };
            auto * q = ggml_get_tensor(ctx, "q");
            std::vector<float> fq(ggml_nbytes(q) / sizeof(float));
            for (auto & value : fq) value = sample(next());
            ggml_backend_tensor_set(q, fq.data(), 0, ggml_nbytes(q));

            std::vector<ggml_fp16_t> table(65536);
            for (uint32_t i = 0; i < table.size(); ++i) table[i] = ggml_fp32_to_fp16(sample(i));
            for (const char * name : {"k", "v"}) {
                auto * t = ggml_get_tensor(ctx, name);
                std::vector<ggml_fp16_t> data(ggml_nbytes(t) / sizeof(ggml_fp16_t));
                for (auto & value : data) value = table[next()];
                ggml_backend_tensor_set(t, data.data(), 0, ggml_nbytes(t));
            }
        } else {
            test_flash_attn_ext::initialize_tensors(ctx);
        }
