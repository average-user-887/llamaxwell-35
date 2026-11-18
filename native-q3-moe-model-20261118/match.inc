struct ggml_cuda_occamy_moe_match {
    int n_nodes = 0;                 // window length (first = RMS_NORM, last = residual ADD)
    const ggml_tensor * h = nullptr; // residual stream input [2048]
    const ggml_tensor * norm_w = nullptr;
    float eps = 1e-6f;
    const ggml_tensor * gate_inp = nullptr, * gate_inp_sh = nullptr;
    const ggml_tensor * gate_exps = nullptr, * up_exps = nullptr, * down_exps = nullptr;
    const ggml_tensor * gate_sh = nullptr, * up_sh = nullptr, * down_sh = nullptr;
    ggml_tensor * out = nullptr;
    int nt = 1;                      // D1b: tokens in the window (2 only with GGML_CUDA_OCCAMY_N2SPLIT=1; launched once per row)
};
