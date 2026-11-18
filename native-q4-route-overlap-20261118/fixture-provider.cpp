#include "llama-batch.h"
#include "ggml-backend.h"
#include <cstdio>
#include <cstring>

static ggml_backend_sched_eval_callback callback;
static void * callback_data;
extern "C" llama_context * llama_init_from_model(llama_model * model, llama_context_params) {
    return reinterpret_cast<llama_context *>(0x1234 + reinterpret_cast<uintptr_t>(model));
}
extern "C" void ggml_backend_sched_set_eval_callback(ggml_backend_sched_t, ggml_backend_sched_eval_callback cb, void * data) {
    callback = cb;
    callback_data = data;
}
extern "C" const char * ggml_backend_buffer_name(ggml_backend_buffer_t) {
    return "CUDA0";
}
extern "C" void ggml_backend_tensor_get(const ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    memcpy(data, static_cast<const char *>(tensor->data) + offset, size);
}
extern "C" ggml_status ggml_backend_sched_graph_compute_async(ggml_backend_sched_t, ggml_cgraph *) {
    if (callback) {
        int32_t fixture[4][64] = {};
        for (int row = 0; row < 4; ++row) for (int rank = 0; rank < 8; ++rank) fixture[row][rank] = 4*row + rank;
        ggml_tensor tensor{};
        tensor.type = GGML_TYPE_I32;
        tensor.ne[0]=8; tensor.ne[1]=4; tensor.ne[2]=tensor.ne[3]=1;
        tensor.nb[0]=4; tensor.nb[1]=sizeof(fixture[0]);
        tensor.data = fixture;
        tensor.buffer = reinterpret_cast<ggml_backend_buffer_t>(0x4321);
        for (int layer=0; layer<40; ++layer) {
            std::snprintf(tensor.name,sizeof(tensor.name),"ffn_moe_topk-%d",layer);
            if (callback(&tensor,true,callback_data)) callback(&tensor,false,callback_data);
        }
    }
    return GGML_STATUS_SUCCESS;
}
extern "C" int32_t llama_process(llama_context *,llama_process_type,llama_batch_ext *) {
    return ggml_backend_sched_graph_compute_async(nullptr,nullptr);
}
