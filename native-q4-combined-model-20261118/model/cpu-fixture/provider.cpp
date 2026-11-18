#include "llama-batch.h"
#include <dlfcn.h>
extern "C" int32_t llama_process(llama_context *,llama_process_type,llama_batch_ext *) {
    void * lib=dlopen("/candidate/libggml-cuda.so",RTLD_NOW|RTLD_LOCAL);
    if (!lib) return 2;
    auto fn=reinterpret_cast<void(*)()>(dlsym(lib,"run_fixture"));
    if (!fn) return 3;
    fn();
    return 0;
}
