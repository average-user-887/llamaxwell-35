#include "common.cuh"
#include "ggml-backend-impl.h"
#include "ggml-cuda.h"
#include "compact-readback.h"

#include <cstring>

namespace {
using namespace occamy_compact;
constexpr int vocab = vocab_size;
constexpr int threads = 256;
constexpr int tile = 8 * threads;
constexpr int tiles = (vocab + tile - 1) / tile;
struct Peak { float score; int id; unsigned flags; };
struct MaskIds { int ids[mask_size]; };

__device__ bool is_masked(const MaskIds & mask, int id) {
    if (id < mask.ids[0]) {
        return false;
    }
    for (int k = 0; k < mask_size; ++k) {
        if (mask.ids[k] == id) {
            return true;
        }
    }
    return false;
}

__device__ Peak merge(Peak a, Peak b) {
    const unsigned invalid = (a.flags | b.flags) & nonfinite;
    if (b.id >= 0 && (a.id < 0 || b.score > a.score)) {
        a = b;
    } else if (a.id >= 0 && b.id >= 0 && b.score == a.score) {
        a.flags |= tie;
    }
    a.flags |= invalid;
    return a;
}

__device__ Peak warp_peak(Peak local) {
    for (int step = 16; step > 0; step /= 2) {
        const Peak other{__shfl_down_sync(0xffffffff, local.score, step),
                         __shfl_down_sync(0xffffffff, local.id, step),
                         __shfl_down_sync(0xffffffff, local.flags, step)};
        if ((threadIdx.x & 31) + step < 32) {
            local = merge(local, other);
        }
    }
    return local;
}

__device__ Peak block_peak(Peak local) {
    __shared__ Peak maxima[threads / 32];
    const unsigned lane = threadIdx.x & 31;
    const unsigned warp = threadIdx.x / 32;
    local = warp_peak(local);
    if (lane == 0) {
        maxima[warp] = local;
    }
    __syncthreads();
    if (warp == 0) {
        return warp_peak(lane < blockDim.x / 32 ? maxima[lane] : Peak{0, -1, 0});
    }
    return local;
}

__global__ void scan_tiles(const float * input, const MaskIds masked, Peak * scratch) {
    const float * row = input + blockIdx.y * vocab;
    const int first = blockIdx.x * tile + threadIdx.x;
    Peak local{0, -1, 0};
    for (int i = first; i < min(vocab, int(blockIdx.x + 1) * tile); i += threads) {
        const float score = row[i];
        if (!isfinite(score)) {
            local.flags |= nonfinite;
        } else if (!is_masked(masked, i)) {
            local = merge(local, Peak{score, i, 0});
        }
    }
    const Peak best = block_peak(local);
    if (threadIdx.x == 0) {
        scratch[blockIdx.y * tiles + blockIdx.x] = best;
    }
}

__global__ void finish_rows(const Peak * scratch, Choice * out) {
    const unsigned i = threadIdx.x;
    const Peak local = i < tiles ? scratch[blockIdx.x * tiles + i] : Peak{0, -1, 0};
    const Peak best = block_peak(local);
    if (i == 0) {
        out[blockIdx.x] = {best.id, best.flags | (best.id < 0 ? empty : 0u)};
    }
}

} // namespace

extern "C" __attribute__((visibility("default")))
bool ggml_backend_cuda_occamy_compact_v1(ggml_backend_t backend, const ggml_tensor * logits,
        const occamy_compact::Mask * mask, void * pinned, occamy_compact::Result * result) {
    using namespace occamy_compact;
    if (!ggml_backend_is_cuda(backend) || logits->type != GGML_TYPE_F32 ||
        logits->ne[0] != vocab_size || logits->ne[1] != 2 || logits->ne[2] != 1 || logits->ne[3] != 1 ||
        !ggml_is_contiguous(logits)) {
        return false;
    }
    auto & ctx = *static_cast<ggml_backend_cuda_context *>(backend->context);
    const auto * buffer = logits->view_src ? logits->view_src->buffer : logits->buffer;
    if (ctx.curr_stream_no != 0 || !buffer || buffer->buft != ggml_backend_cuda_buffer_type(ctx.device) ||
        ggml_cuda_info().devices[ctx.device].cc != 520) {
        return false;
    }
    MaskIds ids{};
    std::memcpy(ids.ids, mask->data(), sizeof(ids.ids));
    ggml_cuda_set_device(ctx.device);
    auto stream = ctx.stream();
    ggml_cuda_pool_alloc<Peak> scratch(ctx.pool(), 2 * tiles);
    ggml_cuda_pool_alloc<Choice> output(ctx.pool(), 2);
    scan_tiles<<<dim3(tiles, 2), threads, 0, stream>>>(static_cast<const float *>(logits->data), ids, scratch.get());
    finish_rows<<<2, 128, 0, stream>>>(scratch.get(), output.get());
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpyAsync(pinned, output.get(), sizeof(Choice) * 2, cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    std::memcpy(result->rows.data(), pinned, sizeof(Choice) * 2);
    return true;
}
