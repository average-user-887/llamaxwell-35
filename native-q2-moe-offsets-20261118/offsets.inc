// Fixed contiguous Q4_0 weights. Absolute addresses retain the native pointer width.
constexpr uint32_t EXPERT_PITCH = FF * ROW_GU;
static_assert(NE == 256 && TOPK == 8 && H == 2048 && FF == 512 && FFS == 512);
static_assert(ROW_GU == 1152 && ROW_D == 288 && EXPERT_PITCH == H * ROW_D);
static_assert(uint64_t(NE) * EXPERT_PITCH == 150994944u && uint64_t(NE) * EXPERT_PITCH < UINT32_MAX);

// Fold the per-lane vector offset into the bounded row before adding the wide base.
__device__ __forceinline__ void rowB_offset(const Layer & L, const uint32_t * expert_offsets, unsigned t,
        unsigned lane, const uint8_t *& gate, const uint8_t *& up) {
    const bool shared = t < FFS;
    uint32_t offset = t * ROW_GU;
    if (!shared) {
        const unsigned task = t - FFS;
        offset = expert_offsets[task / FF] + (task % FF) * ROW_GU;
    }
    offset += lane * sizeof(uint4);
    gate = (shared ? L.gate_sh : L.gate_exps) + offset;
    up = (shared ? L.up_sh : L.up_exps) + offset;
}
__device__ __forceinline__ void load_gate_up_offset(const uint8_t * lane_ptr, int lane, uint4 * v) {
    const uint4 * p = reinterpret_cast<const uint4 *>(lane_ptr);
    v[0] = __ldg(p);
    v[1] = __ldg(p + 32);
    if (lane < 8) v[2] = __ldg(p + 64);
}
__device__ __forceinline__ void load_down_offset(const Layer & L, const uint32_t * expert_offsets,
        unsigned row, unsigned chunk, unsigned lane, uint4 * v) {
    if (chunk < 2) {
#pragma unroll
        for (unsigned i = 0; i < 3; ++i) {
            if (i < 2 || lane < 8) {
                const unsigned k = lane + 32 * i;
                const unsigned rank = 4 * chunk + k / 18;
                const uint32_t offset = expert_offsets[rank] + row * ROW_D + (k % 18) * 16u;
                v[i] = __ldg(reinterpret_cast<const uint4 *>(L.down_exps + offset));
            }
        }
    } else if (lane < 18) {
        const uint32_t offset = row * ROW_D + lane * 16u;
        v[0] = __ldg(reinterpret_cast<const uint4 *>(L.down_sh + offset));
    }
}
