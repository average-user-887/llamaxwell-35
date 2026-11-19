// A warp owns four adjacent output rows. Each group of eight lanes
// consumes the original eight 64-input units for its row and expert.
__device__ __forceinline__ void ld_down4(const Layer & L, const int * sel, int r, int k, int lane, uint4 * cv) {
    const uint8_t * p = k == TOPK ? L.down_sh : L.down_exps + size_t(sel[k]) * H * ROW_D;
    ld72(p + size_t(r) * ROW_D, lane, cv);
}
