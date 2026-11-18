// A bounded recurrent checkpoint can rewind at most one row per rejected draft
// token. Replay and every other sequence-removal mode keep the original snapshot.
static constexpr bool may_elide_sampler_snapshot(bool enabled, bool replay, common_context_seq_rm_type mode,
                                                  size_t n_draft, uint32_t n_rs_seq) {
    return enabled && !replay && mode == COMMON_CONTEXT_SEQ_RM_TYPE_RS && n_draft > 0 && n_draft <= n_rs_seq;
}
