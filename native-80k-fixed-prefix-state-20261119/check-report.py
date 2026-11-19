"""Exercise report qualification and negative controls without loading a model."""
from report import EXPECTED_GRAPHS, summarize

lines = [
    "PREFIX positions=80167 bytes=1708666664 offload=42/42 rs=1 outputs=2",
    "ACCEPTED row0_matches_archived_accepted_token=true",
    "N82_ADMISSION original_blocks=3 candidate_blocks=3 grid=72 admitted=1",
    "q2_r2_stride_dispatch width=80384 k_token_bytes=1024 v_token_bytes=1024 splits=12",
    "q2_r2_top16_dispatch n_q=2 n_kv=80384 nkvh=2 plans=4",
    "llama_model_loader: n_expert_used = 8",
    "COMPLETE width_flip_matches_archived=true genuine_two_rejections=true",
]
lines += [f"QUALITY80_MAPPED /candidate/lib{i}.so" for i in range(4)]
lines += [f"RESTORE {name} bitwise=1" for name in ("accepted", "serial", "wrong_a", "wrong_b")]
for name, (width, pos) in EXPECTED_GRAPHS.items():
    t2 = 30 if width == 2 else 0
    moe = "0,0,40,40" if width == 2 else "0,0,0,0"
    lines.append(f"GRAPH {name} input={width} output={width} calls=1 position={pos} sparse=10 "
                 f"invalid_sparse=0 padded_kv=80384 t2={t2} t2_attempts={t2} t2_fallback=0 moe={moe}")
for name in ("accepted_vs_serial", "wrong_untrimmed_a_vs_b", "wrong_retained_a_vs_b",
             "wrong_corrected_a_vs_b", "wrong_retained_a_vs_serial", "wrong_corrected_a_vs_serial"):
    differs = name == "wrong_untrimmed_a_vs_b"
    lines.append(f"STATE {name} kv={'false' if differs else 'true'} conv=true ssm=true a_prefix=true b_prefix=true")
for name in ("accepted_row1_vs_serial", "wrong_row0_a_vs_b", "wrong_row1_a_vs_b",
             "wrong_row0_a_vs_accepted_row0", "wrong_corrected_a_vs_b", "wrong_corrected_a_vs_serial"):
    lines.append(f"LOGITS {name} different_f32_words={int(name == 'wrong_row1_a_vs_b')} max_abs=0")
for name in ("accepted_row1", "serial_row1_conditioning", "correct_a", "correct_b"):
    gap = -0.53683 if name == "accepted_row1" else 0.66038
    lines.append(f"SCORE {name} raw=serial greedy=serial top2_gap=1 serial_logp=-1 native_logp=-2 serial_minus_native={gap}")
lines += [f"TRIM {name} removed=1 genuine=true retained_position=80167" for name in ("wrong_a", "wrong_b")]
good = "\n".join(lines)
assert summarize(good, 0, 0)["same_shape_suffix_isolation"] is True
bad = good.replace("GRAPH accepted input=2", "GRAPH accepted input=1")
assert summarize(bad, 0, 0)["width_only_choice_reproduction"] is None
bad = good.replace("LOGITS wrong_row0_a_vs_b different_f32_words=0", "LOGITS wrong_row0_a_vs_b different_f32_words=1")
assert summarize(bad, 0, 0)["same_shape_suffix_isolation"] is False
bad = good.replace("genuine_two_rejections=true", "genuine_two_rejections=false")
assert summarize(bad, 0, 0)["same_shape_suffix_isolation"] is None
assert summarize(good, 0, 1)["executed_required_paths"] is False
print("Report accepts complete synthetic coverage and rejects missing physical, rewind and cleanup evidence.")
