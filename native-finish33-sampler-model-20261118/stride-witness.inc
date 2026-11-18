PROBE_OPTION = "GGML_CUDA_OCCAMY_FA_Q2_R2_PROBE4"
STRIDE_OPTION = "GGML_CUDA_OCCAMY_FA_Q2_R2_STRIDE"
PROBE4 = re.compile(r"^q2_r2_probe4_dispatch width=(\d+) plans=4 blocks=(\d+) tiles=(\d+)$", re.M)
STRIDE = re.compile(
    r"^q2_r2_stride_dispatch width=(\d+) k_token_bytes=(\d+) v_token_bytes=(\d+) splits=12 nu=4 threads=512 grid=24$",
    re.M,
)


def stride_witness(stage, out, metrics, selector_witness):
    logs = [(out / name).read_text() for name in ("server-start.log", "server-pre-warm.log", "server.log")]
    markers = []
    for log in logs:
        probe = [tuple(map(int, row)) for row in PROBE4.findall(log)]
        stride = [tuple(map(int, row)) for row in STRIDE.findall(log)]
        need(len(probe) == log.count("q2_r2_probe4_dispatch"),
             "missing or invalid tiled probe dispatch geometry")
        need(len(stride) == log.count("q2_r2_stride_dispatch"),
             "missing or invalid fixed-stride dispatch geometry")
        markers.append((probe, stride))
    startup, cold, final = markers
    need(startup == ([], []) and cold == final, "stride markers outside first measured request")
    pitch = None
    if stage == "native":
        need(len(cold[0]) == len(cold[1]) == 1, "tiled probe or fixed-stride path did not dispatch")
        (probe_width, blocks, tiles), = cold[0]
        (stride_width, k_pitch, v_pitch), = cold[1]
        width = selector_witness["branch_first_dispatch_width"]
        need(probe_width == stride_width == width and blocks == width // 256 and tiles == (blocks + 3) // 4,
             "first selector, tiled probe and stride widths differ")
        need(k_pitch == v_pitch and k_pitch in (1024, 2048), "stride pitch outside physical coverage")
        pitch = k_pitch
    else:
        need(cold == ([], []), "serial executed a two-row tiled or stride specialization")

    snapshots = [read(out / "witness" / f"{phase}.json") for phase in ("start", "cold", "warm")]
    counts = [snapshot.get("q2_stride") for snapshot in snapshots]
    need(all(type(count) is int and count >= 0 for count in counts) and counts[0] == 0,
         "invalid initial or typed stride count")
    phases = {}
    for phase, start, end in zip(("cold", "warm"), counts, counts[1:]):
        expected = 10 * (metrics[phase]["draft_n"] or 0)
        need(end - start == expected, f"{phase}: fixed-stride engagement differs from ten per offered round")
        phases[phase] = {"actual_stride_dispatches": end - start, "expected_stride_dispatches": expected}
    return {"first_physical_kv_pitch_bytes": pitch, "phases": phases}

