SNAPSHOT_OPTION = "LLAMA_SPEC_RS_SAMPLER_SNAPSHOT_ELIDE"
SNAPSHOT = re.compile(
    r"^sampler_snapshot: elided=(\d+) elided_rejected=(\d+) kept=(\d+) restored=(\d+)$",
    re.M,
)


def snapshot_witness(stage, out, metrics):
    names = ("server-start.log", "server-pre-warm.log", "server.log")
    observed = []
    for name in names:
        log = (out / name).read_text()
        if name == "server-start.log" and stage == "native":
            need("the context supports bounded partial sequence removal" in log,
                 "native target rollback capability is not bounded partial")
        restores = re.findall(r"^mv1_host: rounds=\d+ restores=(\d+)\b", log, re.M)
        need(all(int(count) == 0 for count in restores), "legacy full-checkpoint restore occurred")
        if name == "server.log" and stage == "native":
            need(restores, "legacy restore counter missing")
        rows = [tuple(map(int, row)) for row in SNAPSHOT.findall(log)]
        need(len(rows) == log.count("sampler_snapshot:"), "malformed snapshot witness")
        observed.append(rows)
    startup, cold, final = observed
    need(not startup and len(cold) == 1 and len(final) == 2 and final[:1] == cold,
         "snapshot witness does not match the two completion boundaries")
    result = {}
    previous = (0, 0, 0, 0)
    for phase, current in zip(("cold", "warm"), final):
        delta = tuple(a - b for a, b in zip(current, previous))
        offered = metrics[phase]["draft_n"] or 0
        accepted = metrics[phase]["draft_n_accepted"] or 0
        expected = (offered, offered - accepted, 0, 0) if stage == "native" else (0, 0, 0, 0)
        need(delta == expected, f"{phase}: sampler snapshots, rejected drafts or restores differ")
        result[phase] = dict(zip(("elided", "elided_rejected", "kept", "restored"), delta))
        previous = current
    return result

