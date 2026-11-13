#!/usr/bin/env python3
"""NC1 workload census from llama-server docker logs (read-only). Emits counts/timings only, no prompt text.
usage: census.py OUTDIR   (writes census.json)"""
import re, sys, json, subprocess, datetime as dt, collections, bisect

CONTAINERS = ["llamacpp-m6000", "llamacpp-m6000-gdn4-prev", "llamacpp-m6000-night-prev",
              "llamacpp-m6000-call-prev", "llamacpp-m6000-downx-prev", "llamacpp-m6000-ub1024-prev"]
SWAP_UNIT = "m6000-llama-swap"
TS = r'^(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d\.\d+)Z '
def ts(s):
    return dt.datetime.fromisoformat(s[:26]).replace(tzinfo=dt.timezone.utc).timestamp()

R_NEW   = re.compile(TS + r'.*slot\s+operator\(\): id\s+(\d+) \| task (\d+) \| new prompt,.*task\.n_tokens = (\d+)')
R_CACHED= re.compile(TS + r'.*slot\s+operator\(\): id\s+(\d+) \| task (\d+) \| cached n_tokens = (\d+)')
R_RCP   = re.compile(TS + r'.*slot\s+operator\(\): id\s+(\d+) \| task (\d+) \| restored context checkpoint .*n_past = (\d+)')
R_ERASEI= re.compile(TS + r'.*slot\s+operator\(\): id\s+(\d+) \| task (\d+) \| erased invalidated context checkpoint')
R_PEVAL = re.compile(TS + r'.*print_timing: id\s+(\d+) \| task (\d+) \| prompt eval time =\s*([\d.]+) ms /\s*(\d+) tokens')
R_EVAL  = re.compile(TS + r'.*print_timing: id\s+(\d+) \| task (\d+) \|\s+eval time =\s*([\d.]+) ms /\s*(\d+) tokens')
R_TOTAL = re.compile(TS + r'.*print_timing: id\s+(\d+) \| task (\d+) \|\s+total time =\s*([\d.]+) ms /\s*(\d+) tokens')
R_REL   = re.compile(TS + r'.*slot\s+release: id\s+(\d+) \| task (\d+) \| stop processing: n_tokens = (\d+), truncated = (\d+)')
R_CKPT  = re.compile(TS + r'.*create_check: id\s+(\d+) \| task (\d+) \| created context checkpoint')
R_ERASEC= re.compile(TS + r'.*create_check: id\s+(\d+) \| task (\d+) \| erasing context checkpoint')
R_SLOTRESTORE = re.compile(TS + r'.*load_slot_ch: id\s+(\d+) \| task -?\d+ \| restored (\d+) context checkpoint')
R_SLOTSAVE    = re.compile(TS + r'.*save_slot_ch: id\s+(\d+) \|')
R_PCLOAD = re.compile(TS + r'.*srv\s+load:\s+- prompt with length\s+(\d+), lcp =\s+(\d+)')
R_SHIFT = re.compile(r'context shift|context full|exceeds the available context|truncat(ing|ed) the prompt|n_discard', re.I)

def get_logs(c):
    p = subprocess.run(["docker", "logs", "-t", c], capture_output=True, text=True, errors="replace")
    return (p.stdout + p.stderr).splitlines()

def get_swap():
    """llama-swap POST /slots/N entries: (epoch_end, slot, seconds)."""
    p = subprocess.run(["journalctl", "--user", "-u", SWAP_UNIT, "--no-pager", "-o", "short-unix"],
                       capture_output=True, text=True, errors="replace")
    out = []
    dur = re.compile(r'^(\d+)\.\d+ .*POST /upstream/[^/ ]+/slots/(\d+) HTTP/1.1" (\d+) \d+ ".*?" ([\d.]+)(µs|ms|s|m[\d.]+s)')
    for l in p.stdout.splitlines():
        m = re.match(r'^(\d+)\.\d+ .*POST /upstream/[^/ ]+/slots/(\d+) HTTP/1\.1" \d+ \d+ ".*?" (\S+)$', l)
        if not m: continue
        d = m.group(3); sec = None
        mm = re.fullmatch(r'(?:(\d+)m)?([\d.]+)(µs|ms|s)', d)
        if mm:
            sec = float(mm.group(2)) * {"µs": 1e-6, "ms": 1e-3, "s": 1}[mm.group(3)] + 60 * int(mm.group(1) or 0)
        if sec is not None: out.append((int(m.group(1)), int(m.group(2)), sec))
    return out

def parse(c):
    tasks = {}; ev = collections.Counter(); restores = []; saves = 0; pcloads = []; shifts = 0
    first = last = None
    for l in get_logs(c):
        m = R_NEW.match(l)
        if m:
            t, s, k, n = ts(m[1]), int(m[2]), int(m[3]), int(m[4])
            tasks[(s, k)] = dict(container=c, slot=s, task=k, t0=t, prompt_n=n, reuse=None, rcp=None, ckpt=0,
                                 erased_inval=0, erased_close=0, truncated=None)
            first = t if first is None else first; continue
        for rx, key in ((R_CACHED, "cached"), (R_RCP, "rcp"), (R_ERASEI, "ei"), (R_CKPT, "ck"), (R_ERASEC, "ec")):
            m = rx.match(l)
            if m:
                d = tasks.get((int(m[2]), int(m[3])))
                if d is not None:
                    if key == "cached" and d["reuse"] is None: d["reuse"] = int(m[4])
                    elif key == "rcp": d["rcp"] = int(m[4])
                    elif key == "ei": d["erased_inval"] += 1
                    elif key == "ck": d["ckpt"] += 1
                    elif key == "ec": d["erased_close"] += 1
                break
        else:
            m = R_PEVAL.match(l)
            if m:
                d = tasks.get((int(m[2]), int(m[3])))
                if d is not None: d["pf_ms"] = float(m[4]); d["pf_tok"] = int(m[5])
                continue
            m = R_EVAL.match(l)
            if m:
                d = tasks.get((int(m[2]), int(m[3])))
                if d is not None: d["dec_ms"] = float(m[4]); d["dec_tok"] = int(m[5])
                continue
            m = R_TOTAL.match(l)
            if m:
                d = tasks.get((int(m[2]), int(m[3])))
                if d is not None: d["total_ms"] = float(m[4])
                continue
            m = R_REL.match(l)
            if m:
                d = tasks.get((int(m[2]), int(m[3])))
                if d is not None: d["truncated"] = int(m[5]); d["t1"] = ts(m[1]); d["end_n"] = int(m[4])
                last = ts(m[1]); continue
            m = R_SLOTRESTORE.match(l)
            if m: restores.append(dict(container=c, t=ts(m[1]), slot=int(m[2]), ckpts=int(m[3]))); continue
            if R_SLOTSAVE.match(l): saves += 1; continue
            m = R_PCLOAD.match(l)
            if m: pcloads.append((int(m[2]), int(m[3]))); continue
            if R_SHIFT.search(l) and "n_discard" not in l.split("sampler")[0][:0]: shifts += 1
    return tasks, restores, saves, pcloads, shifts, first, last

def classify(d):
    """Mutually exclusive prefill class using start depth (reuse) and end position (prompt_n)."""
    reuse = d["reuse"] or 0; proc = d.get("pf_tok", 0)
    if proc > 4096 and reuse < 1024: return "full_reprocess"
    if d["prompt_n"] > 65536: return "deep_prefill_gt64k"
    return "shallow_prefill"

def bucket(v, edges):
    i = bisect.bisect_right(edges, v)
    lo = edges[i-1] if i else 0
    return f"{lo}-{edges[i]-1}" if i < len(edges) else f">={edges[-1]}"

def main(out):
    swap = get_swap()
    allt = []; allr = []; summary = {}; saves_total = 0; shifts = 0; pcl = []
    for c in CONTAINERS:
        tasks, restores, saves, pcloads, sh, f, l = parse(c)
        full = [d for d in tasks.values() if "total_ms" in d]
        summary[c] = dict(requests=len(tasks), completed=len(full), first=f, last=l, slot_saves=saves,
                          slot_restores=len(restores), prompt_cache_loads=len(pcloads))
        allt += list(tasks.values()); allr += restores; saves_total += saves; shifts += sh; pcl += pcloads
    # match slot restores to llama-swap /slots/N POST durations (swap logs at completion, second resolution)
    used = set(); matched = 0
    for r in allr:
        best = None
        for i, (te, sl, sec) in enumerate(swap):
            if i in used or sl != r["slot"]: continue
            if abs(te - r["t"]) <= 1.5 and (best is None or abs(te - r["t"]) < abs(swap[best][0] - r["t"])): best = i
        if best is not None:
            used.add(best); r["sec"] = swap[best][2]; matched += 1
    comp = [d for d in allt if "total_ms" in d]
    # time by class (slot-seconds)
    tm = collections.defaultdict(float); cnt = collections.Counter(); ptok = collections.defaultdict(int)
    for d in comp:
        k = classify(d); pf = d.get("pf_ms", 0) / 1000; dec = d.get("dec_ms", 0) / 1000
        tm[k] += pf; cnt[k] += 1; ptok[k] += d.get("pf_tok", 0); tm["decode"] += dec
        tm["other_task_time"] += max(0.0, d["total_ms"] / 1000 - pf - dec)
    rest_s = sum(r.get("sec", 0) for r in allr)
    tm["slot_file_restore"] = rest_s
    acc = sum(tm.values())
    # distributions
    PE = [1, 16, 64, 256, 1024, 4096, 16384, 32768, 65536]
    DE = [1, 1024, 4096, 16384, 32768, 65536, 90000]
    pfd = collections.Counter(bucket(d.get("pf_tok", 0), PE) for d in comp)
    dpd = collections.Counter(bucket(d["reuse"] or 0, DE) for d in comp)
    epd = collections.Counter(bucket(d["prompt_n"], DE) for d in comp)
    big = [d for d in comp if d.get("pf_tok", 0) > 4096]
    fullre = [d for d in comp if classify(d) == "full_reprocess"]
    # rate of prefill by depth bucket (tok/s)
    rate = {}
    for lo, hi in ((0, 4096), (4096, 32768), (32768, 65536), (65536, 10**9)):
        s = [d for d in comp if d.get("pf_tok", 0) > 1000 and lo <= (d["reuse"] or 0) < hi]
        t = sum(d["pf_ms"] for d in s) / 1000
        rate[f"start_depth_{lo}-{hi}"] = dict(n=len(s), tok=sum(d["pf_tok"] for d in s), sec=round(t, 2),
                                             tok_per_s=round(sum(d["pf_tok"] for d in s) / t, 1) if t else None)
    swap_win = [x for x in swap]
    res = dict(
        window_utc=dict(start=dt.datetime.fromtimestamp(min(d["t0"] for d in allt), dt.timezone.utc).isoformat(),
                        end=dt.datetime.fromtimestamp(max(d.get("t1", d["t0"]) for d in allt), dt.timezone.utc).isoformat()),
        per_container=summary, requests_total=len(allt), requests_completed=len(comp),
        time_seconds={k: round(v, 2) for k, v in tm.items()}, accounted_total_s=round(acc, 2),
        share_pct={k: round(100 * v / acc, 2) for k, v in tm.items()} if acc else {},
        class_counts=dict(cnt), class_prefill_tokens=dict(ptok),
        prefill_tokens_total=sum(d.get("pf_tok", 0) for d in comp),
        reused_tokens_total=sum(d["reuse"] or 0 for d in comp),
        prompt_tokens_total=sum(d["prompt_n"] for d in comp),
        reprocess_gt4k=dict(count=len(big), full_reprocess_count=len(fullre), partial=len(big) - len(fullre),
                            tokens=sum(d["pf_tok"] for d in big), seconds=round(sum(d["pf_ms"] for d in big) / 1000, 1)),
        prefill_size_hist=dict(pfd), start_depth_hist=dict(dpd), end_position_hist=dict(epd),
        prefill_rate_by_start_depth=rate,
        checkpoints=dict(created=sum(d["ckpt"] for d in allt), in_task_restores=sum(1 for d in allt if d["rcp"] is not None),
                         erased_invalidated=sum(d["erased_inval"] for d in allt),
                         erased_too_close=sum(d["erased_close"] for d in allt)),
        slot_file=dict(restores=len(allr), restores_timed=matched, restore_seconds=round(rest_s, 2), saves=saves_total),
        prompt_cache_loads=len(pcl), prompt_cache_loads_lcp0=sum(1 for _, l in pcl if l == 0),
        truncated_tasks=sum(1 for d in allt if d["truncated"]), context_shift_or_full_lines=shifts,
        decode=dict(tokens=sum(d.get("dec_tok", 0) for d in comp), seconds=round(tm["decode"], 1)),
        tasks_ge_90k_end=sum(1 for d in comp if d["prompt_n"] >= 90000),
    )
    json.dump(res, open(f"{out}/census.json", "w"), indent=1)
    print(json.dumps(res, indent=1))
if __name__ == "__main__":
    main(sys.argv[1])
