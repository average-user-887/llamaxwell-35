import re,struct
MAGIC=0x4d4f45345102
RESIDENCY=re.compile(r'^Q4_COHORT residency device=0 sm=24 active_per_sm=(\d+) regs=63 static_smem=96 dynamic_smem=39424 local=0 admitted=1$',re.M)
def snapshot(path):
    b=path.read_bytes()
    if len(b)!=4096: raise ValueError("incomplete observer snapshot")
    w=struct.unpack('<512Q',b)
    if w[:2]!=(MAGIC,1): raise ValueError("wrong PID1 observer layout")
    return w
def attribution(mode,begin,end,full,period):
    a,b=snapshot(begin),snapshot(end)
    d=[j-i for i,j in zip(a,b)]
    if any(x<0 for x in d) or b[2]!=1 or any(b[7:9]) or b[5]!=b[6] or b[5]!=sum(b[16:56]):
        raise ValueError("unresolved original provider, wrong target, or counter discontinuity")
    totals,prompt,decode=d[16:56],d[64:104],d[112:152]
    if d[10] or any(t!=p+c for t,p,c in zip(totals,prompt,decode)) or d[160] or not d[161] or not d[162]:
        raise ValueError("window position classification, prompt entry or decode entry missing")
    on=mode in ("serial-on","depth3-combined","smoke-on")
    native=mode!="serial-on"
    offered=period.get("proposal_width_counts",{}).get(3,0)
    if native and (not offered or d[166]<offered):
        raise ValueError("server target process did not see actual post-prompt width4 calls")
    if not on and (b[5] or RESIDENCY.search(full)):
        raise ValueError("cohort execution leaked into disabled arm")
    if on and b[5] and not any(int(v)>=2 for v in RESIDENCY.findall(full)):
        raise ValueError("physical four-cohort occupancy not proven")
    if native and on and (not sum(decode) or max(decode)>d[166]):
        raise ValueError("post-prompt candidate target layer count missing or inconsistent")
    if not native and (sum(decode) or d[166]):
        raise ValueError("serial-on executed a post-prompt width4 target")
    return {"all_layer_launches":totals,"prefill_layer_launches":prompt,"generation_layer_launches":decode,
            "physical_q4_launches":d[5],"target_prefill_calls":d[161],"target_generation_calls":d[162],
            "width4_target_prefill_calls":d[165],"width4_target_generation_calls":d[166],
            "accepted_trace_width4_rounds":offered,"full40_matches_offered_rounds":bool(on and native and decode==[offered]*40),
            "launches_per_offered_width4_round":sum(decode)/offered if offered else None,
            "unknown_or_non_target_windows_fell_back":d[9],"q1_legacy_launches":d[3],"q2_legacy_row_launches":d[4]}
