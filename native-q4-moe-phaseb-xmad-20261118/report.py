"""The numerical and timing bounds below are fixed before the first GPU use."""
import json
import math
import re
import sys
from pathlib import Path

import legacy
from legacy import CASES, FLOAT, hashes, need

HERE=Path(__file__).resolve().parent
MATERIAL=50.0
POLICY={"candidate_generic_relative_rms_additive":0.005,"candidate_generic_absolute_peak_additive":0.015,
        "candidate_v3_relative_rms_max":0.05,"candidate_v3_absolute_peak_max":0.15}
TIMES=["cohorts","xmad","xmad","cohorts"]
def resident(raw,calls):
    digits=re.findall(r"^Q4_XMAD residency device=0 sm=24 active_per_sm=(\d+) regs=(\d+) static_smem=96 dynamic_smem=39424 local=0 admitted=1$",raw,re.M)
    need(len(digits)==1 and int(digits[0][0])>=2 and int(digits[0][1])==json.loads((HERE/"build.json").read_text())["registers"],"candidate occupancy, register count or scratch differs")
    need(raw.splitlines().count(f"Q4_XMAD FINISH launches={calls}")==1,"candidate physical total")
    return {"sm_count":24,"blocks_per_sm":int(digits[0][0]),"registers":int(digits[0][1]),"threads_per_block":512,
            "launched_ctas":48,"cohorts":4,"ctas_per_cohort":12,"dynamic_shared":39424,"static_shared":96,"local_bytes":0}

def policy(row):
    return (row["generic_rms"] <= row["old_rms"]+POLICY["candidate_generic_relative_rms_additive"]
            and row["generic_peak"] <= row["old_peak"]+POLICY["candidate_generic_absolute_peak_additive"]
            and row["v3_rms"] <= POLICY["candidate_v3_relative_rms_max"]
            and row["v3_peak"] <= POLICY["candidate_v3_absolute_peak_max"])

def candidate_numeric(raw):
    output={}
    for name,overlap in CASES.items():
        need(raw.splitlines().count(f"Q4_XMAD ROUTE {name} matched=32 repeat_equal=1 launches=2")==1,"candidate direct router/repeat mismatch")
        ranks=lambda prefix:[(int(r),list(map(int,ids.split(',')))) for r,ids in re.findall(rf"^{prefix} {name} row=(\d+) ids=([0-9,]+)$",raw,re.M)]
        actual=ranks("Q4_XMAD EXPERTS")
        need(actual==ranks("N104 FIXTURE_IDS") and [r for r,_ in actual]==list(range(4)) and
             all(len(ids)==len(set(ids))==8 and min(ids)>=0 and max(ids)<256 for _,ids in actual),"physical top eight differ from independent reference")
        need([len(set(actual[a][1]) & set(actual[b][1])) for a in range(4) for b in range(a+1,4)]==overlap,"actual route geometry differs")
        out=[]
        for r,oldr,oldp,er,ep,vr,vp,n in re.findall(rf"^Q4_XMAD ROW {name} row=(\d+) finite=2048 old_rms=({FLOAT}) old_peak=({FLOAT}) generic_rms=({FLOAT}) generic_peak=({FLOAT}) v3_rms=({FLOAT}) v3_peak=({FLOAT}) changed_v3=(\d+)$",raw,re.M):
            item=dict(zip(["old_rms","old_peak","generic_rms","generic_peak","v3_rms","v3_peak"],map(float,(oldr,oldp,er,ep,vr,vp))))
            need(all(math.isfinite(v) and v>=0 for v in item.values()) and 0<=int(n)<=2048,"invalid numeric row")
            item.update(row=int(r),finite=2048,changed_vs_v3=int(n),within_policy=policy(item))
            out.append(item)
        need([r["row"] for r in out]==list(range(4)),"missing output differences")
        observed_phase=[{"row":int(r),"finite":4608,"generic_relative_rms":float(e),"generic_absolute_peak":float(p)}
                        for r,e,p in re.findall(rf"^Q4_XMAD PHASE {name} row=(\d+) finite=4608 generic_rms=({FLOAT}) generic_peak=({FLOAT})$",raw,re.M)]
        need([v["row"] for v in observed_phase]==list(range(4)) and all(math.isfinite(v["generic_relative_rms"]) and math.isfinite(v["generic_absolute_peak"]) for v in observed_phase),"direct Phase B reference missing")
        output[name]={"actual_ordered_experts":actual,"rank_overlap":overlap,"full_output":out,"phase_b_generic_comparison":observed_phase,
                      "repeated_candidate_bitwise":True,"all_within_policy":all(r["within_policy"] for r in out)}
    return output

def numeric(raw):
    old_hw,old=legacy.numeric(raw)
    obs=candidate_numeric(raw)
    return {"cohorts_residency":old_hw,"xmad_residency":resident(raw,4),"original_controls":old,"xmad":obs,
            "numeric_pass":all(row["all_within_policy"] for row in obs.values())}

def pairs(raw):
    result={}
    for name in CASES:
        rows=[(m,int(n),int(old),int(new),float(us)) for m,n,old,new,us in re.findall(
            rf"^Q4_XMAD TIME {name} mode=(cohorts|xmad) reps=(\d+) old=(\d+) new=(\d+) us=({FLOAT})$",raw,re.M)]
        need([m for m,*_ in rows]==["cohorts","xmad",*TIMES] and [r[1] for r in rows]==[8,8,96,96,96,96],"incomplete paired warmups or ABBA")
        for mode,n,old,new,us in rows:
            need((old,new)==((n,0) if mode=="cohorts" else (0,n)) and math.isfinite(us) and us>0,"incorrect physical interval or timing")
        us=[r[4] for r in rows[2:]]
        saving=[us[0]-us[1],us[3]-us[2]]
        result[name]={"complete_intervals":rows,"incremental_savings_us":saving,"projected_ms_at_40_layers":[v*.04 for v in saving],
                      "materiality_pass":min(saving)>=MATERIAL,"positive_both_orders":min(saving)>0}
    return result

def timing(raw):
    old=legacy.common(raw,400)
    new=resident(raw,400)
    need("MOE_V3 CHECK" not in raw and "Q4_XMAD ROUTE " not in raw,"diagnostics active in timed intervals")
    return {"cohorts_residency":old,"xmad_residency":new,"timing":pairs(raw)}

def write(path,data):
    payload=json.dumps(data,indent=2,allow_nan=False)+"\n"
    if path.exists(): need(json.loads(path.read_text())==json.loads(payload),"saved result differs")
    else: path.write_text(payload)

def export(root):
    status=dict((k,int(v)) for k,v in (ln.split("=",1) for ln in (root/"status.txt").read_text().splitlines()))
    result={"status":status,"bounded_wall_s":status["finished_epoch"]-status["started_epoch"],"build":json.loads((HERE/"build.json").read_text()),
            "numeric_policy":POLICY,"materiality_us":MATERIAL,"scope":"Synthetic four-row standalone integer gate/up; paired synchronized standalone timing; forty-layer savings are extrapolations."}
    for stage,func in (("check",numeric),("perf",timing)):
        p=root/(stage+".log")
        if not p.exists(): result[stage]={"executed":False};continue
        raw=p.read_text(errors="replace")
        data={"executed":True,"observed_lines":[line for line in raw.splitlines() if line.startswith(("N104 ","Q4_XMAD ","Q4_COHORT ","occamy-fused MOE_V3 CHECK ")) and len(line)<400]}
        try: data["validated"]=func(hashes(root,stage))
        except (ValueError,KeyError,OSError) as exc: data["validation_error"]=str(exc) if isinstance(exc,ValueError) else type(exc).__name__
        result[stage]=data
    result["complete_measurement"]=json.loads((root/"result.json").read_text()) if (root/"result.json").exists() else None
    (root/"portable-receipt.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")

def selftest():
    legacy.selftest()
    record=("old_rms=0.015 old_peak=0.06 generic_rms=0.019 generic_peak=0.074 v3_rms=0.03 v3_peak=0.1 changed_v3=2048")
    raw=(HERE/"fixture.txt").read_text()
    for name in CASES:
        raw+=f"Q4_XMAD ROUTE {name} matched=32 repeat_equal=1 launches=2\n"
        raw+="".join(ln.replace("N104 FIXTURE_IDS","Q4_XMAD EXPERTS")+"\n" for ln in (HERE/"fixture.txt").read_text().splitlines() if ln.startswith("N104 FIXTURE_IDS "+name+" "))
        for r in range(4):
            raw+=f"Q4_XMAD ROW {name} row={r} finite=2048 {record}\nQ4_XMAD PHASE {name} row={r} finite=4608 generic_rms=0.0001 generic_peak=0.0003\n"
        for m,n,t in zip(["cohorts","xmad",*TIMES],[8,8,96,96,96,96],[306,231,300,245,240,302]):
            raw+=f"Q4_XMAD TIME {name} mode={m} reps={n} old={n if m=='cohorts' else 0} new={n if m=='xmad' else 0} us={t}\n"
    need(all(r["all_within_policy"] for r in candidate_numeric(raw).values()),"synthetic numeric positive")
    need(not candidate_numeric(raw.replace("generic_rms=0.019","generic_rms=0.0201",1))["independent"]["all_within_policy"],"unbounded generic error")
    need(not candidate_numeric(raw.replace("v3_peak=0.1","v3_peak=0.151",1))["independent"]["all_within_policy"],"unbounded v3 peak")
    need(all(r["materiality_pass"] for r in pairs(raw).values()),"synthetic materiality")
    need(not pairs(raw.replace("us=245","us=251",1))["independent"]["materiality_pass"],"timing below threshold")
    for method,bad in ((pairs,raw.replace("new=96","new=95",1)),(candidate_numeric,raw.replace("matched=32","matched=31",1))):
        try: method(bad)
        except ValueError: continue
        raise ValueError("false coverage accepted")
    print("CPU-only phase, arithmetic, physical counts, route ordering, and paired speed classifications passed.")

if __name__=="__main__":
    action,*args=sys.argv[1:]
    if action=="--check": selftest()
    elif action=="--export": export(Path(args[0]))
    elif action=="--numeric":
        d=numeric(hashes(Path(args[0]),"check"))
        print(json.dumps(d,indent=2));need(d["numeric_pass"],"fixed full-row error envelope failed")
    elif action=="--report":
        root=Path(args[0]);d=numeric(hashes(root,"check"));t=timing(hashes(root,"perf"))
        need(d["numeric_pass"] and d["xmad_residency"]==t["xmad_residency"],"arithmetic or shared hardware mismatch")
        result={"technical_pass":True,"materiality_pass":all(r["materiality_pass"] for r in t["timing"].values()),"numeric_policy":POLICY,"threshold_us":MATERIAL,"numeric":d,"timing":t}
        write(root/"result.json",result)
        print(json.dumps({"technical_pass":True,"materiality_pass":result["materiality_pass"],"timing":t["timing"]},indent=2))
    else: raise SystemExit("unknown mode")
