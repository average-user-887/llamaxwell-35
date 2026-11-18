"""Validate measured 4x12 residency, actual routed outputs, and matched whole-operator timings."""
import json
import math
from pathlib import Path
import re
import sys
HERE=Path(__file__).resolve().parent
CASES={"independent":[0,0,0,0,0,0],"partial":[5,4,4,2,5,3]}
MATERIAL=190.0
MAPS=("libggml.so.0.26.0","libggml-base.so.0.26.0","libggml-cuda.so")
FLOAT=r"[0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?"
ORDER=["generic","sequential","cohorts","cohorts","sequential","generic"]
def need(ok,message):
    if not ok: raise ValueError(message)

def hashes(directory,stage):
    def pins(file):
        out={}
        for row in (directory/file).read_text().splitlines():
            digest,name=row.split("  ",1)
            out[Path(name).name]=digest
        return out
    expected=pins("RUNTIME.sha256")
    expected={name:expected[name] for name in MAPS}
    expected["operator"]=pins("FROZEN.sha256")["operator"]
    raw=(directory/f"{stage}.log").read_text()
    actual={Path(path).name:value for value,path in re.findall(r"^([0-9a-f]{64})  (/(?:runtime/\S+|operator))$",raw,re.M)}
    need(actual==expected,"mapped runtime or executable hash differs")
    return raw

def common(raw,total):
    need(not re.search(r"^N104 ERROR|CUDA error|GGML_ASSERT|CUDA out of memory|illegal memory access",raw,re.I|re.M),"device or arithmetic failure")
    need(sorted(re.findall(r"^N104 MAP /runtime/(\S+)$",raw,re.M))==sorted(MAPS),"actual backend inventory missing")
    for line in (HERE/"fixture.txt").read_text().splitlines():
        need(raw.splitlines().count(line)==1,"seeded width-four route geometry or margin differs")
    need(raw.splitlines().count("Q4_COHORT MAP /operator counter_provider=/runtime/libggml-cuda.so")==1,"counter provider or running executable mismatch")
    matches=re.findall(r"^Q4_COHORT residency device=0 sm=24 active_per_sm=(\d+) regs=63 static_smem=96 dynamic_smem=39424 local=0 admitted=1$",raw,re.M)
    need(len(matches)==1 and int(matches[0])>=2,"candidate launch occupancy not proven")
    need(raw.splitlines().count(f"Q4_COHORT FINISH q3_launches=0 q4_launches={total}")==1,"legacy Q3 or total Q4 physical call mismatch")
    return {"sm_count":24,"active_ctas_per_sm":int(matches[0]),"resident_capacity":24*int(matches[0]),
            "ctas_launched":48,"threads_per_cta":512,"private_barrier_participants":12,"cohorts":4,
            "registers":63,"static_shared_bytes":96,"dynamic_shared_bytes":39424,"thread_local_bytes":0}

def numeric(raw):
    hardware=common(raw,2)
    need(raw.count("MOE_V3 variant 2 blocks/SM x 512 thr -> grid 48, smem 39424 B")==1,"legacy V3 occupancy missing")
    need(len(re.findall(r"MOE_V3 CHECK ",raw))==8,"missing or extra V3 diagnostics")
    output={}
    for name,overlap in CASES.items():
        for line in (f"N104 ROUTER {name} ids=32/32",f"Q4_COHORT DIAG {name} physical=1"):
            need(raw.splitlines().count(line)==1,"32 actual router IDs or candidate call missing")
        plain=re.findall(rf"^N104 GENERIC_BEGIN {name}$\n(.*?)^N104 GENERIC_END {name}$",raw,re.M|re.S)
        split=re.findall(rf"^N104 SPLIT_BEGIN {name}$\n(.*?)^N104 SPLIT_END {name}$",raw,re.M|re.S)
        need(len(plain)==len(split)==1 and "MOE_V3" not in plain[0],"ordinary width4 graph or split brackets missing")
        v3=re.findall(rf"occamy-fused MOE_V3 CHECK n104_v3_row_(\d+) row 0: bitwise-diff (\d+) / 2048, max_abs ({FLOAT})",split[0])
        need([(int(r),int(n),float(e)) for r,n,e in v3]==[(r,0,0.0) for r in range(4)],"legacy per-row V3 differs from V2 or row missing")
        rows=re.findall(rf"^Q4_COHORT ROW {name} row=(\d+) finite=2048 different=(\d+)$",raw,re.M)
        need([(int(r),int(n)) for r,n in rows]==[(r,0) for r in range(4)],"cohorts differ from sequential all-row reference")
        generic=[]
        for row,n,err,rms in re.findall(rf"^N104 NUMERIC {name} row=(\d+) finite=2048 mismatches=(\d+) max_abs=({FLOAT}) relative_rms=({FLOAT})$",raw,re.M):
            need(0<=int(n)<=2048 and math.isfinite(float(err)) and math.isfinite(float(rms)),"nonfinite generic comparison")
            generic.append({"row":int(row),"different_words":int(n),"finite":2048,"max_abs":float(err),"relative_rms":float(rms)})
        need([r["row"] for r in generic]==list(range(4)),"generic comparisons incomplete")
        output[name]={"pair_overlaps":overlap,"gpu_route_ids_equal":32,"candidate_words_checked":8192,
                      "candidate_vs_v3_different_by_row":[0]*4,"v3_vs_v2_different_by_row":[0]*4,"generic_vs_v3":generic}
    return hardware,output

def timing(raw):
    hardware=common(raw,400)
    need("MOE_V3 CHECK" not in raw,"reference check active during timing")
    output={}
    for name in CASES:
        data=re.findall(rf"^Q4_COHORT TIME {name} mode=(\w+) reps=(\d+) launches=(\d+) us=({FLOAT})$",raw,re.M)
        need([m for m,_,_,_ in data]==ORDER,"six timing phases missing")
        avg,physical=[],[]
        for mode,reps,launches,us in data:
            need(int(reps)==96 and int(launches)==(96 if mode=="cohorts" else 0),"unproven physical measured route")
            need(float(us)>0 and math.isfinite(float(us)),"invalid latency")
            avg.append(float(us));physical.append(int(launches))
        generic=[avg[0]-avg[2],avg[5]-avg[3]]
        over_v3=[avg[1]-avg[2],avg[4]-avg[3]]
        output[name]={"order":ORDER,"mean_us":avg,"q4_launches":physical,"generic_minus_cohorts_us":generic,
             "four_v3_rows_minus_cohorts_us":over_v3,"projected_generic_gain_ms_over_40_layers":[.04*x for x in generic],
             "materiality_pass":min(generic)>=MATERIAL,"positive_generic_all_pairs":min(generic)>0,
             "incremental_30us_all_pairs":min(over_v3)>=30}
    return hardware,output

def verify(root):
    hw,arith=numeric(hashes(root,"check"))
    hw2,rates=timing(hashes(root,"perf"))
    need(hw==hw2,"different hardware between diagnostics and timing")
    result={"technical_pass":True,"materiality_pass":all(r["materiality_pass"] for r in rates.values()),
            "positive_generic_all_pairs":all(r["positive_generic_all_pairs"] for r in rates.values()),
            "incremental_30us_all_pairs":all(r["incremental_30us_all_pairs"] for r in rates.values()),
            "threshold_us_per_window":MATERIAL,"residency":hw,"arithmetic":arith,"timing":rates,
            "scope":"Standalone seeded 4-row/8-expert operator. Each interval uses 96 completed graph runs. Four independent V3 row outputs require three concatenations. Forty-window projections assume representative windows and unchanged accepted prefixes."}
    path=root/"result.json"
    if path.exists(): need(json.loads(path.read_text())==result,"saved result differs")
    else: path.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    return {k:result[k] for k in ("technical_pass","materiality_pass","positive_generic_all_pairs","incremental_30us_all_pairs","timing")}

def selftest():
    base=(HERE/"fixture.txt").read_text()+"".join(f"N104 MAP /runtime/{n}\n" for n in MAPS)
    base+="Q4_COHORT MAP /operator counter_provider=/runtime/libggml-cuda.so\nQ4_COHORT residency device=0 sm=24 active_per_sm=2 regs=63 static_smem=96 dynamic_smem=39424 local=0 admitted=1\n"
    check=base+"MOE_V3 variant 2 blocks/SM x 512 thr -> grid 48, smem 39424 B\n"
    perf=base
    for name in CASES:
        check+=f"N104 ROUTER {name} ids=32/32\nN104 GENERIC_BEGIN {name}\nN104 GENERIC_END {name}\nN104 SPLIT_BEGIN {name}\n"
        for r in range(4): check+=f"occamy-fused MOE_V3 CHECK n104_v3_row_{r} row 0: bitwise-diff 0 / 2048, max_abs 0.0\n"
        check+=f"N104 SPLIT_END {name}\nQ4_COHORT DIAG {name} physical=1\n"
        for r in range(4): check+=f"Q4_COHORT ROW {name} row={r} finite=2048 different=0\nN104 NUMERIC {name} row={r} finite=2048 mismatches=1 max_abs=0.1 relative_rms=0.01\n"
        for mode,us in zip(ORDER,[500,405,300,300,405,500]): perf+=f"Q4_COHORT TIME {name} mode={mode} reps=96 launches={96 if mode=='cohorts' else 0} us={us}.0\n"
    check+="Q4_COHORT FINISH q3_launches=0 q4_launches=2\n"
    perf+="Q4_COHORT FINISH q3_launches=0 q4_launches=400\n"
    numeric(check)
    need(all(x["materiality_pass"] for x in timing(perf)[1].values()),"positive synthetic classification")
    need(not timing(perf.replace("us=500.0","us=489.9",1))[1]["independent"]["materiality_pass"],"partial synthetic classification")
    for method,s,bad in ((numeric,check,check.replace("active_per_sm=2","active_per_sm=1")),(numeric,check,check.replace("ids=32/32","ids=31/32",1)),
             (numeric,check,check.replace("row=3 finite=2048 different=0","row=3 finite=2048 different=1",1)),
             (timing,perf,perf.replace("launches=96","launches=95",1)),(timing,perf,perf.replace("q3_launches=0","q3_launches=1"))):
        need(s!=bad,"ineffective negative check")
        try: method(bad)
        except ValueError: continue
        raise ValueError("defective synthetic receipt passed")
    print("CPU-only synthetic parser checks passed for physical counts, row arithmetic, occupancy and materiality.")
def export(root):
    status={k:int(v) for k,v in (row.split("=",1) for row in (root/"status.txt").read_text().splitlines())}
    report={"materiality_threshold_us":MATERIAL,"status":status,"bounded_wall_s":status["finished_epoch"]-status["started_epoch"],
            "sources":json.loads((HERE/"build.json").read_text()),
            "timing_scope":"Standalone full-graph wall time per synchronized run, two static fixtures. No native-model throughput measurement."}
    for stage,method in (("check",numeric),("perf",timing)):
        log=root/(stage+".log")
        if not log.exists():
            report[stage]={"executed":False}
            continue
        raw=log.read_text(errors="replace")
        observed={"executed":True,"observed_physical_totals":[list(map(int,row)) for row in re.findall(
            r"^Q4_COHORT FINISH q3_launches=(\d+) q4_launches=(\d+)$",raw,re.M)],
            "actual_candidate_row_checks":[[name,int(r),int(n)] for name,r,n in re.findall(
            r"^Q4_COHORT ROW (independent|partial) row=(\d+) finite=2048 different=(\d+)$",raw,re.M)],
            "observed_intervals":[[name,mode,int(reps),int(n),float(us)] for name,mode,reps,n,us in re.findall(
            rf"^Q4_COHORT TIME (independent|partial) mode=(generic|sequential|cohorts) reps=(\d+) launches=(\d+) us=({FLOAT})$",raw,re.M) if math.isfinite(float(us))]}
        error=re.search(r"^N104 ERROR ([A-Za-z0-9 /_,.=-]{1,130})$",raw,re.M)
        if error: observed["operator_refusal"]=error[1]
        try:
            verified=hashes(root,stage)
            hardware,results=method(verified)
        except (ValueError,KeyError,OSError) as err:
            observed["verified"]=False
            observed["validator_refusal"]=str(err) if isinstance(err,ValueError) else type(err).__name__
        else:
            observed.update(verified=True,residency=hardware,results=results)
        report[stage]=observed
    result=root/"result.json"
    report["complete_measurement"]=json.loads(result.read_text()) if result.exists() else None
    out=root/"portable-receipt.json"
    out.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")

if __name__=="__main__":
    action,*args=sys.argv[1:]
    if action=="--check" and not args: selftest()
    elif action=="--numeric" and len(args)==1: print(json.dumps(numeric(hashes(Path(args[0]),"check")),indent=2))
    elif action=="--export" and len(args)==1: export(Path(args[0]))
    elif action=="--report" and len(args)==1: print(json.dumps(verify(Path(args[0])),indent=2))
    else: raise SystemExit("usage: report.py --check | --numeric RECEIPT | --report RECEIPT")
