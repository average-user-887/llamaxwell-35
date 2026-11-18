"""Check aggregate counts, chronology, status, and the serializer-only source patch."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import subprocess

root=Path(__file__).resolve().parent
def check(value, why):
    if not value:
        raise ValueError(why)

def main():
    failed,passed=[json.loads((root/f"{s}.receipt.json").read_text()) for s in ("failed","corrected")]
    for rec,elapsed,exit_code in ((failed,34,2),(passed,58,0)):
        check(rec["bounded_wall_s"]==rec["completed_unix_epoch_s"]-rec["started_unix_epoch_s"]==elapsed,"wall time")
        check(rec["cleanup_exit"]==0 and rec["runner_exit"]==exit_code,"runner status")
    check(not failed["measurement_complete"] and failed["completed_trim_calls"]==0,"first observable")
    check(passed["same_shaped_rejected_suffix_isolation_pass"] and all(passed["rejected_tail_states_different"].values()),"boundary")
    for arm,events in passed["causal_timeline"].items():
        trim=[e for e in events if e["event"]=="ROLL3_REMOVE"]
        check(len(trim)==int(arm!="serial"),"trim count")
        if trim:
            check([trim[0][k] for k in ("before","start","after")]==[2006,2005,2004],"trim position")
        end=events[-1]
        check(end["end"]==2009 and end["stock_cache"]==(0 if arm=="serial" else 30),"observed width")
    for pair,summary in passed["comparisons"].items():
        check(summary["first_choice_equal"] and all(summary["next_choices_equal_after_correction_and_four_replay_steps"]),"choices")
        for stage in ("kept","corrected"):
            rec=summary[stage]["recurrent"];rows=rec["per_layer"]
            check(len(rows)==60 and len({(r["layer"],r["kind"]) for r in rows})==60,"layer coverage")
            check(sum(r["different_words"] for r in rows)==rec["different_f32_words"],"changed words")
            check(sum(r["f32_count"] for r in rows)==rec["payload_f32_words"]==16465920,"word total")
            check(sum(r["different_bytes"] for r in rows)==rec["different_payload_bytes"],"byte total")
            check(all(r["nonfinite_a"]==r["nonfinite_b"]==0 for r in rows),"finite comparison")
            check(rec["raw_bitwise_equal"]==(pair=="wide_a_vs_wide_b"),"same/cross shape")
    with TemporaryDirectory() as tmp:
        dest=Path(tmp)/"regenerated.cpp"
        subprocess.run(["patch","--silent","-o",str(dest),str(root/"state.original.cpp"),str(root/"serializer-fix.patch")],check=True)
        check(dest.read_bytes()==(root/"state.corrected.cpp").read_bytes(),"exact serializer patch")
    print("Verified both statuses, timeline, 240 recurrent layer rows and exact serializer fix.")
if __name__=="__main__": main()
