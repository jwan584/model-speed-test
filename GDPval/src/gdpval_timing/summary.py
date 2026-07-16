from __future__ import annotations
import json, math
from pathlib import Path

def percentile(values:list[float],p:float)->float|None:
    if not values:return None
    x=sorted(values); pos=(len(x)-1)*p; lo=math.floor(pos); hi=math.ceil(pos)
    return x[lo] if lo==hi else x[lo]+(x[hi]-x[lo])*(pos-lo)

def build_summary(records:list[dict])->dict:
    groups={}
    for r in records:
        key=f'{r["provider"]["name"]}/{r["provider"]["model"]}'; groups.setdefault(key,[]).append(r)
    out={}
    for key,rs in groups.items():
        def vals(field): return [float(r["timing"][field]) for r in rs if r.get("timing",{}).get(field) is not None]
        out[key]={"attempted":len(rs),"completed":sum(r["outcome"]=="completed" for r in rs),"outcomes":{x:sum(r["outcome"]==x for r in rs) for x in sorted({r["outcome"] for r in rs})}}
        for field in ["total_wall_seconds","inference_seconds","tool_seconds","harness_overhead_seconds",
                      "overall_request_active_billed_tps","overall_active_generation_billed_tps",
                      "overall_end_to_end_billed_tps","overall_tokens_per_second"]:
            v=vals(field); out[key][field]={"median":percentile(v,.5),"p90":percentile(v,.9)}
        request_tokens=sum(r.get("timing",{}).get("request_active_billed_tokens",0) or 0 for r in rs)
        request_seconds=sum(r.get("timing",{}).get("request_active_seconds",0) or 0 for r in rs)
        request_eligible=sum(r.get("timing",{}).get("request_active_eligible_calls",0) or 0 for r in rs)
        request_total=sum(r.get("timing",{}).get("request_active_total_calls",0) or 0 for r in rs)
        all_billed=sum(r.get("timing",{}).get("billed_output_tokens",0) or 0 for r in rs)
        reasons={}
        for r in rs:
            for reason,count in r.get("timing",{}).get("request_active_exclusion_reasons",{}).items():
                reasons[reason]=reasons.get(reason,0)+count
        active_tokens=sum(r.get("timing",{}).get("active_generation_billed_tokens",0) or 0 for r in rs)
        active_seconds=sum(r.get("timing",{}).get("active_generation_seconds",0) or 0 for r in rs)
        active_eligible=sum(r.get("timing",{}).get("active_generation_eligible_calls",0) or 0 for r in rs)
        active_total=sum(r.get("timing",{}).get("active_generation_total_calls",0) or 0 for r in rs)
        ttfts=[float(c["ttft_seconds"]) for r in rs for c in r.get("timing",{}).get("inference_calls",[])
               if c.get("ttft_seconds") is not None]
        out[key]["request_active_ratio_of_sums"]={
            "billed_output_tokens":request_tokens,"seconds":request_seconds,
            "billed_tps":request_tokens/request_seconds if request_seconds else None,
            "strict_billed_tps":request_tokens/request_seconds if request_seconds and request_total and request_eligible==request_total else None,
            "eligible_calls":request_eligible,"total_calls":request_total,
            "eligible_billed_token_coverage":request_tokens/all_billed if all_billed else None,
            "exclusion_reasons":reasons}
        out[key]["active_generation_ratio_of_sums"]={
            "billed_output_tokens":active_tokens,"seconds":active_seconds,
            "billed_tps":active_tokens/active_seconds if active_seconds else None,
            "strict_billed_tps":active_tokens/active_seconds if active_seconds and active_total and active_eligible==active_total else None,
            "eligible_calls":active_eligible,"total_calls":active_total,
            "eligible_billed_token_coverage":active_tokens/all_billed if all_billed else None}
        out[key]["ttft_seconds"]={"median":percentile(ttfts,.5),"p90":percentile(ttfts,.9),"observed_calls":len(ttfts)}
    return {"providers":out}

def write_summary(run_dir:Path)->dict:
    records=[]
    for path in run_dir.glob("records/*/*.json"):
        try: records.append(json.loads(path.read_text()))
        except Exception: pass
    result=build_summary(records); (run_dir/"summary.json").write_text(json.dumps(result,indent=2)); return result
