from __future__ import annotations
import asyncio, hashlib, json, os, platform, re
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any
from gdpval_timing.dataset import GDPvalDataset, Task
from gdpval_timing.models import InferenceTiming, Message, ToolTiming, jsonable
from gdpval_timing.providers import PROVIDERS
from gdpval_timing.providers.base import InferenceTimeoutError
from gdpval_timing.sandbox import Sandbox
from gdpval_timing.summary import write_summary
from gdpval_timing.tools import SCHEMAS, Tools

def slug(value:str)->str:return re.sub(r"[^A-Za-z0-9_.-]+","_",value)
def utcnow()->str:return datetime.now(timezone.utc).isoformat()

class Harness:
    def __init__(self,config:dict[str,Any]):
        self.config=config; self.run_cfg=config["run"]; self.run_dir=Path(self.run_cfg["output_dir"])
        self.run_dir.mkdir(parents=True,exist_ok=True); (self.run_dir/"records").mkdir(exist_ok=True); (self.run_dir/"workspaces").mkdir(exist_ok=True)
        self.dataset=GDPvalDataset(self.run_cfg["dataset_cache"])
    def select(self,tasks:list[Task])->list[Task]:
        ids=list(self.run_cfg.get("task_ids") or [])
        if ids:
            by_id={t.task_id:t for t in tasks}
            missing=[task_id for task_id in ids if task_id not in by_id]
            if missing: raise ValueError(f"Unknown task_ids: {missing}")
            return [by_id[task_id] for task_id in ids]
        offset=int(self.run_cfg.get("offset",0)); limit=self.run_cfg.get("limit")
        return tasks[offset:] if limit is None else tasks[offset:offset+int(limit)]
    async def run(self)->dict:
        if int(self.run_cfg.get("concurrency",1))!=1: raise ValueError("concurrency must be 1 for timing fidelity")
        await Sandbox.preflight(self.config["sandbox"])
        tasks=self.select(self.dataset.load())
        for name,pconfig in self.config["providers"].items():
            if not pconfig.get("enabled",False): continue
            required_effort=self.run_cfg.get("required_reasoning_effort")
            if required_effort and pconfig.get("reasoning_effort")!=required_effort:
                raise ValueError(f"Provider {name} must set reasoning_effort={required_effort!r} for this run")
            key=os.environ.get(pconfig["api_key_env"])
            if not key: raise RuntimeError(f'Missing environment variable {pconfig["api_key_env"]}')
            provider_type=pconfig.get("type",name)
            if provider_type not in PROVIDERS: raise ValueError(f"Unknown provider type: {provider_type}")
            provider=PROVIDERS[provider_type](pconfig,key)
            warmup=await provider.warmup() if self.run_cfg.get("warmup",True) else {"skipped":True}
            try:
                for task in tasks:
                    repetitions=int(self.run_cfg.get("repetitions",1))
                    if repetitions < 1: raise ValueError("repetitions must be at least 1")
                    structural_failures=0
                    for repetition in range(1,repetitions+1):
                        filename=f"{task.task_id}.json" if repetitions==1 else f"{task.task_id}.run-{repetition:03d}.json"
                        record_path=self.run_dir/"records"/slug(name)/filename
                        if self.run_cfg.get("resume",True) and record_path.exists(): continue
                        record=await self._safe_task(provider,name,pconfig,task,warmup,repetition,repetitions)
                        record_path.parent.mkdir(parents=True,exist_ok=True)
                        tmp=record_path.with_suffix(".tmp"); tmp.write_text(json.dumps(record,indent=2)); tmp.replace(record_path)
                        print(json.dumps({"task_id":task.task_id,"repetition":repetition,"provider":name,"model":provider.model,"outcome":record["outcome"],"timing":record["timing"]},indent=2),flush=True)
                        write_summary(self.run_dir)
                        economy=self.run_cfg.get("economy",{})
                        structural_outcomes=set(economy.get("structural_failure_outcomes",[
                            "max_turns","completion_unlikely","deliverable_unfinished","stalled",
                            "tool_failure_loop","inference_timeout","timeout","error"
                        ]))
                        if record["outcome"] in structural_outcomes and not record["deliverables"]: structural_failures+=1
                        else: structural_failures=0
                        threshold=int(economy.get("skip_repetitions_after_structural_failures",0))
                        if economy.get("enabled",False) and threshold and structural_failures>=threshold:
                            print(json.dumps({"task_id":task.task_id,"provider":name,"skipped_repetitions":repetitions-repetition,
                                              "reason":f"{structural_failures} consecutive structural failures with zero deliverables"}),flush=True)
                            break
            finally: await provider.close()
        return write_summary(self.run_dir)
    async def _safe_task(self,provider,name,pconfig,task,warmup,repetition=1,repetitions=1):
        started=perf_counter(); started_at=utcnow()
        state={"inf":[],"tools":[],"backoff":0.0,"retry_api":0.0,"retries":0,"files":[],"active":None,"output_dir":None}
        try: return await asyncio.wait_for(self._task(provider,name,pconfig,task,warmup,started,started_at,state,repetition,repetitions),float(self.run_cfg["task_timeout_seconds"]))
        except TimeoutError as exc:
            is_inference=isinstance(exc,InferenceTimeoutError)
            outcome="inference_timeout" if is_inference else "timeout"
            error=str(exc) if is_inference else f"Task exceeded {self.run_cfg['task_timeout_seconds']} seconds"
            active=state.get("active")
            if active:
                elapsed=perf_counter()-active["start"]
                if active["kind"]=="inference": state["inf"].append(InferenceTiming(len(state["inf"])+1,1,active["stamp"],elapsed,None,None,None,None))
                else: state["tools"].append(ToolTiming(len(state["tools"])+1,active["turn"],active["name"],active["id"],active["stamp"],elapsed,False))
        except Exception as exc:
            outcome,error="error",f"{type(exc).__name__}: {exc}"
            state["backoff"]+=getattr(exc,"backoff_seconds",0.0); state["retry_api"]+=getattr(exc,"retry_api_seconds",0.0); state["retries"]+=getattr(exc,"retry_count",0)
            active=state.get("active")
            if active and active["kind"]=="inference":
                state["inf"].append(InferenceTiming(len(state["inf"])+1,1,active["stamp"],perf_counter()-active["start"],None,None,None,None))
        total=perf_counter()-started
        if state.get("output_dir"): state["files"]=self._files(state["output_dir"])
        return self._record(task,name,pconfig,warmup,started_at,outcome,error,total,state["inf"],state["tools"],state["backoff"],state["retry_api"],state["retries"],state["files"],repetition)
    async def _task(self,provider,name,pconfig,task,warmup,started,started_at,state,repetition=1,repetitions=1):
        work=self.run_dir/"workspaces"/slug(name)/task.task_id
        if repetitions > 1: work=work/f"run-{repetition:03d}"
        input_dir=work/"input"; output_dir=work/"output"; state["output_dir"]=output_dir
        input_dir.mkdir(parents=True,exist_ok=True); output_dir.mkdir(parents=True,exist_ok=True)
        refs=self.dataset.stage(task,input_dir)
        intro=("You are completing a GDPval professional task in a persistent sandbox. Input files are in /workspace/input "
               "and every final deliverable must be written under /workspace/output. Inspect files with tools; do not merely describe work. "
               "Call finish only when the files exist.\n\nTASK:\n"+task.prompt+"\n\nREFERENCE FILES:\n"+"\n".join(f"- /workspace/input/{x}" for x in refs))
        messages=[Message(role="user",text=intro)]; inf=state["inf"]; tool_times=state["tools"]; outcome="max_turns"; error=None; finished=None
        economy=self.run_cfg.get("economy",{}); context_cfg=self.run_cfg.get("context",{})
        no_tool_turns=0; consecutive_tool_failures=0; repeated_error_signature=None; repeated_error_count=0
        finalize_prompted_turn=None; last_manifest=self._manifest(output_dir); no_artifact_progress_turns=0
        sandbox=Sandbox(self.config["sandbox"],work,float(self.run_cfg["tool_timeout_seconds"]))
        async with sandbox:
            secrets={"brave":os.environ.get(self.config.get("tools",{}).get("web_search",{}).get("api_key_env","BRAVE_API_KEY"),"")}
            tools=Tools(sandbox,work,self.config.get("tools",{}),secrets)
            try:
                for turn in range(1,int(self.run_cfg["max_turns"])+1):
                    self._compact_history(messages,context_cfg)
                    state["active"]={"kind":"inference","start":perf_counter(),"stamp":utcnow(),"turn":turn}
                    response=await provider.generate(messages,SCHEMAS,len(inf)+1); inf.append(response.timing); state["active"]=None
                    state["backoff"]+=response.backoff_seconds; state["retry_api"]+=response.retry_api_seconds; state["retries"]+=response.retry_count
                    messages.append(response.message)
                    if not response.message.tool_calls:
                        no_tool_turns+=1
                        has_output=any(p.is_file() for p in output_dir.rglob("*"))
                        force_turn=int(economy.get("force_finalize_turn",0))
                        stop_turn=int(economy.get("stop_empty_output_turn",0))
                        if economy.get("enabled",False) and stop_turn and turn>=stop_turn and not has_output:
                            outcome="completion_unlikely"; error=f"No files in /workspace/output by economy cutoff turn {stop_turn}"; break
                        stall_limit=int(economy.get("max_consecutive_no_tool_turns",0))
                        if economy.get("enabled",False) and stall_limit and no_tool_turns>=stall_limit:
                            outcome="stalled"; error=f"No tool call for {no_tool_turns} consecutive turns"; break
                        if economy.get("enabled",False) and force_turn and turn>=force_turn and finalize_prompted_turn is None:
                            finalize_prompted_turn=turn
                            messages.append(Message(role="user",text=self._finalize_prompt(has_output)))
                        else: messages.append(Message(role="user",text="Continue using tools. You must create deliverables and call finish."))
                        grace=int(economy.get("finalize_grace_turns",3))
                        if finalize_prompted_turn is not None and turn>=finalize_prompted_turn+grace:
                            outcome="deliverable_unfinished" if has_output else "completion_unlikely"
                            error=f"Model did not call finish within {grace} finalization grace turns"; break
                        continue
                    no_tool_turns=0
                    results=[]
                    for call in response.message.tool_calls:
                        t0=perf_counter(); stamp=utcnow(); state["active"]={"kind":"tool","start":t0,"stamp":stamp,"turn":turn,"name":call.name,"id":call.id}
                        result=await tools.execute(call); duration=perf_counter()-t0; state["active"]=None
                        tool_times.append(ToolTiming(len(tool_times)+1,turn,call.name,call.id,stamp,duration,not result.is_error)); results.append(result)
                        if result.is_error:
                            consecutive_tool_failures+=1
                            signature=self._error_signature(result.output)
                            repeated_error_count=repeated_error_count+1 if signature==repeated_error_signature else 1
                            repeated_error_signature=signature
                        else:
                            consecutive_tool_failures=0; repeated_error_count=0; repeated_error_signature=None
                        if call.name=="finish" and not result.is_error: outcome="completed"; finished=perf_counter()
                    messages.append(Message(role="tool",tool_results=results))
                    if outcome=="completed": break
                    if economy.get("enabled",False):
                        has_output=any(p.is_file() for p in output_dir.rglob("*"))
                        manifest=self._manifest(output_dir)
                        no_artifact_progress_turns=0 if manifest!=last_manifest else no_artifact_progress_turns+1
                        last_manifest=manifest
                        force_turn=int(economy.get("force_finalize_turn",0))
                        stop_turn=int(economy.get("stop_empty_output_turn",0))
                        if stop_turn and turn>=stop_turn and not has_output:
                            outcome="completion_unlikely"; error=f"No files in /workspace/output by economy cutoff turn {stop_turn}"; break
                        warning=int(economy.get("repeated_error_warning_threshold",3))
                        failure_limit=int(economy.get("max_consecutive_tool_failures",5))
                        if failure_limit and consecutive_tool_failures>=failure_limit:
                            outcome="tool_failure_loop"; error=f"{consecutive_tool_failures} consecutive tool failures"; break
                        if warning and repeated_error_count==warning:
                            messages.append(Message(role="user",text="The same tool error has repeated. Stop retrying that approach; use a different method or finalize the best valid deliverable now."))
                        no_progress_limit=int(economy.get("max_no_progress_turns",5))
                        progress_start=int(economy.get("progress_tracking_start_turn",12))
                        should_finalize=(force_turn and turn>=force_turn) or (turn>=progress_start and no_progress_limit and no_artifact_progress_turns>=no_progress_limit)
                        if should_finalize and finalize_prompted_turn is None:
                            finalize_prompted_turn=turn; messages.append(Message(role="user",text=self._finalize_prompt(has_output)))
                        grace=int(economy.get("finalize_grace_turns",3))
                        if finalize_prompted_turn is not None and turn>=finalize_prompted_turn+grace:
                            outcome="deliverable_unfinished" if has_output else "completion_unlikely"
                            error=f"Model did not call finish within {grace} finalization grace turns"; break
            finally: await tools.close()
        total=(finished or perf_counter())-started; files=self._files(output_dir); state["files"]=files
        if outcome=="completed" and not files: outcome="error"; error="finish was called but /workspace/output is empty"
        return self._record(task,name,pconfig,warmup,started_at,outcome,error,total,inf,tool_times,state["backoff"],state["retry_api"],state["retries"],files,repetition)
    def _files(self,directory:Path)->list[dict]:
        out=[]
        for p in sorted(directory.rglob("*")):
            if p.is_file(): out.append({"path":str(p.resolve()),"relative_path":str(p.relative_to(directory)),"bytes":p.stat().st_size,"sha256":hashlib.sha256(p.read_bytes()).hexdigest()})
        return out
    @staticmethod
    def _manifest(directory:Path)->tuple:
        return tuple((str(p.relative_to(directory)),p.stat().st_size,p.stat().st_mtime_ns) for p in sorted(directory.rglob("*")) if p.is_file())
    @staticmethod
    def _error_signature(output:str)->str:
        return re.sub(r"\b\d+(?:\.\d+)?\b","#",output[:500]).strip()
    @staticmethod
    def _finalize_prompt(has_output:bool)->str:
        state="Existing output files must be validated briefly and then submitted." if has_output else "Create the best valid deliverables possible now."
        return ("ECONOMY FINALIZATION DEADLINE: stop research, package installation, and nonessential inspection. "
                f"{state} Use the remaining turns only for /workspace/output and call finish.")
    @staticmethod
    def _compact_history(messages:list[Message],config:dict[str,Any])->None:
        keep=int(config.get("retain_recent_tool_turns",8)); limit=int(config.get("max_historical_tool_result_chars",4000))
        image_keep=int(config.get("image_payload_retention_turns",1))
        tool_indexes=[i for i,m in enumerate(messages) if m.role=="tool"]
        old=set(tool_indexes[:-keep] if keep else tool_indexes)
        image_old=set(tool_indexes[:-image_keep] if image_keep else tool_indexes)
        for i in tool_indexes:
            for result in messages[i].tool_results:
                if i in old and len(result.output)>limit: result.output=result.output[:limit]+"\n[historical tool output truncated; full artifacts remain on disk]"
                if i in image_old and result.image_data_url:
                    result.image_data_url=None
                    result.output += "\n[historical image payload removed; source file remains in workspace]"
    def _record(self,task,name,pconfig,warmup,started_at,outcome,error,total,inf,tools,backoff,retry_api,retries,files,repetition=1):
        inference=sum(x.latency_seconds for x in inf); tool=sum(x.duration_seconds for x in tools)
        overhead=total-inference-tool-backoff-retry_api
        comparable_output=sum(x.comparable_output_tokens or 0 for x in inf)
        reliable_calls=[x for x in inf if x.post_ttft_tokens_per_second_reliable]
        reliable_tokens=sum(x.comparable_output_tokens or 0 for x in reliable_calls)
        observable_seconds=sum((x.last_observable_output_seconds or 0)-(x.first_observable_output_seconds or 0) for x in reliable_calls)
        billed_output=sum(x.billed_output_tokens or 0 for x in inf)
        request_active_calls=[x for x in inf if x.request_active_eligible]
        request_active_billed_output=sum(x.billed_output_tokens or 0 for x in request_active_calls)
        request_active_seconds=sum(x.request_active_seconds or 0 for x in request_active_calls)
        strict_request_active_available=bool(inf) and len(request_active_calls)==len(inf)
        active_calls=[x for x in inf if x.outcome=="completed" and x.active_generation_seconds is not None and x.billed_output_tokens is not None]
        active_billed_output=sum(x.billed_output_tokens or 0 for x in active_calls)
        active_generation_seconds=sum(x.active_generation_seconds or 0 for x in active_calls)
        strict_active_available=bool(inf) and len(active_calls)==len(inf)
        exclusion_reasons={}
        for call in inf:
            if call.request_active_exclusion_reason:
                reason=call.request_active_exclusion_reason
                exclusion_reasons[reason]=exclusion_reasons.get(reason,0)+1
        reasoning_values=[x.reasoning_tokens for x in inf]
        input_tokens=sum(x.input_tokens or 0 for x in inf)
        cached_input_tokens=sum(x.cached_input_tokens or 0 for x in inf)
        non_reasoning_values=[x.non_reasoning_output_tokens for x in inf]
        timing={"total_wall_seconds":total,"inference_seconds":inference,"tool_seconds":tool,"backoff_seconds":backoff,
                "retry_api_seconds":retry_api,"harness_overhead_seconds":overhead,"inference_calls":jsonable(inf),"tool_calls":jsonable(tools),
                "billed_output_tokens":billed_output,"comparable_output_tokens":comparable_output,
                "active_generation_billed_tokens":active_billed_output,
                "active_generation_seconds":active_generation_seconds,
                "active_generation_eligible_calls":len(active_calls),
                "active_generation_total_calls":len(inf),
                "active_generation_eligible_billed_token_coverage":active_billed_output/billed_output if billed_output else None,
                "overall_active_generation_billed_tps":active_billed_output/active_generation_seconds if strict_active_available and active_generation_seconds else None,
                "partial_active_generation_billed_tps":active_billed_output/active_generation_seconds if active_generation_seconds else None,
                "request_active_billed_tokens":request_active_billed_output,
                "request_active_seconds":request_active_seconds,
                "request_active_eligible_calls":len(request_active_calls),
                "request_active_total_calls":len(inf),
                "request_active_excluded_calls":len(inf)-len(request_active_calls),
                "request_active_exclusion_reasons":exclusion_reasons,
                "request_active_eligible_billed_token_coverage":request_active_billed_output/billed_output if billed_output else None,
                "overall_request_active_billed_tps":request_active_billed_output/request_active_seconds if strict_request_active_available and request_active_seconds else None,
                "partial_request_active_billed_tps":request_active_billed_output/request_active_seconds if request_active_seconds else None,
                "overall_end_to_end_billed_tps":request_active_billed_output/request_active_seconds if strict_request_active_available and request_active_seconds else None,
                "output_token_reconciliation_status":"matched" if inf and all(x.output_token_reconciliation_status=="matched" for x in inf) else "mismatched" if any(x.output_token_reconciliation_status=="mismatched" for x in inf) else "unavailable",
                "reasoning_token_reconciliation_status":"matched" if inf and all(x.reasoning_token_reconciliation_status=="matched" for x in inf) else "mismatched" if any(x.reasoning_token_reconciliation_status=="mismatched" for x in inf) else "unavailable",
                "input_tokens":input_tokens,"cached_input_tokens":cached_input_tokens,
                "reasoning_tokens":sum(reasoning_values) if all(x is not None for x in reasoning_values) else None,
                "comparable_tokenizer":"o200k_base",
                "non_reasoning_output_tokens":sum(non_reasoning_values) if all(x is not None for x in non_reasoning_values) else None,
                "overall_tokens_per_second":reliable_tokens/observable_seconds if observable_seconds else None,
                "overall_tokens_per_second_reliable_calls":len(reliable_calls),
                "overall_tokens_per_second_total_calls":len(inf),
                "overall_end_to_end_tokens_per_second":comparable_output/inference if inference else None,
                "inference_to_tool_ratio":inference/tool if tool else None}
        return {"schema_version":"1.3","started_at":started_at,"task":{"task_id":task.task_id,"sector":task.sector,"occupation":task.occupation},
                "provider":{"name":name,"model":pconfig["model"],"endpoint":pconfig["endpoint"],"region":self.run_cfg.get("region"),"reasoning_effort":pconfig.get("reasoning_effort"),"adaptive_thinking":pconfig.get("adaptive_thinking"),"warmup":warmup},
                "environment":{"sandbox_backend":self.config["sandbox"]["backend"],"sandbox_image":self.config["sandbox"].get("image"),"host_platform":platform.platform()},
                "repetition":repetition,"outcome":outcome,"error":error,"turn_count":len(inf),"retry_count":retries,"timing":timing,"deliverables":files}
