import asyncio
from pathlib import Path
from gdpval_timing.models import Message, ToolCall
from gdpval_timing.providers.base import Provider, RetryableProviderError
from gdpval_timing.providers.anthropic import update_anthropic_usage
from gdpval_timing.summary import build_summary, percentile

class FakeProvider(Provider):
    name="fake"
    async def _stream_once(self,messages,tools):
        from time import perf_counter
        await asyncio.sleep(.01); first=perf_counter(); await asyncio.sleep(.01); last=perf_counter()
        stream={"first_event":first-.002,"first_observable":first,"last_observable":last,
                "observable_chunks":2,"observable_text":"hello world"}
        return Message(role="assistant",tool_calls=[ToolCall("1","finish",{"summary":"ok","deliverables":[]})]),stream,{"input_tokens":4,"output_tokens":9,"reasoning_tokens":7},"req"

def test_provider_timing_is_monotonic():
    async def scenario():
        p=FakeProvider({"model":"fake","max_retries":0},"secret")
        try: return await p.generate([Message(role="user",text="x")],[],1)
        finally: await p.close()
    r=asyncio.run(scenario())
    assert r.timing.latency_seconds >= r.timing.ttft_seconds >= 0
    assert r.timing.generation_seconds > 0
    assert r.timing.tokens_per_second > 0
    assert r.timing.comparable_output_tokens == 2
    assert r.timing.billed_output_tokens == 9
    assert r.timing.non_reasoning_output_tokens == 2
    assert r.timing.post_ttft_tokens_per_second_reliable is True
    assert r.timing.first_stream_event_seconds < r.timing.first_observable_output_seconds

class BatchedProvider(Provider):
    name="batched"
    async def _stream_once(self,messages,tools):
        from time import perf_counter
        now=perf_counter()
        return Message(role="assistant",text="hello"), {"first_event":now,"first_observable":now,
            "last_observable":now,"observable_chunks":1,"observable_text":"hello"}, {"output_tokens":1,"no_hidden_reasoning":True}, "req"

def test_batched_stream_does_not_claim_post_ttft_tps():
    async def scenario():
        p=BatchedProvider({"model":"fake","max_retries":0},"secret")
        try: return await p.generate([Message(role="user",text="x")],[],1)
        finally: await p.close()
    timing=asyncio.run(scenario()).timing
    assert timing.post_ttft_tokens_per_second is None
    assert timing.post_ttft_tokens_per_second_reliable is False
    assert "fewer than two" in timing.post_ttft_reliability_reason
    assert timing.end_to_end_tokens_per_second is not None

def test_anthropic_thinking_tokens_are_kept_separate():
    usage={}
    update_anthropic_usage(usage,{"input_tokens":10})
    update_anthropic_usage(usage,{"output_tokens":20,"output_tokens_details":{"thinking_tokens":7}})
    assert usage == {"input_tokens":10,"output_tokens":20,"reasoning_tokens":7}

def test_reasoning_observability_enumeration_is_spec_compliant():
    allowed={"full","phase_boundary_only","none","not_applicable","unavailable"}
    assert "not_observed" not in allowed

class BoundaryProvider(Provider):
    name="boundary"
    async def _stream_once(self,messages,tools):
        from time import perf_counter
        start=perf_counter()
        await asyncio.sleep(.01); generation_start=perf_counter()
        await asyncio.sleep(.02); terminal=perf_counter()
        stream={"first_event":generation_start,"first_observable":terminal-.005,
                "last_observable":terminal-.001,"observable_chunks":2,"observable_text":"visible",
                "generation_start":generation_start,"generation_start_event_type":"reasoning.start",
                "generation_start_event_detail":"item.type=reasoning","generation_start_confidence":"provider_boundary",
                "hidden_reasoning_observability":"phase_boundary_only","terminal_event":terminal,
                "inference_outcome":"completed","stop_reason":"completed"}
        return Message(role="assistant",text="visible"),stream,{"input_tokens":10,"cached_input_tokens":3,
            "output_tokens":30,"authoritative_output_tokens":30,
            "reasoning_tokens":20,"authoritative_reasoning_tokens":20},"req-boundary"

def test_active_generation_uses_provider_boundary_to_terminal():
    async def scenario():
        p=BoundaryProvider({"model":"fake","max_retries":0},"secret")
        try:return await p.generate([Message(role="user",text="x")],[],1)
        finally:await p.close()
    timing=asyncio.run(scenario()).timing
    assert timing.outcome=="completed"
    assert timing.generation_start_event_type=="reasoning.start"
    assert timing.generation_start_confidence=="provider_boundary"
    assert timing.hidden_reasoning_observability=="phase_boundary_only"
    assert timing.cached_input_tokens==3
    assert timing.visible_output_tokens==10
    assert timing.active_generation_seconds is not None
    assert timing.active_generation_billed_tps == timing.billed_output_tokens / timing.active_generation_seconds
    assert abs(timing.latency_seconds-(timing.observed_pre_generation_seconds+timing.active_generation_seconds)) < 1e-9
    assert timing.end_to_end_billed_tps == timing.billed_output_tokens / timing.latency_seconds
    assert timing.request_active_eligible is True
    assert timing.request_active_billed_tps == timing.billed_output_tokens / timing.request_active_seconds
    assert timing.output_token_reconciliation_status == "matched"
    assert timing.reasoning_token_reconciliation_status == "matched"

class IncompleteBoundaryProvider(BoundaryProvider):
    async def _stream_once(self,messages,tools):
        message,stream,usage,request_id=await super()._stream_once(messages,tools)
        stream["inference_outcome"]="max_output_tokens"; stream["stop_reason"]="max_output_tokens"
        return message,stream,usage,request_id

def test_incomplete_call_is_ineligible_for_active_tps():
    async def scenario():
        p=IncompleteBoundaryProvider({"model":"fake","max_retries":0},"secret")
        try:return await p.generate([Message(role="user",text="x")],[],1)
        finally:await p.close()
    timing=asyncio.run(scenario()).timing
    assert timing.outcome=="max_output_tokens"
    assert timing.active_generation_seconds is None
    assert timing.active_generation_billed_tps is None
    assert timing.end_to_end_billed_tps is None
    assert timing.request_active_eligible is False
    assert timing.request_active_exclusion_reason == "outcome:max_output_tokens"

class MismatchedUsageProvider(BoundaryProvider):
    async def _stream_once(self,messages,tools):
        message,stream,usage,request_id=await super()._stream_once(messages,tools)
        usage["authoritative_output_tokens"]=31
        return message,stream,usage,request_id

def test_authoritative_usage_mismatch_disqualifies_strict_tps():
    async def scenario():
        p=MismatchedUsageProvider({"model":"fake","max_retries":0},"secret")
        try:return await p.generate([Message(role="user",text="x")],[],1)
        finally:await p.close()
    timing=asyncio.run(scenario()).timing
    assert timing.output_token_reconciliation_status == "mismatched"
    assert timing.request_active_eligible is False
    assert timing.request_active_exclusion_reason == "output_token_reconciliation_mismatch"
    assert timing.request_active_billed_tps is None
    assert timing.active_generation_billed_tps is None

class RetryOnceProvider(BoundaryProvider):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs); self.calls=0
    async def _stream_once(self,messages,tools):
        self.calls+=1
        if self.calls==1: raise RetryableProviderError("retry me")
        return await super()._stream_once(messages,tools)

def test_success_after_retry_is_excluded_from_strict_tps(monkeypatch):
    async def no_sleep(_): return None
    monkeypatch.setattr("gdpval_timing.providers.base.asyncio.sleep",no_sleep)
    async def scenario():
        p=RetryOnceProvider({"model":"fake","max_retries":1},"secret")
        try:return await p.generate([Message(role="user",text="x")],[],1)
        finally:await p.close()
    response=asyncio.run(scenario())
    assert response.retry_count == 1
    assert response.timing.attempt == 2
    assert response.timing.request_active_eligible is False
    assert response.timing.request_active_exclusion_reason == "retried_call"
    assert response.timing.active_generation_billed_tps is None

def test_percentiles_and_grouping():
    assert percentile([1,2,10],.5)==2
    record={"provider":{"name":"p","model":"m"},"outcome":"completed","timing":{"total_wall_seconds":2,"inference_seconds":1,"tool_seconds":.5,"harness_overhead_seconds":.5,"overall_tokens_per_second":10}}
    assert build_summary([record])["providers"]["p/m"]["completed"]==1

def test_summary_uses_ratio_of_sums_and_reports_coverage():
    base={"provider":{"name":"p","model":"m"},"outcome":"completed"}
    records=[
        {**base,"timing":{"request_active_billed_tokens":100,"request_active_seconds":2,
         "request_active_eligible_calls":1,"request_active_total_calls":1,"billed_output_tokens":100,
         "active_generation_billed_tokens":100,"active_generation_seconds":1,
         "active_generation_eligible_calls":1,"active_generation_total_calls":1,
         "inference_calls":[{"ttft_seconds":.5}]}},
        {**base,"timing":{"request_active_billed_tokens":100,"request_active_seconds":8,
         "request_active_eligible_calls":1,"request_active_total_calls":2,"billed_output_tokens":150,
         "request_active_exclusion_reasons":{"retried_call":1},
         "active_generation_billed_tokens":100,"active_generation_seconds":4,
         "active_generation_eligible_calls":1,"active_generation_total_calls":2,
         "inference_calls":[{"ttft_seconds":1.5},{"ttft_seconds":None}]}}]
    group=build_summary(records)["providers"]["p/m"]
    request=group["request_active_ratio_of_sums"]
    assert request["billed_tps"] == 20
    assert request["strict_billed_tps"] is None
    assert request["eligible_calls"] == 2 and request["total_calls"] == 3
    assert request["eligible_billed_token_coverage"] == .8
    assert request["exclusion_reasons"] == {"retried_call":1}
    assert group["active_generation_ratio_of_sums"]["billed_tps"] == 40
    assert group["ttft_seconds"] == {"median":1.0,"p90":1.4,"observed_calls":2}
