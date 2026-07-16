from __future__ import annotations

import base64
from time import perf_counter
from urllib.parse import quote
import json

from gdpval_timing.models import Message, ToolCall
from gdpval_timing.providers.base import Provider, RetryableProviderError, sse_json


class GoogleProvider(Provider):
    name = "google"

    def _contents(self, messages):
        out=[]
        for m in messages:
            role="model" if m.role == "assistant" else "user"; parts=[]
            if m.text: parts.append({"text":m.text})
            if m.role == "assistant":
                for c in m.tool_calls:
                    p={"functionCall":{"name":c.name,"args":c.arguments}}
                    if sig:=c.provider_metadata.get("thought_signature"): p["thoughtSignature"]=sig
                    parts.append(p)
            elif m.role == "tool":
                for r in m.tool_results:
                    parts.append({"functionResponse":{"name":r.name,"response":{"output":r.output}}})
                    if r.image_data_url:
                        header,data=r.image_data_url.split(",",1); parts.append({"inlineData":{"mimeType":header.split(":",1)[1].split(";",1)[0],"data":data}})
            out.append({"role":role,"parts":parts})
        return out

    async def _stream_once(self, messages, tools):
        body={"contents":self._contents(messages), "tools":[{"functionDeclarations":[{
            "name":t["name"],"description":t["description"],"parameters":t["parameters"]} for t in tools]}],
            "generationConfig":{"maxOutputTokens":self.config.get("max_output_tokens",32000)}}
        if self.config.get("temperature") is not None: body["generationConfig"]["temperature"]=self.config["temperature"]
        url=f'{self.config["endpoint"]}/models/{quote(self.model, safe="")}:streamGenerateContent?alt=sse&key={quote(self.api_key, safe="")}'
        request_dispatch=perf_counter()
        async with self.client.stream("POST",url,json=body) as response:
            if response.status_code in {408,409,429,500,502,503,504}:
                await response.aread(); raise RetryableProviderError(f"Google HTTP {response.status_code}")
            response.raise_for_status(); text=[]; calls=[]; usage={}
            stream={"request_dispatch":request_dispatch,"first_event":None,"first_observable":None,"last_observable":None,
                    "observable_chunks":0,"observable_characters":0,"observable_text":"",
                    "generation_start":None,"generation_start_event_type":None,
                    "generation_start_event_detail":None,"generation_start_confidence":"unavailable",
                    "hidden_reasoning_observability":"unavailable","terminal_event":None,
                    "inference_outcome":"incomplete","stop_reason":None}
            async for e in sse_json(response, self.inference_idle_timeout_seconds):
                now=perf_counter()
                if stream["first_event"] is None: stream["first_event"] = now
                for cand in e.get("candidates",[]):
                    for p in cand.get("content",{}).get("parts",[]):
                        observable=(p.get("text") if "text" in p else
                                    json.dumps(p.get("functionCall"), sort_keys=True, separators=(",", ":"), ensure_ascii=False)
                                    if "functionCall" in p else None)
                        if observable:
                            if stream["generation_start"] is None:
                                stream.update(generation_start=now,generation_start_event_type="candidate.part",
                                              generation_start_event_detail="first non-empty text or functionCall",
                                              generation_start_confidence="delta_fallback",
                                              hidden_reasoning_observability="none")
                            if stream["first_observable"] is None: stream["first_observable"]=now
                            stream["last_observable"]=now; stream["observable_chunks"]+=1
                            stream["observable_characters"]+=len(observable)
                            stream["observable_text"]+=observable
                        if "text" in p: text.append(p["text"])
                        if "functionCall" in p:
                            f=p["functionCall"]; meta={}
                            if "thoughtSignature" in p: meta["thought_signature"]=p["thoughtSignature"]
                            calls.append(ToolCall(f.get("id",f["name"]+str(len(calls))),f["name"],f.get("args",{}),meta))
                if u:=e.get("usageMetadata"):
                    usage={"input_tokens":u.get("promptTokenCount"),"output_tokens":u.get("candidatesTokenCount"),
                           "authoritative_output_tokens":u.get("candidatesTokenCount"),
                           "reasoning_tokens":u.get("thoughtsTokenCount"),
                           # Gemini reports candidate and thought tokens as
                           # separate, non-overlapping usage counters.
                           "comparable_output_tokens":u.get("candidatesTokenCount"),
                           "token_count_comparability":"provider_candidate_tokens_exclude_thoughts"}
                for cand in e.get("candidates",[]):
                    if cand.get("finishReason"):
                        stream["stop_reason"]=cand["finishReason"]
            stream["terminal_event"]=perf_counter()
            stop=stream.get("stop_reason")
            stream["inference_outcome"]="completed" if stop not in {"MAX_TOKENS","SAFETY","RECITATION","BLOCKLIST","PROHIBITED_CONTENT","SPII"} else str(stop).lower()
            return Message(role="assistant",text="".join(text),tool_calls=calls),stream,usage,response.headers.get("x-request-id")
