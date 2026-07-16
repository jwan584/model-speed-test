from __future__ import annotations

import json
from time import perf_counter

from gdpval_timing.models import Message, ToolCall
from gdpval_timing.providers.base import Provider, RetryableProviderError, sse_json


class OpenAIProvider(Provider):
    name = "openai"

    def _input(self, messages):
        out = []
        for m in messages:
            if m.role in {"user", "assistant"}:
                if m.role == "assistant" and m.provider_metadata.get("output_items"):
                    out += m.provider_metadata["output_items"]
                    continue
                if m.text: out.append({"role": m.role, "content": m.text})
                if m.role == "assistant":
                    out += [{"type": "function_call", "call_id": c.id, "name": c.name,
                             "arguments": json.dumps(c.arguments)} for c in m.tool_calls]
            else:
                for r in m.tool_results:
                    output = r.output if not r.image_data_url else [{"type":"input_text","text":r.output},{"type":"input_image","image_url":r.image_data_url,"detail":"auto"}]
                    out.append({"type":"function_call_output","call_id":r.call_id,"output":output})
        return out

    async def _stream_once(self, messages, tools):
        body = {"model": self.model, "input": self._input(messages), "tools": [dict(type="function", **t) for t in tools],
                "stream": True, "max_output_tokens": self.config.get("max_output_tokens", 32000), "store": False,
                "include":["reasoning.encrypted_content"]}
        if self.config.get("reasoning_effort"):
            body["reasoning"]={"effort":self.config["reasoning_effort"]}
        headers = {"authorization": f"Bearer {self.api_key}", "content-type": "application/json"}
        request_dispatch = perf_counter()
        async with self.client.stream("POST", self.config["endpoint"], headers=headers, json=body) as response:
            if response.status_code in {408, 409, 429, 500, 502, 503, 504}:
                await response.aread(); raise RetryableProviderError(f"OpenAI HTTP {response.status_code}")
            response.raise_for_status()
            request_id = response.headers.get("x-request-id")
            text, calls, items, usage = [], [], [], {}
            stream = {"first_event": None, "first_observable": None, "last_observable": None,
                      "request_dispatch": request_dispatch,
                      "observable_chunks": 0, "observable_characters": 0, "observable_text": "",
                      "generation_start": None, "generation_start_event_type": None,
                      "generation_start_event_detail": None, "generation_start_confidence": "unavailable",
                      "hidden_reasoning_observability": "unavailable", "terminal_event": None,
                      "inference_outcome": "incomplete", "stop_reason": None, "saw_refusal": False}
            async for e in sse_json(response, self.inference_idle_timeout_seconds):
                now = perf_counter()
                if stream["first_event"] is None: stream["first_event"] = now
                typ = e.get("type") or e.get("_event")
                if typ == "response.output_item.added" and stream["generation_start"] is None and (e.get("item") or {}).get("type") == "reasoning":
                    stream.update(generation_start=now, generation_start_event_type=typ,
                                  generation_start_event_detail="item.type=reasoning",
                                  generation_start_confidence="provider_boundary",
                                  hidden_reasoning_observability="phase_boundary_only")
                if typ in {"response.output_text.delta", "response.function_call_arguments.delta"}:
                    delta = e.get("delta", "")
                    if delta:
                        if stream["generation_start"] is None:
                            stream.update(generation_start=now, generation_start_event_type=typ,
                                          generation_start_event_detail="first non-empty generated delta",
                                          generation_start_confidence="delta_fallback",
                                          hidden_reasoning_observability="none")
                        if stream["first_observable"] is None: stream["first_observable"] = now
                        stream["last_observable"] = now
                        stream["observable_chunks"] += 1
                        stream["observable_characters"] += len(delta)
                        stream["observable_text"] += delta
                if typ == "response.output_text.delta": text.append(e.get("delta", ""))
                elif typ == "response.output_item.done":
                    x=e.get("item",{}); items.append(x)
                    if x.get("type")=="refusal" or any(c.get("type")=="refusal" for c in x.get("content",[]) if isinstance(c,dict)): stream["saw_refusal"]=True
                    if x.get("type") == "function_call": calls.append(ToolCall(x["call_id"], x["name"], json.loads(x.get("arguments") or "{}")))
                elif typ in {"response.completed", "response.incomplete"}:
                    response_data=e["response"]; u=response_data.get("usage") or {}
                    usage={"input_tokens":u.get("input_tokens"),
                        "cached_input_tokens":(u.get("input_tokens_details") or {}).get("cached_tokens"),
                        "output_tokens":u.get("output_tokens"),
                        "authoritative_output_tokens":u.get("output_tokens"),
                        "reasoning_tokens":u.get("output_tokens_details",{}).get("reasoning_tokens"),
                        "authoritative_reasoning_tokens":u.get("output_tokens_details",{}).get("reasoning_tokens")}
                    stream["terminal_event"]=now
                    incomplete=(response_data.get("incomplete_details") or {}).get("reason")
                    stream["stop_reason"]="refusal" if stream["saw_refusal"] else incomplete or response_data.get("status")
                    stream["inference_outcome"]="completed" if typ=="response.completed" and not stream["saw_refusal"] else (stream["stop_reason"] or "incomplete")
            return Message(role="assistant", text="".join(text), tool_calls=calls, provider_metadata={"output_items":items}), stream, usage, request_id
