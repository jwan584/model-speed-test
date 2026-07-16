from __future__ import annotations

import json
from time import perf_counter
from typing import Any

from gdpval_timing.models import Message, ToolCall
from gdpval_timing.providers.base import Provider, RetryableProviderError, sse_json


def update_anthropic_usage(target: dict[str, Any], usage: dict[str, Any]) -> None:
    """Merge usage fragments emitted at message_start/message_delta."""
    for key in ("input_tokens", "output_tokens", "cache_read_input_tokens", "cache_creation_input_tokens"):
        if usage.get(key) is not None:
            target[key] = usage[key]
    details = usage.get("output_tokens_details") or {}
    if details.get("thinking_tokens") is not None:
        target["reasoning_tokens"] = details["thinking_tokens"]


class AnthropicProvider(Provider):
    name = "anthropic"

    def _messages(self, messages: list[Message]) -> list[dict[str, Any]]:
        out = []
        for m in messages:
            if m.role == "user":
                out.append({"role": "user", "content": m.text})
            elif m.role == "assistant":
                if m.provider_metadata.get("anthropic_content"):
                    out.append({"role": "assistant", "content": m.provider_metadata["anthropic_content"]})
                    continue
                content: list[dict[str, Any]] = []
                if m.text:
                    content.append({"type": "text", "text": m.text})
                content += [{"type": "tool_use", "id": c.id, "name": c.name, "input": c.arguments} for c in m.tool_calls]
                out.append({"role": "assistant", "content": content})
            else:
                blocks = []
                for r in m.tool_results:
                    content: Any = r.output
                    if r.image_data_url:
                        header, data = r.image_data_url.split(",", 1)
                        content = [{"type": "text", "text": r.output}, {"type": "image", "source": {
                            "type": "base64", "media_type": header.split(":", 1)[1].split(";", 1)[0], "data": data}}]
                    blocks.append({"type": "tool_result", "tool_use_id": r.call_id, "content": content, "is_error": r.is_error})
                out.append({"role": "user", "content": blocks})
        return out

    async def _stream_once(self, messages, tools):
        body = {"model": self.model, "max_tokens": self.config.get("max_output_tokens", 32000),
                "messages": self._messages(messages), "stream": True}
        if tools:
            body["tools"] = [{"name": t["name"], "description": t["description"], "input_schema": t["parameters"]} for t in tools]
        if self.config.get("temperature") is not None:
            body["temperature"] = self.config["temperature"]
        if self.config.get("reasoning_effort"):
            body["output_config"] = {"effort": self.config["reasoning_effort"]}
        if self.config.get("adaptive_thinking"):
            body["thinking"] = {"type": "adaptive", "display": "omitted"}
        headers = {"x-api-key": self.api_key, "anthropic-version": "2023-06-01", "content-type": "application/json"}
        request_dispatch = perf_counter()
        async with self.client.stream("POST", self.config["endpoint"], headers=headers, json=body) as response:
            if response.status_code in {408, 409, 429, 500, 502, 503, 504, 529}:
                await response.aread(); raise RetryableProviderError(f"Anthropic HTTP {response.status_code}")
            if response.is_error:
                detail=(await response.aread()).decode(errors="replace")[:2000]
                raise RuntimeError(f"Anthropic HTTP {response.status_code}: {detail}")
            request_id = response.headers.get("request-id")
            text, calls, blocks, usage = [], [], {}, {}
            stream = {"first_event": None, "first_observable": None, "last_observable": None,
                      "request_dispatch": request_dispatch,
                      "observable_chunks": 0, "observable_characters": 0, "observable_text": "",
                      "generation_start": None, "generation_start_event_type": None,
                      "generation_start_event_detail": None, "generation_start_confidence": "unavailable",
                      "hidden_reasoning_observability": "unavailable", "terminal_event": None,
                      "inference_outcome": "incomplete", "stop_reason": None, "saw_refusal": False}
            async for event in sse_json(response, self.inference_idle_timeout_seconds):
                now = perf_counter()
                if stream["first_event"] is None: stream["first_event"] = now
                typ = event.get("type") or event.get("_event")
                if typ == "message_start":
                    update_anthropic_usage(usage, event["message"].get("usage", {}))
                elif typ == "content_block_start":
                    b = event["content_block"]; blocks[event["index"]] = {"block": b, "json": "", "text": b.get("text", ""), "thinking": b.get("thinking", ""), "signature": b.get("signature", "")}
                    if b.get("type") == "refusal": stream["saw_refusal"] = True
                    if stream["generation_start"] is None and b.get("type") in {"thinking", "redacted_thinking", "text", "tool_use", "refusal", "fallback"}:
                        block_type=b.get("type")
                        stream.update(generation_start=now, generation_start_event_type=typ,
                                      generation_start_event_detail=f"index={event['index']};content_block.type={block_type}",
                                      generation_start_confidence="provider_boundary",
                                      hidden_reasoning_observability="phase_boundary_only" if block_type in {"thinking", "redacted_thinking"} else "none")
                elif typ == "content_block_delta":
                    d = event["delta"]
                    if stream["generation_start"] is None and d.get("type") in {"thinking_delta", "text_delta", "input_json_delta"}:
                        delta_value=d.get("thinking") or d.get("text") or d.get("partial_json")
                        if delta_value:
                            stream.update(generation_start=now, generation_start_event_type=typ,
                                          generation_start_event_detail=f"index={event['index']};delta.type={d['type']}",
                                          generation_start_confidence="delta_fallback",
                                          hidden_reasoning_observability="full" if d["type"]=="thinking_delta" else "none")
                    observable = d.get("text") if d["type"] == "text_delta" else d.get("partial_json") if d["type"] == "input_json_delta" else None
                    if observable:
                        if stream["first_observable"] is None: stream["first_observable"] = now
                        stream["last_observable"] = now
                        stream["observable_chunks"] += 1
                        stream["observable_characters"] += len(observable)
                        stream["observable_text"] += observable
                    if d["type"] == "text_delta": text.append(d["text"]); blocks[event["index"]]["text"] += d["text"]
                    elif d["type"] == "input_json_delta": blocks[event["index"]]["json"] += d["partial_json"]
                    elif d["type"] == "thinking_delta": blocks[event["index"]]["thinking"] += d.get("thinking", "")
                    elif d["type"] == "signature_delta": blocks[event["index"]]["signature"] += d.get("signature", "")
                elif typ == "message_delta":
                    u = event.get("usage", {}); update_anthropic_usage(usage, u)
                    if u.get("output_tokens") is not None:
                        usage["authoritative_output_tokens"] = u["output_tokens"]
                    details = u.get("output_tokens_details") or {}
                    if details.get("thinking_tokens") is not None:
                        usage["authoritative_reasoning_tokens"] = details["thinking_tokens"]
                    stream["stop_reason"]=(event.get("delta") or {}).get("stop_reason") or stream["stop_reason"]
                    if not self.config.get("adaptive_thinking"):
                        usage["no_hidden_reasoning"] = True
                elif typ == "message_stop":
                    stream["terminal_event"] = now
                    stop=stream.get("stop_reason")
                    stream["inference_outcome"] = "completed" if stop not in {"max_tokens", "refusal"} and not stream["saw_refusal"] else (stop or "refusal")
                elif typ == "error":
                    raise RetryableProviderError(str(event.get("error")))
            content=[]
            for _, value in sorted(blocks.items()):
                b = value["block"]
                if b["type"] == "tool_use":
                    raw = value["json"]
                    args=json.loads(raw) if raw else b.get("input", {}); calls.append(ToolCall(b["id"], b["name"], args)); content.append({"type":"tool_use","id":b["id"],"name":b["name"],"input":args})
                elif b["type"] == "text": content.append({"type":"text","text":value["text"]})
                elif b["type"] == "thinking":
                    item={"type":"thinking","thinking":value["thinking"],"signature":value["signature"]}; content.append(item)
                elif b["type"] == "redacted_thinking": content.append(b)
            return Message(role="assistant", text="".join(text), tool_calls=calls, provider_metadata={"anthropic_content":content}), stream, usage, request_id
