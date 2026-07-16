from __future__ import annotations

import abc
import asyncio
import json
import random
from datetime import datetime, timezone
from time import perf_counter
from typing import Any, AsyncIterator

import httpx
import tiktoken

from gdpval_timing.models import InferenceTiming, Message, ProviderResponse

COMPARABLE_TOKENIZER = "o200k_base"
_COMPARABLE_ENCODING = tiktoken.get_encoding(COMPARABLE_TOKENIZER)


class RetryableProviderError(RuntimeError):
    pass

class InferenceTimeoutError(TimeoutError):
    """A provider stream went silent or exceeded its per-call ceiling."""


async def sse_json(response: httpx.Response, idle_timeout_seconds: float | None = None) -> AsyncIterator[dict[str, Any]]:
    event = ""
    data: list[str] = []
    lines = response.aiter_lines().__aiter__()
    while True:
        try:
            if idle_timeout_seconds:
                line = await asyncio.wait_for(lines.__anext__(), idle_timeout_seconds)
            else:
                line = await lines.__anext__()
        except StopAsyncIteration:
            break
        except asyncio.TimeoutError as exc:
            raise InferenceTimeoutError(
                f"Provider stream produced no event for {idle_timeout_seconds:g} seconds"
            ) from exc
        if not line:
            if data:
                raw = "\n".join(data)
                if raw != "[DONE]":
                    obj = json.loads(raw)
                    if event and "_event" not in obj:
                        obj["_event"] = event
                    yield obj
            event, data = "", []
        elif line.startswith("event:"):
            event = line[6:].strip()
        elif line.startswith("data:"):
            data.append(line[5:].strip())
    if data:
        raw = "\n".join(data)
        if raw != "[DONE]":
            yield json.loads(raw)


class Provider(abc.ABC):
    name: str

    def __init__(self, config: dict[str, Any], api_key: str):
        self.config = config
        self.api_key = api_key
        self.model = config["model"]
        self.max_retries = int(config.get("max_retries", 5))
        self.inference_idle_timeout_seconds = float(config.get("inference_idle_timeout_seconds", 300))
        self.inference_absolute_timeout_seconds = float(config.get("inference_absolute_timeout_seconds", 1800))
        self.client = httpx.AsyncClient(timeout=httpx.Timeout(None, connect=30))

    async def close(self) -> None:
        await self.client.aclose()

    async def warmup(self) -> dict[str, Any]:
        started = perf_counter()
        try:
            await self.generate([Message(role="user", text="Reply with OK.")], [], 0)
            return {"success": True, "duration_seconds": perf_counter() - started}
        except Exception as exc:
            return {"success": False, "duration_seconds": perf_counter() - started, "error": str(exc)}

    async def generate(self, messages: list[Message], tools: list[dict[str, Any]], call_index: int) -> ProviderResponse:
        backoff = retry_api = 0.0
        for attempt in range(1, self.max_retries + 2):
            started_at = datetime.now(timezone.utc).isoformat()
            started = perf_counter()
            try:
                try:
                    message, stream, usage, request_id = await asyncio.wait_for(
                        self._stream_once(messages, tools), self.inference_absolute_timeout_seconds
                    )
                except asyncio.TimeoutError as exc:
                    raise InferenceTimeoutError(
                        f"Inference call exceeded {self.inference_absolute_timeout_seconds:g} seconds"
                    ) from exc
                completed_at = perf_counter()
                dispatch = stream.get("request_dispatch", started)
                terminal = stream.get("terminal_event")
                latency = (terminal or completed_at) - dispatch
                first_event = stream.get("first_event")
                first = stream.get("first_observable")
                last = stream.get("last_observable")
                ttft = None if first is None else first - dispatch
                generation = None if first is None else latency - ttft
                output = usage.get("output_tokens")
                reasoning = usage.get("reasoning_tokens")
                authoritative_output = usage.get("authoritative_output_tokens")
                authoritative_reasoning = usage.get("authoritative_reasoning_tokens")
                output_reconciliation = ("matched" if authoritative_output is not None and output == authoritative_output
                                         else "mismatched" if authoritative_output is not None else "unavailable")
                reasoning_reconciliation = ("matched" if authoritative_reasoning is not None and reasoning == authoritative_reasoning
                                            else "mismatched" if authoritative_reasoning is not None else "unavailable")
                # A cross-provider numerator is only available when hidden
                # reasoning is absent or explicitly separable. Never silently
                # call a provider's combined billed count "visible" output.
                no_hidden_reasoning = usage.get("no_hidden_reasoning", False)
                non_reasoning = (usage["comparable_output_tokens"] if usage.get("comparable_output_tokens") is not None
                              else max(0, output - reasoning) if output is not None and reasoning is not None
                              else output if output is not None and no_hidden_reasoning else None)
                observable_text = stream.get("observable_text", "")
                comparable = len(_COMPARABLE_ENCODING.encode(observable_text, disallowed_special=()))
                chunks = int(stream.get("observable_chunks", 0))
                observable_span = None if first is None or last is None else last - first
                reliable = chunks >= 2 and observable_span is not None and observable_span >= 0.001
                reason = None
                if chunks < 2: reason = "fewer than two observable streamed chunks"
                elif observable_span is None or observable_span < 0.001: reason = "observable output was batched into an insufficient time span"
                post_tps = comparable / observable_span if reliable else None
                end_to_end_tps = comparable / latency if latency > 0 else None
                generation_start = stream.get("generation_start")
                inference_outcome=stream.get("inference_outcome", "incomplete")
                output_valid=isinstance(output,int) and not isinstance(output,bool) and output >= 0
                reasoning_valid=(reasoning is None or
                                 (isinstance(reasoning,int) and not isinstance(reasoning,bool) and 0 <= reasoning <= output)) if output_valid else False
                request_exclusion=None
                if inference_outcome != "completed": request_exclusion=f"outcome:{inference_outcome}"
                elif terminal is None: request_exclusion="missing_terminal_event"
                elif not output_valid: request_exclusion="missing_or_invalid_billed_output_tokens"
                elif not reasoning_valid: request_exclusion="invalid_reasoning_token_subset"
                elif authoritative_output is not None and output_reconciliation != "matched": request_exclusion="output_token_reconciliation_mismatch"
                elif attempt > 1: request_exclusion="retried_call"
                elif latency <= 0: request_exclusion="nonpositive_request_duration"
                request_eligible=request_exclusion is None
                eligible=(request_eligible and generation_start is not None)
                active_generation_seconds = terminal-generation_start if eligible and terminal > generation_start else None
                active_generation_billed_tps = output / active_generation_seconds if active_generation_seconds else None
                billed_end_to_end_tps = output / latency if request_eligible else None
                chars = len(observable_text)
                byte_count = len(observable_text.encode("utf-8"))
                chars_per_second = chars / observable_span if reliable else None
                bytes_per_second = byte_count / observable_span if reliable else None
                timing = InferenceTiming(call_index=call_index, attempt=attempt, started_at=started_at,
                    latency_seconds=latency, ttft_seconds=ttft, generation_seconds=generation,
                    input_tokens=usage.get("input_tokens"), output_tokens=output, reasoning_tokens=reasoning,
                    throughput_tokens=comparable, tokens_per_second=post_tps, request_id=request_id,
                    first_stream_event_seconds=None if first_event is None else first_event-dispatch,
                    first_observable_output_seconds=ttft, last_observable_output_seconds=None if last is None else last-dispatch,
                    observable_output_chunks=chunks, observable_output_characters=chars, observable_output_bytes=byte_count,
                    billed_output_tokens=output, non_reasoning_output_tokens=non_reasoning,
                    comparable_output_tokens=comparable, post_ttft_tokens_per_second=post_tps,
                    post_ttft_tokens_per_second_reliable=reliable, post_ttft_reliability_reason=reason,
                    end_to_end_tokens_per_second=end_to_end_tps, observable_characters_per_second=chars_per_second,
                    observable_bytes_per_second=bytes_per_second,
                    generation_start_seconds=None if generation_start is None else generation_start-dispatch,
                    generation_start_event_type=stream.get("generation_start_event_type"),
                    generation_start_event_detail=stream.get("generation_start_event_detail"),
                    generation_start_confidence=stream.get("generation_start_confidence", "unavailable"),
                    hidden_reasoning_observability=stream.get("hidden_reasoning_observability", "unavailable"),
                    terminal_event_seconds=None if terminal is None else terminal-dispatch,
                    observed_pre_generation_seconds=None if generation_start is None else generation_start-dispatch,
                    active_generation_seconds=active_generation_seconds,
                    active_generation_billed_tps=active_generation_billed_tps,
                    end_to_end_billed_tps=billed_end_to_end_tps,
                    cached_input_tokens=usage.get("cached_input_tokens", usage.get("cache_read_input_tokens")),
                    visible_output_tokens=non_reasoning, outcome=inference_outcome,
                    stop_reason=stream.get("stop_reason"), request_active_seconds=latency if request_eligible else None,
                    request_active_billed_tps=billed_end_to_end_tps,
                    request_active_eligible=request_eligible,
                    request_active_exclusion_reason=request_exclusion,
                    authoritative_output_tokens=authoritative_output,
                    output_token_reconciliation_status=output_reconciliation,
                    authoritative_reasoning_tokens=authoritative_reasoning,
                    reasoning_token_reconciliation_status=reasoning_reconciliation)
                return ProviderResponse(message, timing, backoff, retry_api, attempt - 1)
            except RetryableProviderError as exc:
                retry_api += perf_counter() - started
                if attempt > self.max_retries:
                    exc.backoff_seconds=backoff; exc.retry_api_seconds=retry_api; exc.retry_count=attempt-1
                    raise
                delay = min(60.0, 1.0 * 2 ** (attempt - 1)) * random.uniform(0.75, 1.25)
                wait_started = perf_counter()
                await asyncio.sleep(delay)
                backoff += perf_counter() - wait_started
        raise AssertionError("unreachable")

    @abc.abstractmethod
    async def _stream_once(self, messages: list[Message], tools: list[dict[str, Any]]) -> tuple[Message, dict[str, Any], dict[str, Any], str | None]: ...
