from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]
    provider_metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolResult:
    call_id: str
    name: str
    output: str
    is_error: bool = False
    image_data_url: str | None = None


@dataclass
class Message:
    role: Literal["user", "assistant", "tool"]
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_results: list[ToolResult] = field(default_factory=list)
    provider_metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class InferenceTiming:
    call_index: int
    attempt: int
    started_at: str
    latency_seconds: float
    ttft_seconds: float | None
    generation_seconds: float | None
    input_tokens: int | None
    output_tokens: int | None
    reasoning_tokens: int | None = None
    throughput_tokens: int | None = None
    tokens_per_second: float | None = None
    request_id: str | None = None
    first_stream_event_seconds: float | None = None
    first_observable_output_seconds: float | None = None
    last_observable_output_seconds: float | None = None
    observable_output_chunks: int = 0
    observable_output_characters: int = 0
    observable_output_bytes: int = 0
    billed_output_tokens: int | None = None
    non_reasoning_output_tokens: int | None = None
    comparable_output_tokens: int | None = None
    comparable_tokenizer: str = "o200k_base"
    token_count_comparability: str = "fixed_local_tokenizer"
    post_ttft_tokens_per_second: float | None = None
    post_ttft_tokens_per_second_reliable: bool = False
    post_ttft_reliability_reason: str | None = None
    end_to_end_tokens_per_second: float | None = None
    observable_characters_per_second: float | None = None
    observable_bytes_per_second: float | None = None
    generation_start_seconds: float | None = None
    generation_start_event_type: str | None = None
    generation_start_event_detail: str | None = None
    generation_start_confidence: str = "unavailable"
    hidden_reasoning_observability: str = "unavailable"
    terminal_event_seconds: float | None = None
    observed_pre_generation_seconds: float | None = None
    active_generation_seconds: float | None = None
    active_generation_billed_tps: float | None = None
    end_to_end_billed_tps: float | None = None
    request_dispatch_seconds: float = 0.0
    cached_input_tokens: int | None = None
    visible_output_tokens: int | None = None
    outcome: str = "incomplete"
    stop_reason: str | None = None
    request_active_seconds: float | None = None
    request_active_billed_tps: float | None = None
    request_active_eligible: bool = False
    request_active_exclusion_reason: str | None = None
    authoritative_output_tokens: int | None = None
    output_token_reconciliation_status: str = "unavailable"
    authoritative_reasoning_tokens: int | None = None
    reasoning_token_reconciliation_status: str = "unavailable"


@dataclass
class ProviderResponse:
    message: Message
    timing: InferenceTiming
    backoff_seconds: float = 0.0
    retry_api_seconds: float = 0.0
    retry_count: int = 0


@dataclass
class ToolTiming:
    call_index: int
    turn: int
    tool_type: str
    call_id: str
    started_at: str
    duration_seconds: float
    success: bool


def jsonable(value: Any) -> Any:
    if hasattr(value, "__dataclass_fields__"):
        return {k: jsonable(v) for k, v in asdict(value).items()}
    if isinstance(value, list):
        return [jsonable(v) for v in value]
    if isinstance(value, dict):
        return {k: jsonable(v) for k, v in value.items()}
    return value
