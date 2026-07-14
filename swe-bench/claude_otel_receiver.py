#!/usr/bin/env python3
"""Minimal loopback-only OTLP/HTTP JSON receiver for Claude Code traces."""

from __future__ import annotations

import gzip
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any


class _TraceServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, receiver: "ClaudeOtelTraceReceiver") -> None:
        super().__init__(("127.0.0.1", 0), _TraceHandler)
        self.receiver = receiver


class _TraceHandler(BaseHTTPRequestHandler):
    server: _TraceServer

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        if self.path.rstrip("/") not in {"/v1/traces", "/traces"}:
            self.send_error(404)
            return
        try:
            content_length = int(self.headers.get("Content-Length", "0"))
            if content_length <= 0 or content_length > 100 * 1024 * 1024:
                raise ValueError("invalid OTLP payload length")
            body = self.rfile.read(content_length)
            if self.headers.get("Content-Encoding", "").lower() == "gzip":
                body = gzip.decompress(body)
            payload = json.loads(body)
            if not isinstance(payload, dict):
                raise ValueError("OTLP payload must be an object")
            self.server.receiver._record(payload)
        except (OSError, ValueError, json.JSONDecodeError) as error:
            self.server.receiver._record_error(f"{type(error).__name__}: {error}")
            self.send_error(400)
            return

        response = b"{}"
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(response)))
        self.end_headers()
        self.wfile.write(response)

    def log_message(self, _format: str, *_args: Any) -> None:
        return


class ClaudeOtelTraceReceiver:
    """Collect Claude Code trace payloads without an external OTel backend."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._payloads: list[dict[str, Any]] = []
        self._errors: list[str] = []
        self._last_received_monotonic: float | None = None
        self._server: _TraceServer | None = None
        self._thread: threading.Thread | None = None

    @property
    def endpoint(self) -> str:
        if self._server is None:
            raise RuntimeError("OTLP receiver is not running")
        host, port = self._server.server_address
        return f"http://{host}:{port}/v1/traces"

    def start(self) -> None:
        if self._server is not None:
            raise RuntimeError("OTLP receiver is already running")
        self._server = _TraceServer(self)
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            name="claude-otel-receiver",
            daemon=True,
        )
        self._thread.start()

    def telemetry_environment(self, base: dict[str, str]) -> dict[str, str]:
        """Return a Claude environment scoped to safe local trace export."""
        environment = base.copy()
        for sensitive_name in (
            "OTEL_LOG_USER_PROMPTS",
            "OTEL_LOG_TOOL_DETAILS",
            "OTEL_LOG_TOOL_CONTENT",
            "OTEL_LOG_RAW_API_BODIES",
            "ENABLE_BETA_TRACING_DETAILED",
            "BETA_TRACING_ENDPOINT",
        ):
            environment.pop(sensitive_name, None)
        environment.update(
            {
                "CLAUDE_CODE_ENABLE_TELEMETRY": "1",
                "CLAUDE_CODE_ENHANCED_TELEMETRY_BETA": "1",
                "OTEL_TRACES_EXPORTER": "otlp",
                "OTEL_METRICS_EXPORTER": "none",
                "OTEL_LOGS_EXPORTER": "none",
                "OTEL_EXPORTER_OTLP_TRACES_PROTOCOL": "http/json",
                "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT": self.endpoint,
                "OTEL_TRACES_EXPORT_INTERVAL": "250",
                "OTEL_BSP_SCHEDULE_DELAY": "250",
                "OTEL_BSP_EXPORT_TIMEOUT": "3000",
                "OTEL_METRICS_INCLUDE_ACCOUNT_UUID": "false",
                "OTEL_METRICS_INCLUDE_SESSION_ID": "false",
            }
        )
        return environment

    def wait_for_quiet(self, quiet_seconds: float = 0.5, timeout: float = 3.0) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self._lock:
                last_received = self._last_received_monotonic
            if last_received is not None and time.monotonic() - last_received >= quiet_seconds:
                return
            time.sleep(0.05)

    def stop(self) -> None:
        server = self._server
        if server is None:
            return
        server.shutdown()
        server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=2)
        self._server = None
        self._thread = None

    def payloads(self) -> list[dict[str, Any]]:
        with self._lock:
            return list(self._payloads)

    def diagnostics(self) -> dict[str, Any]:
        with self._lock:
            return {
                "payload_count": len(self._payloads),
                "receiver_error_count": len(self._errors),
                "receiver_errors": list(self._errors),
                "transport": "loopback_otlp_http_json",
                "raw_payloads_persisted": False,
            }

    def _record(self, payload: dict[str, Any]) -> None:
        with self._lock:
            self._payloads.append(payload)
            self._last_received_monotonic = time.monotonic()

    def _record_error(self, error: str) -> None:
        with self._lock:
            self._errors.append(error)
            self._last_received_monotonic = time.monotonic()
