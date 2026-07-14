#!/usr/bin/env python3
"""Minimal OpenAI Responses API smoke test."""

from __future__ import annotations

import argparse
import json
import os
import ssl
import sys
import urllib.error
import urllib.request

import certifi


DEFAULT_MODEL = "koffing-updated"
DEFAULT_BASE_URL = "https://api.openai.com/v1"


def extract_output_text(response: dict) -> str:
    if response.get("output_text"):
        return str(response["output_text"])

    parts: list[str] = []
    for item in response.get("output", []):
        for content in item.get("content", []):
            if content.get("type") in {"output_text", "text"} and content.get("text"):
                parts.append(str(content["text"]))
    return "\n".join(parts)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prompt", nargs="?", default="Reply with exactly: API test successful")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL", DEFAULT_BASE_URL))
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--json", action="store_true", help="Print the complete JSON response")
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: Set OPENAI_API_KEY in the environment.", file=sys.stderr)
        return 2

    payload = json.dumps({"model": args.model, "input": args.prompt}).encode("utf-8")
    url = f"{args.base_url.rstrip('/')}/responses"
    request = urllib.request.Request(
        url,
        data=payload,
        method="POST",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
    )

    try:
        tls_context = ssl.create_default_context(cafile=certifi.where())
        with urllib.request.urlopen(request, timeout=args.timeout, context=tls_context) as result:
            response = json.load(result)
    except urllib.error.HTTPError as error:
        body = error.read().decode("utf-8", errors="replace")
        print(f"ERROR: OpenAI returned HTTP {error.code}: {body}", file=sys.stderr)
        return 1
    except (urllib.error.URLError, TimeoutError) as error:
        print(f"ERROR: Request failed: {error}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(response, indent=2))
        return 0

    text = extract_output_text(response)
    if not text:
        print("ERROR: Response contained no output text. Re-run with --json.", file=sys.stderr)
        return 1

    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
