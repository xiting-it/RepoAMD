"""Tool call parsing for RepoAgent Agent engine.

Supports two modes:
1. Structured: vLLM returns ``tool_calls`` array in OpenAI-compatible JSON.
2. Text format: llama.cpp or vLLM fallback — extract ``<tool_call>...</tool_call>`` blocks.

Provides a unified ToolCall dataclass and ``parse_tool_calls()`` that handles both.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolCall:
    name: str
    arguments: dict[str, Any]
    raw: str = ""


@dataclass
class ToolCallResult:
    calls: list[ToolCall] = field(default_factory=list)
    text_before: str = ""
    text_after: str = ""
    mode: str = "none"  # "structured" | "text" | "none"


# ── Text-format parsing ──

# Qwen2.5 / Hermes text tool call pattern
_TOOL_CALL_RE = re.compile(
    r"<tool_call>\s*(.*?)\s*</tool_call>",
    re.DOTALL,
)

# Fallback for models that don't use XML tags (plain JSON)
_JSON_OBJECT_RE = re.compile(r'\{[^{}]*"name"\s*:\s*"[^"]+".*?\}', re.DOTALL)


def _extract_json_from_text(block: str) -> dict[str, Any] | None:
    """Try to parse a JSON object from a text block."""
    block = block.strip()
    # Remove markdown code fences if present
    if block.startswith("```"):
        lines = block.split("\n")
        block = "\n".join(lines[1:])
        if block.rstrip().endswith("```"):
            block = block.rstrip()[:-3]
    try:
        obj = json.loads(block.strip())
        if isinstance(obj, dict):
            return obj
    except (json.JSONDecodeError, ValueError):
        pass

    # Try to find a JSON object with "name" key
    for match in _JSON_OBJECT_RE.finditer(block):
        try:
            obj = json.loads(match.group())
            if isinstance(obj, dict) and "name" in obj:
                return obj
        except (json.JSONDecodeError, ValueError):
            continue
    return None


def parse_text_tool_calls(text: str) -> ToolCallResult:
    """Parse text-format tool calls from LLM output (llama.cpp / fallback)."""
    calls: list[ToolCall] = []
    text_before = text
    text_after = ""

    # Try <tool_call>...</tool_call> blocks first
    matches = list(_TOOL_CALL_RE.finditer(text))
    if matches:
        first_start = matches[0].start()
        text_before = text[:first_start].strip()
        text_after = text[matches[-1].end():].strip()
        for m in matches:
            obj = _extract_json_from_text(m.group(1))
            if obj and "name" in obj:
                calls.append(ToolCall(
                    name=obj["name"],
                    arguments=obj.get("arguments", obj.get("parameters", {})),
                    raw=m.group(0),
                ))
    else:
        # Fallback: bare JSON object with "name" key
        obj = _extract_json_from_text(text)
        if obj and "name" in obj:
            calls.append(ToolCall(
                name=obj["name"],
                arguments=obj.get("arguments", obj.get("parameters", {})),
                raw=text,
            ))
            # Split text around the JSON
            text_before = ""

    return ToolCallResult(
        calls=calls,
        text_before=text_before,
        text_after=text_after,
        mode="text" if calls else "none",
    )


def parse_structured_tool_calls(
    message: dict[str, Any],
    full_text: str = "",
) -> ToolCallResult:
    """Parse structured tool_calls from OpenAI-compatible API response.

    ``message`` is the assistant message dict, e.g.::

        {
            "role": "assistant",
            "content": "Let me search...",
            "tool_calls": [
                {
                    "id": "call_abc",
                    "function": {
                        "name": "search_code",
                        "arguments": "{\"query\": \"auth login\"}",
                    }
                }
            ]
        }
    """
    tool_calls_raw = message.get("tool_calls", [])
    calls: list[ToolCall] = []
    text = message.get("content") or full_text or ""

    for tc in tool_calls_raw:
        func = tc.get("function", {})
        name = func.get("name", "")
        args_raw = func.get("arguments", "{}")
        if isinstance(args_raw, str):
            try:
                arguments = json.loads(args_raw)
            except json.JSONDecodeError:
                arguments = {"_raw": args_raw}
        else:
            arguments = args_raw
        if name:
            calls.append(ToolCall(name=name, arguments=arguments, raw=json.dumps(tc)))

    return ToolCallResult(
        calls=calls,
        text_before=text,
        text_after="",
        mode="structured" if calls else "none",
    )


def parse_tool_calls(
    message: dict[str, Any],
    full_text: str = "",
    prefer_structured: bool = True,
) -> ToolCallResult:
    """Unified entry point: try structured first, fall back to text parsing."""
    if prefer_structured and message.get("tool_calls"):
        return parse_structured_tool_calls(message, full_text)

    text = message.get("content") or full_text or ""
    if text:
        return parse_text_tool_calls(text)

    return ToolCallResult(mode="none")
