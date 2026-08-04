"""ReAct Agent engine: multi-step reasoning with streaming tool calls.

Key improvements over v1:
- STREAMING: Uses stream_chat() so users see tokens as they generate
- PROGRESS: Emits events immediately when tools start/finish
- RECOVERY: Handles malformed tool calls gracefully
- BUDGET-AWARE: Truncates tool results based on remaining context
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, AsyncIterator

from ..backend import LLMBackend, ChatMessage, ToolDef
from ..tool_parser import parse_tool_calls, ToolCall
from ..tools.registry import ToolRegistry
from .context import ContextBuilder, BUDGET_16K, BUDGET_32K, estimate_tokens, budget_for_context_len

logger = logging.getLogger(__name__)


class EventType(str, Enum):
    THINKING = "thinking"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    TEXT = "text"
    TEXT_DELTA = "text_delta"
    PROGRESS = "progress"
    DONE = "done"
    ERROR = "error"


@dataclass
class AgentEvent:
    type: EventType
    content: str = ""
    data: dict[str, Any] = field(default_factory=dict)

    def to_sse(self) -> str:
        payload = {"type": self.type.value, "content": self.content}
        payload.update(self.data)
        return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


@dataclass
class AgentConfig:
    max_iterations: int = 8
    temperature: float = 0.3
    top_p: float = 0.9
    max_tokens: int = 4096
    context_budget: dict[str, int] = field(default_factory=lambda: BUDGET_16K)


class AgentEngine:
    """ReAct engine with streaming responses and progress events."""

    def __init__(
        self,
        backend: LLMBackend,
        registry: ToolRegistry,
        repo_path: str,
        config: AgentConfig | None = None,
    ) -> None:
        self.backend = backend
        self.registry = registry
        self.repo_path = repo_path
        self.config = config or AgentConfig()
        self.context_builder = ContextBuilder(self.config.context_budget)

    async def run(
        self,
        query: str,
        history: list[ChatMessage] | None = None,
    ) -> AsyncIterator[AgentEvent]:
        """Run the ReAct loop with streaming."""
        history = history or []
        tools = self.registry.get_definitions()
        messages = self.context_builder.build(self.repo_path, history, query)
        tool_names = self.registry.list_tools()
        tool_result_budget = self.config.context_budget.get("tool_results", 9000)

        for iteration in range(1, self.config.max_iterations + 1):
            logger.info("Agent iteration %d/%d", iteration, self.config.max_iterations)

            # ── Stream the LLM response ──
            full_text = ""
            is_last_iter = iteration >= self.config.max_iterations

            try:
                async for delta in self.backend.stream_chat(
                    messages=messages,
                    tools=tools if not is_last_iter else None,
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens,
                ):
                    chunk = delta.get("content", "")
                    if chunk:
                        full_text += chunk
                        # Stream to UI as text_delta so it appears live
                        yield AgentEvent(
                            type=EventType.TEXT_DELTA,
                            content=chunk,
                            data={"iteration": iteration},
                        )
                    finish = delta.get("finish_reason")
                    if finish:
                        break

            except Exception as e:
                logger.error("LLM stream failed: %s", e)
                yield AgentEvent(type=EventType.ERROR, content=f"LLM error: {e}")
                return

            # ── Parse for tool calls ──
            result = parse_tool_calls(
                message={"content": full_text, "tool_calls": []},
                full_text=full_text,
                prefer_structured=False,
            )

            has_valid_calls = result.calls and any(
                tc.name in tool_names for tc in result.calls
            )

            if not has_valid_calls:
                # No valid tool calls — this is the final answer
                final_text = full_text.strip()
                yield AgentEvent(type=EventType.TEXT, content=final_text)
                yield AgentEvent(type=EventType.DONE, content=final_text)
                return

            # ── Valid tool calls found ──
            # The streamed text was reasoning, not final answer
            yield AgentEvent(
                type=EventType.THINKING,
                content=full_text,
                data={"iteration": iteration},
            )

            # Add assistant message to conversation.
            # MUST include structured tool_calls so vLLM accepts the subsequent
            # role="tool" messages. We construct them from the text-parsed calls.
            structured_tool_calls = []
            for tc in result.calls:
                if tc.name not in tool_names:
                    continue
                call_id = _make_tool_call_id(tc.name, iteration)
                structured_tool_calls.append({
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": json.dumps(tc.arguments, ensure_ascii=False),
                    },
                })

            assistant_msg = ChatMessage(
                role="assistant",
                content=full_text,
                tool_calls=structured_tool_calls,
            )
            messages.append(assistant_msg)

            # ── Execute each tool call ──
            for tc in result.calls:
                if tc.name not in tool_names:
                    logger.warning("Unknown tool: %s", tc.name)
                    continue

                # Emit progress immediately (no dead air)
                yield AgentEvent(
                    type=EventType.PROGRESS,
                    content=_tool_progress(tc.name, tc.arguments),
                    data={"tool": tc.name, "iteration": iteration},
                )

                yield AgentEvent(
                    type=EventType.TOOL_CALL,
                    content=tc.name,
                    data={"arguments": tc.arguments, "iteration": iteration},
                )

                # Execute
                tool_result = await self.registry.execute(tc.name, tc.arguments)

                # Truncate to fit remaining budget
                remaining_budget = tool_result_budget - estimate_tokens(tool_result)
                if remaining_budget < 0:
                    # Need to truncate this result
                    max_chars = tool_result_budget * 4
                    if len(tool_result) > max_chars:
                        tool_result = tool_result[:max_chars] + "\n\n... [truncated to fit context budget]"

                # Emit result (UI display, truncated)
                display = tool_result
                if len(display) > 1500:
                    display = display[:1500] + "\n... [truncated]"
                yield AgentEvent(
                    type=EventType.TOOL_RESULT,
                    content=display,
                    data={"tool": tc.name, "iteration": iteration},
                )

                messages.append(ChatMessage(
                    role="tool",
                    content=tool_result,
                    tool_call_id=_make_tool_call_id(tc.name, iteration),
                    name=tc.name,
                ))
                # Reset accumulated tool call IDs for this iteration

        # Max iterations reached
        logger.warning("Agent reached max iterations (%d)", self.config.max_iterations)
        yield AgentEvent(
            type=EventType.TEXT,
            content=(
                "Based on my exploration, I've gathered the relevant code context. "
                "Let me summarize what I found."
            ),
        )
        yield AgentEvent(type=EventType.DONE)


def _tool_progress(name: str, args: dict) -> str:
    """Human-readable progress message for tool execution."""
    if name == "search_code":
        return f"Searching codebase for: {args.get('query', '?')}"
    elif name == "grep_code":
        return f"Searching for pattern: {args.get('pattern', '?')}"
    elif name == "read_file":
        path = args.get("path", "?")
        lines = ""
        if args.get("start_line", 0) > 1:
            lines = f" (line {args['start_line']})"
        return f"Reading {path}{lines}"
    elif name == "get_symbols":
        return f"Extracting symbols from {args.get('path', '?')}"
    elif name == "find_references":
        return f"Finding references to: {args.get('name', '?')}"
    elif name == "list_directory":
        return f"Listing {args.get('path', '.')}"
    elif name == "run_tests":
        return "Running tests..."
    return f"Executing {name}..."


def _make_tool_call_id(tool_name: str, iteration: int) -> str:
    return f"call_{tool_name}_{iteration}"
