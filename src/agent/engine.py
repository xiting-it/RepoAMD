"""ReAct Agent engine: multi-step reasoning with tool calls.

Orchestrates the LLM + tool execution loop:
1. Build context (system prompt + repo tree + history)
2. Call LLM with tools
3. Parse tool calls (structured JSON or text format)
4. Execute tools
5. Feed results back, repeat (max 8 iterations)
6. Stream events to the UI
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
from .context import ContextBuilder, BUDGET_16K, BUDGET_32K, budget_for_context_len

logger = logging.getLogger(__name__)


class EventType(str, Enum):
    THINKING = "thinking"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    TEXT = "text"
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
    """ReAct engine that orchestrates LLM + tool execution.

    Usage::

        engine = AgentEngine(backend, registry, repo_path)
        async for event in engine.run(query, history):
            handle(event)
    """

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
        """Run the ReAct loop, yielding events.

        Events:
        - THINKING: intermediate reasoning text
        - TOOL_CALL: LLM is calling a tool
        - TOOL_RESULT: tool execution result
        - TEXT: final answer (streamed)
        - DONE: complete
        - ERROR: failure
        """
        history = history or []
        tools = self.registry.get_definitions()

        # The conversation messages for this run
        messages = self.context_builder.build(self.repo_path, history, query)

        for iteration in range(1, self.config.max_iterations + 1):
            logger.info("Agent iteration %d/%d", iteration, self.config.max_iterations)

            try:
                response = await self.backend.chat(
                    messages=messages,
                    tools=tools if iteration < self.config.max_iterations else None,
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens,
                )
            except Exception as e:
                logger.error("LLM call failed: %s", e)
                yield AgentEvent(
                    type=EventType.ERROR,
                    content=f"LLM inference error: {e}",
                )
                return

            # Parse tool calls
            result = parse_tool_calls(
                message={
                    "content": response.content,
                    "tool_calls": response.tool_calls,
                },
                full_text=response.content,
                prefer_structured=bool(response.tool_calls),
            )

            # Emit thinking/reasoning text if present
            if result.text_before:
                yield AgentEvent(
                    type=EventType.THINKING,
                    content=result.text_before,
                    data={"iteration": iteration},
                )

            # If no tool calls, emit final text and finish
            if not result.calls:
                final_text = result.text_before or response.content
                yield AgentEvent(type=EventType.TEXT, content=final_text)
                yield AgentEvent(type=EventType.DONE, content=final_text)
                return

            # Process tool calls
            assistant_msg = ChatMessage(
                role="assistant",
                content=response.content or "",
                tool_calls=response.tool_calls if response.tool_calls else None,
            )
            messages.append(assistant_msg)

            for tc in result.calls:
                # Emit tool call event
                yield AgentEvent(
                    type=EventType.TOOL_CALL,
                    content=tc.name,
                    data={
                        "arguments": tc.arguments,
                        "iteration": iteration,
                    },
                )

                # Execute the tool
                tool_result = await self.registry.execute(tc.name, tc.arguments)

                # Emit tool result (truncate for event)
                display_result = tool_result
                if len(display_result) > 2000:
                    display_result = display_result[:2000] + "\n... [truncated]"
                yield AgentEvent(
                    type=EventType.TOOL_RESULT,
                    content=display_result,
                    data={
                        "tool": tc.name,
                        "iteration": iteration,
                    },
                )

                # Add tool result to conversation
                tool_msg = ChatMessage(
                    role="tool",
                    content=tool_result,
                    tool_call_id=_make_tool_call_id(tc.name, iteration),
                    name=tc.name,
                )
                messages.append(tool_msg)

        # Max iterations reached without a final answer
        logger.warning("Agent reached max iterations (%d)", self.config.max_iterations)
        yield AgentEvent(
            type=EventType.TEXT,
            content=(
                "I've reached the maximum number of reasoning steps. "
                "Here's what I found so far. You may need to refine your question."
            ),
        )
        yield AgentEvent(type=EventType.DONE)


async def run_agent_streaming(
    backend: LLMBackend,
    registry: ToolRegistry,
    repo_path: str,
    query: str,
    history: list[ChatMessage] | None = None,
    config: AgentConfig | None = None,
) -> AsyncIterator[AgentEvent]:
    """Convenience function: create an engine and run it."""
    engine = AgentEngine(backend, registry, repo_path, config)
    async for event in engine.run(query, history):
        yield event


def _make_tool_call_id(tool_name: str, iteration: int) -> str:
    """Generate a tool call ID compatible with OpenAI API format."""
    return f"call_{tool_name}_{iteration}"


