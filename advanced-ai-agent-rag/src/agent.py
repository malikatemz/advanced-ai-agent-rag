"""
Multi-step reasoning agent built on Claude's native tool-use loop.
The agent can call tools repeatedly (retrieval, calculator, web search) until
it has enough information to produce a final answer, up to max_iterations.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import anthropic

from src.config import get_settings
from src.tools.definitions import TOOL_REGISTRY, TOOL_SCHEMAS

SYSTEM_PROMPT = """You are a careful, multi-step reasoning AI agent with access to tools:
- retrieve_documents: search the user's private document knowledge base
- calculator: evaluate arithmetic
- web_search: search the live web (only if configured)
- list_knowledge_base: see what documents are available

Guidelines:
1. Break complex questions into steps. Use tools as many times as needed before answering.
2. Prefer retrieve_documents for anything that could be in the user's documents.
3. Always ground factual claims in tool results; say so explicitly if you're unsure.
4. When you have enough information, give a clear, direct final answer. Do not call
   more tools than necessary.
5. Show your reasoning briefly, then the answer."""


@dataclass
class AgentStep:
    type: str  # "tool_call" | "tool_result" | "text"
    content: str
    tool_name: str | None = None
    tool_input: dict | None = None


@dataclass
class AgentResult:
    final_answer: str
    steps: list[AgentStep] = field(default_factory=list)


def run_agent(user_query: str, max_iterations: int = 6) -> AgentResult:
    settings = get_settings()
    client = anthropic.Anthropic(api_key=settings.anthropic_api_key)

    messages = [{"role": "user", "content": user_query}]
    steps: list[AgentStep] = []

    for _ in range(max_iterations):
        response = client.messages.create(
            model=settings.claude_model,
            max_tokens=1536,
            system=SYSTEM_PROMPT,
            tools=TOOL_SCHEMAS,
            messages=messages,
        )

        text_parts = [b.text for b in response.content if b.type == "text"]
        if text_parts:
            steps.append(AgentStep(type="text", content="".join(text_parts)))

        if response.stop_reason != "tool_use":
            final_text = "".join(text_parts).strip()
            return AgentResult(final_answer=final_text or "(no answer produced)", steps=steps)

        # Execute every tool_use block the model requested, then continue the loop.
        messages.append({"role": "assistant", "content": response.content})
        tool_results = []
        for block in response.content:
            if block.type != "tool_use":
                continue
            handler = TOOL_REGISTRY.get(block.name)
            if handler is None:
                result_text = f"Unknown tool: {block.name}"
            else:
                try:
                    result_text = handler(block.input)
                except Exception as e:  # noqa: BLE001
                    result_text = f"Tool error: {e}"

            steps.append(
                AgentStep(type="tool_call", content=str(block.input), tool_name=block.name, tool_input=block.input)
            )
            steps.append(AgentStep(type="tool_result", content=result_text, tool_name=block.name))

            tool_results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result_text,
                }
            )

        messages.append({"role": "user", "content": tool_results})

    return AgentResult(
        final_answer="Reached the maximum number of reasoning steps without a final answer. "
        "Try rephrasing the question or increasing max_iterations.",
        steps=steps,
    )
