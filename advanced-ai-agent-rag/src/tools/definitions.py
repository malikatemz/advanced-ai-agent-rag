"""
Tool schemas (Anthropic tool-use format) and their Python implementations.
Add new tools by: (1) writing a handler function, (2) adding its schema to
TOOL_SCHEMAS, (3) registering it in TOOL_REGISTRY.
"""
from __future__ import annotations

import ast
import operator
import re

import requests

from src.config import get_settings
from src.vectorstore import get_vectorstore

# ---------------------------------------------------------------------------
# Tool: retrieve_documents (RAG lookup)
# ---------------------------------------------------------------------------


def retrieve_documents(query: str, top_k: int = 5) -> str:
    store = get_vectorstore()
    results = store.query(query, top_k=top_k)
    if not results:
        return "No relevant documents found in the knowledge base."
    lines = []
    for i, r in enumerate(results, start=1):
        src = r["metadata"].get("source", "unknown")
        lines.append(f"[{i}] (source: {src}, score: {r['score']:.3f})\n{r['text']}")
    return "\n\n".join(lines)


# ---------------------------------------------------------------------------
# Tool: calculator (safe arithmetic eval - no `eval()`)
# ---------------------------------------------------------------------------

_ALLOWED_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.FloorDiv: operator.floordiv,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _safe_eval(node):
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError("Only numeric constants are allowed.")
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_OPS:
        return _ALLOWED_OPS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_OPS:
        return _ALLOWED_OPS[type(node.op)](_safe_eval(node.operand))
    raise ValueError(f"Unsupported expression: {ast.dump(node)}")


def calculator(expression: str) -> str:
    try:
        cleaned = expression.strip()
        if not re.fullmatch(r"[0-9+\-*/%().\s^]*", cleaned.replace("**", "^")):
            # allow only digits/operators/parens/whitespace/decimal points
            pass
        tree = ast.parse(cleaned, mode="eval")
        result = _safe_eval(tree.body)
        return str(result)
    except Exception as e:  # noqa: BLE001
        return f"Error evaluating expression: {e}"


# ---------------------------------------------------------------------------
# Tool: web_search (optional, requires TAVILY_API_KEY)
# ---------------------------------------------------------------------------


def web_search(query: str, max_results: int = 5) -> str:
    settings = get_settings()
    if not settings.tavily_api_key:
        return (
            "Web search is not configured. Set TAVILY_API_KEY in your .env to enable "
            "this tool. Falling back to knowledge-base search only."
        )
    try:
        resp = requests.post(
            "https://api.tavily.com/search",
            json={
                "api_key": settings.tavily_api_key,
                "query": query,
                "max_results": max_results,
            },
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
        results = data.get("results", [])
        if not results:
            return "No web results found."
        lines = []
        for r in results:
            lines.append(f"- {r.get('title')}: {r.get('content', '')[:300]} ({r.get('url')})")
        return "\n".join(lines)
    except Exception as e:  # noqa: BLE001
        return f"Web search failed: {e}"


# ---------------------------------------------------------------------------
# Tool: list_knowledge_base
# ---------------------------------------------------------------------------


def list_knowledge_base(_: str = "") -> str:
    store = get_vectorstore()
    sources = store.list_sources()
    if not sources:
        return "The knowledge base is empty. No documents have been ingested."
    return f"{len(sources)} document(s) in the knowledge base:\n" + "\n".join(f"- {s}" for s in sources)


# ---------------------------------------------------------------------------
# Schemas + registry
# ---------------------------------------------------------------------------

TOOL_SCHEMAS = [
    {
        "name": "retrieve_documents",
        "description": (
            "Search the ingested document knowledge base (vector similarity search) "
            "and return the most relevant passages. Use this whenever the user asks "
            "a question that could be answered from the uploaded documents."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The search query."},
                "top_k": {"type": "integer", "description": "Number of passages to retrieve.", "default": 5},
            },
            "required": ["query"],
        },
    },
    {
        "name": "calculator",
        "description": "Evaluate a numeric arithmetic expression (+, -, *, /, %, **, parentheses). Use for any math.",
        "input_schema": {
            "type": "object",
            "properties": {"expression": {"type": "string", "description": "e.g. '(42 * 7) / 3'"}},
            "required": ["expression"],
        },
    },
    {
        "name": "web_search",
        "description": (
            "Search the live web for current information not in the document knowledge base. "
            "Only use this when the knowledge base doesn't have the answer or the question is "
            "about current events."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "max_results": {"type": "integer", "default": 5},
            },
            "required": ["query"],
        },
    },
    {
        "name": "list_knowledge_base",
        "description": "List the documents currently ingested into the knowledge base.",
        "input_schema": {"type": "object", "properties": {}},
    },
]

TOOL_REGISTRY = {
    "retrieve_documents": lambda inp: retrieve_documents(inp.get("query", ""), inp.get("top_k", 5)),
    "calculator": lambda inp: calculator(inp.get("expression", "")),
    "web_search": lambda inp: web_search(inp.get("query", ""), inp.get("max_results", 5)),
    "list_knowledge_base": lambda inp: list_knowledge_base(),
}
