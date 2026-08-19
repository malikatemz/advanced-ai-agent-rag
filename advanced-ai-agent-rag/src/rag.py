"""
Retrieval-Augmented Generation pipeline: retrieve relevant chunks from the
vector store, then ask Claude to answer strictly grounded in that context,
with inline [n] citations back to source chunks.
"""
from __future__ import annotations

from dataclasses import dataclass

import anthropic

from src.config import get_settings
from src.vectorstore import get_vectorstore

SYSTEM_PROMPT = """You are a precise document Q&A assistant. You answer ONLY using the \
numbered context passages provided below. Rules:
1. Every factual claim must be supported by the context and cited inline like [1], [2].
2. If the context does not contain the answer, say so plainly - do not guess or use \
outside knowledge.
3. Be concise and direct. Do not pad the answer with generic commentary.
4. If multiple passages disagree, point out the disagreement rather than picking one \
silently."""


@dataclass
class RAGResult:
    answer: str
    sources: list[dict]
    query: str


def _format_context(chunks: list[dict]) -> str:
    blocks = []
    for i, c in enumerate(chunks, start=1):
        src = c["metadata"].get("source", "unknown")
        blocks.append(f"[{i}] (source: {src})\n{c['text']}")
    return "\n\n".join(blocks)


def answer_query(query: str, top_k: int | None = None, filter_source: str | None = None) -> RAGResult:
    settings = get_settings()
    store = get_vectorstore()

    where = {"source": filter_source} if filter_source else None
    retrieved = store.query(query, top_k=top_k, where=where)

    if not retrieved:
        return RAGResult(
            answer="No documents have been ingested yet, so I have no context to answer from. "
            "Ingest some documents first.",
            sources=[],
            query=query,
        )

    context = _format_context(retrieved)
    client = anthropic.Anthropic(api_key=settings.anthropic_api_key)

    message = client.messages.create(
        model=settings.claude_model,
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": f"Context passages:\n\n{context}\n\nQuestion: {query}",
            }
        ],
    )

    answer_text = "".join(block.text for block in message.content if block.type == "text")

    return RAGResult(
        answer=answer_text,
        sources=[
            {
                "index": i + 1,
                "source": c["metadata"].get("source"),
                "chunk_index": c["metadata"].get("chunk_index"),
                "score": c["score"],
                "preview": c["text"][:220] + ("..." if len(c["text"]) > 220 else ""),
            }
            for i, c in enumerate(retrieved)
        ],
        query=query,
    )
