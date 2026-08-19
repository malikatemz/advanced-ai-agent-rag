"""
Thin wrapper around a persistent ChromaDB collection, using our own
embedding function so we control which provider generates the vectors.
"""
from __future__ import annotations

from functools import lru_cache

import chromadb
from chromadb.api.types import Documents, Embeddings

from src.config import get_settings
from src.embeddings import get_embedder
from src.ingestion import Chunk


class _EmbeddingFunctionAdapter:
    """Adapts our embedder to Chroma's EmbeddingFunction interface."""

    def __call__(self, input: Documents) -> Embeddings:  # noqa: A002 - Chroma's required name
        return get_embedder().embed(list(input))


class VectorStore:
    def __init__(self):
        settings = get_settings()
        self.client = chromadb.PersistentClient(path=settings.chroma_persist_dir)
        self.collection = self.client.get_or_create_collection(
            name=settings.chroma_collection,
            embedding_function=_EmbeddingFunctionAdapter(),
            metadata={"hnsw:space": "cosine"},
        )

    def add_chunks(self, chunks: list[Chunk]) -> int:
        if not chunks:
            return 0
        self.collection.upsert(
            ids=[c.id for c in chunks],
            documents=[c.text for c in chunks],
            metadatas=[{"source": c.source, "doc_id": c.doc_id, "chunk_index": c.chunk_index} for c in chunks],
        )
        return len(chunks)

    def query(self, query_text: str, top_k: int | None = None, where: dict | None = None) -> list[dict]:
        settings = get_settings()
        top_k = top_k or settings.top_k
        results = self.collection.query(
            query_texts=[query_text],
            n_results=top_k,
            where=where,
        )
        out = []
        ids = results.get("ids", [[]])[0]
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        dists = results.get("distances", [[]])[0]
        for i in range(len(ids)):
            out.append(
                {
                    "id": ids[i],
                    "text": docs[i],
                    "metadata": metas[i],
                    "distance": dists[i],
                    "score": 1 - dists[i] if dists[i] is not None else None,
                }
            )
        return out

    def count(self) -> int:
        return self.collection.count()

    def list_sources(self) -> list[str]:
        data = self.collection.get(include=["metadatas"])
        sources = {m.get("source") for m in data.get("metadatas", []) if m}
        return sorted(s for s in sources if s)

    def delete_source(self, source: str) -> int:
        data = self.collection.get(where={"source": source}, include=[])
        ids = data.get("ids", [])
        if ids:
            self.collection.delete(ids=ids)
        return len(ids)

    def reset(self):
        settings = get_settings()
        self.client.delete_collection(settings.chroma_collection)
        self.collection = self.client.get_or_create_collection(
            name=settings.chroma_collection,
            embedding_function=_EmbeddingFunctionAdapter(),
            metadata={"hnsw:space": "cosine"},
        )


@lru_cache
def get_vectorstore() -> VectorStore:
    return VectorStore()
