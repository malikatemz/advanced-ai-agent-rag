"""
Embedding provider abstraction. Defaults to a local sentence-transformers
model (free, no API key). Can be switched to OpenAI embeddings via config.
"""
from __future__ import annotations

from functools import lru_cache

from src.config import get_settings


class LocalEmbedder:
    """Wraps a sentence-transformers model."""

    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors = self.model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
        return [v.tolist() for v in vectors]


class OpenAIEmbedder:
    """Wraps OpenAI's embeddings endpoint."""

    def __init__(self, api_key: str, model_name: str = "text-embedding-3-small"):
        from openai import OpenAI

        self.client = OpenAI(api_key=api_key)
        self.model_name = model_name

    def embed(self, texts: list[str]) -> list[list[float]]:
        response = self.client.embeddings.create(model=self.model_name, input=texts)
        return [item.embedding for item in response.data]


@lru_cache
def get_embedder():
    settings = get_settings()
    if settings.embedding_provider == "openai":
        if not settings.openai_api_key:
            raise RuntimeError("EMBEDDING_PROVIDER=openai requires OPENAI_API_KEY to be set.")
        return OpenAIEmbedder(settings.openai_api_key)
    return LocalEmbedder(settings.embedding_model)
