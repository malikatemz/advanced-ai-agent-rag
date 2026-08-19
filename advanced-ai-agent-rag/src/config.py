"""
Centralized configuration for the RAG + Agent system.
All values are read from environment variables / .env, with sane defaults.
"""
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # LLM
    anthropic_api_key: str = ""
    claude_model: str = "claude-sonnet-4-6"

    # Optional web search tool
    tavily_api_key: str = ""

    # Embeddings
    embedding_provider: str = "local"  # "local" or "openai"
    embedding_model: str = "all-MiniLM-L6-v2"
    openai_api_key: str = ""

    # Vector store
    chroma_persist_dir: str = "./data/chroma"
    chroma_collection: str = "documents"

    # Chunking / retrieval
    chunk_size: int = 800
    chunk_overlap: int = 120
    top_k: int = 5

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000


@lru_cache
def get_settings() -> Settings:
    return Settings()
