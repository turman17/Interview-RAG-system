from __future__ import annotations
import os
from dataclasses import dataclass


def _load_dotenv() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv()


@dataclass(frozen=True)
class Settings:
    llm_base_url: str
    llm_model: str
    llm_api_key: str
    embedding_provider: str
    embedding_model: str
    albums_api_url: str
    rag_source_path: str
    rag_backend: str
    rag_top_k: int
    rag_min_relevance: float
    rag_chunk_size: int
    rag_chunk_overlap: int
    rag_persist_dir: str


def normalize_openai_base_url(base_url: str) -> str:
    normalized = base_url.rstrip("/")
    if not normalized.endswith("/v1"):
        normalized = f"{normalized}/v1"
    return normalized


def get_settings() -> Settings:
    _load_dotenv()
    base_url = os.getenv("LLM_BASE_URL", "http://127.0.0.1:1234")
    model = os.getenv("LLM_MODEL", "qwen2.5-7b-instruct-1m")
    api_key = os.getenv("LLM_API_KEY", "lm-studio")
    embedding_provider = os.getenv("EMBEDDING_PROVIDER", "local")
    embedding_model = os.getenv("EMBEDDING_MODEL", model)
    albums_api_url = os.getenv(
        "ALBUMS_API_URL", "https://jsonplaceholder.typicode.com/albums"
    )
    rag_source_path = os.getenv("RAG_SOURCE_PATH", "data/source.txt")
    rag_backend = os.getenv("RAG_BACKEND", "faiss")
    rag_top_k = int(os.getenv("RAG_TOP_K", "3"))
    rag_min_relevance = float(os.getenv("RAG_MIN_RELEVANCE", "0.6"))
    rag_chunk_size = int(os.getenv("RAG_CHUNK_SIZE", "800"))
    rag_chunk_overlap = int(os.getenv("RAG_CHUNK_OVERLAP", "100"))
    rag_persist_dir = os.getenv("RAG_PERSIST_DIR", "data/index")
    return Settings(
        llm_base_url=base_url,
        llm_model=model,
        llm_api_key=api_key,
        embedding_provider=embedding_provider,
        embedding_model=embedding_model,
        albums_api_url=albums_api_url,
        rag_source_path=rag_source_path,
        rag_backend=rag_backend,
        rag_top_k=rag_top_k,
        rag_min_relevance=rag_min_relevance,
        rag_chunk_size=rag_chunk_size,
        rag_chunk_overlap=rag_chunk_overlap,
        rag_persist_dir=rag_persist_dir,
    )
