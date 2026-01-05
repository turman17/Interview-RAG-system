from __future__ import annotations

from typing import Any

from .config import get_settings, normalize_openai_base_url
from .llm import get_chat_model

_VECTOR_STORE: Any | None = None


def _get_embeddings() -> Any:
    settings = get_settings()
    if settings.embedding_provider.lower() == "local":
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "Local embeddings not installed. "
                "Install with `pip install sentence-transformers`."
            ) from exc

        try:
            from langchain_core.embeddings import Embeddings
        except ImportError:  # pragma: no cover - optional dependency shape
            Embeddings = object  # type: ignore[misc,assignment]

        class LocalSentenceTransformerEmbeddings(Embeddings):
            def __init__(self, model_name: str) -> None:
                self._model = SentenceTransformer(model_name)

            def embed_documents(self, texts: list[str]) -> list[list[float]]:
                vectors = self._model.encode(texts, normalize_embeddings=True)
                return [vector.tolist() for vector in vectors]

            def embed_query(self, text: str) -> list[float]:
                vector = self._model.encode([text], normalize_embeddings=True)[0]
                return vector.tolist()

        return LocalSentenceTransformerEmbeddings(settings.embedding_model)

    try:
        from langchain_openai import OpenAIEmbeddings
    except ImportError as exc:
        raise RuntimeError(
            "LangChain OpenAI embeddings not installed. "
            "Install with `pip install langchain-openai`."
        ) from exc

    return OpenAIEmbeddings(
        base_url=normalize_openai_base_url(settings.llm_base_url),
        api_key=settings.llm_api_key,
        model=settings.embedding_model,
    )


def _load_documents() -> list[Any]:
    settings = get_settings()
    try:
        from langchain_community.document_loaders import TextLoader
        from langchain_text_splitters import RecursiveCharacterTextSplitter
    except ImportError as exc:
        raise RuntimeError(
            "LangChain loaders/splitters not installed. "
            "Install with `pip install langchain-community langchain-text-splitters`."
        ) from exc

    loader = TextLoader(settings.rag_source_path, encoding="utf-8")
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.rag_chunk_size,
        chunk_overlap=settings.rag_chunk_overlap,
    )
    return splitter.split_documents(documents)


def _build_vector_store() -> Any:
    settings = get_settings()
    docs = _load_documents()
    embeddings = _get_embeddings()

    backend = settings.rag_backend.lower()
    if backend == "faiss":
        try:
            from langchain_community.vectorstores import FAISS
        except ImportError as exc:
            raise RuntimeError(
                "FAISS vector store not installed. "
                "Install with `pip install langchain-community faiss-cpu`."
            ) from exc
        return FAISS.from_documents(docs, embeddings)

    if backend == "chroma":
        try:
            from langchain_community.vectorstores import Chroma
        except ImportError as exc:
            raise RuntimeError(
                "Chroma vector store not installed. "
                "Install with `pip install langchain-community chromadb`."
            ) from exc
        return Chroma.from_documents(
            docs,
            embeddings,
            persist_directory=settings.rag_persist_dir,
            collection_name="rag",
        )

    raise ValueError(f"Unsupported RAG backend: {settings.rag_backend}")


def _get_vector_store() -> Any:
    global _VECTOR_STORE
    if _VECTOR_STORE is None:
        _VECTOR_STORE = _build_vector_store()
    return _VECTOR_STORE


def _score_to_relevance(score: float) -> float:
    if 0.0 <= score <= 1.0:
        return score
    return 1.0 / (1.0 + abs(score))


def _retrieve_context(question: str) -> list[str]:
    settings = get_settings()
    store = _get_vector_store()

    results = store.similarity_search_with_score(question, k=settings.rag_top_k)
    scored = [
        (doc.page_content, _score_to_relevance(score))
        for doc, score in results
    ]

    return [
        content for content, score in scored if score >= settings.rag_min_relevance
    ]


def answer_with_rag(question: str) -> str:
    context_chunks = _retrieve_context(question)
    if not context_chunks:
        return "I can't answer that based on the provided data."

    context = "\n\n".join(context_chunks)
    prompt = (
        "Use the context to answer the question. "
        "If the answer is not in the context, say: "
        '"I can\'t answer that based on the provided data."\n\n'
        f"Context:\n{context}\n\nQuestion: {question}"
    )
    model = get_chat_model()
    response = model.invoke(prompt)
    return getattr(response, "content", "") or "I can't answer that based on the provided data."
