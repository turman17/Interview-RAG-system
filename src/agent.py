from __future__ import annotations

from typing import Any

from .api_tool import fetch_albums
from .llm import get_chat_model
from .rag import answer_with_rag


def _route_question(question: str) -> str:
    model = get_chat_model()
    prompt = (
        "You are a router. Choose the best data source to answer the question.\n"
        "If it is about AI history, singularity, or the provided document, answer: rag\n"
        "If it is about albums, users, ids, or titles from the albums API, answer: api\n"
        "Return only one word: rag or api.\n\n"
        f"Question: {question}"
    )
    response = model.invoke(prompt)
    choice = str(getattr(response, "content", "")).strip().lower()
    return "api" if "api" in choice else "rag"


def _answer_with_api(question: str) -> str:
    albums = fetch_albums()
    model = get_chat_model()
    context = albums[:20]
    prompt = (
        "Use the albums data to answer the question. "
        "If the answer is not in the data, say: "
        '"I can\'t answer that based on the albums data."\n\n'
        f"Albums data:\n{context}\n\nQuestion: {question}"
    )
    response = model.invoke(prompt)
    return (
        getattr(response, "content", "")
        or "I can't answer that based on the albums data."
    )


def answer_with_router(question: str) -> tuple[str, str]:
    route = _route_question(question)
    if route == "api":
        return "api", _answer_with_api(question)
    return "rag", answer_with_rag(question)
