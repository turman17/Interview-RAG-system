from __future__ import annotations
from typing import Any
from .config import get_settings, normalize_openai_base_url


def get_chat_model() -> Any:
    settings = get_settings()
    base_url = normalize_openai_base_url(settings.llm_base_url)
    try:
        from langchain_openai import ChatOpenAI
    except ImportError as exc:
        raise RuntimeError(
            "LangChain OpenAI integration not installed. "
            "Install with `pip install langchain-openai`."
        ) from exc

    return ChatOpenAI(
        base_url=base_url,
        api_key=settings.llm_api_key,
        model=settings.llm_model,
    )
