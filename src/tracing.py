"""Трассировка в Langfuse (необязательная).

Включается, если в .env заданы LANGFUSE_PUBLIC_KEY и LANGFUSE_SECRET_KEY
(+ LANGFUSE_BASE_URL, по умолчанию EU-облако https://cloud.langfuse.com).
Без ключей все функции — заглушки, pipeline работает как раньше.

Структура трассы одного вопроса:
    rag-ask (chain)
    ├── retrieve (retriever) — кандидаты из Qdrant/BM25 со скорами
    ├── rerank (span)        — итоговый контекст после фильтров и reranker'а
    └── llm:groq (generation) — промпт, ответ, токены, кэш
В экспериментах (run_generation) поверх — eval-question с оценками судьи (scores).
"""
from __future__ import annotations

import atexit
import logging
import os
from contextlib import nullcontext
from functools import lru_cache

from dotenv import dotenv_values

from src.config import ENV_FILES, get_secret

log = logging.getLogger(__name__)
DEFAULT_BASE_URL = "https://cloud.langfuse.com"


class _NoOp:
    """Заглушка наблюдения: те же методы, что у LangfuseSpan, но ничего не делают."""

    def update(self, **_):
        return self

    def score_trace(self, **_):
        return None


@lru_cache(maxsize=1)
def client():
    pk, sk = get_secret("LANGFUSE_PUBLIC_KEY"), get_secret("LANGFUSE_SECRET_KEY")
    if not (pk and sk):
        return None
    base_url = os.environ.get("LANGFUSE_BASE_URL") or os.environ.get("LANGFUSE_HOST")
    for f in ENV_FILES:
        if not base_url and f.exists():
            env = dotenv_values(f)
            base_url = env.get("LANGFUSE_BASE_URL") or env.get("LANGFUSE_HOST")
    from langfuse import Langfuse
    lf = Langfuse(public_key=pk, secret_key=sk, base_url=base_url or DEFAULT_BASE_URL)
    atexit.register(lf.flush)              # короткоживущие процессы (CLI, эксперименты)
    return lf


def enabled() -> bool:
    return client() is not None


def observation(name: str, as_type: str = "span", **kwargs):
    """Контекстный менеджер наблюдения (дочернего к текущему, если оно есть)."""
    lf = client()
    if lf is None:
        return nullcontext(_NoOp())
    return lf.start_as_current_observation(name=name, as_type=as_type, **kwargs)


def trace_attributes(**kwargs):
    """Атрибуты всей трассы: session_id, tags, metadata, trace_name."""
    if client() is None:
        return nullcontext()
    from langfuse import propagate_attributes
    return propagate_attributes(**kwargs)


def current_trace_url() -> str | None:
    """Ссылка на текущую трассу. SDK узнаёт id проекта сетевым запросом — если Langfuse
    недоступен или ключи неверны, ответ пользователю не должен падать из-за трассировки."""
    lf = client()
    if lf is None:
        return None
    try:
        return lf.get_trace_url(trace_id=lf.get_current_trace_id())
    except Exception as e:
        log.warning("Langfuse недоступен, ссылка на трассу не получена: %s", e)
        return None


def flush() -> None:
    lf = client()
    if lf:
        lf.flush()


def hits_summary(hits) -> list[dict]:
    """Компактное представление найденных chunks для трассы (без полного текста)."""
    return [{"chunk_id": h.chunk.chunk_id, "title": h.chunk.title, "section": h.chunk.section,
             "dense": None if h.dense_score is None else round(h.dense_score, 4),
             "rerank": None if h.rerank_score is None else round(h.rerank_score, 4)} for h in hits]
