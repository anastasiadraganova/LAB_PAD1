"""Генерация ответа по найденному контексту + список источников."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from src.generation.llm import LLM, LLMResponse, build_llm
from src.generation.prompts import REFUSAL, build_messages
from src.retrieval import Hit

_CITE = re.compile(r"\[(\d+)\]")
# gpt-oss иногда ставит ссылки в «широких» скобках: 【1】 → [1]
_WIDE_CITE = re.compile(r"\s?【(\d+)[^】]*】")


@dataclass
class Answer:
    text: str
    refused: bool
    sources: list[dict] = field(default_factory=list)
    llm: LLMResponse | None = None


def is_refusal(text: str) -> bool:
    t = text.lower()
    return "недостаточно информации" in t or "нет информации" in t


def generate(llm: LLM, prompt_key: str, question: str, hits: list[Hit]) -> Answer:
    if not hits:                                   # нечего отдавать LLM → честный отказ без запроса
        return Answer(REFUSAL, True)
    resp = llm.chat(build_messages(prompt_key, question, hits))
    resp.text = _WIDE_CITE.sub(r"[\1]", resp.text)
    cited = sorted({int(n) for n in _CITE.findall(resp.text) if 0 < int(n) <= len(hits)})
    used = cited or list(range(1, len(hits) + 1))  # нет ссылок → показываем весь контекст
    sources, seen = [], set()
    for n in used:
        c = hits[n - 1].chunk
        key = (c.url, c.section)
        if key in seen:
            continue
        seen.add(key)
        sources.append({"n": n, "title": c.title, "section": c.section, "url": c.url,
                        "source": c.source, "cited": bool(cited)})
    refused = is_refusal(resp.text)
    return Answer(resp.text, refused, [] if refused else sources, resp)


__all__ = ["Answer", "generate", "build_llm", "is_refusal", "REFUSAL"]
