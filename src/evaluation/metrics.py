"""Метрики качества поиска.

Эталон (gold) размечен на уровне «документ + секция», а не chunk'а, поэтому
один и тот же датасет подходит для любой стратегии chunking:
chunk релевантен, если он из нужного документа и покрывает нужную секцию.
"""
from __future__ import annotations

import math

from src.models import Chunk


def _section_match(chunk_sections: list[str], gold_section: str | list[str] | None) -> bool:
    """gold_section — одна секция или список равноправных секций (факт повторяется
    в карточке и во введении); подсекции "Атмосфера / Климат" входят в "Атмосфера"."""
    if not gold_section:
        return True
    golds = gold_section if isinstance(gold_section, list) else [gold_section]
    return any(s == g or s.startswith(g + " / ") for s in chunk_sections for g in golds)


def matches(chunk: Chunk, g: dict, level: str = "section") -> bool:
    if chunk.doc_id != g["doc_id"]:
        return False
    return level == "doc" or _section_match(chunk.sections, g.get("section"))


def is_relevant(chunk: Chunk, gold: list[dict], level: str = "section") -> bool:
    return any(matches(chunk, g, level) for g in gold)


def retrieval_metrics(ranked: list[Chunk], gold: list[dict], ks=(1, 3, 5, 10, 20),
                      level: str = "section") -> dict[str, float]:
    rel = [is_relevant(c, gold, level) for c in ranked]
    n_rel = sum(rel)          # идеальный порядок: все найденные релевантные — в начале списка
    out: dict[str, float] = {}
    first = next((i for i, r in enumerate(rel) if r), None)
    out["mrr"] = 1.0 / (first + 1) if first is not None else 0.0
    for k in ks:
        top = ranked[:k]
        covered = sum(any(matches(c, g, level) for c in top) for g in gold)
        out[f"hit@{k}"] = float(any(rel[:k]))
        out[f"recall@{k}"] = covered / len(gold)
        out[f"precision@{k}"] = sum(rel[:k]) / k
        dcg = sum(1 / math.log2(i + 2) for i, r in enumerate(rel[:k]) if r)
        idcg = sum(1 / math.log2(i + 2) for i in range(min(k, n_rel)))
        out[f"ndcg@{k}"] = dcg / idcg if idcg else 0.0
    return out
