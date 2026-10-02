"""Фильтрация контекста между retriever'ом и reranker'ом / LLM."""
from __future__ import annotations

import re
from dataclasses import dataclass

from src.retrieval import Hit

_W = re.compile(r"\w+", re.U)


@dataclass
class FilterConfig:
    score_threshold: float = 0.0      # минимальный dense cosine score (0 — выключено)
    min_chars: int = 60               # слишком короткие chunks ("Таланты\nУровень 10: ...")
    dedup: bool = True                # удалять почти одинаковые chunks
    dedup_jaccard: float = 0.85
    max_per_doc: int = 0              # не больше N chunks из одного документа (0 — без лимита)


def _shingles(text: str, n: int = 3) -> set[tuple]:
    w = _W.findall(text.lower())
    return {tuple(w[i:i + n]) for i in range(max(1, len(w) - n + 1))}


def jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if a and b else 0.0


def apply_filters(hits: list[Hit], cfg: FilterConfig) -> tuple[list[Hit], dict]:
    log = {"input": len(hits), "threshold": 0, "short": 0, "duplicate": 0, "per_doc": 0}
    out: list[Hit] = []
    kept_shingles: list[set] = []
    per_doc: dict[str, int] = {}
    for h in hits:
        if cfg.score_threshold and h.dense_score is not None and h.dense_score < cfg.score_threshold:
            log["threshold"] += 1
            continue
        body = h.chunk.text.split("\n", 1)[-1]           # без заголовка-префикса
        if len(body) < cfg.min_chars:
            log["short"] += 1
            continue
        if cfg.dedup:
            sh = _shingles(body)
            if any(jaccard(sh, k) >= cfg.dedup_jaccard for k in kept_shingles):
                log["duplicate"] += 1
                continue
            kept_shingles.append(sh)
        if cfg.max_per_doc:
            if per_doc.get(h.chunk.doc_id, 0) >= cfg.max_per_doc:
                log["per_doc"] += 1
                continue
            per_doc[h.chunk.doc_id] = per_doc.get(h.chunk.doc_id, 0) + 1
        out.append(h)
    log["output"] = len(out)
    return out, log
