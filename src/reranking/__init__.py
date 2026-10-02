"""Reranking cross-encoder'ом: пара (вопрос, chunk) оценивается совместно,
поэтому точнее bi-encoder'а, но дороже — применяем только к top-K кандидатам."""
from __future__ import annotations

from src.retrieval import Hit

RERANKERS = {
    # мультиязычный MiniLM, дообученный на mMARCO (переведённый MS MARCO) — быстрый, 118M
    "mminilm": "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
    # BGE reranker v2 на базе bge-m3 — точнее, но в ~5 раз тяжелее (568M)
    "bge-m3": "BAAI/bge-reranker-v2-m3",
}


class Reranker:
    def __init__(self, key: str):
        self.key = key
        self.name = RERANKERS[key]

    def rerank(self, query: str, hits: list[Hit], top_n: int) -> list[Hit]:
        if not hits:
            return []
        from src.hf import cross_scores
        scores = cross_scores(self.name, [(query, h.chunk.text) for h in hits])
        for h, s in zip(hits, scores):
            h.rerank_score = float(s)
        return sorted(hits, key=lambda h: -h.rerank_score)[:top_n]
