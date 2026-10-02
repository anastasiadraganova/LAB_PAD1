"""Векторное хранилище (Qdrant), BM25 и retriever (dense / bm25 / hybrid)."""
from __future__ import annotations

import atexit
import re
import uuid
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from qdrant_client import QdrantClient, models
from rank_bm25 import BM25Okapi

from src import config
from src.embeddings import Embedder
from src.models import Chunk


@dataclass
class Hit:
    chunk: Chunk
    score: float                        # dense cosine / bm25 / RRF — в зависимости от режима
    dense_score: float | None = None
    rerank_score: float | None = None


@lru_cache(maxsize=2)
def get_client(location: str) -> QdrantClient:
    """Qdrant: локальный embedded-режим (путь) или сервер (http://...)."""
    if location.startswith("http"):
        return QdrantClient(url=location)
    client = QdrantClient(path=location)
    atexit.register(client.close)          # иначе embedded-клиент закрывается при выгрузке модулей с ошибкой
    return client


def qdrant_location(cfg: dict) -> str:
    q = cfg.get("vector_store", {})
    return q.get("url") or str(config.path(cfg, "qdrant"))


class VectorIndex:
    def __init__(self, cfg: dict, collection: str):
        self.client = get_client(qdrant_location(cfg))
        self.collection = collection

    def count(self) -> int:
        if not self.client.collection_exists(self.collection):
            return 0
        return self.client.count(self.collection, exact=True).count

    def build(self, chunks: list[Chunk], embedder: Embedder, recreate: bool = False) -> None:
        if not recreate and self.count() == len(chunks):
            return                                              # индекс уже актуален
        if self.client.collection_exists(self.collection):
            self.client.delete_collection(self.collection)
        self.client.create_collection(
            self.collection,
            vectors_config=models.VectorParams(size=embedder.dim, distance=models.Distance.COSINE),
        )
        # индексы по metadata для фильтрации
        for field in ("source", "lang", "entity_type", "doc_id"):
            self.client.create_payload_index(self.collection, field, models.PayloadSchemaType.KEYWORD)
        vectors = embedder.embed_documents([c.text for c in chunks])
        for i in range(0, len(chunks), 256):
            batch = chunks[i:i + 256]
            self.client.upsert(self.collection, points=[
                models.PointStruct(id=str(uuid.uuid5(uuid.NAMESPACE_URL, c.chunk_id)),
                                   vector=vectors[i + j].tolist(), payload=c.payload())
                for j, c in enumerate(batch)
            ])

    def search(self, qvec: np.ndarray, k: int, filters: dict | None = None) -> list[Hit]:
        flt = None
        if filters:
            flt = models.Filter(must=[
                models.FieldCondition(key=key, match=models.MatchAny(any=v if isinstance(v, list) else [v]))
                for key, v in filters.items() if v
            ])
        res = self.client.query_points(self.collection, query=qvec.tolist(), limit=k,
                                       query_filter=flt, with_payload=True).points
        return [Hit(Chunk(**p.payload), p.score, dense_score=p.score) for p in res]


_WORD = re.compile(r"\w+", re.U)


def bm25_tokens(text: str) -> list[str]:
    return [w for w in _WORD.findall(text.lower().replace("ё", "е")) if len(w) > 1]


class BM25Index:
    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks
        self.bm25 = BM25Okapi([bm25_tokens(c.text) for c in chunks])

    def search(self, query: str, k: int, filters: dict | None = None) -> list[Hit]:
        scores = self.bm25.get_scores(bm25_tokens(query))
        order = np.argsort(-scores)
        out = []
        for i in order:
            c = self.chunks[i]
            if filters and any(v and getattr(c, key) not in (v if isinstance(v, list) else [v])
                               for key, v in filters.items()):
                continue
            out.append(Hit(c, float(scores[i])))
            if len(out) >= k:
                break
        return out


def rrf(result_lists: list[list[Hit]], k: int, c: int = 60) -> list[Hit]:
    """Reciprocal Rank Fusion: score = Σ 1 / (c + rank)."""
    fused: dict[str, tuple[float, Hit]] = {}
    for hits in result_lists:
        for rank, h in enumerate(hits, 1):
            prev = fused.get(h.chunk.chunk_id)
            score = (prev[0] if prev else 0.0) + 1.0 / (c + rank)
            keep = prev[1] if prev and prev[1].dense_score is not None else h
            fused[h.chunk.chunk_id] = (score, keep)
    ranked = sorted(fused.values(), key=lambda x: -x[0])[:k]
    return [Hit(h.chunk, s, dense_score=h.dense_score) for s, h in ranked]


class Retriever:
    def __init__(self, index: VectorIndex, embedder: Embedder, bm25: BM25Index | None = None,
                 mode: str = "dense"):
        self.index, self.embedder, self.bm25, self.mode = index, embedder, bm25, mode

    def retrieve(self, query: str, k: int, filters: dict | None = None) -> list[Hit]:
        if self.mode == "bm25":
            return self.bm25.search(query, k, filters)
        dense = self.index.search(self.embedder.embed_query(query), k, filters)
        if self.mode == "dense":
            return dense
        return rrf([dense, self.bm25.search(query, k, filters)], k)
