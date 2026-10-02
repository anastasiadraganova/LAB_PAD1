"""Полный RAG pipeline: query → retrieve → filter → rerank → LLM → answer + sources."""
from __future__ import annotations

import time
from dataclasses import dataclass, field

from src import config
from src.embeddings import Embedder
from src.filtering import FilterConfig, apply_filters
from src.generation import Answer, build_llm, generate
from src.generation.prompts import REFUSAL
from src.preprocessing import build_chunks
from src.preprocessing.chunkers import ChunkingConfig
from src.reranking import Reranker
from src.retrieval import BM25Index, Hit, Retriever, VectorIndex


def chunking_config(cfg: dict) -> ChunkingConfig:
    return ChunkingConfig(**cfg["chunking"])


def collection_name(cfg: dict) -> str:
    return f"{chunking_config(cfg).slug}__{cfg['embedding']['model']}"


def build_index(cfg: dict, recreate: bool = False) -> tuple[VectorIndex, Embedder, list]:
    chunks = build_chunks(cfg, chunking_config(cfg))
    embedder = Embedder(cfg["embedding"]["model"], config.path(cfg, "cache"))
    index = VectorIndex(cfg, collection_name(cfg))
    index.build(chunks, embedder, recreate=recreate)
    return index, embedder, chunks


@dataclass
class RetrievalResult:
    candidates: list[Hit]              # после retriever'а
    filtered: list[Hit]                # после фильтров
    final: list[Hit]                   # после reranker'а → в LLM
    filter_log: dict
    timings: dict = field(default_factory=dict)


class RAGPipeline:
    def __init__(self, cfg: dict, with_llm: bool = True):
        self.cfg = cfg
        self.index, self.embedder, self.chunks = build_index(cfg)
        mode = cfg["retrieval"]["mode"]
        bm25 = BM25Index(self.chunks) if mode in ("bm25", "hybrid") else None
        self.retriever = Retriever(self.index, self.embedder, bm25, mode)
        self._rerankers: dict[str, Reranker] = {}
        self.llm = build_llm(cfg) if with_llm else None

    def reranker(self, key: str) -> Reranker:
        if key not in self._rerankers:
            self._rerankers[key] = Reranker(key)
        return self._rerankers[key]

    def retrieve(self, question: str, cfg: dict | None = None) -> RetrievalResult:
        cfg = cfg or self.cfg
        r, f, rr = cfg["retrieval"], cfg["filtering"], cfg["reranker"]
        t = {}
        t0 = time.perf_counter()
        candidates = self.retriever.retrieve(question, r["top_k"], r.get("filters") or None)
        t["retrieve"] = time.perf_counter() - t0

        filtered, flog = apply_filters(list(candidates), FilterConfig(**f))
        t0 = time.perf_counter()
        if rr["enabled"]:
            final = self.reranker(rr["model"]).rerank(question, filtered, rr["top_n"])
            if rr.get("min_score"):
                before = len(final)
                final = [h for h in final if h.rerank_score >= rr["min_score"]]
                flog["rerank_threshold"] = before - len(final)
        else:
            final = filtered[:rr["top_n"]]
        t["rerank"] = time.perf_counter() - t0
        return RetrievalResult(candidates, filtered, final, flog, t)

    def ask(self, question: str, cfg: dict | None = None, llm=None) -> tuple[Answer, RetrievalResult]:
        cfg = cfg or self.cfg
        res = self.retrieve(question, cfg)
        llm = llm or self.llm
        t0 = time.perf_counter()
        answer = generate(llm, cfg["generation"]["prompt"], question, res.final) if res.final \
            else Answer(REFUSAL, True)
        res.timings["generate"] = time.perf_counter() - t0
        return answer, res
