"""Прогон датасета через pipeline и подсчёт метрик для одной конфигурации.
Результат — список словарей (по строке на вопрос)."""
from __future__ import annotations

import time

from tqdm import tqdm

from src.evaluation import Question
from src.evaluation.judge import judge_answer
from src.evaluation.metrics import is_relevant, retrieval_metrics
from src.generation import generate
from src.generation.prompts import REFUSAL
from src.pipeline import RAGPipeline

KS = (1, 3, 5, 10, 20)


def eval_retrieval(rag: RAGPipeline, questions: list[Question], cfg: dict | None = None,
                   use_en: bool = False, stage: str = "final", level: str = "section") -> list[dict]:
    """Метрики поиска по отвечаемым вопросам.

    stage="candidates" — ранжирование retriever'а (top_k кандидатов, без фильтров/rerank);
    stage="final"      — то, что реально уйдёт в LLM (после фильтров и reranker'а).
    """
    cfg = cfg or rag.cfg
    rows = []
    for q in questions:
        if not q.answerable:
            continue
        text = q.question_en if use_en and q.question_en else q.question
        t0 = time.perf_counter()
        res = rag.retrieve(text, cfg)
        latency = time.perf_counter() - t0
        ranked = res.candidates if stage == "candidates" else res.final
        m = retrieval_metrics([h.chunk for h in ranked], q.gold, KS, level)
        rows.append({"id": q.id, "type": q.type, "latency": latency, "n_final": len(res.final), **m})
    return rows


def eval_refusal_gate(rag: RAGPipeline, questions: list[Question], cfg: dict) -> list[dict]:
    """Для порогов: максимальные скоры у отвечаемых и неотвечаемых вопросов."""
    rows = []
    for q in questions:
        res = rag.retrieve(q.question, cfg)
        top_dense = max((h.dense_score or 0 for h in res.candidates), default=0)
        top_rerank = max((h.rerank_score or 0 for h in res.final), default=0)
        gold_ctx = any(is_relevant(h.chunk, q.gold) for h in res.final) if q.answerable else None
        rows.append({"id": q.id, "type": q.type, "answerable": q.answerable, "n_final": len(res.final),
                     "top_dense": round(top_dense, 4), "top_rerank": round(top_rerank, 4),
                     "gold_in_context": gold_ctx})
    return rows


def eval_generation(rag: RAGPipeline, questions: list[Question], llm, judge, cfg: dict | None = None,
                    tag: str = "") -> list[dict]:
    """Полный прогон: ответ LLM + оценка судьёй + правило отказа."""
    cfg = cfg or rag.cfg
    rows = []
    for q in tqdm(questions, desc=f"gen {tag}", unit="q"):
        res = rag.retrieve(q.question, cfg)
        t0 = time.perf_counter()
        ans = generate(llm, cfg["generation"]["prompt"], q.question, res.final)
        latency = time.perf_counter() - t0
        scores = judge_answer(judge, q.question, q.expected_answer if q.answerable else REFUSAL,
                              res.final, ans.text)
        gold_ctx = any(is_relevant(h.chunk, q.gold) for h in res.final) if q.answerable else None
        rows.append({
            "tag": tag, "id": q.id, "type": q.type, "answerable": q.answerable,
            "refused": ans.refused,
            # правильное поведение: отвечаемый → не отказ; неотвечаемый → отказ
            "refusal_correct": ans.refused != q.answerable,
            "gold_in_context": gold_ctx, "n_context": len(res.final), "n_sources": len(ans.sources),
            "cited": bool(ans.sources and ans.sources[0].get("cited")),
            "latency": ans.llm.latency_sec if ans.llm else round(latency, 3),
            "prompt_tokens": ans.llm.prompt_tokens if ans.llm else 0,
            "completion_tokens": ans.llm.completion_tokens if ans.llm else 0,
            **{k: scores[k] for k in ("correctness", "faithfulness", "relevancy")},
            "judge_comment": scores["comment"], "question": q.question, "answer": ans.text,
            "expected": q.expected_answer,
        })
    return rows
