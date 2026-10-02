"""Эксперименты с поиском (без LLM): chunking → embeddings → Top-K → reranker → пороги.

Этапы последовательные: лучший вариант этапа (по MRR, затем Recall@5) фиксируется
и используется дальше. Результаты: experiments/results/retrieval_<этап>.csv

    python experiments/run_retrieval.py            # все этапы
    python experiments/run_retrieval.py --only s2  # один этап (берёт лучшие из best_config.json)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import load_config, merged  # noqa: E402
from src.evaluation import load_questions  # noqa: E402
from src.evaluation.runner import KS, eval_refusal_gate, eval_retrieval  # noqa: E402
from src.evaluation.table import avg, fmt_table, summarize, write_csv  # noqa: E402
from src.pipeline import RAGPipeline  # noqa: E402

RES = ROOT / "experiments" / "results"
RES.mkdir(parents=True, exist_ok=True)
METRICS = [f"{m}@{k}" for m in ("hit", "recall", "precision", "ndcg") for k in KS] + ["mrr", "latency"]
STATE = RES / "best_config.json"

# «чистый» поиск: без фильтров и reranker'а, чтобы сравнивать только retriever
RAW = {"filtering": {"score_threshold": 0.0, "min_chars": 0, "dedup": False, "max_per_doc": 0},
       "reranker": {"enabled": False, "top_n": 20}, "retrieval": {"mode": "dense", "top_k": 20, "filters": {}}}

CHUNKINGS = [  # (strategy, size, overlap)
    ("fixed", 256, 0), ("fixed", 256, 64), ("paragraph", 256, 0),
    ("section", 128, 0), ("section", 256, 0), ("section", 512, 0),
]


def run(cfg: dict, questions, label: dict, stage="candidates") -> dict:
    t0 = time.perf_counter()
    rag = RAGPipeline(cfg, with_llm=False)
    build = time.perf_counter() - t0
    rag.retrieve("прогрев моделей", cfg)              # загрузка моделей не должна попадать в latency
    rows = eval_retrieval(rag, questions, cfg, stage=stage)
    row = {**label, "n_chunks": len(rag.chunks),
           "avg_tokens": round(sum(c.n_tokens for c in rag.chunks) / len(rag.chunks), 1),
           "build_sec": round(build, 1), **summarize(rows, METRICS)}
    print(json.dumps(row, ensure_ascii=False))
    return row


def best(rows: list[dict]) -> dict:
    return max(rows, key=lambda r: (round(r["mrr"], 3), r["recall@5"]))


def save(name: str, rows: list[dict], cols: list[str] | None = None) -> None:
    write_csv(RES / f"retrieval_{name}.csv", rows)
    print(fmt_table(rows, cols))


def main(only: str | None) -> None:
    base = load_config()
    qs = load_questions(base)
    st = json.loads(STATE.read_text(encoding="utf-8")) if STATE.exists() else {}
    st.setdefault("chunking", {"strategy": "section", "size": 256, "overlap": 0, "prefix_title": True})
    st.setdefault("embedding", "e5-small")
    want = (lambda s: only is None or only == s)

    def save_state():
        STATE.write_text(json.dumps(st, ensure_ascii=False, indent=2), encoding="utf-8")

    def cfg_with(**kw):
        c = merged(base, **RAW)
        return merged(c, chunking=kw.get("chunking", st["chunking"]),
                      embedding={"model": kw.get("embedding", st["embedding"])},
                      retrieval={"mode": "dense", "top_k": kw.get("top_k", 20), "filters": {}})

    short = ["hit@1", "hit@5", "mrr", "recall@5", "ndcg@5", "n_chunks", "avg_tokens"]

    # S1: стратегия chunking (e5-small, dense)
    if want("s1"):
        rows = []
        for strategy, size, overlap in CHUNKINGS:
            ch = {"strategy": strategy, "size": size, "overlap": overlap, "prefix_title": True}
            rows.append(run(cfg_with(chunking=ch, embedding="e5-small"), qs,
                            {"chunking": f"{strategy}-{size}" + (f"+ov{overlap}" if overlap else ""),
                             "strategy": strategy, "size": size, "overlap": overlap}))
        save("s1_chunking", rows, ["chunking", *short])
        b = best(rows)
        st["chunking"] = {"strategy": b["strategy"], "size": int(b["size"]), "overlap": int(b["overlap"]),
                          "prefix_title": True}
        save_state()

    # S2: embedding-модели на лучшем chunking; S3: Top-K из того же прогона
    if want("s2"):
        rows = [run(cfg_with(embedding=e), qs, {"embedding": e}) for e in ("e5-small", "bge-m3")]
        save("s2_embeddings", rows, ["embedding", *short, "build_sec", "latency"])
        st["embedding"] = best(rows)["embedding"]
        save_state()
        topk = [{"embedding": r["embedding"], "k": k, "hit": r[f"hit@{k}"], "recall": r[f"recall@{k}"],
                 "precision": r[f"precision@{k}"], "ndcg": r[f"ndcg@{k}"]} for r in rows for k in KS]
        save("s3_topk", topk)

    # S4: reranker (кандидаты top-10 → rerank → top-5), метрики на итоговом контексте
    if want("s4"):
        rows = []
        for rr in (None, "bge-m3"):
            c = merged(cfg_with(top_k=10), reranker={"enabled": rr is not None, "model": rr or "bge-m3", "top_n": 5})
            rows.append(run(c, qs, {"reranker": rr or "none", "candidates_k": 10}, stage="final"))
        save("s4_reranker", rows, ["reranker", "hit@1", "hit@3", "hit@5", "mrr", "recall@5", "ndcg@5", "latency"])
        # в LLM уходят все top-5 сразу, порядок внутри контекста почти не важен →
        # главное — полнота контекста (Recall@5), затем MRR
        b = max(rows, key=lambda r: (round(r["recall@5"], 3), r["mrr"]))
        st["reranker"] = {"model": b["reranker"], "candidates_k": 10}
        save_state()

    # S5: пороги cosine и reranker — по сохранённым скорам, отвечаемые vs неотвечаемые
    if want("s5"):
        c = merged(cfg_with(top_k=10),
                   reranker={"enabled": True, "model": "bge-m3", "top_n": 5, "min_score": 0.0},
                   filtering={"score_threshold": 0.0, "min_chars": 60, "dedup": True, "max_per_doc": 0})
        scores = eval_refusal_gate(RAGPipeline(c, with_llm=False), qs, c)
        write_csv(RES / "retrieval_s5_scores.csv", scores)
        ans = [r for r in scores if r["answerable"]]
        una = [r for r in scores if not r["answerable"]]
        rows = []
        for kind, col in (("dense", "top_dense"), ("rerank", "top_rerank")):
            print(f"{col}: отвечаемые min={min(r[col] for r in ans):.3f} avg={avg(ans, col):.3f}; "
                  f"неотвечаемые max={max(r[col] for r in una):.3f} avg={avg(una, col):.3f}")
            vals = sorted(r[col] for r in scores)
            lo, hi = vals[0], vals[-1]
            grid = sorted({round(lo + (hi - lo) * i / 20, 3) for i in range(21)} | {0.0})
            for t in grid:
                kept = sum(r[col] >= t for r in ans) / len(ans)
                blocked = sum(r[col] < t for r in una) / len(una)
                rows.append({"kind": kind, "threshold": t, "answerable_kept": round(kept, 4),
                             "unanswerable_blocked": round(blocked, 4), "balanced_acc": round((kept + blocked) / 2, 4)})
        write_csv(RES / "retrieval_s5_thresholds.csv", rows)
        for kind in ("dense", "rerank"):
            # при равной balanced accuracy берём меньший порог — меньше риск отсечь нужный ответ
            b = max((r for r in rows if r["kind"] == kind), key=lambda r: (r["balanced_acc"], -r["threshold"]))
            st[f"{kind}_gate"] = b["threshold"]
            print(kind, b)
        save_state()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only")
    main(ap.parse_args().only)
