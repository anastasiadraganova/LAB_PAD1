"""Графики по результатам экспериментов → experiments/figures/*.png (matplotlib).

    python experiments/make_figures.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))

from src.evaluation.table import read_csv  # noqa: E402

RES, FIG = ROOT / "results", ROOT / "figures"
FIG.mkdir(exist_ok=True)

# категориальная палитра в фиксированном порядке (проверена на CVD-различимость)
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2,
    "ytick.color": INK2, "axes.titlecolor": INK, "axes.titlesize": 12, "axes.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "axes.grid.axis": "y",
    "grid.color": GRID, "grid.linewidth": 0.8, "legend.frameon": False, "axes.axisbelow": True,
})


def bars(name: str, title: str, groups: list[str], series: dict[str, list[float]], ymax=1.0,
         subtitle: str = "", fmt="{:.2f}") -> None:
    fig, ax = plt.subplots(figsize=(8, 4.2), dpi=150)
    n = len(series)
    width = 0.8 / n
    x = np.arange(len(groups))
    for i, (label, vals) in enumerate(series.items()):
        pos = x - 0.4 + width * (i + 0.5)
        b = ax.bar(pos, vals, width - 0.02, label=label, color=SERIES[i % len(SERIES)],
                   edgecolor=SURFACE, linewidth=1)
        ax.bar_label(b, [fmt.format(v) for v in vals], fontsize=8, color=INK2, padding=2)
    ax.set_xticks(x, groups)
    ax.set_ylim(0, ymax * 1.08)
    ax.set_title(title, loc="left", pad=20 if subtitle else 6)
    if subtitle:
        ax.text(0, 1.015, subtitle, transform=ax.transAxes, fontsize=9, color=INK2, va="bottom")
    if n > 1:
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=n)
    fig.tight_layout()
    fig.savefig(FIG / f"{name}.png")
    plt.close(fig)


def lines(name: str, title: str, xs: list, series: dict[str, list[float]], xlabel: str, ymax=1.0) -> None:
    fig, ax = plt.subplots(figsize=(7, 4), dpi=150)
    for i, (label, vals) in enumerate(series.items()):
        c = SERIES[i % len(SERIES)]
        ax.plot(xs, vals, color=c, linewidth=2, marker="o", markersize=5, label=label)
        ax.annotate(f"{vals[-1]:.2f}", (xs[-1], vals[-1]), xytext=(6, 0), textcoords="offset points",
                    color=INK2, fontsize=8, va="center")
    ax.set_ylim(0, ymax * 1.05)
    ax.set_xlabel(xlabel)
    ax.set_title(title, loc="left")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG / f"{name}.png")
    plt.close(fig)


def fig_chunking():
    rows = read_csv(RES / "retrieval_s1_chunking.csv")
    if rows:
        bars("s1_chunking", "S1. Стратегия chunking", [r["chunking"] for r in rows],
             {"MRR": [r["mrr"] for r in rows], "Hit@5": [r["hit@5"] for r in rows],
              "Recall@5": [r["recall@5"] for r in rows]},
             subtitle="e5-small, dense, кандидаты до reranker, 25 отвечаемых вопросов")


def fig_embeddings():
    rows = read_csv(RES / "retrieval_s2_embeddings.csv")
    if rows:
        m = ["hit@1", "hit@5", "mrr", "recall@5", "ndcg@5"]
        bars("s2_embeddings", "S2. Embedding-модель", m, {r["embedding"]: [r[k] for k in m] for r in rows},
             subtitle="лучший chunking из S1")


def fig_topk():
    rows = read_csv(RES / "retrieval_s3_topk.csv")
    if not rows:
        return
    st = json.loads((RES / "best_config.json").read_text(encoding="utf-8"))
    d = [r for r in rows if r["embedding"] == st["embedding"]]
    ks = [r["k"] for r in d]
    lines("s3_topk", f"S3. Top-K ({st['embedding']})", ks,
          {"Recall@K": [r["recall"] for r in d], "Hit@K": [r["hit"] for r in d],
           "Precision@K": [r["precision"] for r in d]}, "K — число найденных chunks")


def fig_reranker():
    rows = read_csv(RES / "retrieval_s4_reranker.csv")
    if rows:
        m = ["hit@1", "hit@3", "mrr", "recall@5", "ndcg@5"]
        bars("s4_reranker", "S4. Reranker: итоговый контекст top-5", m,
             {("без reranker" if r["reranker"] == "none" else "bge-reranker-v2-m3 (10→5)"): [r[k] for k in m]
              for r in rows})


def fig_thresholds():
    rows = read_csv(RES / "retrieval_s5_thresholds.csv")
    for kind, title in (("dense", "cosine similarity"), ("rerank", "reranker score")):
        d = [r for r in rows if r["kind"] == kind and r["threshold"] > 0]
        if d:
            lines(f"s5_threshold_{kind}", f"S5. Порог {title}", [r["threshold"] for r in d],
                  {"отвечаемые: контекст сохранён": [r["answerable_kept"] for r in d],
                   "неотвечаемые: отсечены": [r["unanswerable_blocked"] for r in d]}, "порог")


def fig_generation():
    rows = read_csv(RES / "generation_summary.csv")
    if not rows:
        return
    tags = [r["tag"].replace("/", "\n") for r in rows]
    bars("generation_judge", "LLM × промпт: оценка судьи (1–5)", tags,
         {k: [r[k] for r in rows] for k in ("correctness", "faithfulness", "relevancy")}, ymax=5,
         subtitle="судья — Gemini; 25 отвечаемых + 5 неотвечаемых вопросов", fmt="{:.1f}")
    bars("generation_refusal", "LLM × промпт: поведение при отказе", tags,
         {"ответил, когда ответ есть": [r["answered_ok"] for r in rows],
          "отказал, когда ответа нет": [r["refused_ok"] for r in rows],
          "доля ответов со ссылками": [r["cited_share"] for r in rows]})


if __name__ == "__main__":
    for f in (fig_chunking, fig_embeddings, fig_topk, fig_reranker, fig_thresholds, fig_generation):
        try:
            f()
        except Exception as e:           # один сломанный график не должен мешать остальным
            print("ERROR", f.__name__, repr(e), file=sys.stderr)
    print("figures:", sorted(p.name for p in FIG.glob("*.png")))
