"""Точка входа: python -m src.cli <команда> [опции]

  check                 — проверить ключи API (без вывода значений)
  grab [--source X]     — собрать/обновить документы
"""
from __future__ import annotations

import argparse
import json
import logging
import sys

from src import config


def main(argv: list[str] | None = None) -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(prog="rag")
    parser.add_argument("--config", default=None, help="путь к YAML (по умолчанию configs/default.yaml)")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                        help="переопределить параметр конфига, напр. retrieval.top_k=10")
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("check", help="проверить наличие API-ключей")

    p = sub.add_parser("grab", help="собрать документы из источников")
    p.add_argument("--source", choices=["wikipedia"], help="только один источник")
    p.add_argument("--force", action="store_true", help="игнорировать версии и скачать всё заново")
    p.add_argument("--limit", type=int, help="ограничить число документов (для отладки)")
    p.add_argument("--reparse", action="store_true",
                   help="пересобрать документы из сохранённых сырых ответов (после правки парсера)")

    p = sub.add_parser("stats", help="статистика корпуса и chunks")

    p = sub.add_parser("index", help="построить chunks и векторный индекс для текущего конфига")
    p.add_argument("--recreate", action="store_true")

    p = sub.add_parser("ask", help="задать вопрос RAG-системе")
    p.add_argument("question")
    p.add_argument("--show-context", action="store_true")

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = config.load_config(args.config, args.set)

    if args.cmd == "check":
        for k, v in config.describe_secrets().items():
            print(f"{k}: {v}")
    elif args.cmd == "grab":
        from src.grabber import grab
        res = grab(cfg, only=args.source, force=args.force, limit=args.limit, reparse=args.reparse)
        summary = res.pop("_summary")
        print(json.dumps(res, ensure_ascii=False, indent=2))
        print("\nСостояние манифеста (source, status, count):")
        for row in summary:
            print("  ", *row)
    elif args.cmd == "stats":
        _stats(cfg)
    elif args.cmd == "index":
        from src.pipeline import build_index, collection_name
        index, _, chunks = build_index(cfg, recreate=args.recreate)
        print(f"{collection_name(cfg)}: {len(chunks)} chunks, в индексе {index.count()}")
    elif args.cmd == "ask":
        _ask(cfg, args.question, args.show_context)


def _stats(cfg: dict) -> None:
    from collections import Counter

    from src.models import load_documents
    from src.pipeline import chunking_config
    from src.preprocessing import build_chunks, clean_document
    raw = load_documents(config.path(cfg, "documents"))
    clean = [clean_document(d) for d in raw]
    by = Counter((d.source, d.entity_type) for d in raw)
    print("Документы:", len(raw))
    for (s, t), n in sorted(by.items()):
        print(f"   {s:9} {t:9} {n}")
    before, after = sum(len(d.text) for d in raw), sum(len(d.text) for d in clean)
    print(f"Символов до очистки: {before:,}, после: {after:,} (−{100 * (1 - after / before):.1f}%)")
    chunks = build_chunks(cfg, chunking_config(cfg))
    toks = sorted(c.n_tokens for c in chunks)
    print(f"Chunks ({chunking_config(cfg).slug}): {len(chunks)}, токенов: медиана {toks[len(toks) // 2]}, "
          f"среднее {sum(toks) / len(toks):.0f}, максимум {toks[-1]}")


def _ask(cfg: dict, question: str, show_context: bool) -> None:
    from src.pipeline import RAGPipeline
    rag = RAGPipeline(cfg)
    answer, res = rag.ask(question)
    print(answer.text)
    if answer.sources:
        print("\nИсточники:")
        for s in answer.sources:
            print(f"  [{s['n']}] {s['title']} — {s['section']}  ({s['url']})")
    if show_context:
        print("\nКонтекст (после rerank):")
        for i, h in enumerate(res.final, 1):
            print(f"  [{i}] dense={h.dense_score:.3f} rerank={h.rerank_score or 0:.3f}  {h.chunk.chunk_id}")
    print(f"\nфильтры: {res.filter_log}; время: "
          + ", ".join(f"{k}={v:.2f}s" for k, v in res.timings.items()))


if __name__ == "__main__":
    main()
