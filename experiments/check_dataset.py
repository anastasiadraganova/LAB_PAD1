"""Проверка датасета: типы вопросов и ссылки эталона (doc_id + секция) на реальный корпус.

    python experiments/check_dataset.py
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import config  # noqa: E402
from src.evaluation import QUESTION_TYPES, load_questions  # noqa: E402
from src.models import load_documents  # noqa: E402


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    cfg = config.load_config()
    docs = {d.doc_id: d for d in load_documents(config.path(cfg, "documents"))}
    qs = load_questions(cfg)
    errors = []
    for q in qs:
        if q.type not in QUESTION_TYPES:
            errors.append(f"{q.id}: неизвестный тип {q.type}")
        if q.answerable != bool(q.gold):
            errors.append(f"{q.id}: у отвечаемого вопроса должен быть эталон, у неотвечаемого — нет")
        for g in q.gold:
            d = docs.get(g["doc_id"])
            if d is None:
                errors.append(f"{q.id}: нет документа {g['doc_id']}")
                continue
            titles = [s.title for s in d.sections]
            secs = g["section"] if isinstance(g["section"], list) else [g["section"]]
            for s in secs:
                if not any(t == s or t.startswith(s + " / ") for t in titles):
                    errors.append(f"{q.id}: в «{d.title}» нет секции «{s}»")
    print("Вопросов:", len(qs), dict(Counter(q.type for q in qs)))
    print("\n".join(errors) if errors else "Все ссылки эталона корректны.")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
