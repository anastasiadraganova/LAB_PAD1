"""Evaluation: датасет вопросов, метрики поиска, LLM-as-a-Judge."""
from __future__ import annotations

import json
from dataclasses import dataclass, field

from src import config

QUESTION_TYPES = {
    "factoid": "простой фактологический",
    "specific": "поиск конкретной информации (число, условие)",
    "multi_doc": "по нескольким документам",
    "reasoning": "требует понимания контекста",
    "unanswerable": "ответа нет в базе",
}


@dataclass
class Question:
    id: str
    question: str                     # на русском
    type: str
    expected_answer: str
    gold: list[dict] = field(default_factory=list)   # [{"doc_id": ..., "section": ...}]
    question_en: str = ""             # перевод — для эксперимента с языком запроса
    notes: str = ""

    @property
    def answerable(self) -> bool:
        return self.type != "unanswerable"


def dataset_path(cfg: dict):
    return config.path(cfg, "data") / "eval" / "questions.jsonl"


def load_questions(cfg: dict) -> list[Question]:
    p = dataset_path(cfg)
    return [Question(**json.loads(line)) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
