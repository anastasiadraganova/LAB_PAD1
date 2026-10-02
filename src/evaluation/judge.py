"""LLM-as-a-Judge: оценка ответа по трём критериям (шкала 1–5).

Судья — модель другого семейства (Gemini), чем генератор (Groq: gpt-oss),
чтобы снизить эффект «модель хвалит свои ответы».
"""
from __future__ import annotations

import json
import re

from src.generation.llm import LLM
from src.generation.prompts import format_context
from src.retrieval import Hit

JUDGE_PROMPT = """Ты — строгий эксперт, оценивающий ответы RAG-системы о Солнечной системе.
Дано: вопрос, эталонный ответ, контекст (фрагменты, которые получила система) и ответ системы.

Оцени ответ системы по трём критериям, целыми числами от 1 до 5:
- correctness: насколько ответ совпадает по смыслу с эталоном (5 — полностью верен и полон; 3 — частично; 1 — неверен или отказ при наличии ответа в эталоне).
- faithfulness: все ли утверждения ответа подтверждаются контекстом (5 — всё подтверждено; 1 — ответ в основном выдуман/не следует из контекста). Отказ «недостаточно информации» считается faithful (5).
- relevancy: отвечает ли ответ именно на заданный вопрос, без лишнего (5 — точно по делу; 1 — не о том).

Верни ТОЛЬКО JSON: {"correctness": n, "faithfulness": n, "relevancy": n, "comment": "краткое пояснение на русском"}"""


def judge_answer(judge: LLM, question: str, expected: str, hits: list[Hit], answer: str) -> dict:
    user = (f"Вопрос: {question}\n\nЭталонный ответ: {expected}\n\n"
            f"Контекст:\n{format_context(hits) if hits else '(пусто)'}\n\nОтвет системы: {answer}")
    resp = judge.chat([{"role": "system", "content": JUDGE_PROMPT}, {"role": "user", "content": user}],
                      json_mode=True)
    return parse_scores(resp.text)


def parse_scores(text: str) -> dict:
    m = re.search(r"\{.*\}", text, re.S)
    try:
        data = json.loads(m.group(0)) if m else {}
    except json.JSONDecodeError:
        data = {}
    out = {}
    for k in ("correctness", "faithfulness", "relevancy"):
        try:
            out[k] = max(1, min(5, int(data.get(k))))
        except (TypeError, ValueError):
            out[k] = None
    out["comment"] = str(data.get("comment", ""))[:400]
    return out
