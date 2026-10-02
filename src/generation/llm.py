"""Клиент LLM: Groq и Gemini через OpenAI-совместимый HTTP API + кэш ответов.

Запросы идут напрямую через requests (POST /chat/completions) — без SDK openai,
чья нативная зависимость (jiter) блокируется Smart App Control на Windows.

Кэш (SQLite) делает эксперименты воспроизводимыми и экономит бесплатную квоту:
повторный прогон той же конфигурации не тратит запросы.
"""
from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

import requests

from src.config import get_secret

log = logging.getLogger(__name__)

PROVIDERS = {
    "groq": ("https://api.groq.com/openai/v1", "GROQ_API_KEY"),
    "gemini": ("https://generativelanguage.googleapis.com/v1beta/openai/", "GEMINI_API_KEY"),
}


@dataclass
class LLMResponse:
    text: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_sec: float = 0.0
    cached: bool = False


class LLM:
    def __init__(self, provider: str, model: str, cache_dir: Path, temperature: float = 0.0,
                 max_tokens: int = 800, extra: dict | None = None):
        base_url, key_name = PROVIDERS[provider]
        key = get_secret(key_name)
        if not key:
            raise RuntimeError(f"не найден ключ {key_name} (см. .env.example)")
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.session = requests.Session()
        self.session.headers.update({"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
        self.provider, self.model = provider, model
        self.temperature, self.max_tokens, self.extra = temperature, max_tokens, extra or {}
        cache_dir.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(cache_dir / "llm_cache.sqlite", check_same_thread=False)
        self.db.execute("CREATE TABLE IF NOT EXISTS llm (h TEXT PRIMARY KEY, response TEXT)")

    @property
    def name(self) -> str:
        return f"{self.provider}:{self.model}"

    def chat(self, messages: list[dict], use_cache: bool = True, json_mode: bool = False) -> LLMResponse:
        req = {"model": self.model, "messages": messages, "temperature": self.temperature,
               "max_tokens": self.max_tokens, **self.extra}
        if json_mode:
            req["response_format"] = {"type": "json_object"}
        h = hashlib.sha256(json.dumps([self.provider, req], ensure_ascii=False, sort_keys=True).encode()).hexdigest()
        if use_cache:
            row = self.db.execute("SELECT response FROM llm WHERE h=?", (h,)).fetchone()
            if row:
                return LLMResponse(**json.loads(row[0]) | {"cached": True})

        t0 = time.perf_counter()
        for attempt in range(8):
            try:
                r = self.session.post(self.url, json=req, timeout=180)
            except (requests.ConnectionError, requests.Timeout) as e:
                log.warning("%s: сетевая ошибка %s, повтор", self.name, e)
                time.sleep(5 * (attempt + 1))
                continue
            if r.status_code == 429 or r.status_code >= 500:     # бесплатный тариф: ждём и повторяем
                wait = _retry_after(r) or min(60, 5 * 2 ** attempt)
                log.warning("%s: HTTP %s, жду %.0f с", self.name, r.status_code, wait)
                time.sleep(wait)
                continue
            if r.status_code != 200:
                raise RuntimeError(f"{self.name}: HTTP {r.status_code}: {r.text[:300]}")
            data = r.json()
            break
        else:
            raise RuntimeError(f"{self.name}: лимит запросов не снялся")
        usage = data.get("usage") or {}
        out = LLMResponse(
            text=(data["choices"][0]["message"].get("content") or "").strip(), model=self.name,
            prompt_tokens=usage.get("prompt_tokens", 0) or 0,
            completion_tokens=usage.get("completion_tokens", 0) or 0,
            latency_sec=round(time.perf_counter() - t0, 2),
        )
        if out.text:
            self.db.execute("INSERT OR REPLACE INTO llm VALUES (?, ?)",
                            (h, json.dumps({k: v for k, v in out.__dict__.items() if k != "cached"})))
            self.db.commit()
        return out


def _retry_after(r: requests.Response) -> float | None:
    try:
        return min(120.0, float(r.headers.get("retry-after")))
    except (TypeError, ValueError):
        return None


def build_llm(cfg: dict, section: str = "generation", **overrides) -> LLM:
    from src import config
    g = {**cfg[section], **overrides}
    models_cfg = cfg.get("llm_models", {}).get(g["model"], {})
    return LLM(g["provider"], g["model"], config.path(cfg, "cache"),
               temperature=g.get("temperature", 0.0), max_tokens=g.get("max_tokens", 800),
               extra=models_cfg.get("extra"))
