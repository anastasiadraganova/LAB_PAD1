"""Загрузка конфигурации (YAML) и секретов (.env)."""
from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parent.parent          # lab1/
DEFAULT_CONFIG = ROOT / "configs" / "default.yaml"

# .env ищем в lab1/ и в корне репозитория (на уровень выше)
ENV_FILES = [ROOT / ".env", ROOT.parent / ".env"]

# Допустимые имена переменных для ключей и префиксы самих ключей —
# чтобы работал и .env вида "GROQ_API_KEY=gsk_...", и файл, где ключи
# записаны просто строками без имён.
KEY_ALIASES = {
    "GROQ_API_KEY": (["GROQ_API_KEY", "GROQ_KEY", "GROQ"], "gsk_"),
    "GEMINI_API_KEY": (["GEMINI_API_KEY", "GOOGLE_API_KEY", "GEMINI_KEY", "GEMINI", "GOOGLE_AI_API_KEY"], "AIza"),
}


def load_config(path: str | Path | None = None, overrides: list[str] | None = None) -> dict[str, Any]:
    with open(path or DEFAULT_CONFIG, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    for item in overrides or []:
        key, _, raw = item.partition("=")
        set_by_path(cfg, key.strip(), yaml.safe_load(raw))
    return cfg


def set_by_path(cfg: dict, dotted: str, value: Any) -> None:
    node = cfg
    *parents, last = dotted.split(".")
    for p in parents:
        node = node.setdefault(p, {})
    node[last] = value


def merged(cfg: dict, **section_overrides: dict) -> dict:
    """Копия конфига с переопределёнными секциями (удобно для экспериментов)."""
    out = copy.deepcopy(cfg)
    for section, values in section_overrides.items():
        out.setdefault(section, {}).update(values)
    return out


def path(cfg: dict, name: str) -> Path:
    p = ROOT / cfg["paths"][name]
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def _read_env_files() -> tuple[dict[str, str], list[str]]:
    named: dict[str, str] = {}
    bare: list[str] = []
    for env_file in ENV_FILES:
        if not env_file.exists():
            continue
        for k, v in dotenv_values(env_file).items():
            if v:
                named.setdefault(k.strip(), v.strip())
            elif k and k.strip():               # строка без "=": просто ключ
                bare.append(k.strip().strip("'\""))
    return named, bare


def get_secret(name: str) -> str | None:
    """Возвращает API-ключ по каноническому имени (GROQ_API_KEY / GEMINI_API_KEY)."""
    aliases, prefix = KEY_ALIASES[name]
    for alias in aliases:
        if os.environ.get(alias):
            return os.environ[alias]
    named, bare = _read_env_files()
    for alias in aliases:
        if named.get(alias):
            return named[alias]
    # ключ записан под другим именем или без имени — узнаём по префиксу
    for value in list(named.values()) + bare:
        if value.startswith(prefix):
            return value
    # имя переменной содержит слово-подсказку (GEMINI_TOKEN, MY_GOOGLE_KEY, ...)
    hints = {"GROQ_API_KEY": ("GROQ",), "GEMINI_API_KEY": ("GEMINI", "GOOGLE")}[name]
    for k, v in named.items():
        if any(h in k.upper() for h in hints):
            return v
    # последняя попытка: единственная строка без имени, не похожая на другие ключи
    other_prefixes = tuple(p for n, (_, p) in KEY_ALIASES.items() if n != name)
    leftovers = [b for b in bare if not b.startswith(other_prefixes)]
    if len(leftovers) == 1:
        return leftovers[0]
    return None


def describe_secrets() -> dict[str, str]:
    """Статус ключей без раскрытия значений (для `cli check`)."""
    out = {}
    for name in KEY_ALIASES:
        v = get_secret(name)
        out[name] = f"найден ({v[:4]}…, длина {len(v)})" if v else "НЕ найден"
    named, bare = _read_env_files()
    out["переменные в .env"] = ", ".join(named) or "—"
    out["строк без имени"] = str(len(bare))
    return out
