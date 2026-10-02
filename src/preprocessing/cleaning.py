"""Очистка текста: HTML-разметка, мусор вики, дубли строк, нормализация."""
from __future__ import annotations

import html
import re
import unicodedata

_TAG = re.compile(r"<[^>]+>")
_BR = re.compile(r"<br\s*/?>", re.I)
_H = re.compile(r"<h\d[^>]*>(.*?)</h\d>", re.I | re.S)
_ZERO_WIDTH = re.compile(r"[​-‏⁠﻿­]")
_SPACES = re.compile(r"[ \t   ]+")
_MANY_NL = re.compile(r"\n{3,}")

# строки-мусор, которые остаются от вики после парсинга
NOISE_LINE_PATTERNS = [
    re.compile(r"^Lua error", re.I),
    re.compile(r"^File:", re.I),
    re.compile(r"^\[?edit\]?$", re.I),
    re.compile(r"^[\W_]{1,3}$"),                 # одиночные символы: "(", "†", ",", ":"
    re.compile(r"^\{\\displaystyle"),            # остатки TeX-формул
    re.compile(r"^(Play|Link)▶?$", re.I),
]

QUOTES = {"«": '"', "»": '"', "„": '"', "“": '"', "”": '"', "‘": "'", "’": "'", "′": "'"}
DASHES = {"‒": "-", "–": "–", "—": "—", "−": "-"}


def strip_html(text: str) -> str:
    """Официальный источник отдаёт описания с <h1>, <br>, <b>, <font>."""
    text = _BR.sub("\n", text)
    text = _H.sub(lambda m: m.group(1).strip() + ": ", text)
    text = _TAG.sub("", text)
    return html.unescape(text)


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = _ZERO_WIDTH.sub("", text)
    text = "".join(QUOTES.get(c, DASHES.get(c, c)) for c in text)
    text = _SPACES.sub(" ", text)
    return text


def is_noise_line(line: str) -> bool:
    return any(p.search(line) for p in NOISE_LINE_PATTERNS)


def merge_orphan_lines(lines: list[str]) -> list[str]:
    """После HTML-парсинга инлайн-ссылки иногда оказываются на отдельных строках:
    'thrown at him and his' / 'Allies' / ', returning them...'. Склеиваем короткие
    фрагменты с предыдущей строкой, если предыдущая не закончена."""
    out: list[str] = []
    for ln in lines:
        table_row = " | " in ln or (out and " | " in out[-1])
        if out and not table_row and (ln[:1] in ",.;:)!?%" or (
                len(ln) < 25 and not out[-1].endswith((".", ":", "!", "?")) and not ln.startswith(("-", "•")))):
            sep = "" if ln[:1] in ",.;:)!?%" else " "
            out[-1] = out[-1] + sep + ln
        else:
            out.append(ln)
    return out


def clean_text(text: str, merge_orphans: bool = True) -> str:
    text = normalize(strip_html(text))
    lines, seen = [], set()
    for raw in text.split("\n"):
        ln = raw.strip()
        if not ln or is_noise_line(ln):
            continue
        key = ln.lower()
        if len(ln) > 30 and key in seen:           # повтор длинной строки (табберы вики)
            continue
        seen.add(key)
        lines.append(ln)
    if merge_orphans:
        lines = merge_orphan_lines(lines)
    return _MANY_NL.sub("\n\n", "\n".join(lines)).strip()
