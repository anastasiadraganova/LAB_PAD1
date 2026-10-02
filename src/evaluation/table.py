"""Минимальные табличные утилиты на стандартной библиотеке (вместо pandas,
DLL которого блокирует Windows Smart App Control)."""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


def _parse(v: str):
    if v in ("True", "False"):
        return v == "True"
    if v == "":
        return None
    try:
        f = float(v)
        return int(f) if f.is_integer() and "." not in v and "e" not in v.lower() else f
    except ValueError:
        return v


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return [{k: _parse(v) for k, v in r.items()} for r in csv.DictReader(f)]


def avg(rows: list[dict], key: str) -> float:
    vals = [float(r[key]) for r in rows if r.get(key) is not None]
    return round(mean(vals), 4) if vals else float("nan")


def summarize(rows: list[dict], cols: list[str]) -> dict:
    return {c: avg(rows, c) for c in cols if rows and c in rows[0]}


def group_by(rows: list[dict], key: str) -> dict:
    out: dict = defaultdict(list)
    for r in rows:
        out[r[key]].append(r)
    return dict(out)


def fmt_table(rows: list[dict], cols: list[str] | None = None, digits: int = 3) -> str:
    """Markdown-таблица (для отчёта и вывода в консоль)."""
    if not rows:
        return "(пусто)"
    cols = cols or list(rows[0])

    def cell(v):
        return f"{v:.{digits}f}" if isinstance(v, float) else str(v)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(cell(r.get(c, "")) for c in cols) + " |" for r in rows]
    return "\n".join(lines)
