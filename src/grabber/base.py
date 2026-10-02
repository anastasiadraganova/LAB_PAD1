"""Общий интерфейс источника и инкрементальный прогон grabber'а."""
from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from tqdm import tqdm

from src.grabber.http import HttpClient
from src.grabber.manifest import Manifest, now
from src.models import Document, safe_name

log = logging.getLogger(__name__)


@dataclass
class ItemRef:
    """Ссылка на документ у источника (результат листинга, без скачивания контента)."""
    doc_id: str
    title: str
    url: str
    version_hint: str | None = None     # если источник отдаёт версию дёшево (revid) — не качаем зря
    meta: dict = field(default_factory=dict)


class Source(ABC):
    name: str

    def __init__(self, http: HttpClient, raw_dir: Path):
        self.http = http
        self.raw_dir = raw_dir / self.name
        self.raw_dir.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def list_items(self) -> list[ItemRef]:
        """Перечислить документы источника."""

    @abstractmethod
    def download(self, ref: ItemRef) -> dict:
        """Скачать сырой ответ источника (JSON/HTML)."""

    @abstractmethod
    def parse(self, ref: ItemRef, raw: dict) -> Document:
        """Превратить сырой ответ в Document (без сети)."""

    def fetch(self, ref: ItemRef, reparse: bool = False) -> Document:
        """reparse=True — взять сохранённый сырой ответ вместо скачивания:
        после исправления парсера не нужно заново нагружать источник."""
        raw_file = self.raw_dir / f"{safe_name(ref.doc_id)}.json"
        if reparse and raw_file.exists():
            raw = json.loads(raw_file.read_text(encoding="utf-8"))
        else:
            raw = self.download(ref)
            raw_file.write_text(json.dumps(raw, ensure_ascii=False), encoding="utf-8")
        return self.parse(ref, raw)


def run_source(source: Source, manifest: Manifest, docs_dir: Path, force: bool = False,
               limit: int | None = None, reparse: bool = False) -> Counter:
    started = now()
    stats: Counter = Counter()
    refs = source.list_items()
    if limit:
        refs = refs[:limit]
    stats["listed"] = len(refs)
    seen: set[str] = set()

    for ref in tqdm(refs, desc=source.name, unit="doc"):
        if ref.doc_id in seen:                       # один документ попал в листинг дважды
            stats["duplicate"] += 1
            continue
        seen.add(ref.doc_id)
        rec = manifest.get(ref.doc_id)
        doc_file = docs_dir / f"{safe_name(ref.doc_id)}.json"

        # 1) дешёвая проверка по версии источника: ничего не скачиваем
        if (not force and not reparse and rec and rec.status == "active" and ref.version_hint
                and rec.version == ref.version_hint and doc_file.exists()):
            manifest.touch(ref.doc_id)
            stats["skipped"] += 1
            continue

        try:
            doc = source.fetch(ref, reparse=reparse)
        except Exception as e:                       # ошибка одного документа не роняет прогон
            log.error("%s: %s", ref.doc_id, e)
            manifest.mark_failed(ref.doc_id, source.name, ref.title, ref.url, repr(e))
            stats["failed"] += 1
            continue

        h = doc.content_hash
        # 2) проверка по содержимому: версия могла смениться, а текст — нет
        if rec and rec.status == "active" and rec.content_hash == h and doc_file.exists():
            manifest.upsert(doc.doc_id, source.name, doc.title, doc.url, doc.version, h, changed=False)
            stats["unchanged"] += 1
            continue

        # 3) защита от дубликатов: тот же контент под другим id
        dup = manifest.find_by_hash(h, exclude=doc.doc_id)
        if dup:
            manifest.upsert(doc.doc_id, source.name, doc.title, doc.url, doc.version, h,
                            changed=False, status="duplicate", duplicate_of=dup)
            stats["duplicate"] += 1
            continue

        doc.save(docs_dir)
        manifest.upsert(doc.doc_id, source.name, doc.title, doc.url, doc.version, h, changed=True)
        stats["updated" if rec and rec.status == "active" else "new"] += 1

    if not limit:                                    # при частичном прогоне удалённые не ищем
        removed = manifest.mark_removed(source.name, seen)
        for doc_id in removed:
            (docs_dir / f"{safe_name(doc_id)}.json").unlink(missing_ok=True)
        stats["removed"] = len(removed)
    manifest.log_run(source.name, started, stats, source.http.requests_made)
    return stats
