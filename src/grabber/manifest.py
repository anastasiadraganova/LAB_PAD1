"""Манифест grabber'а (SQLite): что скачано, какая версия, когда менялось.

Позволяет повторно запускать сбор: неизменённые документы пропускаются,
изменённые перезаписываются, исчезнувшие у источника помечаются removed.
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    doc_id        TEXT PRIMARY KEY,
    source        TEXT NOT NULL,
    title         TEXT,
    url           TEXT,
    version       TEXT,
    content_hash  TEXT,
    status        TEXT NOT NULL,          -- active | failed | removed | duplicate
    duplicate_of  TEXT,
    first_seen    TEXT,
    last_checked  TEXT,
    last_changed  TEXT,
    error         TEXT
);
CREATE INDEX IF NOT EXISTS idx_documents_hash ON documents(content_hash);
CREATE TABLE IF NOT EXISTS runs (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    source       TEXT,
    started_at   TEXT,
    finished_at  TEXT,
    listed INTEGER, new INTEGER, updated INTEGER, unchanged INTEGER,
    skipped INTEGER, duplicate INTEGER, failed INTEGER, removed INTEGER,
    http_requests INTEGER
);
"""


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class Record:
    doc_id: str
    source: str
    version: str | None
    content_hash: str | None
    status: str


class Manifest:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path)
        self.db.executescript(SCHEMA)

    def get(self, doc_id: str) -> Record | None:
        row = self.db.execute(
            "SELECT doc_id, source, version, content_hash, status FROM documents WHERE doc_id=?", (doc_id,)
        ).fetchone()
        return Record(*row) if row else None

    def find_by_hash(self, content_hash: str, exclude: str) -> str | None:
        row = self.db.execute(
            "SELECT doc_id FROM documents WHERE content_hash=? AND doc_id<>? AND status='active'",
            (content_hash, exclude),
        ).fetchone()
        return row[0] if row else None

    def upsert(self, doc_id, source, title, url, version, content_hash, changed: bool,
               status="active", duplicate_of=None) -> None:
        ts = now()
        self.db.execute(
            """INSERT INTO documents(doc_id, source, title, url, version, content_hash, status,
                                     duplicate_of, first_seen, last_checked, last_changed, error)
               VALUES(?,?,?,?,?,?,?,?,?,?,?,NULL)
               ON CONFLICT(doc_id) DO UPDATE SET
                   title=excluded.title, url=excluded.url, version=excluded.version,
                   content_hash=excluded.content_hash, status=excluded.status,
                   duplicate_of=excluded.duplicate_of, last_checked=excluded.last_checked,
                   last_changed=CASE WHEN ? THEN excluded.last_changed ELSE documents.last_changed END,
                   error=NULL""",
            (doc_id, source, title, url, version, content_hash, status, duplicate_of, ts, ts, ts, changed),
        )
        self.db.commit()

    def touch(self, doc_id: str) -> None:
        self.db.execute("UPDATE documents SET last_checked=? WHERE doc_id=?", (now(), doc_id))
        self.db.commit()

    def mark_failed(self, doc_id, source, title, url, error: str) -> None:
        ts = now()
        # если документ уже был успешно скачан раньше — не теряем его, только пишем ошибку
        self.db.execute(
            """INSERT INTO documents(doc_id, source, title, url, status, first_seen, last_checked, error)
               VALUES(?,?,?,?, 'failed', ?, ?, ?)
               ON CONFLICT(doc_id) DO UPDATE SET last_checked=excluded.last_checked, error=excluded.error""",
            (doc_id, source, title, url, ts, ts, error[:500]),
        )
        self.db.commit()

    def mark_removed(self, source: str, seen_ids: set[str]) -> list[str]:
        rows = self.db.execute(
            "SELECT doc_id FROM documents WHERE source=? AND status='active'", (source,)
        ).fetchall()
        gone = [r[0] for r in rows if r[0] not in seen_ids]
        for doc_id in gone:
            self.db.execute("UPDATE documents SET status='removed', last_checked=? WHERE doc_id=?", (now(), doc_id))
        self.db.commit()
        return gone

    def log_run(self, source: str, started_at: str, stats: dict, http_requests: int) -> None:
        cols = ["listed", "new", "updated", "unchanged", "skipped", "duplicate", "failed", "removed"]
        self.db.execute(
            f"INSERT INTO runs(source, started_at, finished_at, {', '.join(cols)}, http_requests) "
            f"VALUES(?,?,?,{','.join('?' * len(cols))},?)",
            (source, started_at, now(), *[stats.get(c, 0) for c in cols], http_requests),
        )
        self.db.commit()

    def active_ids(self) -> set[str]:
        return {r[0] for r in self.db.execute("SELECT doc_id FROM documents WHERE status='active'")}

    def summary(self) -> list[tuple]:
        return self.db.execute(
            "SELECT source, status, COUNT(*) FROM documents GROUP BY source, status ORDER BY source, status"
        ).fetchall()
