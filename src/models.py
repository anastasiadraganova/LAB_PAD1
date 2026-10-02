"""Общие структуры данных pipeline'а."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class Section:
    title: str
    text: str
    level: int = 2


@dataclass
class Document:
    doc_id: str                 # стабильный id: "wiki:12345" (pageid)
    source: str                 # wikipedia
    lang: str                   # ru
    title: str
    url: str
    entity_type: str            # planet | moon | mission | small_body | ...
    sections: list[Section]
    updated_at: str             # ISO-дата версии у источника (или время сбора)
    version: str = ""           # revid страницы
    extra: dict = field(default_factory=dict)

    @property
    def content_hash(self) -> str:
        payload = json.dumps([self.title, [(s.title, s.text) for s in self.sections]], ensure_ascii=False)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @property
    def text(self) -> str:
        return "\n\n".join(f"{s.title}\n{s.text}" if s.title else s.text for s in self.sections)

    def save(self, directory: Path) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        p = directory / f"{safe_name(self.doc_id)}.json"
        data = asdict(self) | {"content_hash": self.content_hash}
        p.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
        return p

    @classmethod
    def load(cls, p: Path) -> "Document":
        data = json.loads(p.read_text(encoding="utf-8"))
        data.pop("content_hash", None)
        data["sections"] = [Section(**s) for s in data["sections"]]
        return cls(**data)


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    text: str                   # текст, который векторизуется (с заголовком-контекстом)
    title: str
    sections: list[str]         # секции документа, которые покрывает chunk
    source: str
    lang: str
    url: str
    entity_type: str
    updated_at: str
    position: int
    n_tokens: int
    strategy: str

    @property
    def section(self) -> str:
        return self.sections[0] if self.sections else ""

    def payload(self) -> dict:
        return asdict(self)


def safe_name(doc_id: str) -> str:
    return doc_id.replace(":", "__").replace("/", "_")


def load_documents(directory: Path) -> list[Document]:
    return [Document.load(p) for p in sorted(directory.glob("*.json"))]
