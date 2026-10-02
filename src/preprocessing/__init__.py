"""Preprocessing: Document → очистка → нормализация → chunking → Chunk (+ metadata)."""
from __future__ import annotations

import json
from dataclasses import asdict, replace
from pathlib import Path

from tqdm import tqdm

from src import config
from src.models import Chunk, Document, Section, load_documents
from src.preprocessing.chunkers import ChunkingConfig, chunk_document
from src.preprocessing.cleaning import clean_text

MIN_SECTION_CHARS = 20


def clean_document(doc: Document) -> Document:
    sections = []
    for s in doc.sections:
        text = clean_text(s.text, merge_orphans=True)
        if len(text) >= MIN_SECTION_CHARS:
            sections.append(Section(clean_text(s.title), text, s.level))
    return replace(doc, sections=sections)


def load_clean_documents(cfg: dict) -> list[Document]:
    docs = load_documents(config.path(cfg, "documents"))
    return [d for d in (clean_document(x) for x in docs) if d.sections]


def build_chunks(cfg: dict, ccfg: ChunkingConfig, docs: list[Document] | None = None,
                 use_cache: bool = True) -> list[Chunk]:
    """Chunks для заданной стратегии; кэшируются в data/chunks/<slug>.jsonl."""
    cache = config.path(cfg, "data") / "chunks" / f"{ccfg.slug}.jsonl"
    docs_dir = config.path(cfg, "documents")
    if use_cache and cache.exists() and cache.stat().st_mtime > _latest_mtime(docs_dir):
        return [Chunk(**json.loads(line)) for line in cache.read_text(encoding="utf-8").splitlines()]

    docs = docs if docs is not None else load_clean_documents(cfg)
    chunks: list[Chunk] = []
    for d in tqdm(docs, desc=f"chunking {ccfg.slug}", unit="doc", leave=False):
        chunks += chunk_document(d, ccfg)
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text("\n".join(json.dumps(asdict(c), ensure_ascii=False) for c in chunks), encoding="utf-8")
    return chunks


def _latest_mtime(directory: Path) -> float:
    return max((p.stat().st_mtime for p in directory.glob("*.json")), default=0.0)
