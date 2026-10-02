"""Embedding-модели (HuggingFace transformers) с дисковым кэшем векторов."""
from __future__ import annotations

import hashlib
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class EmbeddingSpec:
    key: str
    name: str
    dim: int
    query_prefix: str = ""
    doc_prefix: str = ""
    params_m: int = 0             # число параметров, млн (для отчёта)
    note: str = ""
    pooling: str = "mean"         # mean — E5 / SBERT; cls — BGE


# Обе модели мультиязычные (корпус и вопросы на русском), токенизатор XLM-R,
# лимит контекста 512 токенов. Сравниваются в эксперименте S2.
MODELS: dict[str, EmbeddingSpec] = {
    "e5-small": EmbeddingSpec("e5-small", "intfloat/multilingual-e5-small", 384,
                              "query: ", "passage: ", 118, "быстрая базовая модель"),
    "bge-m3": EmbeddingSpec("bge-m3", "BAAI/bge-m3", 1024, "", "", 568, "крупная SOTA-модель, 100+ языков",
                            pooling="cls"),
}


class Embedder:
    def __init__(self, key: str, cache_dir: Path, batch_size: int = 32):
        self.spec = MODELS[key]
        self.batch_size = batch_size
        cache_dir.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(cache_dir / f"emb_{key}.sqlite")
        self.db.execute("CREATE TABLE IF NOT EXISTS emb (h TEXT PRIMARY KEY, v BLOB)")

    @property
    def dim(self) -> int:
        return self.spec.dim

    def _encode(self, texts: list[str]) -> np.ndarray:
        from src.hf import encode
        return encode(self.spec.name, texts, self.spec.pooling, self.batch_size, progress=len(texts) > 200)

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        texts = [self.spec.doc_prefix + t for t in texts]
        hashes = [hashlib.sha1(t.encode("utf-8")).hexdigest() for t in texts]
        found: dict[str, np.ndarray] = {}
        for i in range(0, len(hashes), 900):                 # лимит параметров SQLite
            part = hashes[i:i + 900]
            q = f"SELECT h, v FROM emb WHERE h IN ({','.join('?' * len(part))})"
            found.update({h: np.frombuffer(v, dtype=np.float32) for h, v in self.db.execute(q, part)})
        missing = [i for i, h in enumerate(hashes) if h not in found]
        if missing:
            vecs = self._encode([texts[i] for i in missing])
            rows = [(hashes[i], vecs[j].tobytes()) for j, i in enumerate(missing)]
            self.db.executemany("INSERT OR REPLACE INTO emb VALUES (?, ?)", rows)
            self.db.commit()
            for (h, _), v in zip(rows, vecs):
                found[h] = v
        return np.stack([found[h] for h in hashes]) if hashes else np.zeros((0, self.dim), np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        return self._encode([self.spec.query_prefix + query])[0]
