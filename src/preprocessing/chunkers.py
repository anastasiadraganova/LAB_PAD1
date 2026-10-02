"""Стратегии разбиения документов на chunks.

fixed      — окно фиксированного размера по токенам (overlap=0)
fixed      — то же с перекрытием (overlap>0)
paragraph  — упаковка абзацев (строк) до лимита, границы секций игнорируются
section    — chunk = секция документа; длинные секции делятся по абзацам

Размер считается в токенах токенизатора XLM-R (общий для multilingual-e5,
bge-m3 и paraphrase-multilingual-MiniLM), поэтому chunk гарантированно
помещается в контекст embedding-модели.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache

from src.models import Chunk, Document

TOKENIZER_NAME = "intfloat/multilingual-e5-small"
_SENT = re.compile(r"(?<=[.!?])\s+")


@lru_cache(maxsize=1)
def tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(TOKENIZER_NAME)


def n_tokens(text: str) -> int:
    return len(tokenizer()(text, add_special_tokens=False)["input_ids"])


@dataclass
class ChunkingConfig:
    strategy: str = "section"          # fixed | paragraph | section
    size: int = 256                    # максимум токенов в chunk
    overlap: int = 0                   # перекрытие в токенах (для fixed)
    prefix_title: bool = True          # добавлять "Документ — Секция" в начало chunk

    @property
    def slug(self) -> str:
        p = "" if self.prefix_title else "-notitle"
        return f"{self.strategy}-{self.size}-{self.overlap}{p}"


def chunk_document(doc: Document, cfg: ChunkingConfig) -> list[Chunk]:
    if cfg.strategy == "fixed":
        pieces = _fixed(doc, cfg.size, cfg.overlap)
    elif cfg.strategy == "paragraph":
        pieces = _paragraph(doc, cfg.size)
    elif cfg.strategy == "section":
        pieces = _section(doc, cfg.size)
    else:
        raise ValueError(f"неизвестная стратегия {cfg.strategy}")

    chunks = []
    for i, (body, sections) in enumerate(pieces):
        body = body.strip()
        if not body:
            continue
        if cfg.prefix_title:
            head = doc.title if cfg.strategy != "section" else f"{doc.title} — {sections[0]}"
            text = f"{head}\n{body}"
        else:
            text = body
        chunks.append(Chunk(
            chunk_id=f"{doc.doc_id}#{cfg.slug}#{i}", doc_id=doc.doc_id, text=text, title=doc.title,
            sections=sections, source=doc.source, lang=doc.lang, url=doc.url,
            entity_type=doc.entity_type, updated_at=doc.updated_at, position=i,
            n_tokens=n_tokens(body), strategy=cfg.slug,
        ))
    return chunks


# ---------------------------------------------------------------- fixed
def _fixed(doc: Document, size: int, overlap: int) -> list[tuple[str, list[str]]]:
    # склеиваем документ, запоминая диапазоны символов каждой секции
    parts, spans, pos = [], [], 0
    for s in doc.sections:
        block = f"{s.title}\n{s.text}\n\n"
        parts.append(block)
        spans.append((pos, pos + len(block), s.title))
        pos += len(block)
    full = "".join(parts)
    enc = tokenizer()(full, add_special_tokens=False, return_offsets_mapping=True)
    offsets = enc["offset_mapping"]
    step = max(1, size - overlap)
    out = []
    for start in range(0, len(offsets), step):
        window = offsets[start:start + size]
        if not window:
            break
        a, b = window[0][0], window[-1][1]
        covered = [t for (s0, s1, t) in spans if s0 < b and s1 > a]
        out.append((full[a:b], covered))
        if start + size >= len(offsets):
            break
    return out


# ---------------------------------------------------------------- paragraph
def _split_long(text: str, size: int) -> list[str]:
    """Абзац длиннее лимита режем по предложениям, а если и те длинные — по токенам."""
    if n_tokens(text) <= size:
        return [text]
    out, cur = [], ""
    for sent in _SENT.split(text):
        cand = f"{cur} {sent}".strip()
        if n_tokens(cand) <= size:
            cur = cand
            continue
        if cur:
            out.append(cur)
        if n_tokens(sent) <= size:
            cur = sent
        else:                                   # очень длинное "предложение" (таблица)
            ids = tokenizer()(sent, add_special_tokens=False, return_offsets_mapping=True)["offset_mapping"]
            for i in range(0, len(ids), size):
                w = ids[i:i + size]
                out.append(sent[w[0][0]:w[-1][1]])
            cur = ""
    if cur:
        out.append(cur)
    return out


def _pack(units: list[tuple[str, str]], size: int) -> list[tuple[str, list[str]]]:
    """Жадно упаковываем (текст, секция) в chunks не длиннее size токенов."""
    out, buf, buf_secs, buf_tok = [], [], [], 0
    for text, sec in units:
        for piece in _split_long(text, size):
            t = n_tokens(piece)
            if buf and buf_tok + t > size:
                out.append(("\n".join(buf), buf_secs))
                buf, buf_secs, buf_tok = [], [], 0
            buf.append(piece)
            buf_tok += t
            if sec not in buf_secs:
                buf_secs.append(sec)
    if buf:
        out.append(("\n".join(buf), buf_secs))
    return out


def _paragraph(doc: Document, size: int) -> list[tuple[str, list[str]]]:
    units = []
    for s in doc.sections:
        units.append((s.title, s.title))       # заголовок — отдельный абзац, но без спец. обработки
        units += [(p, s.title) for p in s.text.split("\n") if p.strip()]
    return _pack(units, size)


# ---------------------------------------------------------------- section
def _section(doc: Document, size: int) -> list[tuple[str, list[str]]]:
    out = []
    for s in doc.sections:
        paras = [(p, s.title) for p in s.text.split("\n") if p.strip()]
        out += _pack(paras, size)
    return out
