"""Тонкая обёртка над HuggingFace transformers: bi-encoder и cross-encoder.

Используем transformers напрямую (без sentence-transformers): так видно, что
происходит внутри, и нет зависимости от scikit-learn, DLL которого на Windows
может блокировать Smart App Control.
"""
from __future__ import annotations

import os
from functools import lru_cache

import numpy as np
import torch
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer

torch.set_num_threads(os.cpu_count() or 4)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@lru_cache(maxsize=4)
def load_encoder(name: str):
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModel.from_pretrained(name).to(DEVICE).eval()
    return tok, model


@lru_cache(maxsize=2)
def load_cross_encoder(name: str):
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForSequenceClassification.from_pretrained(name).to(DEVICE).eval()
    return tok, model


@torch.inference_mode()
def encode(name: str, texts: list[str], pooling: str = "mean", batch_size: int = 32,
           max_length: int = 512, progress: bool = False) -> np.ndarray:
    """Эмбеддинги текстов, L2-нормированные (cosine = dot product)."""
    tok, model = load_encoder(name)
    # сортируем по длине — меньше паддинга в батчах, заметно быстрее на CPU
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
    out = np.zeros((len(texts), model.config.hidden_size), dtype=np.float32)
    batches = range(0, len(texts), batch_size)
    if progress:
        from tqdm import tqdm
        batches = tqdm(batches, desc=f"embed {name.split('/')[-1]}", unit="batch")
    for b in batches:
        idx = order[b:b + batch_size]
        enc = tok([texts[i] for i in idx], padding=True, truncation=True, max_length=max_length,
                  return_tensors="pt").to(DEVICE)
        hidden = model(**enc).last_hidden_state
        if pooling == "cls":
            vec = hidden[:, 0]
        else:                                             # mean pooling по реальным токенам
            mask = enc["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            vec = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        vec = torch.nn.functional.normalize(vec, dim=-1)
        out[idx] = vec.float().cpu().numpy()
    return out


@torch.inference_mode()
def cross_scores(name: str, pairs: list[tuple[str, str]], batch_size: int = 16, max_length: int = 512) -> list[float]:
    """Релевантность пар (запрос, документ) ∈ [0, 1] — sigmoid от логита."""
    tok, model = load_cross_encoder(name)
    scores: list[float] = []
    for b in range(0, len(pairs), batch_size):
        batch = pairs[b:b + batch_size]
        enc = tok([q for q, _ in batch], [d for _, d in batch], padding=True, truncation="only_second",
                  max_length=max_length, return_tensors="pt").to(DEVICE)
        logits = model(**enc).logits
        logits = logits[:, 0] if logits.shape[-1] == 1 else logits[:, -1]
        scores += torch.sigmoid(logits).float().cpu().tolist()
    return scores
