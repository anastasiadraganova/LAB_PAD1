from src.evaluation.metrics import retrieval_metrics
from src.filtering import FilterConfig, apply_filters
from src.models import Chunk, Document, Section
from src.preprocessing.chunkers import ChunkingConfig, chunk_document, n_tokens
from src.preprocessing.cleaning import clean_text
from src.retrieval import Hit, rrf

DOC = Document(
    "wiki:1", "wikipedia", "ru", "Марс", "u", "planet",
    [Section("Введение", "Марс — четвёртая планета. " * 60), Section("Атмосфера / Состав", "Углекислый газ преобладает. " * 40),
     Section("Спутники", "Фобос и Деймос.")],
    "2026-01-01",
)


def test_clean_text_removes_html_and_noise():
    raw = "<h1>Орбита</h1> Эллипс<br>Lua error in Module:X\nFile:icon.png\n(\nтекст  с   пробелами"
    out = clean_text(raw, merge_orphans=False)
    assert "Lua error" not in out and "File:" not in out and "<" not in out
    assert "Орбита: Эллипс" in out and "текст с пробелами" in out


def test_merge_orphans_keeps_table_rows():
    out = clean_text("Планета | Радиус\nМарс | 3390\nЗемля | 6371", merge_orphans=True)
    assert out.count("\n") == 2


def test_chunk_sizes_respected():
    for strategy in ("fixed", "paragraph", "section"):
        cfg = ChunkingConfig(strategy, size=64, overlap=16 if strategy == "fixed" else 0, prefix_title=False)
        chunks = chunk_document(DOC, cfg)
        assert chunks, strategy
        assert all(c.n_tokens <= 64 + 2 for c in chunks), strategy


def test_section_chunks_do_not_cross_sections():
    chunks = chunk_document(DOC, ChunkingConfig("section", size=64))
    assert all(len(c.sections) == 1 for c in chunks)
    assert chunks[0].text.startswith("Марс — Введение\n")


def test_fixed_overlap_produces_more_chunks():
    a = chunk_document(DOC, ChunkingConfig("fixed", 64, 0))
    b = chunk_document(DOC, ChunkingConfig("fixed", 64, 32))
    assert len(b) > len(a)
    assert any(len(c.sections) > 1 for c in a)          # fixed-окна пересекают границы секций


def _chunk(cid, doc, sec, text="some long enough text " * 5):
    return Chunk(cid, doc, text, "T", [sec], "wikipedia", "ru", "u", "planet", "", 0, n_tokens(text), "s")


def test_retrieval_metrics():
    gold = [{"doc_id": "d1", "section": "Атмосфера"}]
    ranked = [_chunk("a", "d2", "Bio"), _chunk("b", "d1", "Атмосфера / Состав"), _chunk("c", "d1", "Bio")]
    m = retrieval_metrics(ranked, gold, ks=(1, 3))
    assert m["hit@1"] == 0 and m["hit@3"] == 1
    assert m["mrr"] == 0.5
    assert abs(m["precision@3"] - 1 / 3) < 1e-9
    assert retrieval_metrics(ranked, gold, ks=(3,), level="doc")["precision@3"] == 2 / 3


def test_filters_dedup_threshold_short():
    hits = [Hit(_chunk("a", "d1", "S"), 0.9, 0.9), Hit(_chunk("b", "d2", "S"), 0.8, 0.8),
            Hit(_chunk("c", "d3", "S", "short"), 0.85, 0.85), Hit(_chunk("d", "d4", "S", "other words " * 10), 0.3, 0.3)]
    out, log = apply_filters(hits, FilterConfig(score_threshold=0.5, min_chars=20, dedup=True))
    assert [h.chunk.chunk_id for h in out] == ["a"]
    assert log["duplicate"] == 1 and log["short"] == 1 and log["threshold"] == 1


def test_rrf_prefers_items_in_both_lists():
    a, b, c = (_chunk(x, x, "S") for x in "abc")
    fused = rrf([[Hit(a, 1), Hit(b, 1)], [Hit(b, 1), Hit(c, 1)]], k=3)
    assert fused[0].chunk.chunk_id == "b"
