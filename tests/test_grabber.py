from collections import Counter

from src.grabber.base import ItemRef, Source, run_source
from src.grabber.wikipedia import html_to_sections
from src.grabber.manifest import Manifest
from src.models import Document, Section


# ---------- парсинг Википедии ----------
HTML = """
<div class="mw-parser-output">
  <table class="infobox"><tr><th>Масса</th><td>6,4171·10<sup>23</sup> кг<sup class="reference">[1]</sup></td></tr></table>
  <p>Марс — четвёртая по удалённости от Солнца планета.<sup class="reference">[2]</sup></p>
  <div class="mw-heading mw-heading2"><h2 id="x">Атмосфера</h2></div>
  <p>Атмосфера Марса состоит в основном из углекислого газа.</p>
  <h3><span class="mw-headline">Климат</span></h3>
  <table class="wikitable"><tr><th>Параметр</th><th>Значение</th></tr><tr><td>Давление</td><td>0,6 кПа</td></tr></table>
  <h2><span class="mw-headline">Примечания</span></h2>
  <p>Этот текст должен быть пропущен целиком, это сноски.</p>
</div>"""


def test_wiki_sections_and_noise():
    secs = html_to_sections(HTML, skip_sections={"Примечания"})
    titles = [s.title for s in secs]
    assert titles == ["Характеристики", "Введение", "Атмосфера", "Атмосфера / Климат"]
    assert "Масса | 6,4171" in secs[0].text and "[1]" not in secs[0].text
    assert "[2]" not in secs[1].text
    assert "Параметр | Значение" in secs[3].text and "Давление | 0,6 кПа" in secs[3].text


# ---------- инкрементальность и дубликаты ----------
class FakeHttp:
    requests_made = 0


class FakeSource(Source):
    name = "fake"

    def __init__(self, tmp_path, pages: dict, versions: dict):
        super().__init__(FakeHttp(), tmp_path / "raw")
        self.pages, self.versions, self.fetched = pages, versions, Counter()

    def list_items(self):
        return [ItemRef(k, "Same title", "u", self.versions.get(k)) for k in self.pages]

    def download(self, ref):
        return {}

    def parse(self, ref, raw):
        return self.fetch(ref)

    def fetch(self, ref, reparse=False):
        self.fetched[ref.doc_id] += 1
        if self.pages[ref.doc_id] is None:
            raise RuntimeError("network down")
        return Document(ref.doc_id, "fake", "ru", ref.title, "u", "other",
                        [Section("s", self.pages[ref.doc_id])], "2026-01-01", ref.version_hint or "")


def test_incremental_run(tmp_path):
    m = Manifest(tmp_path / "m.db")
    docs = tmp_path / "docs"
    src = FakeSource(tmp_path, {"a": "text A", "b": "text B", "c": "text A", "d": None}, {"a": "1", "b": "1"})
    s1 = run_source(src, m, docs)
    assert s1["new"] == 2 and s1["duplicate"] == 1 and s1["failed"] == 1   # c — копия a

    s2 = run_source(src, m, docs)
    assert s2["skipped"] == 2                  # a, b: версия не менялась → не скачиваем
    assert src.fetched["a"] == 1

    src.pages["b"], src.versions["b"] = "text B v2", "2"
    del src.pages["a"]
    s3 = run_source(src, m, docs)
    assert s3["updated"] == 1 and s3["removed"] == 1
    assert not (docs / "a.json").exists()
