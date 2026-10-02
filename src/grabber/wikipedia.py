"""Источник: русская Википедия (MediaWiki API).

Листинг:  список статей из конфига (группы: планеты, спутники, миссии, ...).
Версия:   lastrevid страницы — берётся пачками по 50 через prop=info,
          поэтому неизменённые страницы вообще не скачиваются.
Контент:  action=parse → HTML → парсинг по заголовкам h2/h3 в секции.
          Карточка (infobox) с числовыми характеристиками сохраняется
          отдельной секцией «Характеристики».
"""
from __future__ import annotations

import re

from bs4 import BeautifulSoup, NavigableString, Tag

from src.grabber.base import ItemRef, Source
from src.models import Document, Section

# элементы, которые не несут знаний: навигация, сноски, медиа, служебные плашки
NOISE_SELECTORS = [
    "script", "style", "noscript", ".navbox", ".navbox-styles", ".toc", ".mw-editsection",
    "sup.reference", "sup.noprint", ".reference", ".references", ".reflist", ".hatnote", ".dablink",
    ".ambox", ".metadata", ".noprint", ".gallery", "figure", ".thumb", ".mwe-math-mathml-inline",
    ".mwe-math-mathml-display", "img", ".mw-empty-elt", ".sistersitebox", ".plainlinks.metadata",
    ".ts-Начало_и_конец", "table.infobox", ".infobox", "table.navigation-box", ".navigation-box",
    ".ruwikiShortDescription", ".printfooter",
]
BLOCK_TAGS = {"p", "div", "li", "dd", "dt", "tr", "br", "ul", "ol", "dl", "table", "blockquote", "h4", "h5", "h6"}
HEADINGS = {"h2": 2, "h3": 3}


def infobox_text(root: Tag) -> str:
    """Карточка статьи → строки «параметр | значение»."""
    box = root.select_one("table.infobox")
    if not box:
        return ""
    for el in box.select("sup.reference, .mw-editsection, style, img"):
        el.decompose()
    rows = []
    for tr in box.find_all("tr"):
        cells = [re.sub(r"\s+", " ", c.get_text(" ", strip=True)) for c in tr.find_all(["th", "td"])]
        cells = [c for c in cells if c]
        if len(cells) >= 2:
            rows.append(" | ".join(cells[:2]))
    return "\n".join(rows)


def html_to_sections(html: str, skip_sections: set[str]) -> list[Section]:
    soup = BeautifulSoup(html, "lxml")
    root = soup.select_one(".mw-parser-output") or soup.body or soup
    card = infobox_text(root)
    for sel in NOISE_SELECTORS:
        for el in root.select(sel):
            el.decompose()

    sections: list[Section] = []
    if card:
        sections.append(Section("Характеристики", card, 1))
    current = Section("Введение", "", 1)
    buf: list[str] = []
    skipping = False
    skip_level = 0
    parent = ""

    def flush():
        text = "\n".join(buf).strip()
        if text and not skipping:
            sections.append(Section(current.title, text, current.level))

    for el in root.children:
        if not isinstance(el, Tag):
            if isinstance(el, NavigableString) and el.strip():
                buf.append(el.strip())
            continue
        if el.name == "div" and "mw-heading" in (el.get("class") or []):   # новый формат MediaWiki
            el = el.find(list(HEADINGS)) or el
        if el.name in HEADINGS:
            level = HEADINGS[el.name]
            title = el.get_text(" ", strip=True)
            flush()
            buf = []
            if skipping and level > skip_level:     # подсекция пропускаемой секции
                continue
            skipping = title in skip_sections
            skip_level = level
            if level == 2:
                parent = title
            elif parent:                            # h3 получает контекст родительской h2
                title = f"{parent} / {title}"
            current = Section(title, "", level)
            continue
        if skipping:
            continue
        buf.append(block_text(el))
    flush()
    return [s for s in sections if len(s.text) > 20]


def block_text(el: Tag) -> str:
    """Текст с сохранением блочной структуры: строки таблиц → 'a | b | c'."""
    for table in el.find_all("table") if el.name != "table" else [el]:
        rows = []
        for tr in table.find_all("tr"):
            cells = [c.get_text(" ", strip=True) for c in tr.find_all(["th", "td"])]
            cells = [c for c in cells if c]
            if cells:
                rows.append(" | ".join(cells))
        table.replace_with(NavigableString("\n" + "\n".join(rows) + "\n"))
        if el.name == "table":
            return "\n".join(rows)
    for br in el.find_all("br"):
        br.replace_with(NavigableString("\n"))
    for tag in el.find_all(BLOCK_TAGS):
        tag.insert_before(NavigableString("\n"))
        tag.insert_after(NavigableString("\n"))
    text = el.get_text("")
    lines = [re.sub(r"[ \t ]+", " ", ln).strip() for ln in text.split("\n")]
    return "\n".join(ln for ln in lines if ln)


class WikipediaSource(Source):
    name = "wikipedia"

    def __init__(self, http, raw_dir, cfg: dict):
        super().__init__(http, raw_dir)
        self.api = cfg["api_url"]
        self.page_url = cfg["page_url"]
        self.cfg = cfg
        self.skip_sections = set(cfg.get("skip_sections", []))

    def _api(self, **params) -> dict:
        return self.http.get_json(self.api, {"format": "json", "formatversion": 2, **params})

    def list_items(self) -> list[ItemRef]:
        wanted: dict[str, str] = {}                       # title → entity_type
        for entity_type, titles in self.cfg["articles"].items():
            for t in titles:
                wanted.setdefault(t, entity_type)

        refs, titles = [], list(wanted)
        for i in range(0, len(titles), 50):               # prop=info принимает до 50 названий
            batch = titles[i:i + 50]
            d = self._api(action="query", prop="info", titles="|".join(batch), redirects=1)
            redirects = {r["to"]: r["from"] for r in d["query"].get("redirects", [])}
            normalized = {n["to"]: n["from"] for n in d["query"].get("normalized", [])}
            for p in d["query"]["pages"]:
                if p.get("missing") or "pageid" not in p:
                    continue
                orig = redirects.get(p["title"], p["title"])
                orig = normalized.get(orig, orig)
                refs.append(ItemRef(
                    doc_id=f"wiki:{p['pageid']}", title=p["title"],
                    url=self.page_url + p["title"].replace(" ", "_"),
                    version_hint=str(p["lastrevid"]),
                    meta={"pageid": p["pageid"], "entity_type": wanted.get(orig, "other"),
                          "touched": p.get("touched", "")},
                ))
        return refs

    def download(self, ref: ItemRef) -> dict:
        d = self._api(action="parse", pageid=ref.meta["pageid"], prop="text|revid",
                      disableeditsection=1, disabletoc=1, disablelimitreport=1)["parse"]
        return {"title": d["title"], "revid": d["revid"], "html": d["text"]}

    def parse(self, ref: ItemRef, parse: dict) -> Document:
        sections = html_to_sections(parse["html"], self.skip_sections)
        if not sections:
            raise ValueError("пустая страница после очистки")
        return Document(
            doc_id=ref.doc_id, source=self.name, lang="ru", title=parse["title"], url=ref.url,
            entity_type=ref.meta["entity_type"], sections=sections,
            updated_at=(ref.meta.get("touched") or "")[:10], version=str(parse["revid"]),
        )
