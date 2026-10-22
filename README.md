# Лабораторная работа. RAG-система «Солнечная система»

Вопросно-ответная система на основе Retrieval-Augmented Generation: отвечает на
вопросы о Солнце, планетах, спутниках, малых телах и космических миссиях **только
по собранной базе знаний** (статьи русской Википедии) и указывает источники.
Если ответа в базе нет — честно сообщает об этом.

- **Источник:** ru.wikipedia.org через MediaWiki API, 112 статей (7 групп).
- **Стек:** Python, HuggingFace transformers (GPU), Qdrant (embedded), Groq / Gemini API, Streamlit,
  Langfuse (трассировка), pixi (окружение).

Результаты экспериментов и выводы — в [experiments/REPORT.md](experiments/REPORT.md),
отчёт Word — `report/Отчёт_RAG_Солнечная_система.docx`.

---

## Архитектура

```
                    ru.wikipedia.org (MediaWiki API)
                                │
 ┌──────────────── Grabber (src/grabber) ────────────────────────────────────┐
 │ список статей из конфига → prop=info (lastrevid, пачками по 50)           │
 │ → action=parse (HTML) → секции по h2/h3 + карточка → Document             │
 │ манифест SQLite: new / updated / unchanged / skipped / duplicate / removed│
 └──────────────────────────────┬────────────────────────────────────────────┘
                                ▼
 ┌──────────── Preprocessing (src/preprocessing) ────────────┐
 │ очистка HTML и вики-мусора → нормализация Unicode →       │
 │ chunking: fixed / fixed+overlap / paragraph / section     │
 │ + заголовок «Документ — Секция» + metadata: source, url,  │
 │   title, section, doc_id, lang, entity_type, updated_at   │
 └──────────────────────────────┬────────────────────────────┘
                                ▼
      Embeddings (src/embeddings: e5-small / bge-m3, GPU, кэш векторов в SQLite)
                                ▼
      Vector Store: Qdrant (embedded или Docker), коллекция на конфигурацию
                                │
          User Query ───────────┤
                                ▼
      Retriever (src/retrieval): dense / BM25 / hybrid (RRF), фильтр по metadata
                                ▼  Top-K кандидатов
      Filters (src/filtering): порог cosine, мин. длина, дедупликация
                                ▼
      Reranker (src/reranking): bge-reranker-v2-m3 → Top-N, порог релевантности
                                ▼
      LLM (src/generation): системный промпт + контекст [1..N] + вопрос → Groq
                                ▼
      Ответ со ссылками [n] + список источников ──► Streamlit (app.py) / CLI
```

### Структура репозитория

```
lab1/
├── app.py                    — веб-интерфейс (Streamlit)
├── pixi.toml, pixi.lock      — окружение pixi и задачи (pixi run app / ask / test …)
├── configs/default.yaml      — все параметры pipeline'а и список статей
├── src/
│   ├── grabber/              — источник Википедии, HTTP-клиент с повторами, манифест
│   ├── preprocessing/        — очистка, нормализация, 4 стратегии chunking
│   ├── embeddings/           — embedding-модели, кэш векторов
│   ├── retrieval/            — Qdrant, BM25, RRF
│   ├── filtering/            — фильтры контекста
│   ├── reranking/            — cross-encoder'ы
│   ├── generation/           — промпты, клиент LLM с кэшем, источники
│   ├── evaluation/           — датасет, метрики, LLM-as-a-Judge
│   ├── hf.py                 — bi-/cross-encoder на transformers (CUDA, если есть)
│   ├── tracing.py            — трассировка в Langfuse (выключена без ключей)
│   ├── pipeline.py           — сборка RAG pipeline
│   └── cli.py                — команды check / grab / stats / index / ask
├── data/eval/questions.jsonl — 30 вопросов с эталоном (doc_id + секция)
├── experiments/              — скрипты экспериментов, results/*.csv, figures/*.png, REPORT.md
├── report/                   — отчёт .docx
└── tests/                    — pytest
```

---

## Запуск

NVIDIA GPU желательна (на CPU тоже работает, медленнее). Окружение — один из двух вариантов.

**Вариант A — pixi** (рекомендуется: Python ставить не нужно, версии зафиксированы в `pixi.lock`).
Установка pixi: `winget install prefix-dev.pixi` (Linux/macOS: `curl -fsSL https://pixi.sh/install.sh | sh`).

```bash
pixi install                               # окружение .pixi/ по pixi.lock (Python 3.13, torch cu128, …)
pixi run check                             # ключи найдены?
pixi run app                               # веб-интерфейс → http://localhost:8501
pixi run ask "Почему Плутон перестали считать планетой?"
pixi task list                             # все задачи: grab, index, test, retrieval, generation, figures, docx…
```

**Вариант B — venv + pip** (Python 3.12+):

```bash
python -m venv .venv
.venv\Scripts\activate                     # Linux/macOS: source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

Дальше команды даны для варианта B; в pixi то же самое — `pixi run <задача>`.

Ключи API — в `.env` (в `lab1/` или уровнем выше), формат в [.env.example](.env.example).
Бесплатные ключи: [Groq](https://console.groq.com/keys), [Google AI Studio](https://aistudio.google.com/apikey).

```bash
python -m src.cli check                    # ключи найдены?
python -m src.cli grab                     # сбор статей (~2 мин; повторно — только изменённые)
python -m src.cli stats                    # статистика корпуса и chunks
python -m src.cli index                    # chunks + эмбеддинги + коллекция Qdrant
python -m src.cli ask "Почему Плутон перестали считать планетой?" --show-context
streamlit run app.py                       # веб-интерфейс → http://localhost:8501
```

Qdrant по умолчанию работает во встроенном режиме (`data/qdrant`, без сервера).
Для Qdrant в Docker: `docker compose up -d` и `--set vector_store.url=http://localhost:6333`.
Встроенный режим блокирует папку — не запускайте одновременно интерфейс и эксперименты.

Эксперименты, отчёт и тесты:

```bash
python experiments/check_dataset.py        # проверка эталонных ссылок датасета
python experiments/run_retrieval.py        # S1–S5: chunking, embeddings, Top-K, reranker, пороги
python experiments/run_generation.py       # LLM × промпт + судья
python experiments/make_figures.py         # графики → experiments/figures/*.png
python experiments/build_docx.py           # REPORT.md → report/*.docx
python -m pytest tests
```

Повторный `grab` не скачивает неизменённое: `lastrevid` каждой статьи сверяется
с манифестом одним запросом на 50 статей, затем — SHA-256 содержимого.
Убранные из конфига статьи помечаются `removed` и удаляются из базы;
`grab --reparse` пересобирает документы из сохранённых сырых ответов (после правки парсера).

### Трассировка в Langfuse

Каждый вопрос можно записывать в [Langfuse](https://langfuse.com) — видно, что нашёл поиск, что
отобрал reranker, какой промпт ушёл в LLM, ответ, токены и время каждого этапа.
Трассировка необязательная: без ключей система работает как обычно.

1. Зарегистрироваться на https://cloud.langfuse.com (бесплатно), создать проект →
   Settings → API Keys → Create new API keys.
2. Добавить в `.env` (регион US — `LANGFUSE_BASE_URL=https://us.cloud.langfuse.com`):
   ```
   LANGFUSE_PUBLIC_KEY=pk-lf-...
   LANGFUSE_SECRET_KEY=sk-lf-...
   LANGFUSE_BASE_URL=https://cloud.langfuse.com
   ```
3. `python -m src.cli check` покажет, что ключи найдены; в интерфейсе в боковой панели —
   «трассировка включена», под ответом — ссылка на трассу.

Структура трассы одного вопроса (`src/tracing.py`):

```
rag-ask (chain)            вопрос → ответ, источники, отказ, время этапов
├── retrieve (retriever)   кандидаты из Qdrant: chunk, документ, секция, cosine
├── rerank (span)          итоговые 5 chunks со скорами reranker'а, работа фильтров
└── llm:groq (generation)  промпт, ответ, модель, токены input/output, попадание в кэш
```

Вопросы из одной вкладки браузера объединяются в сессию. `run_generation.py` пишет каждый вопрос
датасета как трассу `eval-question` (сессия `eval:<вариант>`), а оценки судьи —
correctness, faithfulness, relevancy, refusal_correct — как scores этой трассы.
Если Langfuse недоступен, ответ всё равно возвращается (ошибка трассировки только логируется).

---

## Ключевые решения

| Компонент | Выбор | Почему | Альтернативы |
|---|---|---|---|
| Источник | MediaWiki API ru.wikipedia | стабильный API, HTML уже отрендерен (`action=parse`), есть дешёвая версия `lastrevid` | парсинг HTML страниц, дампы Википедии (избыточно для 112 статей) |
| Инкрементальность | `lastrevid` + SHA-256 + манифест SQLite | проверка без скачивания контента; дубликаты по хэшу | ETag / Last-Modified |
| Карточка статьи | отдельная секция «Характеристики» | числовые факты (радиус, масса, даты запуска) лежат именно в infobox | выбросить как шум (теряются факты) |
| Chunking | см. S1 в REPORT.md | | fixed, fixed+overlap, paragraph, section |
| Токенизация | токенизатор XLM-R | общий у e5 и bge-m3 → chunk точно влезает в модель | символы / слова |
| Embeddings | см. S2 | корпус и вопросы на русском → мультиязычные модели | e5-small, bge-m3 |
| Vector DB | Qdrant | фильтры по payload (metadata), коллекция на конфигурацию, embedded-режим без сервера | FAISS (нет metadata-фильтров), Chroma |
| Reranker | см. S4 | cross-encoder видит вопрос и chunk вместе | без reranker, mMiniLM |
| LLM | Groq: gpt-oss-120b / gpt-oss-20b | бесплатно, быстро, open-weight | локальная модель (Ollama) |
| Судья | Gemini Flash-Lite | другое семейство → меньше «самооценки» | та же модель, что генерирует |
| Интерфейс | Streamlit, один файл | минимум кода, всё нужное (ответ, источники, фрагменты со скорами) | FastAPI + фронтенд |
| Наблюдаемость | Langfuse Cloud, SDK на OpenTelemetry | дерево этапов каждого запроса, токены, сессии, оценки судьи как scores; бесплатный облачный тариф | LangSmith (заточен под LangChain), Phoenix, логи в файл |
| Окружение | pixi (+ venv/pip как запасной вариант) | одна команда `pixi install`, Python не нужен заранее, точные версии в `pixi.lock`, задачи `pixi run …` | conda, Poetry, uv |

### Защита от галлюцинаций

1. **Строгий промпт:** только контекст, ссылка `[n]` после каждого утверждения, фиксированная фраза отказа.
2. **Порог релевантности:** `reranker.min_score` — если ни один chunk не прошёл порог, LLM не вызывается и система сразу отвечает «недостаточно информации» (подбор порога — эксперимент S5).
3. **Источники:** в ответе — только реально процитированные фрагменты (заголовок, секция, URL).
