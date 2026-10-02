"""Веб-интерфейс RAG-системы «Солнечная система».

    streamlit run app.py
"""
from __future__ import annotations

import json

import streamlit as st

from src import config
from src.config import merged
from src.pipeline import RAGPipeline

st.set_page_config(page_title="RAG: Солнечная система", page_icon="🪐", layout="wide")


@st.cache_resource(show_spinner="Загрузка индекса и моделей…")
def get_pipeline() -> RAGPipeline:
    cfg = config.load_config()
    best = config.ROOT / "experiments" / "results" / "best_config.json"
    if best.exists():                                  # лучшая конфигурация по экспериментам
        b = json.loads(best.read_text(encoding="utf-8"))
        cfg = merged(cfg, chunking=b["chunking"], embedding={"model": b["embedding"]})
    return RAGPipeline(cfg)


rag = get_pipeline()

with st.sidebar:
    st.header("Настройки")
    use_rr = st.toggle("Reranker (bge-reranker-v2-m3)", value=True)
    prompt = st.radio("Промпт", ["strict", "basic"], help="strict — только контекст, ссылки [n], явный отказ")
    model = st.selectbox("LLM (Groq)", ["openai/gpt-oss-20b", "openai/gpt-oss-120b"])
    top_k = st.slider("Кандидатов из индекса (Top-K)", 5, 20, rag.cfg["retrieval"]["top_k"])
    st.caption(f"chunking: {rag.cfg['chunking']['strategy']}-{rag.cfg['chunking']['size']}, "
               f"embeddings: {rag.cfg['embedding']['model']}, chunks: {len(rag.chunks)}")

st.title("🪐 Вопросы о Солнечной системе")
st.caption("Ответы строятся только по статьям русской Википедии из базы знаний; источники — ссылками [n].")

question = st.text_input("Вопрос", placeholder="Например: Почему Плутон перестали считать планетой?")
if st.button("Спросить", type="primary") and question.strip():
    cfg = merged(rag.cfg, reranker={"enabled": use_rr}, retrieval={"top_k": top_k},
                 generation={"prompt": prompt, "model": model})
    from src.generation import build_llm
    with st.spinner("Поиск и генерация…"):
        answer, res = rag.ask(question, cfg, llm=build_llm(cfg))

    st.subheader("Ответ")
    (st.warning if answer.refused else st.success)(answer.text)

    if answer.sources:
        st.subheader("Источники")
        for s in answer.sources:
            st.markdown(f"**[{s['n']}]** [{s['title']} — {s['section']}]({s['url']})")

    with st.expander(f"Найденные фрагменты ({len(res.final)} в контексте LLM)"):
        for i, h in enumerate(res.final, 1):
            rr = f", rerank={h.rerank_score:.3f}" if h.rerank_score is not None else ""
            st.markdown(f"**[{i}] {h.chunk.title} — {h.chunk.section}**  \n`cosine={h.dense_score:.3f}{rr}`")
            st.text(h.chunk.text[:1200])
    st.caption(f"фильтры: {res.filter_log}; время: "
               + ", ".join(f"{k}={v:.2f} с" for k, v in res.timings.items()))
