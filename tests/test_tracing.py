from src import config, tracing


def _env(tmp_path, monkeypatch, text: str) -> None:
    env = tmp_path / ".env"
    env.write_text(text, encoding="utf-8")
    monkeypatch.setattr(config, "ENV_FILES", [env])
    for k in ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY"):
        monkeypatch.delenv(k, raising=False)
    tracing.client.cache_clear()


def test_tracing_disabled_without_keys(tmp_path, monkeypatch):
    # ключи без имён: Groq и Gemini; ключ Gemini не должен «угадаться» как ключ Langfuse
    _env(tmp_path, monkeypatch, "gsk_groqkey\nAQ.geminikey\n")
    assert config.get_secret("GEMINI_API_KEY") == "AQ.geminikey"
    assert config.get_secret("LANGFUSE_PUBLIC_KEY") is None
    assert not tracing.enabled()
    with tracing.observation("x", "generation", model="m") as obs, tracing.trace_attributes(tags=["t"]):
        obs.update(output="ok", usage_details={"input": 1})
        obs.score_trace(name="correctness", value=5.0)
    assert tracing.current_trace_url() is None
    tracing.flush()


def test_langfuse_keys_by_name_or_prefix(tmp_path, monkeypatch):
    _env(tmp_path, monkeypatch, "LANGFUSE_PUBLIC_KEY=pk-lf-123\nsk-lf-456\n")
    assert config.get_secret("LANGFUSE_PUBLIC_KEY") == "pk-lf-123"
    assert config.get_secret("LANGFUSE_SECRET_KEY") == "sk-lf-456"
    tracing.client.cache_clear()
