"""Unit tests for ont_agent.llm — construction only, no network."""

from __future__ import annotations

from langchain_openrouter import ChatOpenRouter

from ont_agent.config import Settings
from ont_agent.llm import build_model


def _settings(
    openrouter_api_key: str = "sk-or-test-key",
    openrouter_model: str = "google/gemini-3.7-flash",
) -> Settings:
    return Settings(
        mcp_url="http://fake:1/mcp",
        mcp_bearer_key="sekret",
        openrouter_api_key=openrouter_api_key,
        openrouter_model=openrouter_model,
    )


def test_build_model_returns_chatopenrouter_with_configured_model() -> None:
    model = build_model(_settings())
    assert isinstance(model, ChatOpenRouter)
    assert model.model_name == "google/gemini-3.7-flash"


def test_build_model_applies_api_key_secretly() -> None:
    model = build_model(_settings(openrouter_api_key="sk-or-abc"))
    assert model.openrouter_api_key is not None
    assert model.openrouter_api_key.get_secret_value() == "sk-or-abc"
    # The key must never appear in the model's repr (secret hygiene).
    assert "sk-or-abc" not in repr(model)


def test_build_model_is_pure_config() -> None:
    # Model choice is runtime config, never hardcoded: a different setting
    # must flow straight through.
    model = build_model(_settings(openrouter_model="qwen/some-other-model"))
    assert model.model_name == "qwen/some-other-model"


def test_fallback_models_flow_into_model_kwargs() -> None:
    settings = Settings(
        mcp_url="http://fake:1/mcp",
        mcp_bearer_key="sekret",
        openrouter_api_key="sk-or-x",
        openrouter_fallback_models=["deepseek/deepseek-v4-flash-0731", "b/model"],
    )
    model = build_model(settings)
    assert model.model_kwargs["models"] == ["deepseek/deepseek-v4-flash-0731", "b/model"]


def test_no_fallbacks_means_no_models_kwarg() -> None:
    model = build_model(_settings())
    assert "models" not in model.model_kwargs


# --- uploads use their own, deliberately separate, fallback list -----------


def _with_fallbacks(text: list[str], upload: list[str]) -> Settings:
    return Settings(
        mcp_url="http://fake:1/mcp",
        mcp_bearer_key="sekret",
        openrouter_api_key="sk-or-test-key",
        openrouter_model="google/gemini-3.7-flash",
        openrouter_fallback_models=text,
        upload_fallback_models=upload,
    )


def test_a_text_turn_keeps_the_ordinary_fallbacks() -> None:
    model = build_model(_with_fallbacks(["a/one", "b/two"], ["c/vision"]))
    assert model.model_kwargs["models"] == ["a/one", "b/two"]


def test_an_upload_turn_never_inherits_the_text_fallbacks() -> None:
    """Measured 2026-09-02: one configured text fallback accepts text ONLY.
    Routed a file, it would not error — it would answer from the caption
    alone, sounding confident about a document it never saw."""
    model = build_model(_with_fallbacks(["a/text-only"], ["c/vision"]), for_upload=True)
    assert model.model_kwargs["models"] == ["c/vision"]
    assert "a/text-only" not in model.model_kwargs["models"]


def test_an_upload_turn_with_no_upload_fallbacks_sends_no_list_at_all() -> None:
    # Primary model or nothing. Failing is the correct outcome; being
    # answered by a model that cannot see the file is not.
    model = build_model(_with_fallbacks(["a/one", "b/two"], []), for_upload=True)
    assert "models" not in model.model_kwargs


def test_the_pdf_engine_plugin_is_sent_only_on_upload_turns() -> None:
    settings = Settings(
        mcp_url="http://fake:1/mcp",
        mcp_bearer_key="sekret",
        openrouter_api_key="sk-or-test-key",
        upload_pdf_engine="mistral-ocr",
    )
    # ChatOpenRouter exposes plugins explicitly and REFUSES them inside
    # model_kwargs, so this also pins WHERE the value has to go.
    assert build_model(settings).plugins is None
    assert build_model(settings, for_upload=True).plugins == [
        {"id": "file-parser", "pdf": {"engine": "mistral-ocr"}}
    ]


def test_no_engine_configured_means_no_plugin_block() -> None:
    # Empty leaves OpenRouter on its own default rather than pinning us to a
    # chargeable engine nobody chose.
    settings = Settings(
        mcp_url="http://fake:1/mcp", mcp_bearer_key="sekret", openrouter_api_key="sk-or-test-key"
    )
    assert build_model(settings, for_upload=True).plugins is None
