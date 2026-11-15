"""Large language model construction — OpenRouter only.

The company's standing rule: every LLM call goes through OpenRouter, and the
model is runtime configuration (``OPENROUTER_MODEL``), never a hardcoded
constant.
"""

from __future__ import annotations

from langchain_openrouter import ChatOpenRouter
from pydantic import SecretStr

from ont_agent.config import Settings
from ont_agent.log import get_logger

logger = get_logger(__name__)


def build_model(settings: Settings, *, for_upload: bool = False) -> ChatOpenRouter:
    """Build the chat model from settings; no network happens here.

    :param for_upload: this turn carries an uploaded image or PDF file. Such
        a turn uses its OWN fallback list, because most models cannot read a
        file at all — the configured text fallbacks were measured on
        2026-09-02 and one of them accepts text only. Falling back to it
        would not fail: the model would answer from the caption alone,
        sounding confident about a document it never saw. An empty list
        (the default) means the primary model or nothing.
    """
    logger.info("Using OpenRouter model %s", settings.openrouter_model)
    model_kwargs: dict[str, object] = {}
    fallbacks = (
        settings.upload_fallback_models if for_upload else settings.openrouter_fallback_models
    )
    if fallbacks:
        # OpenRouter server-side failover: primary from `model`, then these
        # in order when providers are down / rate-limited / moderation-block.
        model_kwargs["models"] = fallbacks
        logger.info("Fallback models: %s", ", ".join(fallbacks))
    elif for_upload:
        logger.debug("Upload turn: no fallback models configured, primary only")
    plugins: list[dict[str, object]] | None = None
    if for_upload and settings.upload_pdf_engine:
        # OpenRouter parses PDF files server-side; the engine is chargeable
        # and only meaningful when a file is actually attached. ChatOpenRouter
        # exposes `plugins` explicitly and REFUSES it inside model_kwargs.
        plugins = [{"id": "file-parser", "pdf": {"engine": settings.upload_pdf_engine}}]
    return ChatOpenRouter(
        model=settings.openrouter_model,
        api_key=SecretStr(settings.openrouter_api_key),
        timeout=90_000,  # milliseconds (maps to the SDK's timeout_ms)
        max_retries=2,
        plugins=plugins,
        model_kwargs=model_kwargs,
    )
