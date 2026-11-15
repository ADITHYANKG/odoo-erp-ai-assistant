"""Langfuse observability: tracing, monitoring, and the feedback loop.

Wraps the Langfuse v3+ Python SDK so the rest of the agent never imports
``langfuse`` directly. ``init_tracing(settings)`` returns ``None`` when no
API keys are configured — the disabled path never even imports the SDK, and
every enabled-path failure degrades to a log line: the bot must keep
answering when Langfuse is down.

Design (per the official LangGraph cookbook):

- One ``Langfuse(...)`` singleton at startup, explicit constructor args
  (they beat any ambient ``LANGFUSE_*`` environment variables).
- Per turn: a predefined trace id + ``start_as_current_observation`` span,
  with ``propagate_attributes`` supplying trace name / user / session; the
  LangChain ``CallbackHandler`` inside joins the same trace.
- The predefined trace id travels to Telegram inline buttons so a later
  👍/👎 tap can attach a ``user-feedback`` score to that exact trace.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from ont_agent.config import Settings
from ont_agent.log import get_logger

if TYPE_CHECKING:  # pragma: no cover - typing only
    from langchain_core.callbacks import BaseCallbackHandler
    from langchain_core.outputs import LLMResult

logger = get_logger(__name__)

#: Score name used for the 👍/👎 evaluation loop (cookbook Example 6).
FEEDBACK_SCORE_NAME = "user-feedback"
#: Score attached to the PROPOSING trace when the user decides on a write.
DECISION_SCORE_NAME = "action-decision"


@dataclass
class Turn:
    """One traced turn: its predefined trace id plus an output recorder."""

    trace_id: str
    _span: Any | None = None

    def set_output(self, text: str) -> None:
        """Record the turn's final answer on the root span (never raises)."""
        if self._span is None:
            return
        try:
            self._span.update(output=text)
        except Exception:
            logger.debug("Recording turn output failed", exc_info=True)


def extract_cost_details(response: LLMResult) -> dict[str, float] | None:
    """Actual charged USD cost from an OpenRouter response, or ``None``.

    OpenRouter returns the real billed amount per request (offers and price
    changes included) and ``langchain_openrouter`` surfaces it as
    ``response_metadata["cost"]``. Only ``total`` is ingested: OpenRouter's
    cost breakdown keys don't map onto Langfuse's usage-type buckets, and a
    mismatched breakdown would double-count; the total alone is exact.
    """
    try:
        generation = response.generations[-1][-1]
    except (AttributeError, IndexError, TypeError):
        return None
    message = getattr(generation, "message", None)
    metadata = getattr(message, "response_metadata", None)
    cost = metadata.get("cost") if isinstance(metadata, dict) else None
    if isinstance(cost, int | float) and cost >= 0:
        return {"total": float(cost)}
    return None


class Tracing:
    """A live Langfuse connection plus the helpers the bot needs."""

    def __init__(self, client: Any, handler: BaseCallbackHandler) -> None:
        self._client = client
        self.handler = handler

    @contextmanager
    def turn(
        self,
        *,
        trace_name: str,
        user_id: str,
        session_id: str,
        input_text: str | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Iterator[Turn]:
        """Open one traced turn; yields a :class:`Turn`.

        Everything invoked inside (with ``callbacks=[self.handler]``) lands
        in a single Langfuse trace named ``trace_name``, attributed to the
        Odoo user and grouped by chat session. ``input_text`` becomes the
        trace's input and ``Turn.set_output`` its output — the fields the
        UI evaluators (LLM-as-judge) map their variables to. ``metadata``
        lands on the root span. A tracing failure yields an inert Turn and
        lets the work run untraced.
        """
        from langfuse import Langfuse, propagate_attributes

        turn = Turn("", None)
        stack = ExitStack()
        try:
            trace_id = str(Langfuse.create_trace_id())
            span = stack.enter_context(
                self._client.start_as_current_observation(
                    as_type="span",
                    name=trace_name,
                    trace_context={"trace_id": trace_id},
                )
            )
            stack.enter_context(
                propagate_attributes(
                    trace_name=trace_name,
                    user_id=user_id,
                    session_id=session_id,
                    tags=tags if tags is not None else ["telegram"],
                )
            )
            enrichment: dict[str, Any] = {}
            if input_text is not None:
                enrichment["input"] = input_text
            if metadata:
                enrichment["metadata"] = metadata
            if enrichment:
                span.update(**enrichment)
            turn = Turn(trace_id, span)
        except Exception:
            logger.warning("Langfuse turn setup failed — running untraced", exc_info=True)
            turn = Turn("", None)
        with stack:
            yield turn

    def record_score(
        self, trace_id: str, name: str, value: float, comment: str | None = None
    ) -> None:
        """Attach an arbitrary numeric score to a trace (never raises)."""
        try:
            self._client.create_score(
                trace_id=trace_id,
                name=name,
                value=value,
                data_type="NUMERIC",
                comment=comment,
            )
            logger.info("Score %s=%s recorded on trace %s", name, value, trace_id)
        except Exception:
            logger.warning("Failed to record %s on trace %s", name, trace_id, exc_info=True)

    def record_decision(self, trace_id: str, decision: str) -> None:
        """Score the user's Approve/Cancel/Edit verdict onto the proposing
        trace: approved=1, anything else=0, verdict kept as the comment —
        the action-proposal-precision metric accumulating from real usage."""
        self.record_score(
            trace_id,
            DECISION_SCORE_NAME,
            1 if decision == "approved" else 0,
            comment=decision,
        )

    def record_feedback(self, trace_id: str, positive: bool) -> None:
        """Attach a user-feedback score to an existing trace (never raises)."""
        try:
            self._client.create_score(
                trace_id=trace_id,
                name=FEEDBACK_SCORE_NAME,
                value=1 if positive else 0,
                data_type="NUMERIC",
            )
            logger.info("Feedback recorded: %s on trace %s", "👍" if positive else "👎", trace_id)
        except Exception:
            logger.warning("Failed to record feedback on trace %s", trace_id, exc_info=True)

    def flush(self) -> None:
        """Flush buffered events (shutdown); never raises."""
        try:
            self._client.flush()
        except Exception:
            logger.warning("Langfuse flush failed", exc_info=True)


def _build_cost_aware_handler() -> BaseCallbackHandler:
    """Langfuse LangChain handler that also ingests OpenRouter's real cost.

    The stock handler maps output/usage/model but drops the ``cost`` field
    OpenRouter returns in ``response_metadata``. This subclass attaches it as
    ``cost_details`` on the generation observation *before* delegating —
    Langfuse prioritizes ingested cost over price-table inference, so traces
    show the actual charged amount.
    """
    from uuid import UUID

    from langfuse.langchain import CallbackHandler

    class CostAwareCallbackHandler(CallbackHandler):
        def on_llm_end(
            self,
            response: LLMResult,
            *,
            run_id: UUID,
            parent_run_id: UUID | None = None,
            **kwargs: Any,
        ) -> Any:
            try:
                cost = extract_cost_details(response)
                if cost is not None and run_id in self._runs:
                    self._runs[run_id].update(cost_details=cost)
            except Exception:  # cost is a bonus — never break the trace
                logger.debug("Cost ingestion failed for run %s", run_id, exc_info=True)
            return super().on_llm_end(
                response, run_id=run_id, parent_run_id=parent_run_id, **kwargs
            )

    return CostAwareCallbackHandler()


def init_tracing(settings: Settings) -> Tracing | None:
    """Build the Langfuse singleton, or ``None`` when keys are not set.

    The disabled path returns before importing the SDK. An unreachable
    Langfuse server logs a warning but never blocks bot startup — traces
    buffer client-side and events are dropped by the SDK if it stays down.
    """
    if not settings.langfuse_enabled:
        logger.info("Langfuse tracing disabled (no API keys configured)")
        return None
    try:
        from langfuse import Langfuse

        client = Langfuse(
            public_key=settings.langfuse_public_key,
            secret_key=settings.langfuse_secret_key,
            base_url=settings.langfuse_base_url,
        )
        tracing = Tracing(client, _build_cost_aware_handler())
    except Exception:
        logger.warning("Langfuse initialization failed — tracing disabled", exc_info=True)
        return None
    # The check is diagnostic only. After a host reboot the bot can start
    # before Langfuse answers; disabling tracing then would leave it off until
    # the next bot restart, while the SDK exports fine once Langfuse is up.
    try:
        authenticated = client.auth_check()
    except Exception:
        logger.warning(
            "Langfuse not reachable at %s yet — tracing stays on and exports once it is",
            settings.langfuse_base_url,
            exc_info=True,
        )
        return tracing
    if authenticated:
        logger.info("Langfuse tracing enabled (%s)", settings.langfuse_base_url)
    else:
        logger.warning(
            "Langfuse credentials rejected by %s — tracing will not record",
            settings.langfuse_base_url,
        )
    return tracing
