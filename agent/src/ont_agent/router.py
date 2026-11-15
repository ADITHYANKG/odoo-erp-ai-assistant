"""Semantic routing for inbound messages via Aurelio Semantic Router + FastEmbed.

Pre-classifies messages using fast ONNX local embeddings (avoiding expensive LLM calls
and network latency for greetings, chit-chat, and out-of-scope entertainment queries).
Falls back to full agent reasoning for actual business questions.
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

# Prevent litellm (imported transitively by semantic_router) from auto-loading .env into os.environ
if "LITELLM_MODE" not in os.environ:
    os.environ["LITELLM_MODE"] = "PRODUCTION"

_root_handlers_before = list(logging.getLogger().handlers)

from semantic_router import Route, SemanticRouter  # noqa: E402
from semantic_router.encoders import FastEmbedEncoder  # noqa: E402

# aurelio_sdk (a semantic_router dependency) adds its own root handler on import, which
# duplicates every log line next to ours.
for _handler in list(logging.getLogger().handlers):
    if _handler not in _root_handlers_before:
        logging.getLogger().removeHandler(_handler)

if TYPE_CHECKING:
    from ont_agent.config import Settings

logger = logging.getLogger(__name__)

# Out-of-scope response aligned with agent guidelines
OUT_OF_SCOPE_RESPONSE = (
    "I am an executive assistant focused on your company's Odoo ERP data and operations. "
    "I cannot assist with general entertainment, jokes, poems, or unrelated topics. "
    "Please ask me about your contacts, sales, invoices, inventory, or other business data."
)

GREETINGS_RESPONSE = "Hello! How can I assist you with your Odoo business data today?"

NO_INTENT_RESPONSE = (
    "I didn't catch a question there. Ask me about your contacts, sales, invoices, "
    "inventory, or other business data."
)

# Canonical utterances for routes
GREETINGS_UTTERANCES = [
    "hello",
    "hi",
    "hey",
    "hey there",
    "good morning",
    "good afternoon",
    "good evening",
    "howdy",
    "greetings",
    "how are you",
    "how are you doing",
    "what's up",
]

# Filler and keyboard noise. Excludes yes/no/ok: those can answer the agent's question.
NO_INTENT_UTTERANCES = [
    "man",
    "bro",
    "dude",
    "lol",
    "haha",
    "hmm",
    "umm",
    "huh",
    "pooy",
    "asdf",
    "asdfgh",
    "qwerty",
    "test",
    "testing",
    "blah",
    "hello world",
    "???",
    "...",
    "mann",
    "wow",
    "meh",
    "idk",
    "xyz",
    "abc",
    "aaaa",
    "zzz",
    "ttt",
    "jkjk",
    "random",
    "nothing",
    "nonsense",
    "bruh",
    "lmao",
    "hehe",
    "lolz",
    "bhai",
    "machan",
    "macha",
    "da",
    "poda",
    "dei",
    "aiyo",
    "ggg",
    "qqq",
    "kkk",
    "fgfg",
]

OUT_OF_SCOPE_UTTERANCES = [
    "tell me a joke",
    "tell me a funny story",
    "write a poem",
    "write a song",
    "sing a song",
    "who won the world cup",
    "what is the capital of france",
    "write code in python for snake game",
    "what is the meaning of life",
    "tell me a bedtime story",
    "compose a rap",
    "write an essay on climate change",
    "play a trivia game with me",
]


class AgentSemanticRouter:
    """Wrapper around Aurelio Semantic Router with FastEmbed local ONNX encoder."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        score_threshold: float = 0.55,
    ) -> None:
        self.model_name = model_name
        self.score_threshold = score_threshold
        self._router: SemanticRouter | None = None

    def _init_router(self) -> SemanticRouter:
        if self._router is not None:
            return self._router

        logger.info("Initializing SemanticRouter with FastEmbed (%s)...", self.model_name)
        encoder = FastEmbedEncoder(name=self.model_name)

        greetings_route = Route(
            name="greetings",
            utterances=GREETINGS_UTTERANCES,
            score_threshold=self.score_threshold,
        )
        out_of_scope_route = Route(
            name="out_of_scope",
            utterances=OUT_OF_SCOPE_UTTERANCES,
            score_threshold=self.score_threshold,
        )

        router = SemanticRouter(
            encoder=encoder,
            routes=[],
            aggregation="max",
            top_k=5,
        )
        # Add routes to populate index embeddings
        no_intent_route = Route(
            name="no_intent",
            utterances=NO_INTENT_UTTERANCES,
            score_threshold=self.score_threshold,
        )

        router.add([greetings_route, out_of_scope_route, no_intent_route])
        self._router = router
        return self._router

    def route(self, text: str) -> str | None:
        """Route input text to a route name, or None if none matched."""
        clean_text = text.strip()
        if not clean_text:
            return None

        router = self._init_router()
        try:
            choice = router(clean_text)
            # choice is RouteChoice with .name and .similarity_score
            name = getattr(choice, "name", None)
            return str(name) if name else None
        except Exception as exc:
            logger.warning("Semantic routing failed for '%s': %s", clean_text, exc)
            return None

    def get_static_reply(self, route_name: str | None) -> str | None:
        """Return static response for known fast routes, or None for agent processing."""
        if route_name == "greetings":
            return GREETINGS_RESPONSE
        if route_name == "out_of_scope":
            return OUT_OF_SCOPE_RESPONSE
        if route_name == "no_intent":
            return NO_INTENT_RESPONSE
        return None


# Module-level singleton cache
_ROUTER_INSTANCE: AgentSemanticRouter | None = None


def get_semantic_router(settings: Settings) -> AgentSemanticRouter | None:
    """Return configured semantic router instance if enabled, else None."""
    if not settings.semantic_router:
        return None

    global _ROUTER_INSTANCE
    if _ROUTER_INSTANCE is None:
        _ROUTER_INSTANCE = AgentSemanticRouter(
            model_name=settings.semantic_router_model,
            score_threshold=settings.semantic_router_threshold,
        )
    return _ROUTER_INSTANCE
