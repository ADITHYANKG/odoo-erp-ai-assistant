"""Long-term, per-USER memory — the layer the checkpointer deliberately is not.

The checkpointer (``agent/data/memory.db``) holds one CONVERSATION and is
wiped by ``/new`` and by ``/logout``; both are correct. This module adds a
second, independent store (``agent/data/facts.db``) keyed by the person's
Odoo login, so a stated preference ("always show amounts in lakhs") survives
``/new``, survives summarization, and is still there in a brand-new chat
tomorrow.

Identity boundary: memories are namespaced by ``odoo_login``, NEVER by chat
id. One Telegram chat can change identity (logout, someone else logs in), so
a chat-keyed namespace would leak one executive's preferences to the next.

Embeddings go through OpenRouter like every other model call in this project
(``check_embedding_ctx_length=False`` — OpenRouter needs raw text, not the
tiktoken arrays langchain-openai sends by default).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from langchain_core.tools import BaseTool
from langgraph.store.base import BaseStore

from ont_agent.config import DEFAULT_MEMORY_MAX_RECALL, Settings
from ont_agent.log import get_logger

if TYPE_CHECKING:  # pragma: no cover - typing only
    from langchain_core.embeddings import Embeddings

logger = get_logger(__name__)

#: Root namespace label; the second label is the escaped Odoo login.
MEMORY_ROOT = "memories"


def safe_label(odoo_login: str) -> str:
    """Escape a login for LangGraph's namespace rules (no periods allowed).

    Escaping ``_`` first keeps this injective, so ``john.doe@x.com`` and
    ``john_doe@x.com`` can never collide into one namespace — a real risk
    inside one company, and a cross-user leak if it happened.
    """
    return odoo_login.replace("_", "__").replace(".", "_")


def namespace_for(odoo_login: str) -> tuple[str, str]:
    """The one place a memory namespace is built. Always per user."""
    return (MEMORY_ROOT, safe_label(odoo_login))


def build_embeddings(settings: Settings) -> Embeddings | None:
    """OpenRouter-backed embeddings, or ``None`` when not configured."""
    if not settings.memory_embed_model or not settings.openrouter_api_key:
        return None
    from langchain_openai import OpenAIEmbeddings

    return OpenAIEmbeddings(
        model=settings.memory_embed_model,
        base_url=settings.openrouter_base_url,
        api_key=settings.openrouter_api_key,  # type: ignore[arg-type]
        # OpenRouter (like most OpenAI-compatible providers) needs raw text.
        check_embedding_ctx_length=False,
    )


def build_store_index(settings: Settings) -> Any | None:
    """Vector-search config for the store, or ``None`` for exact listing."""
    embeddings = build_embeddings(settings)
    if embeddings is None:
        logger.info("Long-term memory: no embedding model — semantic search off")
        return None
    return {
        "dims": settings.memory_embed_dims,
        "embed": embeddings,
        "fields": ["content"],
    }


#: Deliberately narrower than langmem's default: this assistant should learn
#: how an executive wants answers shaped, not trivia about them, and never
#: business facts (those live in Odoo and must be re-fetched every time).
_MANAGE_INSTRUCTIONS = (
    "Call this tool when the user states a durable PREFERENCE about how they "
    "want answers presented — currency or number formatting, preferred chart "
    "type, report layout, level of detail, recipients they routinely email, "
    "or an explicit 'remember that ...' request. Also call it to correct or "
    "delete a preference that is now wrong. NEVER store business facts "
    "(totals, counts, statuses, who owns what) — those live in Odoo and must "
    "be looked up fresh every time. Never store personal or sensitive "
    "details that are not about working preferences."
)

_SEARCH_INSTRUCTIONS = (
    "Search the preferences you have stored for this user, for example when "
    "they ask what you remember about them, or when you need to check how "
    "they like something formatted before answering."
)


def build_memory_tools(store: BaseStore, odoo_login: str) -> list[BaseTool]:
    """The manage/search memory tools, bound to this user's namespace."""
    from langmem import (  # type: ignore[import-untyped]
        create_manage_memory_tool,
        create_search_memory_tool,
    )

    namespace = namespace_for(odoo_login)
    return [
        create_manage_memory_tool(
            namespace=namespace, store=store, instructions=_MANAGE_INSTRUCTIONS
        ),
        create_search_memory_tool(
            namespace=namespace, store=store, instructions=_SEARCH_INSTRUCTIONS
        ),
    ]


def _memory_text(value: Any) -> str:
    """A stored memory's text, whatever shape langmem wrote it in."""
    if isinstance(value, dict):
        return str(value.get("content") or value.get("memory") or value)
    return str(value)


async def recall(
    store: BaseStore | None,
    odoo_login: str,
    query: str,
    limit: int = DEFAULT_MEMORY_MAX_RECALL,
) -> list[str]:
    """This user's most relevant stored preferences (never raises).

    Memory is an enhancement: if the store or the embedding provider is
    down, the bot must still answer, just without personalization.

    ``limit`` comes from Settings (AGENT_MEMORY_MAX_RECALL); past it, the
    semantic index decides which preferences make the cut.
    """
    if store is None:
        return []
    try:
        items = await store.asearch(namespace_for(odoo_login), query=query, limit=limit)
    except Exception:
        logger.warning("Memory recall failed for %s — answering without it", odoo_login)
        logger.debug("recall failure detail", exc_info=True)
        return []
    return [_memory_text(item.value) for item in items]


async def forget_all(store: BaseStore | None, odoo_login: str) -> int:
    """Delete every stored preference for this user; returns how many."""
    if store is None:
        return 0
    namespace = namespace_for(odoo_login)
    try:
        items = await store.asearch(namespace, limit=1000)
        for item in items:
            await store.adelete(namespace, item.key)
        return len(items)
    except Exception:
        logger.error("Failed clearing memories for %s", odoo_login, exc_info=True)
        return 0
