"""Tests for ont_agent.router (Semantic Router with FastEmbed)."""

from __future__ import annotations

import pytest

from ont_agent.config import Settings
from ont_agent.router import (
    GREETINGS_RESPONSE,
    NO_INTENT_RESPONSE,
    OUT_OF_SCOPE_RESPONSE,
    AgentSemanticRouter,
    get_semantic_router,
)


@pytest.fixture(scope="module")
def router() -> AgentSemanticRouter:
    """Instantiate a test router with local FastEmbed encoder once."""
    return AgentSemanticRouter(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        score_threshold=0.55,
    )


def test_router_greetings(router: AgentSemanticRouter) -> None:
    for query in ["hello", "hi there", "hey", "good morning", "howdy"]:
        route_name = router.route(query)
        assert route_name == "greetings", f"Expected greetings for '{query}', got {route_name}"
        assert router.get_static_reply(route_name) == GREETINGS_RESPONSE


def test_router_out_of_scope(router: AgentSemanticRouter) -> None:
    for query in [
        "tell me a joke",
        "write a poem about flowers",
        "sing a song for me",
        "what is the capital of france",
    ]:
        route_name = router.route(query)
        msg = f"Expected out_of_scope for '{query}', got {route_name}"
        assert route_name == "out_of_scope", msg
        assert router.get_static_reply(route_name) == OUT_OF_SCOPE_RESPONSE


def test_router_no_intent(router: AgentSemanticRouter) -> None:
    for query in ["Man", "Pooy", "lol", "asdf", "hmm"]:
        route_name = router.route(query)
        assert route_name == "no_intent", f"Expected no_intent for '{query}', got {route_name}"
        assert router.get_static_reply(route_name) == NO_INTENT_RESPONSE


def test_router_short_followups_fall_through(router: AgentSemanticRouter) -> None:
    for query in ["yes", "ok", "invoices", "why?", "approve"]:
        route_name = router.route(query)
        assert route_name is None, f"Expected None for '{query}', got {route_name}"


def test_router_erp_queries_fall_through(router: AgentSemanticRouter) -> None:
    erp_queries = [
        "How many invoices are overdue?",
        "List all customers with unpaid bills",
        "Create a sales quotation for Azure Interior",
        "What is our total revenue this quarter?",
        "Show me employee leaves for this week",
    ]
    for query in erp_queries:
        route_name = router.route(query)
        # Should not match greetings or out_of_scope
        msg = f"Expected None (fall-through to agent) for '{query}', got {route_name}"
        assert route_name is None, msg
        assert router.get_static_reply(route_name) is None


def test_router_empty_or_whitespace(router: AgentSemanticRouter) -> None:
    assert router.route("") is None
    assert router.route("   ") is None


def test_get_semantic_router_respects_settings() -> None:
    # Disabled by default
    off_settings = Settings(
        mcp_url="http://localhost:8069/mcp",
        mcp_bearer_key="k",
        openrouter_api_key="k",
        agent_db_url="postgresql://user:pass@localhost/db",
        semantic_router=False,
    )
    assert get_semantic_router(off_settings) is None

    on_settings = Settings(
        mcp_url="http://localhost:8069/mcp",
        mcp_bearer_key="k",
        openrouter_api_key="k",
        agent_db_url="postgresql://user:pass@localhost/db",
        semantic_router=True,
    )
    r = get_semantic_router(on_settings)
    assert r is not None
    assert isinstance(r, AgentSemanticRouter)
