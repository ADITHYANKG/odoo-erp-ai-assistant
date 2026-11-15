"""Tests for ont_agent.analyze_tool_payload."""

from __future__ import annotations

import json
from pathlib import Path

from ont_agent.analyze_tool_payload import (
    analyze_payload,
    estimate_tokens,
    format_bytes,
    generate_mock_odoo_records,
    prune_payload,
)


def test_estimate_tokens():
    assert estimate_tokens("") == 0
    assert estimate_tokens("hello world") > 0


def test_format_bytes():
    assert format_bytes(500) == "500 B"
    assert format_bytes(2048) == "2.0 KB"


def test_generate_mock_odoo_records():
    emp = generate_mock_odoo_records("hr.employee", count=3)
    assert len(emp) == 3
    assert "name" in emp[0]
    assert "work_email" in emp[0]
    assert "create_date" in emp[0]

    inv = generate_mock_odoo_records("account.move", count=2)
    assert len(inv) == 2
    assert "amount_total" in inv[0]


def test_analyze_payload_identifies_bloat():
    data = generate_mock_odoo_records("hr.employee", count=5)
    report = analyze_payload(data, query="List all employees and their emails")

    assert report.record_count == 5
    assert report.total_tokens > 0
    assert report.bloat_tokens > 0
    assert report.savings_pct > 50.0  # At least 50% bloat identified in raw ERP dump

    categories = {fa.category for fa in report.field_analyses}
    assert "Audit Metadata" in categories
    assert "Relational ID List" in categories


def test_prune_payload_savings():
    data = generate_mock_odoo_records("hr.employee", count=5)
    pruned, tokens, num_bytes = prune_payload(data, ["id", "name", "work_email"])

    assert len(pruned) == 5
    assert all(set(r.keys()) == {"id", "name", "work_email"} for r in pruned)
    assert tokens < 200
