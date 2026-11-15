"""Tool Call Result Analyzer & Mocking Utility for LLM Agent.

Inspects, mocks, and analyzes the exact raw tool payloads passed to the LLM
inside `ToolMessage(content=...)`. Demonstrates how much extraneous JSON
data is fed into the context window, breaks down token consumption per field,
and calculates potential token/cost savings by pruning unneeded fields.

Usage from `agent/`:
    # 1. Run with default realistic mock (hr.employee "list all employees")
    uv run python -m ont_agent.analyze_tool_payload

    # 2. Custom query and Odoo model mock
    uv run python -m ont_agent.analyze_tool_payload --query "Show customer unpaid invoices" --model account.move --records 15

    # 3. Analyze an existing raw JSON dump from a tool
    uv run python -m ont_agent.analyze_tool_payload --from-json path/to/dump.json

    # 4. Save the exact simulated ToolMessage payload to a file
    uv run python -m ont_agent.analyze_tool_payload --export-payload tool_message.json

    # 5. Connect to live Odoo MCP (if online) to capture real tool output
    uv run python -m ont_agent.analyze_tool_payload --live --model hr.employee --fields name work_email
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def estimate_tokens(text: str) -> int:
    """Estimate token count for a text string.

    Uses ~3.7 - 4 characters per token heuristic for typical JSON code/data,
    with additional token penalty for punctuation and brackets.
    """
    if not text:
        return 0
    # JSON has many quotes, colons, brackets which tokenize as separate tokens.
    # A standard rule of thumb for JSON is len(text) / 3.6.
    return max(1, math.ceil(len(text) / 3.6))


def format_bytes(num_bytes: int) -> str:
    """Format byte counts nicely."""
    if num_bytes < 1024:
        return f"{num_bytes} B"
    elif num_bytes < 1024 * 1024:
        return f"{num_bytes / 1024:.1f} KB"
    return f"{num_bytes / (1024 * 1024):.2f} MB"


# ---------------------------------------------------------------------------
# Realistic Mock Generators for Common Odoo Models
# ---------------------------------------------------------------------------

def generate_mock_odoo_records(model: str, count: int = 10, include_bloat: bool = True) -> list[dict[str, Any]]:
    """Generate realistic Odoo search_read payloads including standard ERP bloat."""
    records: list[dict[str, Any]] = []

    first_names = ["Alice", "Bob", "Charlie", "Diana", "Evan", "Fiona", "George", "Hannah", "Ian", "Julia"]
    last_names = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis", "Rodriguez", "Martinez"]
    departments = [[1, "Management"], [2, "Sales"], [3, "Engineering"], [4, "Human Resources"], [5, "Operations"]]
    job_titles = ["Sales Representative", "Software Engineer", "HR Manager", "Account Executive", "Operations Lead"]
    companies = [[1, "Acme Corp"], [2, "Acme Europe B.V."]]

    for i in range(1, count + 1):
        idx = (i - 1) % len(first_names)
        name = f"{first_names[idx]} {last_names[idx]}"

        if model in {"hr.employee", "employee"}:
            rec: dict[str, Any] = {
                "id": i,
                "name": name,
                "work_email": f"{first_names[idx].lower()}.{last_names[idx].lower()}@example.com",
                "job_title": job_titles[idx % len(job_titles)],
                "department_id": departments[idx % len(departments)],
                "company_id": companies[idx % len(companies)],
                "active": True,
                "work_phone": f"+1 (555) 010-{1000 + i}",
            }
            if include_bloat:
                # Typical default fields returned by Odoo search_read when fields=None
                rec.update({
                    "create_uid": [1, "Administrator"],
                    "create_date": f"2024-01-{10 + (i % 15):02d} 09:15:32",
                    "write_uid": [1, "Administrator"],
                    "write_date": f"2024-08-{10 + (i % 15):02d} 14:22:05",
                    "__last_update": f"2024-08-{10 + (i % 15):02d} 14:22:05",
                    "resource_calendar_id": [1, "Standard 40 hours/week"],
                    "tz": "America/New_York",
                    "parent_id": [1, "Subin Krishna"] if i > 1 else False,
                    "coach_id": False,
                    "color": 0,
                    "pinned": False,
                    "employee_type": "employee",
                    "message_follower_ids": [100 + i * 2, 101 + i * 2, 105],
                    "message_partner_ids": [1, 3, 7],
                    "message_ids": list(range(5000 + i * 10, 5010 + i * 10)),
                    "activity_ids": [],
                    "child_ids": [i + 1, i + 2] if i == 1 else [],
                    "category_ids": [1, 3],
                    "avatar_128": f"data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAEAAAABACAYAAACqaXHeAAADR{i}xM9...",
                    "work_location_id": [2, "Building A, Floor 3"],
                    "currency_id": [1, "USD"],
                    "spouse_complete_name": False,
                    "emergency_contact": f"Contact for {name}",
                    "emergency_phone": "+1 555-999-0000",
                    "km_home_work": 14.5,
                })

        elif model in {"account.move", "invoice"}:
            rec = {
                "id": i,
                "name": f"INV/2026/{1000 + i}",
                "move_type": "out_invoice",
                "partner_id": [10 + i, f"Client {chr(65 + (i % 26))} LLC"],
                "invoice_date": f"2026-09-{1 + (i % 25):02d}",
                "invoice_date_due": f"2026-10-{1 + (i % 25):02d}",
                "state": "posted",
                "payment_state": "not_paid" if i % 2 == 0 else "paid",
                "amount_total": round(1250.0 * i + 45.5, 2),
                "amount_residual": round(1250.0 * i + 45.5, 2) if i % 2 == 0 else 0.0,
                "currency_id": [1, "USD"],
            }
            if include_bloat:
                rec.update({
                    "create_uid": [1, "Administrator"],
                    "create_date": "2026-09-01 10:00:00",
                    "write_uid": [1, "Administrator"],
                    "write_date": "2026-09-05 11:23:45",
                    "__last_update": "2026-09-05 11:23:45",
                    "invoice_line_ids": list(range(2000 + i * 5, 2005 + i * 5)),
                    "line_ids": list(range(3000 + i * 8, 3008 + i * 8)),
                    "message_ids": list(range(8000 + i * 4, 8004 + i * 4)),
                    "message_follower_ids": [1, 2, 5],
                    "narration": False,
                    "auto_post": "no",
                    "to_check": False,
                    "posted_before": True,
                    "commercial_partner_id": [10 + i, f"Client {chr(65 + (i % 26))} LLC"],
                })

        elif model in {"sale.order", "order"}:
            rec = {
                "id": i,
                "name": f"SO{200 + i}",
                "partner_id": [20 + i, f"Acme Corporation {i}"],
                "date_order": f"2026-09-{5 + (i % 20):02d} 14:30:00",
                "state": "sale",
                "amount_total": round(3400.0 * i + 120.0, 2),
                "user_id": [5, "Alice Smith"],
            }
            if include_bloat:
                rec.update({
                    "create_uid": [5, "Alice Smith"],
                    "create_date": "2026-09-05 14:00:00",
                    "write_uid": [5, "Alice Smith"],
                    "write_date": "2026-09-05 14:35:10",
                    "__last_update": "2026-09-05 14:35:10",
                    "order_line": list(range(7000 + i * 6, 7006 + i * 6)),
                    "invoice_ids": [1000 + i],
                    "delivery_count": 1,
                    "tag_ids": [2, 4],
                    "message_ids": [9001, 9002],
                    "origin": f"Opportunity #{i}",
                    "team_id": [1, "Direct Sales"],
                })

        else:  # res.partner / default
            rec = {
                "id": i,
                "name": f"Partner {chr(65 + (i % 26))} Co.",
                "is_company": True,
                "email": f"info@partner{chr(97 + (i % 26))}.example.com",
                "phone": f"+1 555-01{i:02d}",
                "city": "New York",
                "country_id": [233, "United States"],
            }
            if include_bloat:
                rec.update({
                    "create_uid": [1, "Administrator"],
                    "create_date": "2024-05-01 10:00:00",
                    "write_date": "2024-07-01 12:00:00",
                    "__last_update": "2024-07-01 12:00:00",
                    "child_ids": list(range(400 + i * 3, 403 + i * 3)),
                    "message_ids": [1000 + i],
                    "category_id": [1],
                    "vat": f"US98765432{i}",
                    "commercial_partner_id": [i, f"Partner {chr(65 + (i % 26))} Co."],
                })

        records.append(rec)
    return records


# ---------------------------------------------------------------------------
# Analysis Core
# ---------------------------------------------------------------------------

@dataclass
class FieldAnalysis:
    name: str
    sample_value: Any
    total_chars: int
    total_tokens: int
    percentage: float
    category: str


@dataclass
class PayloadAnalysisReport:
    query: str
    tool_name: str
    record_count: int
    raw_json_str: str
    total_bytes: int
    total_chars: int
    total_tokens: int
    field_analyses: list[FieldAnalysis]
    bloat_tokens: int
    essential_tokens: int
    savings_pct: float


def categorize_field(field_name: str, sample_val: Any) -> str:
    """Classify field into category to detect bloat."""
    lower = field_name.lower()
    if lower.startswith("image_") or lower.startswith("avatar_") or "base64" in str(sample_val):
        return "BLOB / Image"
    if lower in {"create_uid", "create_date", "write_uid", "write_date", "__last_update"}:
        return "Audit Metadata"
    if lower.endswith("_ids") or lower in {"order_line", "invoice_line_ids", "line_ids"}:
        return "Relational ID List"
    if sample_val is False or sample_val is None or sample_val == []:
        return "Empty / Null"
    return "Domain Data"


def analyze_payload(
    raw_data: Any,
    query: str = "list all records",
    tool_name: str = "search_read",
    essential_field_hints: set[str] | None = None,
) -> PayloadAnalysisReport:
    """Analyze the raw data dictionary/list that gets placed into ToolMessage."""
    raw_json_str = json.dumps(raw_data, indent=2, default=str)
    compact_json_str = json.dumps(raw_data, default=str)

    total_chars = len(compact_json_str)
    total_bytes = len(compact_json_str.encode("utf-8"))
    total_tokens = estimate_tokens(compact_json_str)

    records: list[dict[str, Any]] = []
    if isinstance(raw_data, list):
        records = [r for r in raw_data if isinstance(r, dict)]
    elif isinstance(raw_data, dict):
        if "records" in raw_data and isinstance(raw_data["records"], list):
            records = [r for r in raw_data["records"] if isinstance(r, dict)]
        else:
            records = [raw_data]

    record_count = len(records)
    field_char_counts: Counter[str] = Counter()
    sample_values: dict[str, Any] = {}

    for rec in records:
        for k, v in rec.items():
            val_json = json.dumps(v, default=str)
            # count key + colon + value + commas
            field_char_counts[k] += len(k) + 4 + len(val_json)
            if k not in sample_values:
                sample_values[k] = v

    field_analyses: list[FieldAnalysis] = []
    bloat_chars = 0
    essential_chars = 0

    # Auto-infer essential fields if hints not provided
    if essential_field_hints is None:
        essential_field_hints = set()
        q_lower = query.lower()
        for k in field_char_counts:
            # simple keyword match
            if k.lower() in q_lower or (k == "name") or (k == "id"):
                essential_field_hints.add(k)
        if not essential_field_hints:
            essential_field_hints = {"id", "name"}

    for k, chars in field_char_counts.most_common():
        pct = (chars / total_chars * 100.0) if total_chars > 0 else 0.0
        tokens = estimate_tokens(" " * chars)
        cat = categorize_field(k, sample_values.get(k))

        if cat in {"Audit Metadata", "Relational ID List", "BLOB / Image", "Empty / Null"}:
            bloat_chars += chars
        elif k not in essential_field_hints:
            bloat_chars += chars
        else:
            essential_chars += chars

        field_analyses.append(
            FieldAnalysis(
                name=k,
                sample_value=sample_values.get(k),
                total_chars=chars,
                total_tokens=tokens,
                percentage=pct,
                category=cat,
            )
        )

    bloat_tokens = estimate_tokens(" " * bloat_chars)
    essential_tokens = estimate_tokens(" " * essential_chars)
    savings_pct = (bloat_tokens / total_tokens * 100.0) if total_tokens > 0 else 0.0

    return PayloadAnalysisReport(
        query=query,
        tool_name=tool_name,
        record_count=record_count,
        raw_json_str=raw_json_str,
        total_bytes=total_bytes,
        total_chars=total_chars,
        total_tokens=total_tokens,
        field_analyses=field_analyses,
        bloat_tokens=bloat_tokens,
        essential_tokens=essential_tokens,
        savings_pct=savings_pct,
    )


def prune_payload(
    records: list[dict[str, Any]],
    keep_fields: Sequence[str],
) -> tuple[list[dict[str, Any]], int, int]:
    """Prune records to keep only specified fields and measure token savings."""
    pruned = [
        {k: v for k, v in rec.items() if k in keep_fields}
        for rec in records
    ]
    compact_pruned = json.dumps(pruned, default=str)
    tokens = estimate_tokens(compact_pruned)
    num_bytes = len(compact_pruned.encode("utf-8"))
    return pruned, tokens, num_bytes


def print_report(
    report: PayloadAnalysisReport,
    show_json_preview: bool = True,
    prune_fields: list[str] | None = None,
    raw_records: list[dict[str, Any]] | None = None,
) -> None:
    """Print an executive terminal breakdown of the tool payload passed to LLM."""
    print("\n" + "=" * 80)
    print(" 🔍 TOOL CALL RESULT -> LLM CONTEXT INJECTION ANALYSIS")
    print("=" * 80)
    print(f"User Query      : \"{report.query}\"")
    print(f"Tool Invoked    : {report.tool_name}()")
    print(f"Records Count   : {report.record_count}")
    print(f"Payload Size    : {format_bytes(report.total_bytes)} ({report.total_chars:,} chars)")
    print(f"Estimated Tokens: ~{report.total_tokens:,} tokens injected into LLM")
    print("-" * 80)

    print("\n📊 FIELD-BY-FIELD WEIGHT & BLOAT BREAKDOWN:")
    print(f"{'Field Name':<24} {'Category':<20} {'Size':<10} {'Tokens':<9} {'% Payload':<10} {'Sample Value'}")
    print("-" * 95)

    for fa in report.field_analyses:
        sample_str = str(fa.sample_value)
        if len(sample_str) > 28:
            sample_str = sample_str[:25] + "..."
        flag = "⚠️ " if fa.category in {"Audit Metadata", "Relational ID List", "BLOB / Image"} else "  "
        print(
            f"{flag}{fa.name:<22} "
            f"{fa.category:<20} "
            f"{format_bytes(fa.total_chars):<10} "
            f"~{fa.total_tokens:<8} "
            f"{fa.percentage:5.1f}%    "
            f"{sample_str}"
        )

    print("-" * 95)
    print("\n💡 REDUNDANCY & OPTIMIZATION VERDICT:")
    print(f"  • Essential Data Tokens : ~{report.essential_tokens:,} tokens")
    print(f"  • Extraneous / Bloat    : ~{report.bloat_tokens:,} tokens ({report.savings_pct:.1f}% of total!)")
    print(f"  • Potential Savings     : {report.savings_pct:.1f}% reduction in prompt tokens & latency")

    if raw_records and (prune_fields or report.savings_pct > 20):
        target_fields = prune_fields if prune_fields else [
            fa.name for fa in report.field_analyses
            if fa.category not in {"Audit Metadata", "Relational ID List", "BLOB / Image", "Empty / Null"}
        ][:6]
        pruned_recs, p_tokens, p_bytes = prune_payload(raw_records, target_fields)
        saved_tokens = max(0, report.total_tokens - p_tokens)
        saved_pct = (saved_tokens / report.total_tokens * 100.0) if report.total_tokens > 0 else 0.0

        print("\n" + "=" * 80)
        print(" ✂️ SIMULATED PRUNED PAYLOAD COMPARISON (What the LLM actually needed):")
        print("=" * 80)
        print(f"Kept Fields      : {', '.join(target_fields)}")
        print(f"Original Payload : {format_bytes(report.total_bytes)} | ~{report.total_tokens:,} tokens")
        print(f"Pruned Payload   : {format_bytes(p_bytes)} | ~{p_tokens:,} tokens")
        print(f"Tokens Saved     : ~{saved_tokens:,} tokens ({saved_pct:.1f}% REDUCTION!)")
        print("-" * 80)
        print("Pruned JSON sample (first 2 records):")
        print(json.dumps(pruned_recs[:2], indent=2))

    if show_json_preview:
        print("\n" + "-" * 80)
        print("📝 RAW ToolMessage(content=...) PREVIEW (Passed directly to LLM):")
        print("-" * 80)
        preview_lines = report.raw_json_str.splitlines()[:30]
        for line in preview_lines:
            print(f"  {line}")
        if len(report.raw_json_str.splitlines()) > 30:
            print(f"  ... [{len(report.raw_json_str.splitlines()) - 30} lines truncated] ...")
    print("=" * 80 + "\n")


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------

async def run_live_mcp(model: str, fields: list[str] | None, limit: int) -> list[dict[str, Any]]:
    """Fetch live data from Odoo MCP if configured and online."""
    from ont_agent.config import load_settings
    from ont_agent.tools import get_odoo_tools

    settings = load_settings()
    tools = await get_odoo_tools(settings.mcp_url, settings.mcp_bearer_key)
    search_read_tool = next((t for t in tools if t.name == "search_read"), None)
    if not search_read_tool:
        raise RuntimeError("search_read tool not found in MCP server tools.")

    args: dict[str, Any] = {"model": model, "limit": limit}
    if fields:
        args["fields"] = fields
    res = await search_read_tool.ainvoke(args)
    if isinstance(res, str):
        return json.loads(res)
    return res


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect and analyze tool call results passed into LLM ToolMessage context."
    )
    parser.add_argument(
        "--query",
        default="List all employees and their email addresses",
        help="User question to benchmark relevance against.",
    )
    parser.add_argument(
        "--model",
        default="hr.employee",
        help="Odoo model name for mock generator (hr.employee, account.move, sale.order, res.partner).",
    )
    parser.add_argument(
        "--records",
        type=int,
        default=10,
        help="Number of records to generate for mock analysis (default: 10).",
    )
    parser.add_argument(
        "--from-json",
        type=Path,
        default=None,
        help="Path to a custom JSON file containing a recorded tool result.",
    )
    parser.add_argument(
        "--export-payload",
        type=Path,
        default=None,
        help="Path to save the generated ToolMessage JSON payload.",
    )
    parser.add_argument(
        "--prune-fields",
        nargs="+",
        default=None,
        help="Specific fields to keep in pruning simulation (e.g. --prune-fields id name work_email).",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Connect to live Odoo MCP instead of mock data.",
    )
    parser.add_argument(
        "--fields",
        nargs="+",
        default=None,
        help="Specific fields to query if using live MCP.",
    )
    args = parser.parse_args()

    if args.from_json:
        with open(args.from_json, encoding="utf-8") as f:
            raw_data = json.load(f)
        tool_name = "custom_file_result"
    elif args.live:
        print(f"Connecting to live Odoo MCP to query model '{args.model}'...")
        try:
            raw_data = asyncio.run(run_live_mcp(args.model, args.fields, args.records))
            tool_name = "search_read"
        except Exception as e:
            print(f"Failed to fetch live MCP data ({e}). Falling back to mock generator.")
            raw_data = generate_mock_odoo_records(args.model, args.records)
            tool_name = f"mock_search_read({args.model})"
    else:
        raw_data = generate_mock_odoo_records(args.model, args.records)
        tool_name = f"mock_search_read({args.model})"

    records_list = raw_data if isinstance(raw_data, list) else [raw_data]
    report = analyze_payload(raw_data, query=args.query, tool_name=tool_name)
    print_report(report, prune_fields=args.prune_fields, raw_records=records_list)

    if args.export_payload:
        args.export_payload.write_text(report.raw_json_str, encoding="utf-8")
        print(f" Exported simulated tool payload to {args.export_payload}")


if __name__ == "__main__":
    main()
