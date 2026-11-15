"""Unit tests for ont_agent.render — Plotly figure shapes + fpdf2, temp files.

Chart-validation and lifecycle tests monkeypatch the Kaleido export seam
(``_export_png``) so no Chrome is needed; a single smoke test does one real
PNG export and skips cleanly when the chart engine is unavailable.
"""

from __future__ import annotations

import pytest
from langchain_core.tools import BaseTool

from ont_agent import render
from ont_agent.render import (
    CHART_TYPES,
    MAX_PIE_SLICES,
    MAX_POINTS,
    Artifact,
    build_local_tools,
    discard,
)

FAKE_PNG = b"\x89PNG\r\n\x1a\nfake"


@pytest.fixture
def fake_export(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(render, "_export_png", lambda fig: FAKE_PNG)


def _tools() -> tuple[list[Artifact], dict[str, BaseTool]]:
    collector: list[Artifact] = []
    tools = build_local_tools(collector)
    return collector, {tool.name: tool for tool in tools}


def test_bar_chart_renders_a_png_photo(fake_export: None) -> None:
    collector, tools = _tools()
    message = tools["render_chart"].invoke(
        {
            "chart_type": "bar",
            "title": "Contacts per country",
            "labels": ["Afghanistan", "Belgium"],
            "values": [4, 3],
        }
    )
    assert "attached" in message
    (artifact,) = collector
    assert artifact.kind == "photo"
    assert artifact.caption == "Contacts per country"
    assert artifact.path.read_bytes()[:4] == b"\x89PNG"
    discard(collector)
    assert not artifact.path.exists() and collector == []


def test_every_chart_type_renders(fake_export: None) -> None:
    collector, tools = _tools()
    for chart_type in CHART_TYPES:
        message = tools["render_chart"].invoke(
            {
                "chart_type": chart_type,
                "title": f"{chart_type} demo",
                "labels": ["A", "B", "C"],
                "values": [5, 3, 2],
            }
        )
        assert "attached" in message, f"{chart_type} failed: {message}"
    assert len(collector) == len(CHART_TYPES)
    discard(collector)


def test_figure_builders_produce_expected_trace_shapes() -> None:
    labels, values = ["A", "B"], [1.0, 2.0]
    expectations = {
        "bar": "bar",
        "horizontal_bar": "bar",
        "line": "scatter",
        "area": "scatter",
        "scatter": "scatter",
        "pie": "pie",
        "donut": "pie",
        "treemap": "treemap",
        "funnel": "funnel",
        "waterfall": "waterfall",
    }
    for chart_type, trace_type in expectations.items():
        fig = CHART_TYPES[chart_type](labels, values)
        (trace,) = fig.data
        assert trace.type == trace_type, chart_type
    assert CHART_TYPES["horizontal_bar"](labels, values).data[0].orientation == "h"
    assert CHART_TYPES["donut"](labels, values).data[0].hole > 0
    assert CHART_TYPES["area"](labels, values).data[0].fill == "tozeroy"
    assert CHART_TYPES["scatter"](labels, values).data[0].mode == "markers"
    # Category names must render ON pie/donut slices — a percent-only pie
    # ("35.4%" with no customer name) is useless; regression from live use.
    assert CHART_TYPES["pie"](labels, values).data[0].textinfo == "label+percent"
    assert CHART_TYPES["donut"](labels, values).data[0].textinfo == "label+percent"
    assert CHART_TYPES["funnel"](labels, values).data[0].textinfo == "value"


def test_multi_series_line_builds_one_named_trace_per_series(fake_export: None) -> None:
    # The Odoo Pipeline-Analysis shape: labels = stages, one series per month.
    collector, tools = _tools()
    message = tools["render_chart"].invoke(
        {
            "chart_type": "line",
            "title": "Pipeline by stage per month",
            "labels": ["New", "Qualified", "Proposition", "Won"],
            "series_names": ["June", "July", "August"],
            "series_values": [[1, 2, 1, 0], [2, 3, 2, 1], [3, 4, 3, 1]],
        }
    )
    assert "attached" in message
    (artifact,) = collector
    assert artifact.kind == "photo"
    fig = render._fig_multi(
        "line",
        ["New", "Qualified", "Proposition", "Won"],
        [("June", [1.0, 2.0, 1.0, 0.0]), ("July", [2.0, 3.0, 2.0, 1.0])],
    )
    assert [trace.name for trace in fig.data] == ["June", "July"]
    assert fig.data[0].line.color == render.SERIES[0]
    assert fig.data[1].line.color == render.SERIES[1]
    discard(collector)


def test_multi_series_area_stacks_and_bar_groups() -> None:
    labels = ["A", "B"]
    series = [("one", [1.0, 2.0]), ("two", [3.0, 4.0])]
    area = render._fig_multi("area", labels, series)
    assert all(trace.stackgroup == "one" for trace in area.data)
    bar = render._fig_multi("bar", labels, series)
    assert bar.layout.barmode == "group"
    assert all(trace.type == "bar" for trace in bar.data)
    hbar = render._fig_multi("horizontal_bar", labels, series)
    assert all(trace.orientation == "h" for trace in hbar.data)
    scatter = render._fig_multi("scatter", labels, series)
    assert all(trace.mode == "markers" for trace in scatter.data)


def test_multi_series_gets_a_legend_single_series_does_not() -> None:
    labels, series = ["A", "B"], [("one", [1.0, 2.0])]
    multi = render._fig_multi("line", labels, series)
    render._apply_layout(multi, "t", show_legend=True)
    assert multi.layout.showlegend is True
    single = CHART_TYPES["line"](labels, [1.0, 2.0])
    render._apply_layout(single, "t")
    assert single.layout.showlegend is False


def test_multi_series_validation_errors(fake_export: None) -> None:
    collector, tools = _tools()
    base = {"chart_type": "line", "title": "x", "labels": ["a", "b"]}
    both = tools["render_chart"].invoke(
        {**base, "values": [1, 2], "series_names": ["s"], "series_values": [[1, 2]]}
    )
    assert both.startswith("Error:")
    neither = tools["render_chart"].invoke(base)
    assert neither.startswith("Error:")
    ragged = tools["render_chart"].invoke(
        {**base, "series_names": ["s1", "s2"], "series_values": [[1, 2], [1]]}
    )
    assert ragged.startswith("Error:")
    crowded = tools["render_chart"].invoke(
        {
            **base,
            "series_names": [f"s{i}" for i in range(render.MAX_SERIES + 1)],
            "series_values": [[1.0, 2.0]] * (render.MAX_SERIES + 1),
        }
    )
    assert crowded.startswith("Error:")
    pie = tools["render_chart"].invoke(
        {
            "chart_type": "pie",
            "title": "x",
            "labels": ["a", "b"],
            "series_names": ["s1", "s2"],
            "series_values": [[1, 2], [3, 4]],
        }
    )
    assert pie.startswith("Error:") and "one series only" in pie
    assert collector == []  # every rejection leaves nothing behind


def test_bad_chart_input_returns_error_string_not_artifact(fake_export: None) -> None:
    collector, tools = _tools()
    mismatched = tools["render_chart"].invoke(
        {"chart_type": "bar", "title": "x", "labels": ["a", "b"], "values": [1]}
    )
    assert mismatched.startswith("Error:")
    crowded_pie = tools["render_chart"].invoke(
        {
            "chart_type": "pie",
            "title": "x",
            "labels": [str(i) for i in range(MAX_PIE_SLICES + 1)],
            "values": [1.0] * (MAX_PIE_SLICES + 1),
        }
    )
    assert "treemap" in crowded_pie  # steers the model to a readable form
    too_many = tools["render_chart"].invoke(
        {
            "chart_type": "treemap",
            "title": "x",
            "labels": [str(i) for i in range(MAX_POINTS + 1)],
            "values": [1.0] * (MAX_POINTS + 1),
        }
    )
    assert too_many.startswith("Error:")
    assert collector == []  # nothing rendered, nothing to clean up


def test_treemap_allows_more_slices_than_pie(fake_export: None) -> None:
    # Treemaps stay readable where pies do not: 12 categories is fine.
    collector, tools = _tools()
    message = tools["render_chart"].invoke(
        {
            "chart_type": "treemap",
            "title": "Composition",
            "labels": [f"cat{i}" for i in range(12)],
            "values": [float(i + 1) for i in range(12)],
        }
    )
    assert "attached" in message
    discard(collector)


def test_export_failure_degrades_to_error_string(monkeypatch: pytest.MonkeyPatch) -> None:
    def explode(fig: object) -> bytes:
        raise RuntimeError("no chrome")

    monkeypatch.setattr(render, "_export_png", explode)
    collector, tools = _tools()
    message = tools["render_chart"].invoke(
        {"chart_type": "bar", "title": "x", "labels": ["a"], "values": [1]}
    )
    assert message.startswith("Error:")
    assert "text" in message  # tells the model to fall back to prose
    assert collector == []


def test_real_kaleido_export_smoke() -> None:
    # One genuine Chrome-backed export; skip cleanly where unavailable.
    fig = CHART_TYPES["bar"](["A", "B"], [1.0, 2.0])
    try:
        png = render._export_png(fig)
    except Exception as exc:  # pragma: no cover - environment-dependent
        pytest.skip(f"chart engine unavailable: {exc}")
    assert png[:4] == b"\x89PNG"


def test_pdf_report_renders_a_document() -> None:
    collector, tools = _tools()
    message = tools["create_pdf_report"].invoke(
        {
            "title": "Top companies",
            "summary": "Companies with the most activity.",
            "table_headers": ["Name", "City"],
            "table_rows": [["Azure", "Fremont"], ["Deco", "Chicago"]],
        }
    )
    assert "attached" in message
    (artifact,) = collector
    assert artifact.kind == "document"
    assert artifact.path.read_bytes()[:4] == b"%PDF"
    assert artifact.path.suffix == ".pdf"
    discard(collector)


def test_pdf_rejects_ragged_rows() -> None:
    collector, tools = _tools()
    ragged = tools["create_pdf_report"].invoke(
        {
            "title": "x",
            "summary": "y",
            "table_headers": ["A", "B"],
            "table_rows": [["only-one-cell"]],
        }
    )
    assert ragged.startswith("Error:")
    assert collector == []


def test_pdf_survives_non_latin_text() -> None:
    # Core PDF fonts are latin-1; exotic characters degrade, never crash.
    collector, tools = _tools()
    message = tools["create_pdf_report"].invoke(
        {
            "title": "संपर्क रिपोर्ट",
            "summary": "日本語 summary",
            "table_headers": ["Name"],
            "table_rows": [["Ærøskøbing ✓"]],
        }
    )
    assert "attached" in message
    discard(collector)
