"""Local rendering tools: charts (Plotly -> PNG) and PDF reports (fpdf2).

These run inside the agent process — no Odoo round-trip. The model calls them
AFTER retrieving data over MCP; each rendered file is appended to a
per-message collector that the bot sends to the chat as a photo or document.

Charts are rendered by Plotly and exported to PNG via Kaleido, which drives a
local Chrome/Chromium (auto-detected; installable once with
``uv run plotly_get_chrome``). Styling follows the data-viz reference palette
(light mode): categorical hues in fixed slot order, recessive grid, ink text.
"""

from __future__ import annotations

import re
import tempfile
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import plotly.graph_objects as go
from fpdf import FPDF
from fpdf.enums import TableCellFillMode
from fpdf.fonts import FontFace
from langchain_core.tools import BaseTool, tool

from ont_agent.log import get_logger

logger = get_logger(__name__)

# Reference palette, light mode (see the dataviz skill's palette.md).
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
#: Categorical slots in their validated fixed order — never re-sorted, never cycled.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]

MAX_POINTS = 24
MAX_PIE_SLICES = 8  # past the validated categorical slots, a pie stops being readable
MAX_SERIES = len(SERIES)  # one palette slot per series, never cycled

#: Chart types that show shares of a whole and get unreadable past 8 slices.
_SLICE_TYPES = {"pie", "donut", "funnel"}

#: Chart types where multiple series make sense (one trace per series).
_MULTI_TYPES = {"bar", "horizontal_bar", "line", "area", "scatter"}


@dataclass
class Artifact:
    """A rendered file waiting to be sent to the chat."""

    path: Path
    kind: Literal["photo", "document"]
    caption: str


def discard(artifacts: list[Artifact]) -> None:
    """Delete rendered files that will not be sent (e.g. a turn that paused
    for approval); clears the list in place."""
    for artifact in artifacts:
        artifact.path.unlink(missing_ok=True)
    artifacts.clear()


def _slug(title: str, suffix: str) -> Path:
    name = re.sub(r"[^a-zA-Z0-9]+", "-", title).strip("-").lower() or "report"
    return Path(tempfile.gettempdir()) / f"{name[:40]}-{uuid.uuid4().hex[:8]}{suffix}"


#: Public alias for other artifact producers (e.g. the export bridge).
temp_file_path = _slug


# ---------------------------------------------------------------------------
# Chart builders — every entry consumes the same (labels, values) series.
# ---------------------------------------------------------------------------


def _fig_bar(labels: list[str], values: list[float]) -> go.Figure:
    return go.Figure(go.Bar(x=labels, y=values, marker_color=SERIES[0]))


def _fig_horizontal_bar(labels: list[str], values: list[float]) -> go.Figure:
    fig = go.Figure(go.Bar(x=values, y=labels, orientation="h", marker_color=SERIES[0]))
    fig.update_yaxes(autorange="reversed")  # first item on top, reading order
    return fig


def _fig_line(labels: list[str], values: list[float]) -> go.Figure:
    return go.Figure(
        go.Scatter(x=labels, y=values, mode="lines+markers", line={"color": SERIES[0], "width": 3})
    )


def _fig_area(labels: list[str], values: list[float]) -> go.Figure:
    return go.Figure(
        go.Scatter(
            x=labels, y=values, mode="lines", fill="tozeroy", line={"color": SERIES[0], "width": 3}
        )
    )


def _fig_scatter(labels: list[str], values: list[float]) -> go.Figure:
    return go.Figure(
        go.Scatter(x=labels, y=values, mode="markers", marker={"color": SERIES[0], "size": 12})
    )


def _fig_pie(labels: list[str], values: list[float]) -> go.Figure:
    return go.Figure(
        go.Pie(
            labels=labels,
            values=values,
            marker={"colors": SERIES[: len(labels)]},
            sort=False,
            textinfo="label+percent",  # names ON the slices — never percent-only
            textposition="auto",
        )
    )


def _fig_donut(labels: list[str], values: list[float]) -> go.Figure:
    fig = _fig_pie(labels, values)
    fig.update_traces(hole=0.45)
    return fig


def _fig_treemap(labels: list[str], values: list[float]) -> go.Figure:
    return go.Figure(
        go.Treemap(
            labels=labels,
            parents=[""] * len(labels),
            values=values,
            marker={"colors": [SERIES[i % len(SERIES)] for i in range(len(labels))]},
            textinfo="label+value",
        )
    )


def _fig_funnel(labels: list[str], values: list[float]) -> go.Figure:
    return go.Figure(
        go.Funnel(
            y=labels,
            x=values,
            marker={"color": SERIES[: len(labels)]},
            textinfo="value",  # stage names on the axis, values in the bars
        )
    )


def _fig_waterfall(labels: list[str], values: list[float]) -> go.Figure:
    return go.Figure(
        go.Waterfall(
            x=labels,
            y=values,
            connector={"line": {"color": BASELINE}},
            increasing={"marker": {"color": SERIES[2]}},
            decreasing={"marker": {"color": SERIES[7]}},
            totals={"marker": {"color": SERIES[0]}},
        )
    )


CHART_TYPES: dict[str, Callable[[list[str], list[float]], go.Figure]] = {
    "bar": _fig_bar,
    "horizontal_bar": _fig_horizontal_bar,
    "line": _fig_line,
    "area": _fig_area,
    "scatter": _fig_scatter,
    "pie": _fig_pie,
    "donut": _fig_donut,
    "treemap": _fig_treemap,
    "funnel": _fig_funnel,
    "waterfall": _fig_waterfall,
}

ChartType = Literal[
    "bar",
    "horizontal_bar",
    "line",
    "area",
    "scatter",
    "pie",
    "donut",
    "treemap",
    "funnel",
    "waterfall",
]


def _fig_multi(
    chart_type: str, labels: list[str], series: list[tuple[str, list[float]]]
) -> go.Figure:
    """One trace per named series, palette slots in fixed order.

    Covers the two-dimensional requests ("by stage, split by month") that a
    single labels+values pair cannot express — the shape of Odoo's own
    Pipeline Analysis charts. Area stacks the series; bars group side by
    side; straight line segments (no spline smoothing) by design.
    """
    fig = go.Figure()
    for index, (name, values) in enumerate(series):
        color = SERIES[index]
        if chart_type == "bar":
            fig.add_trace(go.Bar(x=labels, y=values, name=name, marker_color=color))
        elif chart_type == "horizontal_bar":
            fig.add_trace(
                go.Bar(x=values, y=labels, name=name, orientation="h", marker_color=color)
            )
        elif chart_type == "line":
            fig.add_trace(
                go.Scatter(
                    x=labels,
                    y=values,
                    name=name,
                    mode="lines+markers",
                    line={"color": color, "width": 3},
                )
            )
        elif chart_type == "area":
            fig.add_trace(
                go.Scatter(
                    x=labels,
                    y=values,
                    name=name,
                    mode="lines",
                    stackgroup="one",
                    line={"color": color, "width": 2},
                )
            )
        else:  # scatter
            fig.add_trace(
                go.Scatter(
                    x=labels,
                    y=values,
                    name=name,
                    mode="markers",
                    marker={"color": color, "size": 12},
                )
            )
    if chart_type in {"bar", "horizontal_bar"}:
        fig.update_layout(barmode="group")
    if chart_type == "horizontal_bar":
        fig.update_yaxes(autorange="reversed")
    return fig


def _apply_layout(fig: go.Figure, title: str, show_legend: bool = False) -> None:
    fig.update_layout(
        title={"text": title, "font": {"color": INK, "size": 18}, "x": 0.02},
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        font={"color": INK, "size": 13},
        showlegend=show_legend,
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.0, "xanchor": "right", "x": 1.0},
        margin={"l": 60, "r": 30, "t": 60, "b": 50},
    )
    fig.update_xaxes(gridcolor=GRID, linecolor=BASELINE, tickfont={"color": MUTED})
    fig.update_yaxes(gridcolor=GRID, linecolor=BASELINE, tickfont={"color": MUTED})


def _export_png(fig: go.Figure) -> bytes:
    """Single seam for Kaleido/Chrome; monkeypatched in fast tests."""
    return fig.to_image(format="png", width=1000, height=600, scale=2)  # type: ignore[no-any-return]


def _render_chart_png(
    chart_type: str,
    title: str,
    labels: list[str],
    values: list[float] | None,
    series: list[tuple[str, list[float]]] | None = None,
) -> Path:
    if series is not None:
        fig = _fig_multi(chart_type, labels, series)
    else:
        assert values is not None  # validated by the tool before calling
        fig = CHART_TYPES[chart_type](labels, values)
    _apply_layout(fig, title, show_legend=series is not None)
    png = _export_png(fig)
    path = _slug(title, ".png")
    path.write_bytes(png)
    return path


def _render_pdf(
    title: str, summary: str, table_headers: list[str], table_rows: list[list[str]]
) -> Path:
    def latin(text: str) -> str:  # core PDF fonts are latin-1 only
        return text.encode("latin-1", errors="replace").decode("latin-1")

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=18)
    pdf.add_page()
    pdf.set_font("helvetica", style="B", size=16)
    pdf.set_text_color(11, 11, 11)  # primary ink
    pdf.multi_cell(0, 9, latin(title))
    pdf.ln(2)
    pdf.set_font("helvetica", size=11)
    pdf.set_text_color(82, 81, 78)  # secondary ink
    pdf.multi_cell(0, 6, latin(summary))
    pdf.ln(4)

    pdf.set_font("helvetica", size=10)
    pdf.set_text_color(11, 11, 11)
    pdf.set_draw_color(225, 224, 217)  # hairline grid
    pdf.set_fill_color(240, 239, 236)  # neutral stripe
    with pdf.table(
        cell_fill_color=(249, 249, 247),
        cell_fill_mode=TableCellFillMode.ROWS,
        line_height=7,
        headings_style=FontFace(emphasis="BOLD", fill_color=(240, 239, 236)),
    ) as table:
        header_row = table.row()
        for heading in table_headers:
            header_row.cell(latin(heading))
        for row in table_rows:
            table_row = table.row()
            for cell in row:
                table_row.cell(latin(cell))

    path = _slug(title, ".pdf")
    pdf.output(str(path))
    return path


def build_local_tools(collector: list[Artifact]) -> list[BaseTool]:
    """The two rendering tools, bound to a per-message artifact collector."""

    @tool
    def render_chart(
        chart_type: ChartType,
        title: str,
        labels: list[str],
        values: list[float] | None = None,
        series_names: list[str] | None = None,
        series_values: list[list[float]] | None = None,
    ) -> str:
        """Render a chart image that is attached to your reply automatically.

        Use AFTER retrieving the data with the Odoo tools; all numbers must
        come from those tool results. Single series: pass ``values``, one per
        label. Multi-series (a breakdown like "by stage, split by month" —
        supported for bar, horizontal_bar, line, area, scatter): instead of
        ``values``, pass ``series_names`` (one per series, shown in the
        legend) and ``series_values`` (one list per series, each aligned with
        ``labels``); "area" stacks the series. Pick the type by intent:
        "bar" compares categories ("horizontal_bar" when labels are long),
        "line"/"area" show change over time, "scatter" shows individual
        points, "pie"/"donut" show shares of a whole (max 8 slices, single
        series only), "treemap" shows composition when there are many
        categories, "funnel" shows pipeline stages, "waterfall" shows how
        positive and negative contributions build to a total. Hierarchy, map,
        and matrix charts (sunburst, heatmap, sankey, candlestick) are NOT
        supported — offer the closest supported type instead.
        """
        if chart_type not in CHART_TYPES:
            supported = ", ".join(sorted(CHART_TYPES))
            return f"Error: unsupported chart_type; use one of: {supported}."
        multi = series_names is not None or series_values is not None
        if multi == (values is not None):
            return (
                "Error: pass EITHER values (single series) OR series_names + "
                "series_values (multi-series), not both and not neither."
            )
        if not labels:
            return "Error: labels must be a non-empty list."
        if len(labels) > MAX_POINTS:
            return (
                f"Error: too many data points ({len(labels)}); aggregate to {MAX_POINTS} or fewer."
            )
        if chart_type in _SLICE_TYPES and len(labels) > MAX_PIE_SLICES:
            return (
                f"Error: a {chart_type} chart is unreadable with {len(labels)} slices; "
                f"use chart_type='treemap' or 'horizontal_bar', or fold small "
                f"categories into 'Other'."
            )

        series: list[tuple[str, list[float]]] | None = None
        floats: list[float] | None = None
        if multi:
            if chart_type not in _MULTI_TYPES:
                return (
                    f"Error: a {chart_type} chart takes one series only; pass values, "
                    "or use bar, horizontal_bar, line, area, or scatter for a breakdown."
                )
            if not series_names or not series_values or len(series_names) != len(series_values):
                return (
                    "Error: series_names and series_values must be non-empty lists "
                    "of the same length (one name per series)."
                )
            if len(series_names) > MAX_SERIES:
                return (
                    f"Error: too many series ({len(series_names)}); aggregate to "
                    f"{MAX_SERIES} or fewer."
                )
            if any(len(row) != len(labels) for row in series_values):
                return "Error: every series_values list must have exactly one value per label."
            series = [
                (name, [float(v) for v in row])
                for name, row in zip(series_names, series_values, strict=True)
            ]
        else:
            assert values is not None  # by the either/or check above
            if len(labels) != len(values):
                return "Error: labels and values must be non-empty lists of the same length."
            floats = [float(v) for v in values]
        try:
            path = _render_chart_png(chart_type, title, labels, floats, series)
        except Exception:
            logger.error("Chart rendering failed", exc_info=True)
            return (
                "Error: the chart could not be rendered (chart engine unavailable). "
                "Present the numbers as text instead."
            )
        collector.append(Artifact(path=path, kind="photo", caption=title))
        logger.info(
            "Rendered %s chart '%s' (%d points, %d series)",
            chart_type,
            title,
            len(labels),
            len(series) if series else 1,
        )
        return (
            f"Chart '{title}' rendered; it will be attached to your reply as an "
            "image. Keep the accompanying text to one or two sentences."
        )

    @tool
    def create_pdf_report(
        title: str,
        summary: str,
        table_headers: list[str],
        table_rows: list[list[str]],
    ) -> str:
        """Create a PDF report that is attached to your reply automatically.

        Use AFTER retrieving the data with the Odoo tools; the table content
        must come from those tool results. ``summary`` is a short paragraph
        above the table; every row must have exactly one cell per header.
        """
        if not table_headers or not table_rows:
            return "Error: the report needs table_headers and at least one row."
        if any(len(row) != len(table_headers) for row in table_rows):
            return "Error: every row must have exactly one cell per header."
        try:
            path = _render_pdf(title, summary, table_headers, table_rows)
        except Exception:
            logger.error("PDF rendering failed", exc_info=True)
            return "Error: the PDF could not be created."
        collector.append(Artifact(path=path, kind="document", caption=title))
        logger.info("Rendered PDF '%s' (%d rows)", title, len(table_rows))
        return (
            f"PDF report '{title}' created; it will be attached to your reply. "
            "Keep the accompanying text to one or two sentences."
        )

    return [render_chart, create_pdf_report]
