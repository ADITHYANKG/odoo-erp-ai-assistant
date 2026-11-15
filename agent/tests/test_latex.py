"""Unit tests for ont_agent.latex — stray LaTeX must never reach a reader."""

from __future__ import annotations

from ont_agent.latex import strip_latex
from ont_agent.telegram_format import render_markdown


def test_the_live_travel_policy_answer() -> None:
    """Verbatim from Telegram, 2026-09-02. The reader saw the markup."""
    text = (
        "Premium Economy for segments $\\ge$ 6 hours (incremental cost up to $750 above economy)."
    )
    rendered = strip_latex(text)

    assert "≥" in rendered
    assert "\\ge" not in rendered
    assert "$\\ge$" not in rendered


def test_a_comparison_keeps_its_meaning() -> None:
    """Deleting the operator would quietly change what the policy says — on
    a reimbursement rule that is worse than showing the markup."""
    assert strip_latex("amounts $> \\$200$ require approval") == "amounts > $200 require approval"
    assert strip_latex("total air time $> 14$ hours") == "total air time > 14 hours"


def test_each_translated_symbol() -> None:
    for source, expected in (
        ("$\\ge$", "≥"),
        ("$\\leq$", "≤"),
        ("$\\ne$", "≠"),
        ("$\\approx$", "≈"),
        ("$\\pm$", "±"),
        ("$\\times$", "×"),  # noqa: RUF001 - the real symbol
        ("$\\rightarrow$", "→"),
    ):
        assert strip_latex(source) == expected


def test_display_and_bracket_delimiters() -> None:
    assert strip_latex("$$x \\ge y$$") == "x ≥ y"
    assert strip_latex("\\(a \\le b\\)") == "a ≤ b"
    assert strip_latex("\\[c \\ne d\\]") == "c ≠ d"


def test_an_untranslated_command_keeps_its_word() -> None:
    # Better a stray word than a vanished one: "\textbf{Total}" must not
    # silently delete "Total".
    assert "Total" in strip_latex("\\textbf{Total}")


def test_escaped_punctuation_loses_only_the_backslash() -> None:
    assert strip_latex("a 20\\% tip") == "a 20% tip"
    assert strip_latex("costs \\$50") == "costs $50"


def test_ordinary_text_is_untouched() -> None:
    """The common case, and it must stay cheap: no dollar sign and no
    backslash means the text is returned as-is."""
    for text in (
        "Overdue is 46,266.50 across 19 invoices.",
        "Costs $50 per person for the meal.",
        "",
    ):
        assert strip_latex(text) == text


def test_a_windows_path_is_not_mistaken_for_latex() -> None:
    # Backslashes appear in real business text; only known commands change.
    text = "The file is at C:\\Users\\reports\\march.xlsx"
    assert "Users" in strip_latex(text)
    assert "march.xlsx" in strip_latex(text)


def test_stripping_happens_before_rendering() -> None:
    """Wired into render_markdown, so every user-facing message is covered —
    answers, morning briefs, watcher alerts and the weekly digest."""
    html = render_markdown("Flights $\\ge$ 6 hours need **approval**.")

    assert "≥" in html
    assert "\\ge" not in html
    assert "<b>approval</b>" in html
