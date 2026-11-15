"""Unit tests for ont_agent.whatsapp_format."""

from __future__ import annotations

from ont_agent.whatsapp_format import (
    clamp,
    html_to_whatsapp,
    italic,
    render_whatsapp,
    split_for_body,
)


def test_bold_and_italic_tags() -> None:
    assert html_to_whatsapp("<b>Total</b> and <i>note</i>") == "*Total* and _note_"


def test_bold_does_not_hug_whitespace_or_span_lines() -> None:
    assert html_to_whatsapp("<b> Label: </b>") == "*Label:*"
    assert html_to_whatsapp("<b>one\ntwo</b>") == "*one*\n*two*"


def test_links_become_text_with_url() -> None:
    assert (
        html_to_whatsapp('<a href="https://x.io/1">Invoice 1</a>') == "Invoice 1 (https://x.io/1)"
    )


def test_entities_are_unescaped() -> None:
    assert html_to_whatsapp("Tom &amp; Jerry &lt;3") == "Tom & Jerry <3"


def test_pre_block_is_literal() -> None:
    markup = '<pre><code class="language-text">a *b* &lt;c&gt;\nx  y</code></pre>'
    assert html_to_whatsapp(markup) == "```\na *b* <c>\nx  y\n```"


def test_inline_code_is_literal() -> None:
    assert html_to_whatsapp("<code>*x*</code> <b>y</b>") == "`*x*` *y*"


def test_unknown_tags_are_stripped() -> None:
    assert html_to_whatsapp("<u>keep</u> text") == "keep text"


def test_render_markdown_end_to_end() -> None:
    text = "### Sales\n\n**Total:** 120\n\n- one\n- two\n\n| a | b |\n|---|---|\n| 1 | 2 |\n"
    out = render_whatsapp(text)
    assert "*Sales*" in out
    assert "*Total:*" in out
    assert "• one" in out
    assert "```" in out  # the table
    assert "<" not in out  # no HTML left over


def test_approval_prompt_html() -> None:
    prompt = (
        "⚠️ <b>Approval needed</b> — I want to make this change in Odoo:\n\n"
        "▶ <b>send_email</b>\n<b>To:</b> a@b.com\n\nNothing happens until you decide."
    )
    out = html_to_whatsapp(prompt)
    assert "*Approval needed*" in out
    assert "*send_email*" in out
    assert "<" not in out


def test_clamp_keeps_footer_and_cuts_body() -> None:
    out = clamp("x" * 5000, italic("1K tokens"))
    assert len(out) <= 4096
    assert out.endswith("_1K tokens_")
    assert "…" in out


def test_clamp_leaves_short_text_alone() -> None:
    assert clamp("hello") == "hello"
    assert clamp("hello", "_f_") == "hello\n\n_f_"


def test_split_for_body_keeps_short_text_whole() -> None:
    assert split_for_body("short answer") == ("", "short answer")


def test_split_for_body_cuts_at_a_line_break_and_keeps_the_end() -> None:
    text = "intro line\n\n" + "\n".join(f"row {i:03d}" for i in range(300))
    split = split_for_body(text, limit=200)
    assert split is not None
    head, tail = split
    assert 150 < len(tail) <= 200  # as much as fits, so the end stays with the buttons
    assert head.startswith("intro line")
    assert tail.splitlines()[-1] == "row 299"
    assert not tail.startswith("\n")


def test_split_for_body_loses_no_text() -> None:
    text = "\n".join(f"row {i:03d}" for i in range(300))
    split = split_for_body(text, limit=200)
    assert split is not None
    head, tail = split
    assert f"{head}\n{tail}" == text


def test_split_for_body_never_cuts_inside_a_code_block() -> None:
    block = "```\n" + "\n".join(f"| {i:03d} | x |" for i in range(80)) + "\n```"
    # Every cut near the end falls inside the block, so there is no clean one.
    assert split_for_body("title\n\n" + block, limit=200) is None


def test_split_for_body_returns_none_without_any_line_break() -> None:
    assert split_for_body("x" * 3000) is None
