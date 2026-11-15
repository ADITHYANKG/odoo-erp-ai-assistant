"""Unit tests for ont_agent.telegram_format — Markdown to Telegram HTML."""

from __future__ import annotations

from ont_agent.bot import BotReply, prepare_reply_text
from ont_agent.budget import TurnUsage
from ont_agent.telegram_format import render_markdown


def test_headers_become_bold_lines() -> None:
    assert render_markdown("### Customer Balances") == "<b>Customer Balances</b>"
    assert render_markdown("# Top") == "<b>Top</b>"


def test_bold_and_inline_code() -> None:
    assert render_markdown("You have **10 active** contacts.") == (
        "You have <b>10 active</b> contacts."
    )
    assert render_markdown("run `uv sync` now") == "run <code>uv sync</code> now"


def test_single_asterisk_italics() -> None:
    # Live incident: "status: *In Process*" showed literal asterisks.
    assert render_markdown("status: *In Process* / *Not Reconciled*") == (
        "status: <i>In Process</i> / <i>Not Reconciled</i>"
    )
    assert render_markdown("**bold** and *italic* mixed") == "<b>bold</b> and <i>italic</i> mixed"
    assert render_markdown("2*3 stays") == "2*3 stays"  # unpaired: untouched


def test_html_is_escaped_never_injected() -> None:
    rendered = render_markdown("totals: 5 < 10 & <b>fake</b>")
    assert "<b>fake</b>" not in rendered
    assert "&lt;b&gt;fake&lt;/b&gt;" in rendered
    assert "&amp;" in rendered


def test_tables_become_aligned_monospace_blocks() -> None:
    text = (
        "| Customer | Due |\n"
        "| :--- | ---: |\n"
        "| Northwind Traders | $14,950.00 |\n"
        "| Nova | $2,484.00 |"
    )
    rendered = render_markdown(text)
    # Language pinned to "text" so Telegram Desktop can't guess "sql".
    prefix = '<pre><code class="language-text">'
    assert rendered.startswith(prefix) and rendered.endswith("</code></pre>")
    lines = rendered.removeprefix(prefix).removesuffix("</code></pre>").split("\n")
    assert lines[0].startswith("Customer")
    assert ":---" not in rendered  # separator row dropped
    # Columns align: "Due" starts at the same offset in every row.
    offset = lines[0].index("Due")
    assert lines[1].index("$14,950.00") == offset
    assert lines[2].index("$2,484.00") == offset


def test_table_cells_drop_markdown_markers() -> None:
    # Bold cannot render inside <pre>; **markers** must not leak through.
    text = "| Invoice | Due |\n| --- | --- |\n| **INV/2026/00015** | `$1,104.00` |"
    rendered = render_markdown(text)
    assert "**" not in rendered
    inner = rendered.removeprefix('<pre><code class="language-text">').removesuffix(
        "</code></pre>"
    )
    assert "`" not in inner
    assert "INV/2026/00015" in rendered


def test_bullets_and_rules() -> None:
    rendered = render_markdown("- first\n* second\n---\nafter")
    assert "• first" in rendered
    assert "• second" in rendered
    assert "---" not in rendered
    assert "after" in rendered


def test_fenced_code_block_becomes_pre() -> None:
    rendered = render_markdown("```\nSELECT 1 <> 2\n```")
    assert rendered == '<pre><code class="language-text">SELECT 1 &lt;&gt; 2</code></pre>'


def test_fence_language_labels_are_stripped_both_styles() -> None:
    # Live incident: a table fenced as ```sql showed "sql" in the chat.
    backtick = render_markdown("```sql\nBill   Vendor\n```")
    assert "sql" not in backtick and backtick.startswith("<pre>")
    tilde = render_markdown("~~~sql\nBill   Vendor\n~~~")
    assert "sql" not in tilde and tilde.startswith("<pre>")
    assert "Bill   Vendor" in tilde


def test_blank_line_runs_collapse() -> None:
    assert render_markdown("a\n\n\n\nb") == "a\n\nb"


def test_prepare_reply_text_renders_normal_answers() -> None:
    text, parse_mode = prepare_reply_text(BotReply(text="### Hi\n**bold**"))
    assert parse_mode == "HTML"
    assert text == "<b>Hi</b>\n<b>bold</b>"


def test_prepare_reply_text_appends_turn_usage() -> None:
    usage = TurnUsage(
        input_tokens=3_000,
        output_tokens=482,
        cost_usd=0.004217,
        call_count=2,
        tokens_available=True,
        cost_available=True,
    )
    text, parse_mode = prepare_reply_text(BotReply(text="Done.", usage=usage))

    assert parse_mode == "HTML"
    assert text == "Done.\n\n<i>3.5K tokens · Cost: $0.004217</i>"


def test_prepare_reply_text_uses_exact_k_scale_without_trailing_decimal() -> None:
    usage = TurnUsage(
        input_tokens=1_800,
        output_tokens=200,
        cost_usd=0.001,
        call_count=1,
        tokens_available=True,
        cost_available=True,
    )
    text, _parse_mode = prepare_reply_text(BotReply(text="Done.", usage=usage))

    assert text.endswith("<i>2K tokens · Cost: $0.001000</i>")


def test_prepare_reply_text_marks_missing_usage_metadata_unavailable() -> None:
    usage = TurnUsage(call_count=1)
    text, _parse_mode = prepare_reply_text(BotReply(text="Done.", usage=usage))

    assert text.endswith("<i>Tokens: unavailable · Cost: unavailable</i>")


def test_prepare_reply_text_keeps_pending_html_untouched() -> None:
    from ont_agent.agent import PendingAction

    pending = PendingAction(actions=["x"], names=["send_email"], arguments=[{}])
    prebuilt = "<b>To:</b> a@b.c"
    text, parse_mode = prepare_reply_text(BotReply(text=prebuilt, pending=pending))
    assert (text, parse_mode) == (prebuilt, "HTML")


def test_prepare_reply_text_appends_usage_to_pending_html() -> None:
    from ont_agent.agent import PendingAction

    pending = PendingAction(actions=["x"], names=["send_email"], arguments=[{}])
    usage = TurnUsage(
        input_tokens=900,
        output_tokens=100,
        cost_usd=0.001,
        call_count=1,
        tokens_available=True,
        cost_available=True,
    )
    text, parse_mode = prepare_reply_text(
        BotReply(text="<b>To:</b> a@b.c", pending=pending, usage=usage)
    )

    assert parse_mode == "HTML"
    assert text.endswith("<i>1K tokens · Cost: $0.001000</i>")


def test_prepare_reply_text_falls_back_to_plain_when_oversized() -> None:
    text, parse_mode = prepare_reply_text(BotReply(text="x" * 5000))
    assert parse_mode is None
    assert len(text) <= 4096


def test_oversized_reply_preserves_turn_usage() -> None:
    usage = TurnUsage(
        input_tokens=10_000,
        output_tokens=500,
        cost_usd=0.0000004,
        call_count=1,
        tokens_available=True,
        cost_available=True,
    )
    text, parse_mode = prepare_reply_text(BotReply(text="x" * 5000, usage=usage))

    assert parse_mode == "HTML"
    assert len(text) == 4096
    assert text.endswith("<i>10.5K tokens · Cost: &lt;$0.000001</i>")


def test_oversized_table_fallback_has_no_pipe_syntax() -> None:
    rows = [
        "| Name | Job Title | Department | Work Email |",
        "| --- | --- | --- | --- |",
        *[
            f"| Employee {index} | Developer | IT | user{index}@example.com |"
            for index in range(81)
        ],
    ]
    text, parse_mode = prepare_reply_text(BotReply(text="\n".join(rows)))

    assert parse_mode is None
    assert "|" not in text
    assert "• Employee 0 — Developer — IT — user0@example.com" in text


# --- Odoo deep links -------------------------------------------------------

LINKS = {
    "INV/2026/00026": "https://odoo.test/web#model=account.move&id=26&view_type=form",
    "Acme Corporation": "https://odoo.test/web#model=res.partner&id=7&view_type=form",
    "Acme": "https://odoo.test/web#model=res.partner&id=99&view_type=form",
}


def test_a_known_record_becomes_a_link() -> None:
    out = render_markdown("Invoice INV/2026/00026 is overdue.", LINKS)
    assert (
        '<a href="https://odoo.test/web#model=account.move&amp;id=26&amp;view_type=form">INV/2026/00026</a>'
        in out
    )


def test_without_links_the_output_is_byte_identical_to_before() -> None:
    text = "**Bold** and INV/2026/00026 and `code`"
    assert render_markdown(text) == render_markdown(text, None)
    assert "<a href" not in render_markdown(text)


def test_escaping_still_happens_first_so_markup_cannot_be_injected() -> None:
    # The whole safety property of this renderer: model output is escaped BEFORE
    # anything else runs, linking included.
    links = {"Smith & Co": "https://odoo.test/web#model=res.partner&id=9&view_type=form"}
    out = render_markdown("Paid by Smith & Co <b>not bold</b>", links)
    assert "&lt;b&gt;not bold&lt;/b&gt;" in out
    # ...and the link text carries the ESCAPED name.
    assert ">Smith &amp; Co</a>" in out


def test_a_record_named_like_markup_cannot_escape_its_link() -> None:
    # The poisoned probe contact is exactly this shape.
    nasty = {"Harborview <script>": "https://odoo.test/web#model=res.partner&id=85&view_type=form"}
    out = render_markdown("Contact Harborview <script> is odd.", nasty)
    assert "<script>" not in out
    assert "&lt;script&gt;" in out


def test_the_longest_name_wins() -> None:
    # "Acme" is also a known record; it must not carve up "Acme Corporation".
    out = render_markdown("Billed to Acme Corporation today.", LINKS)
    expected = (
        '<a href="https://odoo.test/web#model=res.partner&amp;id=7&amp;view_type=form">'
        "Acme Corporation</a>"
    )
    assert expected in out
    assert "res.partner/99" not in out


def test_each_record_is_linked_only_once() -> None:
    # A link on every repetition is noise, not navigation.
    out = render_markdown("INV/2026/00026 is overdue. Chase INV/2026/00026 today.", LINKS)
    assert out.count("<a href") == 1
    assert out.count("INV/2026/00026") == 2


def test_records_only_inside_a_table_get_a_footer_instead() -> None:
    """Telegram renders no link inside <pre>, and tables ARE <pre> blocks —
    which is exactly where invoice numbers cluster. Without this fallback
    the feature would silently do nothing on the commonest answer shape."""
    table = "| Invoice | Amount |\n|---|---|\n| INV/2026/00026 | 1,840 |"
    out = render_markdown(table, LINKS)

    assert "<pre>" in out
    assert "Open in Odoo" in out
    assert (
        '<a href="https://odoo.test/web#model=account.move&amp;id=26&amp;view_type=form">INV/2026/00026</a>'
        in out
    )


def test_no_footer_when_everything_was_linked_inline() -> None:
    out = render_markdown("Invoice INV/2026/00026 is overdue.", LINKS)
    assert "Open in Odoo" not in out


def test_unmentioned_records_are_never_listed() -> None:
    # Only what the answer actually talks about.
    out = render_markdown("Nothing to report today.", LINKS)
    assert "<a href" not in out
    assert "Open in Odoo" not in out


def test_links_survive_bold_and_bullets() -> None:
    out = render_markdown("- **INV/2026/00026** is overdue", LINKS)
    assert "•" in out
    assert "<a href" in out


# --- malformed HTML is what actually produced "a lot of asterisks" ---------


def test_bold_italic_does_not_cross_its_tags() -> None:
    """ "***x***" rendered as "<b><i>x</b></i>" — crossed tags, which
    Telegram rejects outright, dropping the WHOLE reply to unformatted text.
    That is how a tag bug surfaced as asterisks rather than as an error."""
    assert render_markdown("***Bold and italic***") == "<b><i>Bold and italic</i></b>"


def test_a_bold_heading_is_not_double_wrapped() -> None:
    # "#### **1. Purpose**" produced "<b><b>1. Purpose</b></b>".
    assert render_markdown("#### **1. Purpose**") == "<b>1. Purpose</b>"
    assert render_markdown("## Plain Heading") == "<b>Plain Heading</b>"


def test_no_stray_asterisk_survives_a_realistic_answer() -> None:
    answer = (
        "### **Travel claims**\n"
        "* **Air Travel (Section 9):**\n"
        "  * Lowest logical airfare.\n"
        "* ***Important:*** receipts required.\n"
    )
    rendered = render_markdown(answer)

    assert "*" not in rendered
    assert rendered.count("<b>") == rendered.count("</b>")
    assert rendered.count("<i>") == rendered.count("</i>")


# --- the fallback must degrade to prose, not to markup --------------------


def test_plain_text_strips_every_marker() -> None:
    from ont_agent.telegram_format import plain_text

    text = (
        "### **Travel claims**\n"
        "* **Air Travel:** lowest fare\n"
        "* ***Important:*** keep `receipts`\n"
        "---\n"
    )
    stripped = plain_text(text)

    assert "*" not in stripped
    assert "#" not in stripped
    assert "`" not in stripped
    assert "Travel claims" in stripped
    assert "keep receipts" in stripped


def test_plain_text_keeps_the_words() -> None:
    from ont_agent.telegram_format import plain_text

    # A formatting failure should cost formatting, not legibility.
    assert plain_text("**Deadline:** submit within 10 days") == "Deadline: submit within 10 days"
