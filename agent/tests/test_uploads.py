"""Unit tests for ont_agent.uploads — turning a file into model input."""

from __future__ import annotations

import base64

from ont_agent.config import Settings
from ont_agent.uploads import (
    Attachment,
    check,
    content_block,
    describe,
    family,
    message_content,
    thread_placeholder,
)

SETTINGS = Settings(mcp_url="http://odoo:8160/mcp", mcp_bearer_key="k")

PDF = Attachment(kind="file", mime="application/pdf", filename="bill.pdf", data=b"%PDF-1.4 x")
IMAGE = Attachment(kind="image", mime="image/png", filename="photo.jpg", data=b"\x89PNG rest")


# --- content blocks ---------------------------------------------------------


def test_an_image_becomes_a_base64_image_block() -> None:
    block = content_block(IMAGE)
    assert block["type"] == "image"
    assert block["source_type"] == "base64"
    assert block["mime_type"] == "image/png"
    assert base64.b64decode(block["data"]) == IMAGE.data


def test_a_pdf_becomes_a_file_block_carrying_its_name() -> None:
    # OpenRouter shows the filename to the model, so a real name is context.
    block = content_block(PDF)
    assert block["type"] == "file"
    assert block["filename"] == "bill.pdf"
    assert base64.b64decode(block["data"]) == PDF.data


def test_an_image_block_carries_no_filename() -> None:
    # The image path has nowhere to put one; sending it would be ignored at
    # best and rejected at worst.
    assert "filename" not in content_block(IMAGE)


def test_the_blocks_are_the_shape_langchain_openrouter_recognises() -> None:
    """The key names are load-bearing: an unrecognised block is passed to
    OpenRouter raw instead of being converted, and rejected."""
    from langchain_openrouter.chat_models import (  # type: ignore[attr-defined]
        is_data_content_block,
    )

    assert is_data_content_block(content_block(IMAGE))
    assert is_data_content_block(content_block(PDF))


def test_message_content_puts_the_question_first() -> None:
    content = message_content("is this in Odoo?", PDF)
    assert content[0] == {"type": "text", "text": "is this in Odoo?"}
    assert content[1]["type"] == "file"


# --- validation -------------------------------------------------------------


def test_supported_types_pass() -> None:
    assert check("application/pdf", 1024, SETTINGS) is None
    assert check("image/jpeg", 1024, SETTINGS) is None
    assert check("image/png", 1024, SETTINGS) is None


def test_an_unsupported_type_is_refused_BY_NAME() -> None:
    """ "I can't read that" teaches nobody where the boundary is."""
    refusal = check(
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", 1024, SETTINGS
    )
    assert refusal is not None
    assert "spreadsheet" in refusal
    assert "images and PDF" in refusal


def test_oversize_is_refused_with_both_numbers() -> None:
    limit = SETTINGS.upload_max_mb
    refusal = check("application/pdf", (limit + 3) * 1024 * 1024, SETTINGS)
    assert refusal is not None
    assert str(limit) in refusal  # what is allowed
    assert f"{limit + 3}" in refusal  # and what they sent


def test_the_size_limit_is_configurable() -> None:
    generous = SETTINGS.model_copy(update={"upload_max_mb": 50})
    assert check("application/pdf", 20 * 1024 * 1024, SETTINGS) is not None
    assert check("application/pdf", 20 * 1024 * 1024, generous) is None


def test_type_is_checked_before_size() -> None:
    # A huge spreadsheet should be told it is a spreadsheet: the type is the
    # reason it can never work, whereas the size is merely today's answer.
    refusal = check("text/csv", 500 * 1024 * 1024, SETTINGS)
    assert refusal is not None
    assert "spreadsheet" in refusal


def test_family_names_the_common_refusals() -> None:
    assert family("application/vnd.ms-excel") == "spreadsheet"
    assert family("application/msword") == "Word document"
    assert family("video/mp4") == "video"
    assert family("application/zip") == "archive"
    assert family("application/x-something-unheard-of") == "file"


# --- description and thread placeholder -------------------------------------


def test_describe_names_the_kind_and_size() -> None:
    assert describe(PDF) == "bill.pdf (PDF, 0 KB)"
    big = Attachment(kind="file", mime="application/pdf", filename="q.pdf", data=b"x" * 2_500_000)
    assert describe(big) == "q.pdf (PDF, 2.4 MB)"


def test_the_thread_placeholder_keeps_the_question_and_drops_the_bytes() -> None:
    """What replaces the file in conversation memory: the encoded payload
    must not survive the turn, or every later question re-sends it."""
    placeholder = thread_placeholder("is this in Odoo?", PDF)
    assert "is this in Odoo?" in placeholder
    assert "bill.pdf" in placeholder
    assert base64.b64encode(PDF.data).decode() not in placeholder
