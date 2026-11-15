"""Turning an uploaded Telegram file into something the model can read.

Images and Portable Document Format (PDF) files both travel to the model as
LangChain *data content blocks*, which ``langchain-openrouter`` converts into
OpenRouter's wire format for us — an image becomes ``image_url`` with a
``data:`` Uniform Resource Identifier (URI), a PDF becomes OpenRouter's
``file`` block. OpenRouter parses the PDF server-side, which is why nothing
here reads PDFs itself and no PDF library is a dependency.

Verified against the live application programming interface (API) on
2026-09-02: a generated PDF and a generated image were both read back
correctly, and each turn's real cost arrived through the usual callback, so
uploads join the same spend ledger as every other model call.

Like ``transcribe.py``, this module never raises into the handler: a refusal
is a sentence for the user, not an exception.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass
from typing import Any

from ont_agent.config import Settings
from ont_agent.log import get_logger

logger = get_logger(__name__)

#: Media type -> the LangChain content-block kind it becomes. Internal to
#: this module's logic rather than configuration: adding a type means
#: teaching the code how to describe and block-encode it, not flipping a
#: setting. "file" is OpenRouter's PDF path; "image" is the vision path.
SUPPORTED: dict[str, str] = {
    "image/jpeg": "image",
    "image/png": "image",
    "image/webp": "image",
    "image/gif": "image",
    "application/pdf": "file",
}

#: Plain names for the commonest things people send that we cannot read, so
#: the refusal can say WHAT was rejected. "I can't read that" teaches nobody
#: where the boundary is; "that's a spreadsheet" does.
_FAMILIES: dict[str, str] = {
    "spreadsheetml": "spreadsheet",
    "ms-excel": "spreadsheet",
    "csv": "spreadsheet",
    "wordprocessingml": "Word document",
    "msword": "Word document",
    "presentationml": "slide deck",
    "powerpoint": "slide deck",
    "zip": "archive",
    "rar": "archive",
    "video/": "video",
    "audio/": "audio file",
    "text/": "text file",
}

#: Asked on the user's behalf when they send a file with no caption. Stated
#: back to them in the placeholder, so a surprising answer is traceable to
#: the question rather than looking like the bot inventing a task.
DEFAULT_QUESTION = "What is in this document?"


@dataclass(frozen=True)
class Attachment:
    """One uploaded file, held in memory for the length of a single turn."""

    kind: str
    mime: str
    filename: str
    data: bytes


def family(mime: str) -> str:
    """A plain-English name for an unsupported media type."""
    lowered = mime.lower()
    for marker, name in _FAMILIES.items():
        if marker in lowered:
            return name
    return "file"


def check(mime: str, size_bytes: int, settings: Settings) -> str | None:
    """Validate before downloading; a sentence to send back, or None.

    Both checks run on what Telegram DECLARES about the file, so an
    unsupported or oversized upload costs no download and no model call —
    the same order the voice path uses.
    """
    if mime not in SUPPORTED:
        return (
            f"I can read images and PDF files — that looks like a {family(mime)}. "
            "Send a photo or a PDF, or ask me in text."
        )
    limit = settings.upload_max_mb * 1024 * 1024
    if size_bytes > limit:
        return (
            f"That file is {size_bytes / 1048576:.1f} MB — I can handle up to "
            f"{settings.upload_max_mb} MB. Send a smaller one, or a few pages of it."
        )
    return None


def describe(attachment: Attachment) -> str:
    """``invoice.pdf (PDF, 1.2 MB)`` — for the placeholder, log and thread."""
    kind = "PDF" if attachment.mime == "application/pdf" else "image"
    size = len(attachment.data)
    shown = f"{size / 1048576:.1f} MB" if size >= 1048576 else f"{size // 1024} KB"
    return f"{attachment.filename} ({kind}, {shown})"


def content_block(attachment: Attachment) -> dict[str, Any]:
    """The LangChain data content block for this file.

    ``langchain-openrouter`` recognises these and rewrites them into
    OpenRouter's own shapes; an unrecognised block would be passed through
    raw and rejected, so the key names here are load-bearing.
    """
    block: dict[str, Any] = {
        "type": attachment.kind,
        "source_type": "base64",
        "data": base64.b64encode(attachment.data).decode("ascii"),
        "mime_type": attachment.mime,
    }
    if attachment.kind == "file":
        # OpenRouter shows the filename to the model. A real name ("ACME
        # bill March.pdf") is context; "document.pdf" throws it away.
        block["filename"] = attachment.filename
    return block


def message_content(question: str, attachment: Attachment) -> list[dict[str, Any]]:
    """The user turn: the question, then the file."""
    return [{"type": "text", "text": question}, content_block(attachment)]


def thread_placeholder(question: str, attachment: Attachment) -> str:
    """What replaces the file in conversation memory once the turn is done.

    The encoded bytes must not stay on the thread: they would be written to
    ``memory.db`` AND re-sent to the model on every later question in that
    chat, so one upload would quietly tax the whole conversation.
    """
    return f"{question}\n\n[uploaded: {describe(attachment)}]"
