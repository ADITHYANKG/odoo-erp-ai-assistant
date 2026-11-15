"""Speech to text for Telegram voice notes, through OpenRouter.

Uses OpenRouter's dedicated ``/audio/transcriptions`` endpoint (an
OpenAI-compatible multipart upload), NOT ``/chat/completions``: a
transcription model posted to the chat endpoint is rejected outright —

    "openai/gpt-transcribe is a transcription model and cannot be used with
     the chat/completions endpoint."

Verified against the live API on 2026-09-01: Telegram's OGG/Opus is accepted
as-is, so no ffmpeg transcoding step is needed, and the response carries
``usage.cost`` so spend is recorded from the real billed amount like every
other model call in this service.

One request, no tools, so this talks to httpx directly rather than going
through LangChain.
"""

from __future__ import annotations

from dataclasses import dataclass

import httpx

from ont_agent.config import Settings
from ont_agent.log import get_logger

logger = get_logger(__name__)

#: Telegram voice notes are always OGG/Opus; the endpoint infers the format
#: from the filename and content type, both of which we supply.
VOICE_FILENAME = "voice.ogg"
VOICE_MIME = "audio/ogg"


@dataclass(frozen=True)
class Transcript:
    """What was heard, and what hearing it cost.

    ``text`` is empty when transcription failed OR when the audio simply
    contained no speech — the caller treats both the same way, since to the
    user they are the same experience.
    """

    text: str = ""
    cost_usd: float = 0.0


async def transcribe(
    settings: Settings,
    audio: bytes,
    client: httpx.AsyncClient | None = None,
    *,
    filename: str = VOICE_FILENAME,
    mime: str = VOICE_MIME,
) -> Transcript:
    """Transcribe a voice note. Never raises.

    A failed transcription must degrade to "I couldn't make that out"
    rather than an exception: the person is in a car, and a stack trace
    helps nobody.
    """
    if not settings.openrouter_api_key or not audio:
        return Transcript()

    async def call(active: httpx.AsyncClient) -> httpx.Response:
        return await active.post(
            f"{settings.openrouter_base_url}/audio/transcriptions",
            headers={"Authorization": f"Bearer {settings.openrouter_api_key}"},
            files={"file": (filename, audio, mime)},
            data={"model": settings.transcribe_model},
        )

    try:
        if client is not None:
            response = await call(client)
        else:
            async with httpx.AsyncClient(timeout=120.0) as fresh:
                response = await call(fresh)
    except httpx.HTTPError:
        logger.warning("Transcription request failed", exc_info=True)
        return Transcript()

    if response.status_code != 200:
        logger.warning(
            "Transcription refused (HTTP %s): %s", response.status_code, response.text[:200]
        )
        return Transcript()

    try:
        payload = response.json()
    except ValueError:
        logger.warning("Transcription returned a non-JSON body")
        return Transcript()

    usage = payload.get("usage") or {}
    cost = usage.get("cost")
    return Transcript(
        text=str(payload.get("text") or "").strip(),
        cost_usd=float(cost) if isinstance(cost, int | float) else 0.0,
    )
