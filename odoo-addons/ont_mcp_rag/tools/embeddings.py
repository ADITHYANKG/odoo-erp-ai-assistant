"""Turning text into vectors, through OpenRouter.

OpenRouter is the only language-model provider this project uses, and its
``/embeddings`` endpoint is OpenAI-compatible. Verified live on 2026-09-02:
``openai/text-embedding-3-small`` returns 1536 dimensions and reports
``usage.cost``, so indexing spend is measurable rather than guessed at.

Deliberately plain ``requests`` rather than any provider software
development kit: one endpoint, one call shape, and Odoo already ships
``requests``.
"""

from __future__ import annotations

import logging

import requests

from odoo import _
from odoo.exceptions import UserError

_logger = logging.getLogger(__name__)

#: System parameters. Every one of these is configuration, never a literal
#: in the code: the model and its width are COUPLED, and changing the model
#: without the width silently corrupts the index rather than failing.
PARAM_API_KEY = 'ont_mcp_rag.openrouter_api_key'
PARAM_BASE_URL = 'ont_mcp_rag.openrouter_base_url'
PARAM_MODEL = 'ont_mcp_rag.embed_model'
PARAM_DIMS = 'ont_mcp_rag.embed_dims'

DEFAULT_BASE_URL = 'https://openrouter.ai/api/v1'
DEFAULT_MODEL = 'openai/text-embedding-3-small'
DEFAULT_DIMS = 1536
#: One request per batch; OpenRouter accepts a list of inputs. Kept modest so
#: a failure costs one batch, not an entire document.
BATCH_SIZE = 32
TIMEOUT_SECONDS = 120


def embed_config(env) -> tuple[str, str, str, int]:
    """(api_key, base_url, model, dims) from system parameters."""
    params = env['ir.config_parameter'].sudo()
    api_key = params.get_param(PARAM_API_KEY, '')
    base_url = params.get_param(PARAM_BASE_URL, DEFAULT_BASE_URL).rstrip('/')
    model = params.get_param(PARAM_MODEL, DEFAULT_MODEL)
    dims = int(params.get_param(PARAM_DIMS, DEFAULT_DIMS))
    return api_key, base_url, model, dims


def embed_texts(env, texts: list[str]) -> tuple[list[list[float]], float]:
    """Embed a batch of texts. Returns (vectors, cost_in_usd).

    :raise UserError: no key configured, the request failed, or the provider
        returned vectors of a width that does not match ``embed_dims``.

    The width check is not paranoia. A mismatched model writes vectors that
    PostgreSQL rejects or, worse, that compare meaninglessly against the
    ones already stored - a corruption that shows up as bad answers rather
    than as an error.
    """
    if not texts:
        return [], 0.0
    api_key, base_url, model, dims = embed_config(env)
    if not api_key:
        raise UserError(_(
            'No OpenRouter API key configured. Set the system parameter %s '
            '(Settings > Technical > Parameters > System Parameters).',
            PARAM_API_KEY,
        ))

    try:
        response = requests.post(
            f'{base_url}/embeddings',
            headers={'Authorization': f'Bearer {api_key}'},
            json={'model': model, 'input': texts},
            timeout=TIMEOUT_SECONDS,
        )
    except requests.RequestException as error:
        raise UserError(_('Embedding request failed: %s', error)) from error

    if response.status_code != 200:
        raise UserError(_(
            'Embedding request refused (HTTP %(code)s): %(body)s',
            code=response.status_code, body=response.text[:300],
        ))

    payload = response.json()
    # The provider may return the batch out of order; index is authoritative.
    rows = sorted(payload.get('data') or [], key=lambda row: row.get('index', 0))
    vectors = [row['embedding'] for row in rows]
    if len(vectors) != len(texts):
        raise UserError(_(
            'Embedding provider returned %(got)s vectors for %(want)s inputs.',
            got=len(vectors), want=len(texts),
        ))
    for vector in vectors:
        if len(vector) != dims:
            raise UserError(_(
                'Embedding model %(model)s returned %(got)s dimensions but '
                '%(param)s says %(want)s. Set both together, or the stored '
                'index becomes meaningless.',
                model=model, got=len(vector), param=PARAM_DIMS, want=dims,
            ))

    cost = float((payload.get('usage') or {}).get('cost') or 0.0)
    return vectors, cost
