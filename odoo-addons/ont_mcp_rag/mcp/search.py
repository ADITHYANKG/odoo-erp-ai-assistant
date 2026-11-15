"""The ``search_documents`` MCP tool.

Everything in this file exists to stop one specific bad outcome: someone
asks *"what is the harassment policy?"* and receives a confident answer
that is not this company's policy. That happens in three ways, and each is
answered here rather than left to luck.

- The assistant never searches, and answers from its own general
  knowledge. The tool DESCRIPTION is the first line of defence, so it says
  so explicitly.
- The search finds the wrong thing, or only part of it. Hence hybrid
  ranking and neighbouring-section expansion.
- The search finds nothing, and the silence is filled with invention.
  Hence an explicit, unambiguous empty result that also distinguishes
  "no such policy" from "not indexed yet".
"""

from __future__ import annotations

import logging

from odoo import _, api, models
from odoo.addons.muk_mcp.core.tool import mcp_tool

from ..tools.embeddings import embed_texts

_logger = logging.getLogger(__name__)

PARAM_TOP_K = 'ont_mcp_rag.top_k'
PARAM_MIN_SIMILARITY = 'ont_mcp_rag.min_similarity'
PARAM_MIN_TOP_SIMILARITY = 'ont_mcp_rag.min_top_similarity'
#: Per-passage floor: below this a passage adds noise rather than support.
DEFAULT_MIN_SIMILARITY = 0.25
#: Topic gate: below this the best passage is obvious noise and the honest
#: answer is "I could not find it".
#:
#: Deliberately LOW, and this is the important design decision here.
#: Measured against the live policy on 2026-09-02 over 14 questions it does
#: answer and 6 it does not, the two populations barely separate:
#:
#:     lowest genuine question  0.379  ("can I take a second job?")
#:     highest absent question  0.357  ("what is our crypto policy?")
#:
#: A threshold in that 0.022-wide band would "work" on the sample and break
#: on the next phrasing. Cosine similarity is simply not a reliable relevance
#: classifier: an absent question sharing the document's vocabulary
#: ("policy", "company") scores higher than a genuine one that does not
#: ("work from home" against a section headed "Remote and Hybrid Work").
#:
#: So the gate only removes what is unambiguously unrelated - coffee
#: machines (0.137), laser cutters (0.180), scuba depths (0.184), bitcoin
#: refunds (0.230), van warranties (0.264) - leaving comfortable headroom
#: below every genuine question. Judging the borderline is the MODEL's job:
#: it can read a passage headed "1. Purpose" and tell that it does not
#: answer a question about crypto trading, which no scalar threshold can.
#: The SYSTEM_PROMPT rule requiring it to decline when the passages do not
#: address the question is the other half of this, and the more important
#: half.
DEFAULT_MIN_TOP_SIMILARITY = 0.30
DEFAULT_TOP_K = 6
#: Neighbours are pulled only for the strongest few hits. Expanding all six
#: produced eighteen passages per question against the live document -
#: enough to crowd the answer and bury the passage that mattered.
NEIGHBOUR_EXPANSION_HITS = 3
#: How much of a chunk to show. Enough to answer from, short enough that six
#: of them do not crowd out the conversation.
SNIPPET_CHARS = 1200
#: Reciprocal rank fusion uses 1 / (60 + rank), so a passage ranked FIRST by
#: either the vector or the keyword side scores 1/61. A passage that topped
#: one ranking is a real hit even when its cosine score is modest - an exact
#: clause number or defined term matches on keywords and scores poorly on
#: similarity - so it survives the floor below.
RRF_TOP_RANK_SCORE = 1.0 / 61.0


class MCPMixin(models.AbstractModel):
    """Add the ``search_documents`` MCP tool."""

    _inherit = 'muk_mcp.mixin'

    @api.model
    @mcp_tool(
        name='search_documents',
        description=(
            'Search the company\'s own policy, procedure and contract '
            'documents (Human Resources policy, employee handbook, approval '
            'rules, contract terms) and return the relevant passages. USE '
            'THIS whenever the user asks what a rule, entitlement, '
            'procedure, limit or contractual term IS - for example notice '
            'periods, leave, harassment, conduct, expenses, approval '
            'authority, probation, termination, or anything phrased as '
            '"what is our policy on ...". NEVER answer such questions from '
            'your own knowledge: a plausible answer that is not this '
            'company\'s actual policy is worse than saying you could not '
            'find it. Returns passages with the document name and section '
            'heading so you can cite them. An empty "results" list means '
            'nothing matched - say so plainly rather than guessing.'
        ),
        # Declared explicitly so "query" is REQUIRED. Left to inference the
        # model opened with search_documents({}), got told there was no
        # search text, and retried with variations - three calls where one
        # would do, and on slower questions enough to exhaust the recursion
        # limit and degrade the answer entirely.
        input_schema={
            'type': 'object',
            'properties': {
                'query': {
                    'type': 'string',
                    'description': (
                        'What to look for, in natural language - the user\'s '
                        'question is usually the best query.'
                    ),
                },
                'limit': {
                    'type': 'integer',
                    'description': 'Passages to return (default 6).',
                },
                'folder': {
                    'type': 'string',
                    'description': 'Restrict the search to one folder by name.',
                },
            },
            'required': ['query'],
        },
        category='read',
    )
    def _mcp_search_documents(self, query: str, limit: int = 0, folder: str = '') -> dict:
        """Find passages in indexed documents that answer ``query``.

        :param query: a natural-language question or phrase.
        :param limit: passages to return; defaults to the configured top-k.
        :param folder: restrict to one folder by name.
        """
        query = (query or '').strip()
        if not query:
            return {
                'results': [],
                'searched_folders': [],
                'pending_documents': 0,
                'note': 'No search text was given.',
            }

        params = self.env['ir.config_parameter'].sudo()
        limit = int(limit) or int(params.get_param(PARAM_TOP_K, DEFAULT_TOP_K))
        floor = float(params.get_param(PARAM_MIN_SIMILARITY, DEFAULT_MIN_SIMILARITY))
        gate = float(params.get_param(PARAM_MIN_TOP_SIMILARITY, DEFAULT_MIN_TOP_SIMILARITY))

        # Read as the CALLING user, so documents.document record rules decide
        # what is findable. A folder the user cannot open contributes nothing,
        # without this tool knowing anything about who they are.
        folders = self.env['documents.folder'].search([
            ('ont_rag_indexed', '=', True),
        ] + ([('name', 'ilike', folder)] if folder else []))
        visible = self.env['documents.document'].search([
            ('folder_id', 'child_of', folders.ids),
        ])
        if not visible:
            return {
                'results': [],
                'searched_folders': folders.mapped('name'),
                'pending_documents': 0,
                'note': _('No AI-searchable documents are available to you.'),
            }

        chunks = self.env['ont.document.chunk'].sudo()
        pending = chunks.search_count([
            ('document_id', 'in', visible.ids), ('state', '!=', 'indexed'),
        ])

        vectors, _cost = embed_texts(self.env, [query])
        ranked = chunks._hybrid_search(query, vectors[0], visible.ids, limit)

        # Topic gate FIRST. Without it the top-ranked passage always
        # survives however poor it is, and a question the documents simply
        # do not answer comes back looking answered.
        best = max((similarity for _id, _fused, similarity in ranked), default=0.0)
        if best < gate:
            _logger.info(
                'search_documents(%r): best similarity %.3f below gate %.2f - nothing found',
                query[:60], best, gate,
            )
            return {
                'results': [],
                'searched_folders': folders.mapped('name'),
                'pending_documents': pending,
                'note': _('Nothing on that subject was found in the documents searched.'),
            }

        keep = [
            (chunk_id, fused, similarity)
            for chunk_id, fused, similarity in ranked
            if similarity >= floor or fused >= RRF_TOP_RANK_SCORE
        ]
        hits = chunks.browse([chunk_id for chunk_id, _f, _s in keep])

        results = [
            self._mcp_document_passage(chunk, similarity)
            for chunk, (_id, _fused, similarity) in zip(hits, keep, strict=True)
        ]
        results += self._mcp_neighbouring_passages(hits[:NEIGHBOUR_EXPANSION_HITS], hits)

        _logger.info(
            'search_documents(%r) -> %s passage(s), %s pending',
            query[:60], len(results), pending,
        )
        return {
            'results': results,
            'searched_folders': folders.mapped('name'),
            'pending_documents': pending,
        }

    @api.model
    def _mcp_document_passage(self, chunk, similarity=None, adjacent=False):
        return {
            'document': chunk.document_id.display_name,
            'document_id': chunk.document_id.id,
            'folder': chunk.document_id.folder_id.display_name,
            'section': chunk.heading or '',
            'text': chunk.content[:SNIPPET_CHARS],
            'score': round(similarity, 3) if similarity is not None else None,
            'adjacent': adjacent,
        }

    @api.model
    def _mcp_neighbouring_passages(self, expand, hits):
        """The sections immediately around each hit.

        "What happens if I am harassed?" is genuinely answered by the
        definition, the grievance procedure AND the non-retaliation clause.
        A top-ranked passage alone would read as a complete answer while
        being an incomplete one, which on an HR question is the damaging
        kind of wrong.
        """
        if not expand:
            return []
        chunks = self.env['ont.document.chunk'].sudo()
        wanted = chunks
        for hit in expand:
            wanted |= chunks.search([
                ('document_id', '=', hit.document_id.id),
                ('sequence', 'in', [hit.sequence - 1, hit.sequence + 1]),
                ('state', '=', 'indexed'),
            ])
        return [
            self._mcp_document_passage(chunk, adjacent=True)
            for chunk in wanted - hits
        ]
