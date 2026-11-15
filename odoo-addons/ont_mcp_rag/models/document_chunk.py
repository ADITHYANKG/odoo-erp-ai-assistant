"""Embedded pieces of Odoo Documents, and the job that keeps them current.

Shaped after Odoo's own ``ai.embedding`` (attachment foreign key with
cascade delete, ``sequence``, ``content``, a pgvector column, an ivfflat
cosine index) because that shape is proven. It could not be reused
directly: ``ai.embedding`` is driven by ``ai.agent.source`` records and its
``embedding_model`` is a Selection restricted to first-party providers,
which does not include OpenRouter.
"""

from __future__ import annotations

import logging

from odoo import _, api, fields, models
from odoo.exceptions import UserError
from odoo.tools import config

from ..tools.chunker import CHUNKER_VERSION, chunk_text, clean_text
from ..tools.embeddings import BATCH_SIZE, embed_config, embed_texts

_logger = logging.getLogger(__name__)

PARAM_MAX_CHARS = 'ont_mcp_rag.chunk_max_chars'
PARAM_MIN_CHARS = 'ont_mcp_rag.chunk_min_chars'
PARAM_OVERLAP = 'ont_mcp_rag.overlap_chars'
#: How many times a failing document is retried before it is left alone.
#: Without a ceiling one unreadable file is re-embedded on every cron run,
#: for ever, at cost.
MAX_ATTEMPTS = 3


class DocumentChunk(models.Model):
    _name = 'ont.document.chunk'
    _description = 'Searchable Document Chunk'
    _order = 'document_id, sequence'

    document_id = fields.Many2one(
        'documents.document', required=True, ondelete='cascade', index=True,
    )
    attachment_id = fields.Many2one('ir.attachment', required=True, ondelete='cascade')
    #: The attachment's checksum at index time. Drives BOTH deduplication
    #: (the same file uploaded three times is indexed once) and staleness
    #: (a replaced file has a new checksum, so its chunks are rebuilt).
    checksum = fields.Char(required=True, index=True)
    sequence = fields.Integer(default=0)
    #: "44. Harassment", or "76. Organizational Change - 79. Notice Period"
    #: when short sections were merged. Empty when the document has no
    #: detectable headings.
    heading = fields.Char()
    content = fields.Text(required=True)
    embedding_model = fields.Char(required=True)
    #: The width is fixed in the column type, so it CANNOT be a runtime
    #: setting - the same constraint Odoo's own ai.embedding lives with.
    #: ont_mcp_rag.embed_dims exists to VALIDATE that the configured model
    #: agrees with this number (see tools/embeddings.py), because a model of
    #: a different width would otherwise write vectors that compare
    #: meaninglessly against the ones already stored. Moving to a
    #: differently-sized model means changing this and re-indexing.
    #: The column itself is ``embedding vector(1536)``, created by ``init()``
    #: below (Odoo 17 has no pgvector ORM field), and only ever read or
    #: written through SQL.
    state = fields.Selection(
        [('pending', 'Pending'), ('indexed', 'Indexed'), ('failed', 'Failed')],
        default='pending', required=True, index=True,
    )
    attempts = fields.Integer(default=0)
    error = fields.Char()
    #: Which chunker built this row. A change to the splitter must rebuild
    #: existing chunks, and the file checksum cannot detect that - the file
    #: did not change, the code did.
    chunker_version = fields.Integer(default=0, index=True)

    def init(self):
        """Create the pgvector column and the two search indexes.

        Odoo 17 cannot declare either through the ORM: ``models.Index`` does
        not exist and no field type maps to ``vector``. The ``vector``
        extension itself is created by the module's ``pre_init_hook``.
        """
        super().init()
        cr = self.env.cr
        cr.execute(
            'ALTER TABLE ont_document_chunk ADD COLUMN IF NOT EXISTS embedding vector(1536)'
        )
        cr.execute(
            'CREATE INDEX IF NOT EXISTS ont_document_chunk_embedding_idx '
            'ON ont_document_chunk USING ivfflat (embedding vector_cosine_ops) '
            'WITH (lists = 100)'
        )
        cr.execute(
            'CREATE INDEX IF NOT EXISTS ont_document_chunk_content_fts_idx '
            "ON ont_document_chunk USING gin (to_tsvector('english', content))"
        )

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def _checkpoint(self):
        """Commit progress so far, unless we are inside a test.

        Batching matters in production: embedding a large folder can take
        minutes, and without intermediate commits a failure at the end
        throws away every document already done, to be paid for again on the
        next run. Odoo forbids committing inside a test (it breaks the
        rollback that isolates one test from the next), and
        ``ir.cron._commit_progress`` is no help - outside a cron it simply
        commits too.
        """
        if not config['test_enable']:
            self.env.cr.commit()

    @api.model
    def _cron_index_documents(self, batch_size=20):
        """Chunk and embed everything in AI-searchable folders.

        Runs on a schedule rather than on upload so that a slow or failing
        embedding provider can never block someone saving a file, and so a
        provider outage is retried by itself rather than losing documents.
        """
        documents = self._documents_needing_indexing()
        if not documents:
            return False
        _logger.info('RAG indexing: %s document(s) need chunking', len(documents))
        for document in documents[:batch_size]:
            try:
                self._index_document(document)
                self._checkpoint()
            except Exception as error:  # noqa: BLE001 - one bad file must not stop the batch
                self.env.cr.rollback()
                _logger.exception('RAG indexing failed for document %s', document.id)
                self._mark_failed(document, str(error)[:500])
                self._checkpoint()
        self._embed_pending()
        return True

    @api.model
    def _searchable_folders(self):
        return self.env['documents.folder'].sudo().search(
            [('ont_rag_indexed', '=', True)]
        )

    @api.model
    def _documents_needing_indexing(self):
        """Documents in searchable folders whose chunks are absent or stale."""
        folders = self._searchable_folders()
        if not folders:
            return self.env['documents.document']
        candidates = self.env['documents.document'].sudo().search([
            ('folder_id', 'child_of', folders.ids),
            ('attachment_id', '!=', False),
        ])
        # One set of chunks per DISTINCT FILE. Three identical copies of the
        # same policy would otherwise each claim a slot in every result.
        # Only chunks from the CURRENT chunker count as done; anything
        # older is stale even though its file is unchanged.
        indexed = set(
            self.sudo()
            .search([('chunker_version', '>=', CHUNKER_VERSION)])
            .mapped('checksum')
        )
        seen: set[str] = set()
        needed = self.env['documents.document']
        for document in candidates:
            checksum = document.attachment_id.checksum
            content = document.attachment_id.index_content
            if not checksum or not content:
                continue  # nothing extracted (an image, or a scanned PDF)
            if checksum in indexed or checksum in seen:
                continue
            seen.add(checksum)
            needed |= document
        return needed

    @api.model
    def _index_document(self, document):
        """Replace one document's chunks. Embeddings follow separately."""
        params = self.env['ir.config_parameter'].sudo()
        attachment = document.attachment_id
        _, _, model, _dims = embed_config(self.env)

        self.sudo().search([('document_id', '=', document.id)]).unlink()
        chunks = chunk_text(
            clean_text(attachment.index_content or ''),
            max_chars=int(params.get_param(PARAM_MAX_CHARS, 1800)),
            min_chars=int(params.get_param(PARAM_MIN_CHARS, 250)),
            overlap_chars=int(params.get_param(PARAM_OVERLAP, 200)),
        )
        self.sudo().create([{
            'document_id': document.id,
            'attachment_id': attachment.id,
            'checksum': attachment.checksum,
            'sequence': chunk.sequence,
            'heading': chunk.heading,
            'content': chunk.content,
            'embedding_model': model,
            'chunker_version': CHUNKER_VERSION,
            'state': 'pending',
        } for chunk in chunks])
        _logger.info(
            'RAG indexing: %s -> %s chunk(s)', document.display_name, len(chunks),
        )

    @api.model
    def _embed_pending(self):
        """Embed pending chunks in batches, recording real spend."""
        pending = self.sudo().search([
            ('state', '=', 'pending'), ('attempts', '<', MAX_ATTEMPTS),
        ])
        if not pending:
            return
        total_cost = 0.0
        for start in range(0, len(pending), BATCH_SIZE):
            batch = pending[start:start + BATCH_SIZE]
            for record in batch:
                record.attempts += 1
            try:
                vectors, cost = embed_texts(self.env, [record.content for record in batch])
            except UserError as error:
                # A configuration problem fails the whole run identically for
                # every batch, so stop rather than burn the attempt budget.
                _logger.warning('RAG embedding batch failed: %s', error)
                batch.write({'error': str(error)[:500]})
                for record in batch:
                    if record.attempts >= MAX_ATTEMPTS:
                        record.state = 'failed'
                self._checkpoint()
                return
            total_cost += cost
            for record, vector in zip(batch, vectors, strict=True):
                self.env.cr.execute(
                    'UPDATE ont_document_chunk SET embedding = %s::vector WHERE id = %s',
                    (str(list(vector)), record.id),
                )
                record.write({'state': 'indexed', 'error': False})
            self._checkpoint()
        _logger.info(
            'RAG embedding: %s chunk(s) embedded, cost $%.6f', len(pending), total_cost,
        )

    @api.model
    def _mark_failed(self, document, message):
        existing = self.sudo().search([('document_id', '=', document.id)])
        if existing:
            existing.write({'state': 'failed', 'error': message})

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    @api.model
    def _hybrid_search(self, query, query_vector, allowed_document_ids, limit):
        """Rank chunks by cosine similarity AND keyword match, fused.

        Neither half is sufficient. Cosine similarity misses exact terms - a
        clause number, a defined term, a form name - and keyword search
        misses paraphrase, which is most of how people actually ask ("can I
        work from home?" against a section titled "Remote and Hybrid Work").

        Reciprocal rank fusion combines them without needing the two scores
        to be on comparable scales, which they are not.
        """
        if not allowed_document_ids:
            return []
        query_sql = (
            """
            WITH semantic AS (
                SELECT id, ROW_NUMBER() OVER (ORDER BY embedding <=> %(vector)s::vector) AS rank,
                       1 - (embedding <=> %(vector)s::vector) AS score
                  FROM ont_document_chunk
                 WHERE document_id = ANY(%(docs)s)
                   AND state = 'indexed' AND embedding IS NOT NULL
                 ORDER BY embedding <=> %(vector)s::vector
                 LIMIT %(pool)s
            ), keyword AS (
                SELECT id, ROW_NUMBER() OVER (
                           ORDER BY ts_rank(to_tsvector('english', content),
                                            plainto_tsquery('english', %(query)s)) DESC
                       ) AS rank
                  FROM ont_document_chunk
                 WHERE document_id = ANY(%(docs)s)
                   AND state = 'indexed'
                   AND to_tsvector('english', content)
                       @@ plainto_tsquery('english', %(query)s)
                 LIMIT %(pool)s
            )
            SELECT COALESCE(s.id, k.id) AS id,
                   COALESCE(1.0 / (60 + s.rank), 0) + COALESCE(1.0 / (60 + k.rank), 0) AS fused,
                   COALESCE(s.score, 0) AS similarity
              FROM semantic s
              FULL OUTER JOIN keyword k ON k.id = s.id
             ORDER BY fused DESC
             LIMIT %(limit)s
            """
        )
        self.env.cr.execute(query_sql, {
            'vector': str(query_vector),
            'docs': list(allowed_document_ids),
            'query': query,
            'pool': max(limit * 4, 20),
            'limit': limit,
        })
        rows = self.env.cr.fetchall()
        return [(row[0], float(row[1]), float(row[2])) for row in rows]
