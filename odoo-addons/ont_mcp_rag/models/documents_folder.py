"""Opt a folder in to AI search."""

from __future__ import annotations

from odoo import _, api, fields, models
from odoo.exceptions import AccessError


class DocumentsFolder(models.Model):
    _inherit = 'documents.folder'

    #: Indexing is opt-in per folder, not automatic for everything. A folder
    #: of scanned receipts competing with the policy library for a match
    #: makes every answer worse, and embedding it costs money for no gain.
    #: Documents inherit the flag from the folder they sit in.
    ont_rag_indexed = fields.Boolean(
        string='Searchable by AI',
        help='Chunk and embed the documents in this folder so the AI '
             'assistant can answer questions from their contents. '
             'Subfolders are included.',
    )
    #: Shown beside the checkbox so "is it working?" is answerable without
    #: opening a database console — the first question anyone asks after
    #: ticking the box.
    ont_rag_chunk_count = fields.Integer(
        string='Indexed Passages', compute='_compute_ont_rag_chunk_count',
    )

    @api.depends('ont_rag_indexed')
    def _compute_ont_rag_chunk_count(self):
        chunks = self.env['ont.document.chunk'].sudo()
        for record in self:
            record.ont_rag_chunk_count = chunks.search_count([
                ('document_id.folder_id', 'child_of', record.id), ('state', '=', 'indexed'),
            ])

    def write(self, vals):
        """Only administrators decide what leaves for the embedding provider.

        Opting a folder out deletes its passages, so text of documents that
        are no longer meant to be searchable does not linger in the index.
        """
        if 'ont_rag_indexed' in vals and not (
            self.env.su or self.env.user.has_group('base.group_system')
        ):
            raise AccessError(_('Only administrators can make a folder searchable by AI.'))
        result = super().write(vals)
        if 'ont_rag_indexed' in vals and not vals['ont_rag_indexed']:
            self.env['ont.document.chunk'].sudo().search([
                ('document_id.folder_id', 'child_of', self.ids),
            ]).unlink()
        return result

    @api.model_create_multi
    def create(self, vals_list):
        if any(vals.get('ont_rag_indexed') for vals in vals_list) and not (
            self.env.su or self.env.user.has_group('base.group_system')
        ):
            raise AccessError(_('Only administrators can make a folder searchable by AI.'))
        return super().create(vals_list)
