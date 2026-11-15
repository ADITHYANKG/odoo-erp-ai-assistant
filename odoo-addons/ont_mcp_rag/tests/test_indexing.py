"""Indexing: which documents get chunked, once, and when they are rebuilt."""

import base64
from unittest.mock import patch

from odoo.tests import TransactionCase, tagged

TEXT = (
    '1. Purpose\n\n' + 'This policy sets out the rules for everyone here. ' * 8 +
    '\n\n44. Harassment\n\n' + 'Harassment includes unwanted conduct of any kind. ' * 8
)
EMBED_PATH = 'odoo.addons.ont_mcp_rag.models.document_chunk.embed_texts'


@tagged('post_install', '-at_install')
class TestIndexing(TransactionCase):

    def setUp(self):
        super().setUp()
        self.chunks = self.env['ont.document.chunk']
        self.folder = self.env['documents.folder'].create({
            'name': 'Policies (test)', 'ont_rag_indexed': True,
        })
        self.other = self.env['documents.folder'].create({
            'name': 'Receipts (test)', 'ont_rag_indexed': False,
        })

    def _document(self, folder, name='POLICY.pdf', text=TEXT):
        document = self.env['documents.document'].create({
            'name': name,
            'folder_id': folder.id,
            'datas': base64.b64encode(text.encode()),
            'mimetype': 'text/plain',
        })
        # index_content is normally filled by attachment_indexation on
        # upload; set it directly so the test does not depend on a PDF.
        document.attachment_id.write({'index_content': text})
        return document

    def _fake_embeddings(self, texts):
        return [[0.1] * 1536 for _ in texts], 0.0

    def test_only_flagged_folders_are_indexed(self):
        wanted = self._document(self.folder)
        ignored = self._document(self.other, name='RECEIPT.pdf')
        needed = self.chunks._documents_needing_indexing()

        self.assertIn(wanted, needed)
        self.assertNotIn(ignored, needed)

    def test_the_same_file_uploaded_three_times_is_indexed_once(self):
        """Three identical copies of the policy already exist in the live
        database; without this each would claim a slot in every result."""
        copies = [self._document(self.folder, name=f'POLICY-{n}.pdf') for n in range(3)]
        checksums = {copy.attachment_id.checksum for copy in copies}
        self.assertEqual(len(checksums), 1, 'copies must share a checksum')

        needed = self.chunks._documents_needing_indexing()
        self.assertEqual(len(needed), 1)

    def test_indexing_creates_pending_chunks_with_headings(self):
        document = self._document(self.folder)
        self.chunks._index_document(document)
        created = self.chunks.search([('document_id', '=', document.id)])

        self.assertTrue(created)
        self.assertEqual(set(created.mapped('state')), {'pending'})
        self.assertTrue(any('Harassment' in (c.heading or '') for c in created))
        self.assertEqual(set(created.mapped('checksum')), {document.attachment_id.checksum})

    def test_replacing_the_file_rebuilds_its_chunks(self):
        document = self._document(self.folder)
        self.chunks._index_document(document)
        before = self.chunks.search([('document_id', '=', document.id)])
        self.assertTrue(before)

        new_text = '1. Purpose\n\n' + 'A completely different policy body here. ' * 10
        # Write the bytes first, then index_content on its own: writing datas
        # makes attachment_indexation re-derive index_content from the
        # re-guessed mimetype, which would clobber a value set in the same call.
        document.attachment_id.write({'datas': base64.b64encode(new_text.encode())})
        document.attachment_id.write({'index_content': new_text})
        self.chunks._index_document(document)
        after = self.chunks.search([('document_id', '=', document.id)])

        self.assertFalse(before.exists(), 'stale chunks must be removed, not appended to')
        self.assertTrue(after)
        self.assertIn('different policy', after[0].content)

    def test_deleting_a_document_removes_its_chunks(self):
        document = self._document(self.folder)
        self.chunks._index_document(document)
        created = self.chunks.search([('document_id', '=', document.id)])
        self.assertTrue(created)

        document.unlink()
        self.assertFalse(created.exists())

    def test_embedding_marks_chunks_indexed(self):
        document = self._document(self.folder)
        self.chunks._index_document(document)
        with patch(EMBED_PATH, side_effect=lambda env, texts: self._fake_embeddings(texts)):
            self.chunks._embed_pending()

        created = self.chunks.search([('document_id', '=', document.id)])
        self.assertEqual(set(created.mapped('state')), {'indexed'})

    def test_a_document_with_no_extracted_text_is_skipped(self):
        # A scanned PDF or an image: index_content is empty, so there is
        # nothing to embed and it must not be retried for ever.
        document = self._document(self.folder, name='SCAN.pdf')
        document.attachment_id.write({'index_content': False})

        self.assertNotIn(document, self.chunks._documents_needing_indexing())
