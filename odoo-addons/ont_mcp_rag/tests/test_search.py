"""The search tool: what it returns, and what it refuses to return."""

import base64
from unittest.mock import patch

from odoo.exceptions import AccessError
from odoo.tests import TransactionCase, new_test_user, tagged

EMBED_MODEL_PATH = 'odoo.addons.ont_mcp_rag.models.document_chunk.embed_texts'
EMBED_SEARCH_PATH = 'odoo.addons.ont_mcp_rag.mcp.search.embed_texts'

#: Enough sections that a narrow search leaves some of them UNMATCHED —
#: otherwise every chunk is a hit and "neighbouring section" means nothing.
SECTIONS = [
    ('20. Remote and Hybrid Work', 'Hybrid work may be agreed with a manager. '),
    ('27. Leave Policy', 'Annual leave accrues monthly for all staff. '),
    ('44. Harassment', 'Harassment includes unwanted conduct and remarks. '),
    ('45. Sexual Harassment', 'Unwelcome advances of a sexual nature are prohibited. '),
    ('53. Gifts and Hospitality', 'Gifts above a nominal value must be declared. '),
    ('63. Grievance Procedure', 'Raise a grievance with your manager first. '),
    ('65. Non-Retaliation', 'No employee shall face retaliation for reporting. '),
    ('79. Notice Period', 'Thirty days written notice is required to resign. '),
]
TEXT = '\n\n'.join(f'{heading}\n\n{body * 7}' for heading, body in SECTIONS)


@tagged('post_install', '-at_install')
class TestSearchDocuments(TransactionCase):

    def setUp(self):
        super().setUp()
        self.mixin = self.env['muk_mcp.mixin']
        self.chunks = self.env['ont.document.chunk']
        self.folder = self.env['documents.folder'].create({
            'name': 'Policies (test)', 'ont_rag_indexed': True,
        })
        self.document = self.env['documents.document'].create({
            'name': 'HR-POLICY.pdf',
            'folder_id': self.folder.id,
            'datas': base64.b64encode(TEXT.encode()),
            'mimetype': 'text/plain',
        })
        self.document.attachment_id.write({'index_content': TEXT})
        self.chunks._index_document(self.document)
        with patch(EMBED_MODEL_PATH, side_effect=self._fake_embed):
            self.chunks._embed_pending()

    @staticmethod
    def _fake_embed(env, texts):
        return [[0.1] * 1536 for _ in texts], 0.0

    def test_an_empty_query_returns_an_explicit_nothing(self):
        result = self.mixin._mcp_search_documents('')
        self.assertEqual(result['results'], [])
        self.assertIn('note', result)

    def test_a_match_carries_the_document_and_section_for_citation(self):
        with patch(EMBED_SEARCH_PATH, side_effect=self._fake_embed):
            result = self.mixin._mcp_search_documents('harassment')

        self.assertTrue(result['results'], 'the harassment section should be found')
        first = result['results'][0]
        self.assertEqual(first['document'], 'HR-POLICY.pdf')
        self.assertTrue(first['section'], 'a passage must name its section')
        self.assertIn('Harassment', first['text'])

    def test_keyword_matching_survives_a_useless_vector(self):
        """Every chunk here has an IDENTICAL fake embedding, so similarity
        ranking is meaningless. Full-text search is what still finds the
        right passage — which is the whole reason retrieval is hybrid."""
        with patch(EMBED_SEARCH_PATH, side_effect=self._fake_embed):
            result = self.mixin._mcp_search_documents('retaliation')

        sections = ' '.join(row['section'] or '' for row in result['results'])
        self.assertIn('Non-Retaliation', sections)

    def test_neighbouring_sections_come_back_too(self):
        """"What happens if I am harassed?" is answered by the definition,
        the grievance procedure AND the non-retaliation clause together. One
        passage alone would read as a complete answer while being an
        incomplete one, which on an HR question is the damaging kind of
        wrong."""
        with patch(EMBED_SEARCH_PATH, side_effect=self._fake_embed):
            result = self.mixin._mcp_search_documents('grievance', limit=1)

        self.assertTrue(
            [row for row in result['results'] if row['adjacent']],
            'the sections around a hit must come back too',
        )
        sections = ' '.join(row['section'] or '' for row in result['results'])
        self.assertIn('Non-Retaliation', sections)

    def test_pending_documents_are_reported_so_absence_can_be_explained(self):
        """"We have no such policy" and "it is not indexed yet" are
        different answers, and only one of them is honest."""
        second = self.env['documents.document'].create({
            'name': 'EXPENSES.pdf',
            'folder_id': self.folder.id,
            'datas': base64.b64encode(b'1. Expenses\n\nClaim within 30 days. ' * 20),
            'mimetype': 'text/plain',
        })
        second.attachment_id.write({'index_content': '1. Expenses\n\nClaim within 30 days. ' * 20})
        self.chunks._index_document(second)  # chunked but NOT embedded

        with patch(EMBED_SEARCH_PATH, side_effect=self._fake_embed):
            result = self.mixin._mcp_search_documents('harassment')

        self.assertGreater(result['pending_documents'], 0)

    def test_an_unflagged_folder_is_never_searched(self):
        self.folder.ont_rag_indexed = False
        with patch(EMBED_SEARCH_PATH, side_effect=self._fake_embed):
            result = self.mixin._mcp_search_documents('harassment')

        self.assertEqual(result['results'], [])
        self.assertEqual(result['searched_folders'], [])

    def test_unflagging_a_folder_deletes_its_passages(self):
        self.assertTrue(self.chunks.search_count([('document_id', '=', self.document.id)]))
        self.folder.ont_rag_indexed = False
        self.assertFalse(self.chunks.search_count([('document_id', '=', self.document.id)]))

    def test_employees_cannot_read_passages_directly(self):
        # Only search_documents may serve passages: it filters by the documents
        # the caller can open, a direct read of the table would not.
        employee = new_test_user(self.env, login='rag_employee', groups='base.group_user')
        with self.assertRaises(AccessError):
            self.chunks.with_user(employee).search_read([], ['content'])

    def test_only_administrators_can_flag_a_folder(self):
        employee = new_test_user(
            self.env, login='rag_editor', groups='base.group_user,documents.group_documents_user',
        )
        with self.assertRaises(AccessError):
            self.folder.with_user(employee).write({'ont_rag_indexed': False})

    def test_the_searched_folders_are_reported(self):
        # So the assistant can say WHERE it looked when it finds nothing,
        # instead of an unqualified "there is no such policy".
        with patch(EMBED_SEARCH_PATH, side_effect=self._fake_embed):
            result = self.mixin._mcp_search_documents('harassment')

        self.assertIn('Policies (test)', result['searched_folders'])
