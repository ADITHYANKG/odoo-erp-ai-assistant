"""Chunker unit tests — pure functions, no database needed."""

from odoo.tests import TransactionCase, tagged

from ..tools.chunker import chunk_text, clean_text, looks_like_heading


@tagged('post_install', '-at_install')
class TestChunker(TransactionCase):
    """The splitter must work on documents it has never seen the shape of."""

    def test_heading_shapes_are_recognised_without_knowing_the_document(self):
        for line in (
            '44. Harassment',
            '7.2 Payment Terms',
            '12.3.1 Scope',
            'ARTICLE V - TERMINATION',
            'APPENDIX B',
            'Agenda Item 3',
            'Payment Terms',
        ):
            self.assertTrue(looks_like_heading(line), f'{line!r} should be a heading')

    def test_prose_is_not_mistaken_for_a_heading(self):
        for line in (
            'Employees must act honestly, treat others respectfully.',
            'The purpose of this policy is to establish a framework which is',
            '',
            '42',
            'a' * 120,
        ):
            self.assertFalse(looks_like_heading(line), f'{line!r} should not be a heading')

    def test_sections_become_chunks_carrying_their_heading(self):
        text = (
            '44. Harassment\n\n' + 'Harassment includes unwanted conduct. ' * 12 +
            '\n\n45. Bullying\n\n' + 'Bullying includes repeated humiliation. ' * 12
        )
        chunks = chunk_text(text, min_chars=50)
        headings = [chunk.heading for chunk in chunks]
        self.assertIn('44. Harassment', headings)
        # The heading is EMBEDDED, not just stored: at a few hundred
        # characters it carries much of what makes the question match.
        harassment = next(c for c in chunks if c.heading == '44. Harassment')
        self.assertTrue(harassment.content.startswith('44. Harassment'))

    def test_a_merged_chunk_reports_every_section_it_covers(self):
        """Otherwise the assistant cites the section a chunk STARTS at, and
        answers a notice-period question by citing 'Organizational Change'."""
        text = (
            '76. Organizational Change\n\nChanges may occur.\n\n'
            '77. Redundancy\n\nRoles may be removed.\n\n'
            '78. Resignation\n\nEmployees may resign.\n\n'
            '79. Notice Period\n\nThirty days notice is required.\n'
        )
        chunks = chunk_text(text, min_chars=400)
        self.assertEqual(len(chunks), 1, 'short sections should merge')
        self.assertEqual(chunks[0].heading, '76. Organizational Change - 79. Notice Period')
        self.assertIn('Thirty days', chunks[0].body)

    def test_an_oversized_section_is_split_below_the_cap(self):
        text = '1. Scope\n\n' + ('This clause is long. ' * 500)
        chunks = chunk_text(text, max_chars=600)
        self.assertGreater(len(chunks), 1)
        for chunk in chunks:
            self.assertLessEqual(len(chunk.content), 900)

    def test_documents_without_headings_fall_back_to_paragraphs_with_overlap(self):
        text = '\n\n'.join(f'Paragraph number {n} carries some text.' * 6 for n in range(8))
        chunks = chunk_text(text, max_chars=300, min_chars=50, overlap_chars=80)
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(chunk.heading == '' for chunk in chunks))
        # Overlap is what stops a clause being cut in half where the
        # boundary is arbitrary rather than a real section break.
        joined = ''.join(chunk.content for chunk in chunks)
        self.assertGreater(len(joined), len(text) * 0.95)

    def test_running_headers_are_stripped_but_repeated_content_is_kept(self):
        """A page header repeats on a rhythm; real repeated text clusters."""
        # Realistic page sizes: an extracted PDF page runs to tens of lines,
        # and that wide, regular spacing is what identifies page furniture.
        pages = []
        for page in range(12):
            pages.append('MyCompany | Human Resources Policy')
            pages += [f'Page {page} line {n} of body text, all different.' for n in range(20)]
        # Real content repeating in one tight cluster, not once per page.
        pages[30:30] = ['Approve'] * 5
        cleaned = clean_text('\n'.join(pages))

        self.assertNotIn('MyCompany | Human Resources Policy', cleaned)
        self.assertIn('Approve', cleaned)

    def test_symbol_font_glyphs_are_removed(self):
        # Wingdings bullets extract as private-use codepoints and mean
        # nothing to a reader or to a search index.
        cleaned = clean_text('1. Scope\n\n' + chr(0xF0B7) + ' First item')
        self.assertNotIn(chr(0xF0B7), cleaned)
        self.assertIn('First item', cleaned)

    def test_empty_input_is_not_a_chunk(self):
        self.assertEqual(chunk_text(''), [])
        self.assertEqual(chunk_text('   \n\n  '), [])

    def test_a_page_marker_is_not_a_heading(self):
        # Every "Page 2", "Page 3" is a DIFFERENT string appearing once, so
        # the repeat-based boilerplate detector cannot see them.
        for line in ('Page 2', 'Page 12 of 40', 'p. 7'):
            self.assertFalse(looks_like_heading(line), f'{line!r} is page furniture')

    def test_a_bare_number_is_not_a_section_number(self):
        """"125 USD" labelled a chunk of the expense policy "125 USD", which
        is what a citation would then have quoted (live, 2026-09-02)."""
        self.assertFalse(looks_like_heading('125 USD'))
        self.assertFalse(looks_like_heading('30 days'))
        # Real section numbering is dotted or sub-numbered.
        self.assertTrue(looks_like_heading('44. Harassment'))
        self.assertTrue(looks_like_heading('7.2 Payment Terms'))

    def test_a_title_case_line_needs_prose_after_it(self):
        """A heading introduces prose; a Title Case line followed by more
        short lines is a table row, which is how the expense policy gained
        a chunk headed "Additional Evidence"."""
        table = ['', 'Meals >= 25 USD', 'Itemized receipt', 'Taxi >= 25 USD']
        prose = ['', 'This section sets out what evidence must accompany a claim in each case.']

        self.assertFalse(looks_like_heading('Additional Evidence', table))
        self.assertTrue(looks_like_heading('Additional Evidence', prose))

    def test_a_numbered_heading_is_trusted_without_prose_following(self):
        # Numbering is a strong enough signal on its own; a numbered section
        # whose body happens to be a table is still a section.
        table = ['', 'Meals >= 25 USD', 'Itemized receipt']
        self.assertTrue(looks_like_heading('17. Expense Claims', table))

    def test_a_money_range_is_not_a_section_number(self):
        # "250.01 to 1,000 USD" parsed as section 250.01 and labelled a chunk
        # of the expense policy (live, 2026-09-02). A real section title
        # starts with a capital.
        self.assertFalse(looks_like_heading('250.01 to 1,000 USD'))
        self.assertFalse(looks_like_heading('1.5 million in revenue'))
        self.assertTrue(looks_like_heading('9. Air Travel'))
        self.assertTrue(looks_like_heading('7.2 Payment Terms'))
