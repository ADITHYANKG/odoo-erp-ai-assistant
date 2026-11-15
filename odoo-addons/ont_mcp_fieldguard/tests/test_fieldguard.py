"""Unknown fields are dropped from MCP reads instead of raising."""

from odoo.tests import TransactionCase, tagged


@tagged('post_install', '-at_install')
class TestFieldGuard(TransactionCase):

    def setUp(self):
        super().setUp()
        self.mixin = self.env['muk_mcp.mixin']
        self.partner = self.env['res.partner'].create({
            'name': 'Fieldguard Probe', 'email': 'probe@example.test',
        })

    # -- helper split ---------------------------------------------------
    def test_valid_fields_splits_known_from_unknown(self):
        valid, ignored = self.mixin._mcp_valid_fields(
            'res.partner', ['name', 'email', 'no_such_field', 'phone'],
        )
        self.assertEqual(valid, ['name', 'email', 'phone'])
        self.assertEqual(ignored, ['no_such_field'])

    def test_dotted_path_is_kept(self):
        valid, ignored = self.mixin._mcp_valid_fields(
            'res.partner', ['name', 'parent_id.name'],
        )
        self.assertEqual(valid, ['name', 'parent_id.name'])
        self.assertFalse(ignored)

    # -- read_records ---------------------------------------------------
    def test_read_records_drops_unknown_field_no_error(self):
        rows = self.mixin._mcp_read_records(
            'res.partner', [self.partner.id],
            ['id', 'name', 'no_such_field', 'email'],
        )
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row['name'], 'Fieldguard Probe')
        self.assertEqual(row['email'], 'probe@example.test')
        self.assertNotIn('no_such_field', row)

    def test_read_records_all_invalid_falls_back_to_display_name(self):
        rows = self.mixin._mcp_read_records(
            'res.partner', [self.partner.id], ['bogus_one', 'bogus_two'],
        )
        self.assertEqual(len(rows), 1)
        # display_name fallback only — not a full-field dump.
        self.assertIn('display_name', rows[0])
        self.assertNotIn('bogus_one', rows[0])
        self.assertLessEqual(set(rows[0]) - {'id'}, {'display_name'})

    # -- search_read ----------------------------------------------------
    def test_search_read_drops_unknown_field_no_error(self):
        rows = self.mixin._mcp_search_read(
            'res.partner',
            domain=[('id', '=', self.partner.id)],
            fields=['id', 'name', 'no_such_field'],
        )
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]['name'], 'Fieldguard Probe')
        self.assertNotIn('no_such_field', rows[0])

    def test_search_read_valid_fields_unaffected(self):
        rows = self.mixin._mcp_search_read(
            'res.partner',
            domain=[('id', '=', self.partner.id)],
            fields=['id', 'name', 'email'],
        )
        self.assertEqual(rows[0]['email'], 'probe@example.test')
