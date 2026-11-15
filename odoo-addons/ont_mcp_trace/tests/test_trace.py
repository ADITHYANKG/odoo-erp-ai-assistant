from __future__ import annotations

import json
from typing import Any
from unittest.mock import patch

from odoo.addons.mail.models.mail_thread import MailThread
from odoo.tests import common, tagged


@tagged('post_install', '-at_install')
class TestMcpChatterTrace(common.TransactionCase):
    """Successful MCP writes leave an internal note on the record's chatter."""

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls.tool_model = cls.env['muk_mcp.tool']
        cls.partner = cls.env['res.partner'].create({'name': 'Trace Target'})
        # The test DB may carry a muk_mcp_access allowlist that denies writes;
        # open up the models this suite touches (rolled back with the test).
        if 'muk_mcp_access.model' in cls.env:
            Access = cls.env['muk_mcp_access.model']
            for model_name in ('res.partner', 'res.partner.category'):
                entry = Access.with_context(active_test=False).search(
                    [('model_name', '=', model_name)], limit=1)
                values = {'allow_read': True, 'allow_write': True, 'active': True}
                if entry:
                    entry.write(values)
                else:
                    model = cls.env['ir.model'].search([('model', '=', model_name)], limit=1)
                    Access.create({'model_id': model.id, **values})

    def _call(self, name: str, arguments: dict[str, Any]) -> Any:
        text, _info = self.tool_model._call(name, arguments, self.env)
        return json.loads(text)

    def _notes(self, model: str, res_id: int):
        return self.env['mail.message'].search([
            ('model', '=', model),
            ('res_id', '=', res_id),
            ('body', 'like', 'via MCP'),
        ])

    def test_update_posts_note_with_field_names(self) -> None:
        self._call('update_records', {
            'model': 'res.partner',
            'ids': [self.partner.id],
            'values': {'phone': '12345', 'city': 'Kochi'},
        })
        (note,) = self._notes('res.partner', self.partner.id)
        body = str(note.body)
        self.assertIn('update_records', body)
        self.assertIn('city, phone', body)  # sorted field names
        self.assertIn(self.env.user.name, body)
        self.assertEqual(note.author_id, self.env.user.partner_id)
        # Internal note subtype: followers get no email from agent activity.
        self.assertEqual(note.subtype_id, self.env.ref('mail.mt_note'))

    def test_create_posts_note_on_the_new_record(self) -> None:
        result = self._call('create_records', {
            'model': 'res.partner',
            'values': {'name': 'Freshly Created'},
        })
        (note,) = self._notes('res.partner', result['id'])
        self.assertIn('create_records', str(note.body))
        self.assertIn('record created', str(note.body))

    def test_call_method_posts_note_naming_the_method(self) -> None:
        self._call('call_method', {
            'model': 'res.partner',
            'method': 'toggle_active',
            'ids': [self.partner.id],
        })
        (note,) = self._notes('res.partner', self.partner.id)
        self.assertIn('call_method', str(note.body))
        self.assertIn('toggle_active', str(note.body))

    def test_reads_stay_silent(self) -> None:
        self._call('search_read', {
            'model': 'res.partner',
            'domain': json.dumps([['id', '=', self.partner.id]]),
            'fields': ['name'],
        })
        self.assertFalse(self._notes('res.partner', self.partner.id))

    def test_delete_neither_crashes_nor_orphans_a_note(self) -> None:
        victim = self.env['res.partner'].create({'name': 'Doomed'})
        victim_id = victim.id
        result = self._call('delete_records', {
            'model': 'res.partner', 'ids': [victim_id],
        })
        self.assertTrue(result['success'])
        self.assertFalse(self._notes('res.partner', victim_id))

    def test_chatterless_model_write_succeeds_without_note(self) -> None:
        result = self._call('create_records', {
            'model': 'res.partner.category',
            'values': {'name': 'No Chatter Here'},
        })
        self.assertIn('id', result)  # the write itself succeeded
        self.assertFalse(self._notes('res.partner.category', result['id']))

    def test_posting_failure_never_breaks_the_tool_call(self) -> None:
        def explode(self, *args: Any, **kwargs: Any) -> Any:
            raise RuntimeError('chatter down')

        with patch.object(MailThread, 'message_post', explode):
            result = self._call('update_records', {
                'model': 'res.partner',
                'ids': [self.partner.id],
                'values': {'phone': '999'},
            })
        self.assertTrue(result['success'])  # the write went through regardless
        self.assertEqual(self.partner.phone, '999')
