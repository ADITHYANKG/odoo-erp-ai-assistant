from __future__ import annotations

import base64
import json
from typing import Any

from markupsafe import escape

from odoo.addons.mail.tests.common import MailCommon
from odoo.addons.muk_mcp.tools.exception import MCPScopeDenied
from odoo.exceptions import UserError
from odoo.tests import tagged


@tagged('post_install', '-at_install')
class TestSendEmail(MailCommon):
    """send_email tool: validation, sending (captured, never real SMTP),
    attachment rendering, and write-scope enforcement."""

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls.tool_model = cls.env['muk_mcp.tool']
        # qweb-text report so the attachment path needs no wkhtmltopdf.
        cls.env['ir.ui.view'].create({
            'type': 'qweb',
            'name': 'ont_mcp_mail.test_send_report',
            'key': 'ont_mcp_mail.test_send_report',
            'arch': (
                '<t t-name="ont_mcp_mail.test_send_report">'
                '<t t-foreach="docs" t-as="d"><t t-esc="d.name"/>|</t>'
                '</t>'
            ),
        })
        cls.report = cls.env['ir.actions.report'].create({
            'name': 'Send Test Report',
            'report_name': 'ont_mcp_mail.test_send_report',
            'report_type': 'qweb-text',
            'model': 'res.partner',
        })
        cls.partner = cls.env['res.partner'].create({'name': 'Mail Target Partner'})
        # Every existing test mails @example.com; allow that domain so they
        # exercise sending, not the allowlist (which has its own tests).
        cls.env['ir.config_parameter'].sudo().set_param(
            'ont_mcp_mail.allowed_recipients', '@example.com')

    def _call(self, arguments: dict[str, Any], enforce_scope: str | None = None) -> dict[str, Any]:
        text, _info = self.tool_model._call(
            'send_email', arguments, self.env, enforce_scope=enforce_scope)
        return json.loads(text)

    def test_invalid_recipient_rejected(self) -> None:
        for bad in ('', 'not-an-email', 'a@b.c, garbage'):
            with self.assertRaises(UserError):
                self._call({'to': bad, 'subject': 's', 'body': 'b'})

    def test_entry_hiding_a_second_address_is_rejected(self) -> None:
        # Odoo's mailer splits on ';' and spaces too; the guard must never
        # judge "outsider; insider" by its last domain alone.
        for smuggled in (
            'stranger@evil.test; ceo@example.com',
            'stranger@evil.test;ceo@example.com',
            'stranger@evil.test ceo@example.com',
        ):
            with self.mock_mail_gateway(), self.assertRaises(UserError):
                self._call({'to': smuggled, 'subject': 's', 'body': 'b'})
            self.assertFalse(self._mails)

    def test_semicolon_list_of_allowed_recipients_is_split(self) -> None:
        with self.mock_mail_gateway():
            result = self._call({
                'to': 'a@example.com; B <b@example.com>, a@example.com',
                'subject': 's', 'body': 'b',
            })
        self.assertEqual(result['to'], 'a@example.com, b@example.com')

    def test_semicolon_cannot_smuggle_an_outsider_past_the_allowlist(self) -> None:
        with self.mock_mail_gateway(), self.assertRaises(UserError):
            self._call({'to': 'ceo@example.com; stranger@evil.test', 'subject': 's', 'body': 'b'})
        self.assertFalse(self._mails)

    def test_empty_subject_rejected(self) -> None:
        with self.assertRaises(UserError):
            self._call({'to': 'x@example.com', 'subject': '  ', 'body': 'b'})

    def test_happy_path_sends_and_escapes_body(self) -> None:
        with self.mock_mail_gateway():
            result = self._call({
                'to': 'ceo@example.com',
                'subject': 'Quarterly numbers',
                'body': 'Line one <script>\nLine two',
            })
        self.assertEqual(result['status'], 'sent')
        self.assertEqual(result['to'], 'ceo@example.com')
        self.assertIsNone(result['attached'])
        self.assertEqual(len(self._mails), 1)
        sent = self._mails[0]
        self.assertIn('ceo@example.com', sent['email_to'])
        self.assertEqual(sent['subject'], 'Quarterly numbers')
        self.assertIn('&lt;script&gt;', sent['body'])  # body escaped, never raw HTML
        self.assertIn('Line two', sent['body'])

    def test_sender_is_stable_assistant_identity(self) -> None:
        # One deterministic sender for EVERY agent mail — never the
        # triggering user's formatted address (identity churn reads phishy).
        with self.mock_mail_gateway():
            self._call({'to': 'x@example.com', 'subject': 's', 'body': 'b'})
        (sent,) = self._mails
        self.assertIn('AI Assistant', sent['email_from'])
        self.assertIn(self.env.company.name, sent['email_from'])
        self.assertNotEqual(sent['email_from'], self.env.user.email_formatted)

    def test_sender_name_configurable_via_system_parameter(self) -> None:
        self.env['ir.config_parameter'].sudo().set_param(
            'ont_mcp_mail.sender_name', 'Ontash Bot')
        with self.mock_mail_gateway():
            self._call({'to': 'x@example.com', 'subject': 's', 'body': 'b'})
        (sent,) = self._mails
        self.assertIn('Ontash Bot', sent['email_from'])
        self.assertNotIn('AI Assistant', sent['email_from'])

    def test_sender_email_configurable_via_system_parameter(self) -> None:
        self.env['ir.config_parameter'].sudo().set_param(
            'ont_mcp_mail.sender_email', 'bot@example.com')
        with self.mock_mail_gateway():
            self._call({'to': 'x@example.com', 'subject': 's', 'body': 'b'})
        (sent,) = self._mails
        self.assertIn('bot@example.com', sent['email_from'])

    def test_footer_names_the_triggering_user(self) -> None:
        # The standard footer provides provenance: assistant sender + the
        # human it acted for. Present on every mail, escaped like the body.
        with self.mock_mail_gateway():
            self._call({'to': 'x@example.com', 'subject': 's', 'body': 'b'})
        (sent,) = self._mails
        self.assertIn('Sent by', sent['body'])
        self.assertIn('on behalf of', sent['body'])
        self.assertIn(str(escape(self.env.user.name)), sent['body'])

    def test_record_report_is_attached(self) -> None:
        with self.mock_mail_gateway():
            result = self._call({
                'to': 'ceo@example.com',
                'subject': 'Here is the document',
                'body': 'Attached.',
                'record_model': 'res.partner',
                'record_id': self.partner.id,
                'report_ref': self.report.id,
            })
        self.assertTrue(result['attached'])
        self.assertNotIn(' ', result['attached'])
        (sent,) = self._mails
        (attachment,) = sent['attachments']
        name, content = attachment[0], attachment[1]
        self.assertEqual(name, result['attached'])
        raw = content if isinstance(content, bytes) else base64.b64decode(content)
        self.assertIn(b'Mail Target Partner', raw)

    def test_record_export_is_attached(self) -> None:
        second = self.env['res.partner'].create({'name': 'Second Mail Target'})
        with self.mock_mail_gateway():
            result = self._call({
                'to': 'ceo@example.com',
                'subject': 'Complete contact list',
                'body': 'The complete result is attached.',
                'export': {
                    'model': 'res.partner',
                    'fields': ['name'],
                    'ids': [self.partner.id, second.id],
                    'format': 'xlsx',
                },
            })
        self.assertEqual(result['attached_rows'], 2)
        self.assertTrue(result['attached'].endswith('.xlsx'))
        (sent,) = self._mails
        (attachment,) = sent['attachments']
        self.assertEqual(attachment[0], result['attached'])
        raw = attachment[1] if isinstance(attachment[1], bytes) \
            else base64.b64decode(attachment[1])
        self.assertTrue(raw.startswith(b'PK\x03\x04'))

    def test_empty_record_export_is_attached(self) -> None:
        with self.mock_mail_gateway():
            result = self._call({
                'to': 'ceo@example.com',
                'subject': 'Empty contact list',
                'body': 'No matching contacts were found.',
                'export': {
                    'model': 'res.partner',
                    'fields': ['name'],
                    'domain': [['id', '=', 0]],
                    'format': 'xlsx',
                },
            })
        self.assertEqual(result['attached_rows'], 0)
        self.assertTrue(result['attached'].endswith('.xlsx'))
        (sent,) = self._mails
        (attachment,) = sent['attachments']
        self.assertEqual(attachment[0], result['attached'])

    def test_export_schema_requires_lists_even_when_empty(self) -> None:
        send_email = next(
            tool for tool in self.tool_model.get_tools()
            if tool['name'] == 'send_email'
        )
        description = send_email['inputSchema']['properties']['export']['description']
        self.assertIn('REQUIRED whenever emailing a list or table', description)
        self.assertIn('zero rows', description)

    def test_export_requires_model_and_fields(self) -> None:
        with self.assertRaises(UserError):
            self._call({
                'to': 'x@example.com', 'subject': 's', 'body': 'b',
                'export': {'model': 'res.partner'},
            })

    def test_record_report_and_export_are_mutually_exclusive(self) -> None:
        with self.assertRaises(UserError):
            self._call({
                'to': 'x@example.com', 'subject': 's', 'body': 'b',
                'record_model': 'res.partner', 'record_id': self.partner.id,
                'export': {'model': 'res.partner', 'fields': ['name']},
            })

    def test_default_report_lookup_when_no_ref(self) -> None:
        # No qweb-pdf report for res.partner in this DB slice — the fallback
        # finds our text report by model.
        with self.mock_mail_gateway():
            result = self._call({
                'to': 'x@example.com',
                'subject': 's',
                'body': 'b',
                'record_model': 'res.partner',
                'record_id': self.partner.id,
            })
        self.assertTrue(result['attached'])

    def test_record_email_lands_on_the_records_chatter(self) -> None:
        # With a record attached, the sent mail must be visible in that
        # record's message thread (the chatter), authored by the sender.
        with self.mock_mail_gateway():
            self._call({
                'to': 'ceo@example.com',
                'subject': 'Document for you',
                'body': 'See attachment.',
                'record_model': 'res.partner',
                'record_id': self.partner.id,
                'report_ref': self.report.id,
            })
        message = self.env['mail.message'].search([
            ('model', '=', 'res.partner'),
            ('res_id', '=', self.partner.id),
            ('subject', '=', 'Document for you'),
        ])
        self.assertEqual(len(message), 1)
        self.assertEqual(message.author_id, self.env.user.partner_id)
        self.assertIn('See attachment.', str(message.body))  # body visible in chatter
        (attachment,) = message.attachment_ids
        self.assertEqual(attachment.res_model, 'res.partner')
        self.assertEqual(attachment.res_id, self.partner.id)

    def test_plain_email_stays_off_any_chatter(self) -> None:
        # Without a record, no thread linkage: nothing appears on any record.
        with self.mock_mail_gateway():
            self._call({'to': 'x@example.com', 'subject': 'Standalone', 'body': 'b'})
        self.assertFalse(self.env['mail.message'].search([
            ('subject', '=', 'Standalone'), ('model', '!=', False),
        ]))

    def test_record_args_must_come_together(self) -> None:
        with self.assertRaises(UserError):
            self._call({
                'to': 'x@example.com', 'subject': 's', 'body': 'b',
                'record_model': 'res.partner',
            })

    def test_missing_record_rejected(self) -> None:
        with self.assertRaises(UserError):
            self._call({
                'to': 'x@example.com', 'subject': 's', 'body': 'b',
                'record_model': 'res.partner', 'record_id': 99999999,
            })

    def _set_param(self, key: str, value: str) -> None:
        self.env['ir.config_parameter'].sudo().set_param(key, value)

    def test_unlisted_recipient_is_refused_before_sending(self) -> None:
        # Exfiltration guard: approval is a human check, this is the wall.
        with self.assertRaises(UserError) as caught:
            self._call({'to': 'attacker@evil.test', 'subject': 's', 'body': 'b'})
        message = str(caught.exception)
        self.assertIn('attacker@evil.test', message)
        # The error names the parameter to change — self-documenting.
        self.assertIn('ont_mcp_mail.allowed_recipients', message)

    def test_internal_user_address_is_always_allowed(self) -> None:
        # "Email me the report" must work for colleagues with no config.
        self.env['res.users'].sudo().create({
            'name': 'Insider', 'login': 'insider@internal.test',
            'email': 'insider@internal.test',
        })
        with self.mock_mail_gateway():
            result = self._call({
                'to': 'insider@internal.test', 'subject': 's', 'body': 'b'})
        self.assertEqual(result['status'], 'sent')

    def test_exact_address_and_domain_rules(self) -> None:
        self._set_param(
            'ont_mcp_mail.allowed_recipients', 'board@partner.test, @trusted.test')
        with self.mock_mail_gateway():
            self._call({'to': 'board@partner.test', 'subject': 's', 'body': 'b'})
            self._call({'to': 'anyone@trusted.test', 'subject': 's', 'body': 'b'})
        with self.assertRaises(UserError):
            self._call({'to': 'other@partner.test', 'subject': 's', 'body': 'b'})

    def test_star_disables_the_allowlist(self) -> None:
        self._set_param('ont_mcp_mail.allowed_recipients', '*')
        with self.mock_mail_gateway():
            result = self._call({
                'to': 'anyone@anywhere.test', 'subject': 's', 'body': 'b'})
        self.assertEqual(result['status'], 'sent')

    def test_known_contacts_allowed_until_the_flag_is_off(self) -> None:
        self.env['res.partner'].create(
            {'name': 'Known Customer', 'email': 'customer@known.test'})
        with self.mock_mail_gateway():
            self._call({'to': 'customer@known.test', 'subject': 's', 'body': 'b'})
        self._set_param('ont_mcp_mail.allow_partner_recipients', '0')
        with self.assertRaises(UserError):
            self._call({'to': 'customer@known.test', 'subject': 's', 'body': 'b'})

    def test_one_bad_recipient_refuses_the_whole_send(self) -> None:
        # Fail closed: no partial delivery to the good addresses.
        with self.mock_mail_gateway():
            with self.assertRaises(UserError):
                self._call({
                    'to': 'ceo@example.com, attacker@evil.test',
                    'subject': 's', 'body': 'b',
                })
            self.assertFalse(self._mails)

    def test_read_scope_cannot_send(self) -> None:
        # Write-category tool: read-scoped callers are refused before any work.
        with self.assertRaises(MCPScopeDenied):
            self._call(
                {'to': 'x@example.com', 'subject': 's', 'body': 'b'},
                enforce_scope='read',
            )
