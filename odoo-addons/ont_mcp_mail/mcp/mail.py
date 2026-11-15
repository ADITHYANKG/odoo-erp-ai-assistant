from __future__ import annotations

import base64
import logging
from typing import Any

from markupsafe import escape

from odoo import _, api, models
from odoo.exceptions import UserError
from odoo.http import request
from odoo.tools import email_split
from odoo.tools.mail import email_normalize, formataddr

_logger = logging.getLogger(__name__)

#: Comma-separated extra recipients: exact addresses and/or "@domain.tld"
#: rules. The single value "*" disables the allowlist entirely.
ALLOWED_RECIPIENTS_PARAM = 'ont_mcp_mail.allowed_recipients'
#: Set to 0/false/no to stop treating known contacts as valid recipients.
ALLOW_PARTNERS_PARAM = 'ont_mcp_mail.allow_partner_recipients'
#: Optional From address for every email sent by the assistant.
SENDER_EMAIL_PARAM = 'ont_mcp_mail.sender_email'

from odoo.addons.muk_mcp.core.tool import mcp_tool
from odoo.addons.muk_mcp.tools.descriptions import (
    domain_field,
    fields_field,
    ids_field,
    model_field,
)


class MCPMixin(models.AbstractModel):
    """Add the ``send_email`` MCP tool."""

    _inherit = 'muk_mcp.mixin'

    @api.model
    @mcp_tool(
        name='send_email',
        description=(
            'Send an email through Odoo to one or more addresses. To attach '
            'a record\'s PDF document (e.g. a quotation), pass record_model '
            'and record_id (e.g. record_model="sale.order", record_id=42) — '
            'the report is rendered server-side. To attach a complete list or '
            'table, pass export with its model, fields, and optional domain or '
            'ids; the CSV/XLSX is generated server-side. Do NOT read either '
            'file yourself. The mail goes out via the company mail server and '
            'is logged in Odoo.'
        ),
        input_schema={
            'type': 'object',
            'properties': {
                'to': {
                    'type': 'string',
                    'description': 'Recipient email address(es), comma-separated.',
                },
                'subject': {
                    'type': 'string',
                    'description': 'Email subject line.',
                },
                'body': {
                    'type': 'string',
                    'description': 'Plain-text body; rendered as simple paragraphs.',
                },
                'record_model': model_field(),
                'record_id': {
                    'type': 'integer',
                    'description': (
                        'Record whose PDF report to attach; requires record_model.'
                    ),
                },
                'report_ref': {
                    'type': 'string',
                    'description': (
                        'Optional specific report (XML id or report_name); '
                        "defaults to the model's main PDF report."
                    ),
                },
                'export': {
                    'type': 'object',
                    'description': (
                        'Complete CSV/XLSX result attachment. REQUIRED whenever '
                        'emailing a list or table, including a result with zero '
                        'rows; never send a requested list as body text only.'
                    ),
                    'properties': {
                        'model': model_field(),
                        'fields': fields_field(
                            required_hint=False,
                            extra_note="Use '/' to traverse relations.",
                            example=['name', 'work_email', 'job_id/name'],
                        ),
                        'ids': ids_field(
                            'email attachment',
                            extra_note='Omit to use the domain + limit.',
                        ),
                        'domain': domain_field(
                            extra_note="Used when 'ids' is not supplied.",
                        ),
                        'format': {
                            'type': 'string',
                            'enum': ['csv', 'xlsx'],
                            'default': 'xlsx',
                            'description': 'Attachment format.',
                        },
                        'limit': {
                            'type': 'integer',
                            'default': 1000,
                            'description': 'Maximum records when using domain.',
                        },
                        'order': {
                            'type': 'string',
                            'description': 'Sort order.',
                        },
                    },
                    'required': ['model', 'fields'],
                },
            },
            'required': ['to', 'subject', 'body'],
        },
        category='write',
    )
    def _mcp_send_email(
        self,
        to: str,
        subject: str,
        body: str,
        record_model: str | None = None,
        record_id: int | None = None,
        report_ref: str | int | None = None,
        export: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Validate, optionally render the record's report, then queue + send."""
        recipients = self._mcp_parse_recipients(to)
        # Exfiltration guard, before any work is done (see the helper).
        self._mcp_check_recipients(recipients)
        if not (subject or '').strip():
            raise UserError(_('Subject must not be empty.'))
        if bool(record_model) != bool(record_id):
            raise UserError(_('record_model and record_id must be provided together.'))
        if export and record_model:
            raise UserError(_('Attach either a record report or a data export, not both.'))

        attachment_commands = []
        attached_name = None
        attached_rows = None
        if record_model and record_id:
            attached_name, content = self._mcp_render_record_report(
                record_model, int(record_id), report_ref,
            )
            attachment_commands.append((0, 0, {
                'name': attached_name,
                'datas': base64.b64encode(content).decode(),
                # Linked to the record so it is visible from its chatter.
                'res_model': record_model,
                'res_id': int(record_id),
            }))
        elif export:
            attachment = self._mcp_render_export(export)
            attached_name = attachment['filename']
            attached_rows = attachment['row_count']
            attachment_commands.append((0, 0, {
                'name': attached_name,
                'datas': attachment['content_base64'],
                'mimetype': attachment['mimetype'],
            }))

        lines = [line for line in (body or '').splitlines() if line.strip()]
        body_html = ''.join(f'<p>{escape(line)}</p>' for line in lines) or '<p></p>'
        # Standard footer: every agent mail is visibly uniform AND names the
        # human who triggered it (provenance without sender-identity churn).
        sender_name = self._mcp_sender_name()
        # str(): escape() returns a Markup whose concatenation would
        # re-escape the surrounding (already safe) HTML.
        footer_text = str(escape(_(
            '— Sent by %(sender)s on behalf of %(user)s.',
            sender=sender_name, user=self.env.user.name,
        )))
        body_html += (
            '<p style="color:#888888;font-size:12px;margin-top:16px">'
            + footer_text + '</p>'
        )

        # Everything above (recipient validation, allowlist + record access,
        # report rendering) ran as the authenticated user. The mail queue
        # model itself is admin-only, so the row is created with sudo but
        # stamped with the real user's identity.
        values = {
            'subject': subject,
            # body_html is what the RECIPIENT gets; body is the underlying
            # mail.message field the chatter displays — set both, or the
            # thread shows an empty message with just the attachment.
            'body_html': body_html,
            'body': body_html,
            'email_to': ', '.join(recipients),
            # ONE stable assistant identity for every agent mail — never the
            # triggering user's name (identity churn reads phishy to C-suite
            # recipients). The human stays visible via author_id + footer.
            'email_from': self._mcp_email_from(),
            'author_id': self.env.user.partner_id.id,
            'attachment_ids': attachment_commands,
        }
        if record_model and record_id:
            # mail.mail delegates to mail.message: linking the message to the
            # record makes the sent email show up in that record's chatter.
            values.update({
                'model': record_model,
                'res_id': int(record_id),
                'message_type': 'comment',
                'subtype_id': self.env.ref('mail.mt_comment').id,
            })
        mail = self.env['mail.mail'].sudo().create(values)
        try:
            mail.send(raise_exception=True)
        except UserError:
            raise
        except Exception as exc:
            raise UserError(_('The email could not be sent: %s', exc)) from exc
        return {
            'to': ', '.join(recipients),
            'subject': subject,
            'attached': attached_name,
            'attached_rows': attached_rows,
            'status': 'sent',
        }

    @api.model
    def _mcp_render_export(self, specification: dict[str, Any]) -> dict[str, Any]:
        """Generate an access-controlled export while the outer tool is a write."""
        model = specification.get('model')
        fields = specification.get('fields')
        if not model or not fields:
            raise UserError(_('Export model and fields are required.'))
        previous = getattr(request, '_mcp_tool_category', None) if request else None
        if request:
            request._mcp_tool_category = 'read'
        try:
            return self._mcp_export_records(
                model=model,
                fields=fields,
                ids=specification.get('ids'),
                domain=specification.get('domain'),
                format=specification.get('format', 'xlsx'),
                limit=specification.get('limit', 1000),
                order=specification.get('order'),
            )
        finally:
            if request:
                request._mcp_tool_category = previous

    @api.model
    def _mcp_parse_recipients(self, to: str) -> list[str]:
        """Exactly one normalised address per entry, or a UserError.

        The guard and the mailer must see the same list: Odoo's mailer also
        splits on ';' and whitespace, so an entry holding more than one
        address is refused rather than checked by its last domain only.
        """
        recipients: list[str] = []
        for entry in (to or '').replace(';', ',').split(','):
            entry = entry.strip()
            if not entry:
                continue
            normalized = email_normalize(entry)
            if len(email_split(entry)) != 1 or not normalized:
                raise UserError(_('Invalid email address: %s', entry))
            if normalized not in recipients:
                recipients.append(normalized)
        if not recipients:
            raise UserError(_('No recipient address given.'))
        return recipients

    @api.model
    def _mcp_check_recipients(self, recipients: list[str]) -> None:
        """Refuse any recipient outside the allowlist; fail closed.

        The chat approval prompt is a HUMAN check, and humans skim — a
        look-alike domain (``alwin@0ntash.net``) reads like the real one,
        and a prompt injection hidden in record data can propose exactly
        that. This wall is server-side: an address that is not allowed
        cannot be mailed even after the user tapped Approve.

        An address passes when ANY of these holds: it belongs to an
        internal user; its domain is the company's own sending domain; it
        matches ``ont_mcp_mail.allowed_recipients``; or (unless disabled
        via ``ont_mcp_mail.allow_partner_recipients``) it belongs to a
        contact already in Odoo.

        :raise UserError: on the first disallowed address — no mail is sent.
        """
        params = self.env['ir.config_parameter'].sudo()
        configured = (params.get_param(ALLOWED_RECIPIENTS_PARAM) or '').strip()
        if configured == '*':
            _logger.warning(
                'MCP mail recipient allowlist disabled (%s = "*"): '
                'the assistant may email any address.', ALLOWED_RECIPIENTS_PARAM)
            return

        allowed_addresses: set[str] = set()
        allowed_domains: set[str] = set()
        for entry in configured.replace(';', ',').split(','):
            entry = entry.strip().lower()
            if not entry:
                continue
            if entry.startswith('@'):
                allowed_domains.add(entry[1:])
            else:
                allowed_addresses.add(email_normalize(entry) or entry)
        # The company's own sending domain is always trusted.
        default_from = self.env['ir.mail_server']._get_default_from_address() or ''
        if '@' in default_from:
            allowed_domains.add(default_from.rsplit('@', 1)[1].lower())

        allow_partners = (params.get_param(ALLOW_PARTNERS_PARAM) or '').strip().lower() \
            not in ('0', 'false', 'no', 'off')

        for address in recipients:
            normalized = email_normalize(address) or address.strip().lower()
            domain = normalized.rsplit('@', 1)[-1]
            if normalized in allowed_addresses or domain in allowed_domains:
                continue
            # Colleagues: makes "email me the report" work with no config.
            if self.env['res.users'].sudo().search_count(
                    [('share', '=', False), ('email_normalized', '=', normalized)]):
                continue
            if allow_partners and self.env['res.partner'].sudo().search_count(
                    [('email_normalized', '=', normalized)]):
                continue
            raise UserError(_(
                'Recipient %(address)s is not allowed. The assistant may only '
                'email colleagues, contacts already in Odoo, and addresses '
                'listed in the system parameter "%(param)s".',
                address=address, param=ALLOWED_RECIPIENTS_PARAM,
            ))

    @api.model
    def _mcp_sender_name(self) -> str:
        """Display name of the assistant sender identity.

        Configurable without code via the system parameter
        ``ont_mcp_mail.sender_name``; defaults to "<Company> AI Assistant".
        """
        configured = (
            self.env['ir.config_parameter'].sudo()
            .get_param('ont_mcp_mail.sender_name') or ''
        ).strip()
        return configured or _('%s AI Assistant', self.env.company.name)

    @api.model
    def _mcp_email_from(self) -> str:
        """Stable ``email_from`` for agent mails: assistant name + the
        configured sender address, then Odoo's default or the user's address."""
        configured = (
            self.env['ir.config_parameter'].sudo()
            .get_param(SENDER_EMAIL_PARAM) or ''
        ).strip()
        if configured and not email_split(configured):
            raise UserError(_(
                'The configured MCP sender email is invalid: %s', configured,
            ))
        address = (
            configured
            or self.env['ir.mail_server']._get_default_from_address()
            or self.env.user.email
            or ''
        )
        return formataddr((self._mcp_sender_name(), address))

    @api.model
    def _mcp_render_record_report(
        self, record_model: str, record_id: int, report_ref: str | int | None,
    ) -> tuple[str, bytes]:
        """Render the record's (default) report; returns (filename, bytes).

        Attaching only READS the record, so the allowlist is consulted with
        the read category even though the tool itself is write-category.
        muk_mcp_access (17.x) reads the category from
        ``request._mcp_tool_category``, which the dispatcher set to 'write'
        for this tool, so it is switched to 'read' for the check only.
        """
        previous = getattr(request, '_mcp_tool_category', None) if request else None
        if request:
            request._mcp_tool_category = 'read'
        try:
            self._resolve_model(record_model)
            self._mcp_assert_records_allowed(record_model, [record_id])
        finally:
            if request:
                request._mcp_tool_category = previous
        record = self.env[record_model].browse(record_id)
        if not record.exists():
            raise UserError(
                _('Record %(id)s of %(model)s not found.', id=record_id, model=record_model))

        if report_ref:
            report = self._resolve_report(report_ref)
        else:
            report = self.env['ir.actions.report'].search(
                [('model', '=', record_model), ('report_type', '=', 'qweb-pdf')], limit=1)
            if not report:  # test/dev fallback: any printable report at all
                report = self.env['ir.actions.report'].search(
                    [('model', '=', record_model)], limit=1)
        if not report:
            raise UserError(
                _('No printable report found for %(model)s.', model=record_model))

        content, report_type = report._render(report.report_name, [record_id])
        if isinstance(content, str):
            content = content.encode()
        extensions = {'pdf': 'pdf', 'text': 'txt', 'html': 'html'}
        extension = extensions.get(report_type, report_type or 'bin')
        base_name = (record.display_name or report.name or 'document')
        safe_name = base_name.replace('/', '_').replace(' ', '_')
        return f'{safe_name}.{extension}', content
