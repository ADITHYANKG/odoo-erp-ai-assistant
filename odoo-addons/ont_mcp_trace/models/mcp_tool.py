from __future__ import annotations

import logging
from contextlib import suppress
from html import escape
from typing import Any

from markupsafe import Markup

from odoo import api, models
from odoo.api import Environment

from odoo.addons.muk_mcp.core.tool import get_tool_index

_logger = logging.getLogger(__name__)

#: Write tools that need no extra note: post_message's content IS the chatter
#: entry, send_email posts its own linked message when a record is attached,
#: and a deleted record's chatter is gone with it (MCP Logs keep deletions).
_UNTRACED = frozenset({'post_message', 'send_email', 'delete_records'})

#: A mass write must not storm the chatter of every touched record.
_MAX_TRACED_RECORDS = 10


class MCPTool(models.Model):
    """Chatter trace: successful MCP writes leave a note on the record.

    Layered on top of the OAuth gate's own ``_execute`` override via the
    normal model MRO; posting failures can never break (or roll back) the
    tool call itself.
    """

    _inherit = 'muk_mcp.tool'

    @api.model
    def _execute(
        self,
        name: str,
        arguments: dict[str, Any],
        env: Environment,
        enforce_scope: str | None,
    ) -> tuple[Any, dict[str, Any], str | None]:
        # Snapshot BEFORE execution: write() overrides (e.g. geolocation)
        # mutate the values dict in place, and the note must reflect what the
        # caller actually requested.
        snapshot = dict(arguments)
        if isinstance(snapshot.get('values'), dict):
            snapshot['values'] = dict(snapshot['values'])
        result = super()._execute(name, arguments, env, enforce_scope)
        try:
            self._ont_trace_write(name, snapshot, env, result)
        except Exception:
            _logger.warning('MCP chatter trace failed for %s', name, exc_info=True)
        return result

    @api.model
    def _ont_trace_write(
        self,
        name: str,
        arguments: dict[str, Any],
        env: Environment,
        result: tuple[Any, dict[str, Any], str | None],
    ) -> None:
        if name in _UNTRACED:
            return
        entry = get_tool_index(env).get(name)
        if not entry or entry.get('category') != 'write':
            return
        model_name = arguments.get('model')
        record_ids = (result[1] or {}).get('res_ids') or []
        if not model_name or not record_ids or model_name not in env:
            return
        records = env[model_name].browse(record_ids[:_MAX_TRACED_RECORDS]).exists()
        if not records or not hasattr(records, 'message_post'):
            return
        if len(record_ids) > _MAX_TRACED_RECORDS:
            _logger.info(
                'MCP trace: noting %d of %d records touched by %s on %s',
                _MAX_TRACED_RECORDS, len(record_ids), name, model_name,
            )
        body = self._ont_trace_body(name, arguments, env)
        for record in records:
            # A failed post must neither surface nor poison the transaction.
            with suppress(Exception), env.cr.savepoint():
                record.message_post(
                    body=body,
                    message_type='notification',
                    subtype_xmlid='mail.mt_note',  # internal: no follower emails
                )

    @api.model
    def _ont_trace_body(
        self, name: str, arguments: dict[str, Any], env: Environment
    ) -> Markup:
        detail = ''
        if name == 'create_records':
            detail = 'record created'
        elif name == 'update_records':
            values = arguments.get('values')
            if isinstance(values, dict) and values:
                detail = 'fields: %s' % ', '.join(sorted(values))
        elif name == 'call_method':
            method = arguments.get('method')
            if method:
                detail = 'method: %s' % method
        text = '\U0001f527 via MCP: <b>%s</b> by %s' % (escape(name), escape(env.user.name))
        if detail:
            text += ' — %s' % escape(detail)
        return Markup(text)
