"""Make MCP reads tolerant of unknown field names.

An LLM client frequently guesses Odoo field names that do not exist on the
target model (version drift: ``email`` vs ``work_email``, an invented
``state``/``manager_id``). muk_mcp passes ``fields`` straight to
``search_read``/``read``, so one bad name makes Odoo raise
``Invalid field X on model Y`` and the whole read fails. This mixin extension
drops the unknown names before delegating to the muk_mcp implementation, so a
wrong field is silently ignored (and logged) instead of being fatal.
"""

from __future__ import annotations

import logging

from odoo import api, models

_logger = logging.getLogger(__name__)


class McpFieldGuard(models.AbstractModel):
    _inherit = 'muk_mcp.mixin'

    @api.model
    def _mcp_valid_fields(self, model, fields):
        """Split ``fields`` into those the model has and those it does not.

        Dotted paths (``department_id.name``) are validated on their first
        segment only, matching how ``read``/``search_read`` traverse them.
        """
        target = self._resolve_model(model)
        valid, ignored = [], []
        for field in fields or []:
            root = field.split('.', 1)[0]
            (valid if root in target._fields else ignored).append(field)
        return valid, ignored

    def _mcp_fieldguard_filter(self, tool, model, fields):
        """Return a safe ``fields`` list, or None to read the default set."""
        if not fields:
            return fields
        valid, ignored = self._mcp_valid_fields(model, fields)
        if ignored:
            _logger.info(
                'fieldguard: dropped unknown field(s) %s on %s(%s)',
                ignored, tool, model,
            )
        # Never send [] — muk_mcp/Odoo would then read every field. Fall back
        # to display_name, which exists on every model and is cheap.
        return valid or ['display_name']

    @api.model
    def _mcp_read_records(self, model, ids, fields=None):
        fields = self._mcp_fieldguard_filter('read_records', model, fields)
        return super()._mcp_read_records(model, ids, fields)

    @api.model
    def _mcp_search_read(
        self, model, domain=None, fields=None, limit=80, offset=0, order=None,
    ):
        fields = self._mcp_fieldguard_filter('search_read', model, fields)
        return super()._mcp_search_read(
            model, domain, fields, limit, offset, order,
        )
