from __future__ import annotations

from odoo import api, fields, models


class Connect(models.TransientModel):
    _inherit = 'muk_mcp.connect'

    # Non-stored so nothing sensitive persists in the transient table.
    oauth_server_url = fields.Char(compute='_compute_oauth_server_url')
    oauth_base_is_https = fields.Boolean(compute='_compute_oauth_server_url')

    @api.depends('mcp_url')
    def _compute_oauth_server_url(self):
        for record in self:
            record.oauth_server_url = record.mcp_url
            record.oauth_base_is_https = (record.mcp_url or '').startswith('https://')
