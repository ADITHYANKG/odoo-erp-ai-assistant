from __future__ import annotations

from odoo import api, fields, models


class OntMcpOAuthConsent(models.Model):
    """A remembered per-user, per-client scope grant.

    Lets a returning client skip the consent screen when the previously approved
    scopes already cover what is being requested. Users can revoke a consent
    from their preferences, which also kills the client's live token families.
    """

    _name = 'ont_mcp_oauth.consent'
    _description = 'MCP OAuth Consent'
    _order = 'write_date desc'

    user_id = fields.Many2one('res.users', required=True, ondelete='cascade', index=True)
    client_id = fields.Many2one('ont_mcp_oauth.client', required=True, ondelete='cascade', index=True)
    scope_read = fields.Boolean(default=False)
    scope_write = fields.Boolean(default=False)
    last_used = fields.Datetime()

    _sql_constraints = [
        ('user_client_uniq', 'UNIQUE(user_id, client_id)', 'A user has a single consent record per client.'),
    ]

    def _granted_scopes(self):
        self.ensure_one()
        scopes = set()
        if self.scope_read:
            scopes.add('mcp:read')
        if self.scope_write:
            scopes.add('mcp:write')
        return scopes

    @api.model
    def _covers(self, user, client, requested: set[str]) -> bool:
        """True iff a stored consent already covers ``requested`` scopes."""
        rec = self.sudo().search(
            [('user_id', '=', user.id), ('client_id', '=', client.id)], limit=1,
        )
        return bool(rec) and requested.issubset(rec._granted_scopes())

    @api.model
    def _remember(self, user, client, granted: set[str]):
        """Upsert the consent record with the granted scope set."""
        vals = {
            'scope_read': 'mcp:read' in granted,
            'scope_write': 'mcp:write' in granted,
            'last_used': fields.Datetime.now(),
        }
        rec = self.sudo().search(
            [('user_id', '=', user.id), ('client_id', '=', client.id)], limit=1,
        )
        if rec:
            rec.write(vals)
            return rec
        return self.sudo().create({'user_id': user.id, 'client_id': client.id, **vals})

    def action_revoke(self):
        """Revoke this consent and all live token families for its user+client."""
        Token = self.env['ont_mcp_oauth.token'].sudo()
        for rec in self:
            families = Token.search([
                ('user_id', '=', rec.user_id.id), ('client_id', '=', rec.client_id.id),
            ])
            families.write({'revoked': True})
            rec.unlink()
