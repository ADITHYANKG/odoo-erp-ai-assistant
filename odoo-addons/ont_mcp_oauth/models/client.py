from __future__ import annotations

import hashlib
import hmac
import secrets

from odoo import _, api, fields, models
from odoo.exceptions import UserError, ValidationError

from odoo.addons.ont_mcp_oauth.tools import uri as uri_tools

# Iteration count for PBKDF2 secret hashing.
_PBKDF2_ROUNDS = 200_000


class OntMcpOAuthClient(models.Model):
    """A registered OAuth client (AI connector).

    Public clients (the norm for ChatGPT/Claude connectors) carry no secret and
    rely on PKCE. Confidential clients may set a secret, shown once at creation
    and stored only as a PBKDF2 hash.
    """

    _name = 'ont_mcp_oauth.client'
    _description = 'MCP OAuth Client'
    _order = 'create_date desc'
    _rec_name = 'name'

    name = fields.Char(required=True)
    client_id = fields.Char(
        required=True, readonly=True, index=True, copy=False,
        default=lambda self: 'ont_' + secrets.token_urlsafe(24),
    )
    client_type = fields.Selection(
        selection=[('public', 'Public (PKCE)'), ('confidential', 'Confidential')],
        default='public', required=True,
    )
    client_secret_hash = fields.Char(readonly=True, copy=False, groups='base.group_system')
    has_secret = fields.Boolean(
        string='Secret Set', compute='_compute_has_secret',
        help='A confidential client with no secret can never obtain a token.',
    )
    redirect_uris = fields.Text(
        required=True,
        help='One redirect URI per line. HTTPS, http://localhost, or a native '
             'custom-scheme URI. Matched exactly at authorization time.',
    )
    scope = fields.Selection(
        selection=[('read', 'Read only'), ('write', 'Read & Write')],
        default='write', required=True,
        help='Maximum scope this client may ever be granted.',
    )
    allow_refresh = fields.Boolean(
        string='Allow Refresh Tokens', default=True,
        help='Whether this client may use the refresh_token grant. Dynamically '
             'registered clients only get this if they asked for it.',
    )
    dcr = fields.Boolean(string='Dynamically Registered', readonly=True, copy=False)
    dcr_ip = fields.Char(string='Registered From IP', readonly=True, copy=False)
    rate_limit = fields.Integer(
        default=0,
        help='Per-minute request cap for tokens issued to this client on /mcp. '
             '0 uses the global MCP rate limit.',
    )
    active = fields.Boolean(default=True)
    last_used = fields.Datetime(readonly=True, copy=False)

    _sql_constraints = [
        ('client_id_uniq', 'UNIQUE(client_id)', 'The client id must be unique.'),
    ]

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    @api.constrains('redirect_uris')
    def _check_redirect_uris(self):
        for client in self:
            uris = uri_tools.parse_redirect_uris(client.redirect_uris)
            if not uris:
                raise ValidationError(_('At least one redirect URI is required.'))
            for u in uris:
                if not uri_tools.is_valid_redirect_uri(u):
                    raise ValidationError(_('Invalid redirect URI: %s', u))

    @api.depends('client_secret_hash')
    def _compute_has_secret(self):
        for client in self:
            client.has_secret = bool(client.sudo().client_secret_hash)

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------
    def action_regenerate_secret(self):
        """Mint a new client secret and show it once.

        Without this, a client created through the UI as "Confidential" had no
        way to ever get a secret (only the Dynamic Client Registration path set
        one), so it could never authenticate at /token — it failed closed, but
        silently.
        """
        self.ensure_one()
        if self.client_type != 'confidential':
            raise UserError(_('Only confidential clients use a secret.'))
        secret = secrets.token_urlsafe(32)
        self._set_secret(secret)
        return {
            'type': 'ir.actions.client',
            'tag': 'display_notification',
            'params': {
                'title': _('Client secret'),
                'message': _(
                    'Copy it now — only a hash is stored, so it cannot be shown '
                    'again:\n\n%s', secret),
                'sticky': True,
                'type': 'warning',
                # Reload so the "no secret" banner clears; leaving it up
                # invites a second regeneration, which would invalidate the
                # secret just copied.
                'next': {'type': 'ir.actions.client', 'tag': 'reload'},
            },
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _redirect_uri_list(self):
        self.ensure_one()
        return uri_tools.parse_redirect_uris(self.redirect_uris)

    def _allows_redirect_uri(self, uri):
        self.ensure_one()
        return uri_tools.matches_registered(uri, self._redirect_uri_list())

    def _scope_ceiling(self):
        """Return the set of scope strings this client may be granted."""
        self.ensure_one()
        return {'mcp:read', 'mcp:write'} if self.scope == 'write' else {'mcp:read'}

    @staticmethod
    def _hash_secret(secret: str) -> str:
        salt = secrets.token_bytes(16)
        dk = hashlib.pbkdf2_hmac('sha256', secret.encode('utf-8'), salt, _PBKDF2_ROUNDS)
        return f'pbkdf2_sha256${_PBKDF2_ROUNDS}${salt.hex()}${dk.hex()}'

    def _verify_secret(self, secret: object) -> bool:
        self.ensure_one()
        stored = self.sudo().client_secret_hash
        if not stored or not isinstance(secret, str):
            return False
        try:
            _algo, rounds, salt_hex, dk_hex = stored.split('$')
            dk = hashlib.pbkdf2_hmac(
                'sha256', secret.encode('utf-8'), bytes.fromhex(salt_hex), int(rounds),
            )
        except (ValueError, TypeError):
            return False
        return hmac.compare_digest(dk.hex(), dk_hex)

    def _set_secret(self, secret: str):
        self.ensure_one()
        self.sudo().client_secret_hash = self._hash_secret(secret)

    @api.autovacuum
    def _gc_unused_dcr_clients(self):
        """Remove DCR clients registered >24h ago that were never used.

        A client is "used" once it has any code, token, or consent, or a
        ``last_used`` stamp. Manually-created clients are never reaped.
        """
        cutoff = fields.Datetime.subtract(fields.Datetime.now(), hours=24)
        candidates = self.sudo().search([
            ('dcr', '=', True), ('create_date', '<', cutoff), ('last_used', '=', False),
        ])
        for client in candidates:
            has_children = (
                self.env['ont_mcp_oauth.code'].sudo().search_count([('client_id', '=', client.id)])
                or self.env['ont_mcp_oauth.token'].sudo().search_count([('client_id', '=', client.id)])
                or self.env['ont_mcp_oauth.consent'].sudo().search_count([('client_id', '=', client.id)])
            )
            if not has_children:
                client.unlink()
