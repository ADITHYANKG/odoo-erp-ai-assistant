from __future__ import annotations

import hashlib

from odoo import api, fields, models

# Authorization codes are short-lived by spec recommendation (<= 10 min); we
# use 2 minutes, which is ample for an interactive redirect round-trip.
_CODE_TTL_SECONDS = 120


class OntMcpOAuthCode(models.Model):
    """A one-time authorization code issued by /authorize and spent at /token.

    Only the SHA-256 hash of the opaque code is stored. Consumption is atomic
    (single UPDATE ... RETURNING) so a code can be redeemed exactly once even
    under concurrent replay.
    """

    _name = 'ont_mcp_oauth.code'
    _description = 'MCP OAuth Authorization Code'
    _order = 'create_date desc'

    # Field-level guard mirrors private_pem/client_secret_hash: the model ACL
    # already denies non-admins, but this survives a future ACL loosening.
    code_hash = fields.Char(
        required=True, index=True, copy=False, groups='base.group_system')
    client_id = fields.Many2one('ont_mcp_oauth.client', required=True, ondelete='cascade', index=True)
    user_id = fields.Many2one('res.users', required=True, ondelete='cascade')
    scope = fields.Char(required=True, help='Space-joined granted scopes.')
    code_challenge = fields.Char(required=True, help='PKCE S256 challenge.')
    redirect_uri = fields.Char(required=True)
    resource = fields.Char(help='RFC 8707 requested resource (audience).')
    expires_at = fields.Datetime(required=True, index=True)
    consumed_at = fields.Datetime(copy=False)

    _sql_constraints = [
        ('code_hash_uniq', 'UNIQUE(code_hash)', 'Authorization code hash must be unique.'),
    ]

    @staticmethod
    def _hash(raw: str) -> str:
        return hashlib.sha256(raw.encode('utf-8')).hexdigest()

    @api.model
    def _default_expiry(self):
        return fields.Datetime.add(fields.Datetime.now(), seconds=_CODE_TTL_SECONDS)

    @api.model
    def _consume(self, raw_code: str):
        """Atomically mark a valid, unexpired, unconsumed code as consumed.

        :return: the ``ont_mcp_oauth.code`` record if this call won the race and
            the code was valid; otherwise an empty recordset. Callers must treat
            an empty result as "invalid_grant" and, if a matching-but-already-
            consumed row exists, as a replay attack (revoke the token family).
        """
        code_hash = self._hash(raw_code)
        # Application clock on both sides: expires_at was stamped with the app
        # clock, so comparing against DB now() would make every code dead on
        # arrival once app/DB host clocks drift past the 120 s code TTL.
        now = fields.Datetime.now()
        self.env.cr.execute(
            """
            UPDATE ont_mcp_oauth_code
               SET consumed_at = %s
             WHERE code_hash = %s
               AND consumed_at IS NULL
               AND expires_at > %s
         RETURNING id
            """,
            (now, code_hash, now),
        )
        row = self.env.cr.fetchone()
        if not row:
            return self.browse()
        return self.browse(row[0])

    @api.model
    def _find_by_hash(self, raw_code: str):
        return self.sudo().search([('code_hash', '=', self._hash(raw_code))], limit=1)

    @api.autovacuum
    def _gc_codes(self):
        cutoff = fields.Datetime.subtract(fields.Datetime.now(), days=1)
        self.sudo().search([('expires_at', '<', cutoff)]).unlink()
