from __future__ import annotations

import hashlib

from odoo import api, fields, models


class OntMcpOAuthToken(models.Model):
    """Refresh-token chains and the issued access-token (jti) registry.

    Two kinds share one table so a single ``family_id`` links a whole grant:

    * ``refresh`` rows store only the SHA-256 hash of the opaque refresh token;
      rotation consumes the old row and links a successor. Replaying a consumed
      refresh outside the grace window revokes the entire family.
    * ``access`` rows record the ``jti`` of every issued access JWT so tokens can
      be revoked before their natural expiry (via /revoke or family revocation).
      A JWT whose jti has no live row is rejected at /mcp even if its signature
      is valid.
    """

    _name = 'ont_mcp_oauth.token'
    _description = 'MCP OAuth Token'
    _order = 'create_date desc'

    kind = fields.Selection(
        selection=[('refresh', 'Refresh'), ('access', 'Access (jti)')],
        required=True, index=True,
    )
    # For refresh: sha256(opaque token). For access: the jti value.
    # Field-level guard mirrors private_pem/client_secret_hash: the model ACL
    # already denies non-admins, but this survives a future ACL loosening.
    token_hash = fields.Char(
        required=True, index=True, copy=False, groups='base.group_system')
    family_id = fields.Char(required=True, index=True, copy=False)
    client_id = fields.Many2one('ont_mcp_oauth.client', required=True, ondelete='cascade', index=True)
    user_id = fields.Many2one('res.users', required=True, ondelete='cascade')
    scope = fields.Char()
    resource = fields.Char()
    # Hash of the authorization code this family was originally minted from.
    # Lets a code-replay attack revoke exactly the family derived from that
    # code (RFC 6749 §4.1.2), instead of guessing by user+client.
    code_hash = fields.Char(index=True, copy=False)
    expires_at = fields.Datetime(required=True, index=True)
    consumed_at = fields.Datetime(copy=False)
    revoked = fields.Boolean(default=False, index=True)
    successor_id = fields.Many2one('ont_mcp_oauth.token', copy=False)

    _sql_constraints = [
        ('token_hash_uniq', 'UNIQUE(token_hash)', 'Token hash / jti must be unique.'),
    ]

    @staticmethod
    def _hash(raw: str) -> str:
        return hashlib.sha256(raw.encode('utf-8')).hexdigest()

    # ------------------------------------------------------------------
    # Refresh rotation
    # ------------------------------------------------------------------
    @api.model
    def _consume_refresh(self, raw_token: str):
        """Atomically consume an unconsumed, unrevoked, unexpired refresh token.

        :return: the winning ``ont_mcp_oauth.token`` record, or empty if not
            consumable this call.
        """
        token_hash = self._hash(raw_token)
        # One clock rules: expires_at was stamped by the APPLICATION clock, and
        # the grace-window age in the controller is measured against
        # consumed_at with the application clock too. Using DB now() here
        # instead would let app/DB host clock skew misclassify a benign
        # in-grace retry as a replay (revoking the family), or reject every
        # token as expired.
        now = fields.Datetime.now()
        self.env.cr.execute(
            """
            UPDATE ont_mcp_oauth_token
               SET consumed_at = %s
             WHERE token_hash = %s
               AND kind = 'refresh'
               AND consumed_at IS NULL
               AND revoked = false
               AND expires_at > %s
         RETURNING id
            """,
            (now, token_hash, now),
        )
        row = self.env.cr.fetchone()
        if not row:
            return self.browse()
        return self.browse(row[0])

    @api.model
    def _find_refresh(self, raw_token: str):
        return self.sudo().search(
            [('kind', '=', 'refresh'), ('token_hash', '=', self._hash(raw_token))], limit=1,
        )

    def _revoke_family(self):
        """Revoke every refresh and access row sharing this record's family."""
        for rec in self:
            self.sudo().search([('family_id', '=', rec.family_id)]).write({'revoked': True})

    @api.model
    def _revoke_families_of_code(self, code_hash: str):
        """Revoke every family minted from a given authorization code."""
        if not code_hash:
            return
        for family in set(self.sudo().search(
                [('code_hash', '=', code_hash)]).mapped('family_id')):
            self.sudo().search([('family_id', '=', family)]).write({'revoked': True})

    @api.model
    def _revoke_jti(self, jti: str):
        self.sudo().search(
            [('kind', '=', 'access'), ('token_hash', '=', jti)],
        ).write({'revoked': True})

    @api.autovacuum
    def _gc_tokens(self):
        cutoff = fields.Datetime.subtract(fields.Datetime.now(), days=7)
        self.sudo().search([
            '|', ('expires_at', '<', cutoff), '&', ('revoked', '=', True), ('write_date', '<', cutoff),
        ]).unlink()
