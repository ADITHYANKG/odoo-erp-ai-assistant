from __future__ import annotations

import secrets

from odoo import api, fields, models

# The pending request lives only for the duration of the consent interaction.
_AUTHREQ_TTL_SECONDS = 600


class OntMcpOAuthAuthRequest(models.Model):
    """A server-side pending authorization request awaiting user consent.

    The consent form carries ONLY the opaque ``req_token`` (plus the CSRF
    token) — never the client_id, scope, redirect_uri, etc. On POST the server
    reloads every parameter from this record and re-validates, so a user cannot
    tamper with hidden form fields to widen scope or change the redirect target.
    """

    _name = 'ont_mcp_oauth.authrequest'
    _description = 'MCP OAuth Pending Authorization Request'
    _order = 'create_date desc'

    req_token = fields.Char(
        required=True, index=True, copy=False,
        default=lambda self: secrets.token_urlsafe(32),
    )
    client_id = fields.Many2one('ont_mcp_oauth.client', required=True, ondelete='cascade')
    user_id = fields.Many2one('res.users', required=True, ondelete='cascade')
    scope = fields.Char(required=True, help='Space-joined requested scopes.')
    code_challenge = fields.Char(required=True)
    redirect_uri = fields.Char(required=True)
    resource = fields.Char()
    state = fields.Char()
    expires_at = fields.Datetime(required=True, index=True)

    _sql_constraints = [
        ('req_token_uniq', 'UNIQUE(req_token)', 'Request token must be unique.'),
    ]

    @api.model
    def _default_expiry(self):
        return fields.Datetime.add(fields.Datetime.now(), seconds=_AUTHREQ_TTL_SECONDS)

    @api.model
    def _consume(self, req_token: str, user):
        """Atomically load and delete a valid pending request for ``user``.

        A single ``DELETE ... RETURNING`` does the check and the removal in one
        statement, so two concurrent submissions of the same ``req_token``
        cannot both win: exactly one gets the snapshot. A read-then-unlink would
        let a double-submitted consent form mint two authorization codes from a
        single user interaction.

        ``user_id`` is part of the predicate rather than a check afterwards, so
        someone else's pending request is never deleted — a leaked token cannot
        be used to cancel the owner's authorization. Expired rows are likewise
        left for ``_gc_authrequests`` instead of being deleted by a caller who
        does not own them.

        :return: a dict snapshot of the request fields, or None if not found,
            expired, or owned by a different user.
        """
        if not req_token:
            return None
        # Flush first: the row may have been created earlier in this same
        # transaction and still be pending in the ORM cache.
        self.env.flush_all()
        # App clock, matching the app-stamped expires_at (see code._consume).
        self.env.cr.execute(
            """
            DELETE FROM ont_mcp_oauth_authrequest
             WHERE req_token = %s
               AND user_id = %s
               AND expires_at > %s
         RETURNING client_id, user_id, scope, code_challenge,
                   redirect_uri, resource, state
            """,
            (req_token, user.id, fields.Datetime.now()),
        )
        row = self.env.cr.dictfetchone()
        if not row:
            return None
        # The row was deleted behind the ORM's back; drop any cached copy.
        self.env.invalidate_all()
        return {
            'client_id': self.env['ont_mcp_oauth.client'].browse(row['client_id']),
            'user_id': self.env['res.users'].browse(row['user_id']),
            'scope': row['scope'],
            'code_challenge': row['code_challenge'],
            'redirect_uri': row['redirect_uri'],
            'resource': row['resource'],
            'state': row['state'],
        }

    @api.autovacuum
    def _gc_authrequests(self):
        self.sudo().search([('expires_at', '<', fields.Datetime.now())]).unlink()
