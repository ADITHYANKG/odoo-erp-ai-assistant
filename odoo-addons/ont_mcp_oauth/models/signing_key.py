from __future__ import annotations

import logging
import secrets

from odoo import _, api, fields, models
from odoo.exceptions import UserError

from odoo.addons.ont_mcp_oauth.models.res_config_settings import clamp_param
from odoo.addons.ont_mcp_oauth.tools import jwt as jwt_tools

_logger = logging.getLogger(__name__)

# Advisory-lock namespace so concurrent workers never generate two "first" keys.
_LOCK_KEY = 0x0117_0A01

# How long a retired key stays usable after rotation. While the row exists the
# key still resolves and still verifies its own signatures, so this window must
# be only as long as the access tokens it signed can live, plus leeway.
_RETIRE_LEEWAY_SECONDS = 3600
_RETIRE_MAX_SECONDS = 86400


class OntMcpOAuthSigningKey(models.Model):
    """RSA signing keys for OAuth access-token JWTs.

    The private PEM is readable only by system administrators (see the ACL);
    all runtime access goes through ``sudo()`` helpers on this model so the
    controllers never hold the material directly.

    Rotation is a graceful hand-over, not a cutover: the outgoing key is
    deactivated but deliberately kept resolvable (and published in the JWKS)
    until ``retire_until`` passes, so access tokens signed just before rotation
    keep working for the rest of their lifetime.
    """

    _name = 'ont_mcp_oauth.signing.key'
    _description = 'MCP OAuth Signing Key'
    _order = 'active desc, create_date desc'

    kid = fields.Char(string='Key ID', required=True, readonly=True, index=True)
    private_pem = fields.Char(string='Private Key (PEM)', readonly=True, groups='base.group_system')
    public_pem = fields.Char(string='Public Key (PEM)', readonly=True)
    active = fields.Boolean(default=True)
    retire_until = fields.Datetime(
        string='Valid Until', readonly=True,
        help='When this retired key stops verifying and becomes eligible for '
             'deletion. Fixed at rotation time from the access-token TTL then '
             'in force.',
    )

    _sql_constraints = [
        ('kid_uniq', 'UNIQUE(kid)', 'The signing key id must be unique.'),
    ]
    # At most one active key, enforced in Postgres. The advisory lock in
    # _get_active_key serialises generation, but Odoo runs REPEATABLE READ: a
    # worker that blocked on the lock re-reads its pre-commit snapshot, sees no
    # key, and would insert a second one. With this index that insert fails
    # instead (the request retries and then sees the winner's key), so a
    # duplicate live private key can never linger in the JWKS.
    # Odoo 17 has no models.UniqueIndex and _sql_constraints cannot express a
    # partial index, so it is created by hand in init().
    def init(self):
        super().init()
        self.env.cr.execute(
            'CREATE UNIQUE INDEX IF NOT EXISTS ont_mcp_oauth_signing_key_one_active_idx '
            'ON ont_mcp_oauth_signing_key ((1)) WHERE active IS TRUE'
        )

    # ------------------------------------------------------------------
    # Retirement bookkeeping
    # ------------------------------------------------------------------
    @api.model
    def _retention_seconds(self) -> int:
        """Seconds a key must stay valid after retirement."""
        ttl = clamp_param(
            'ont_mcp_oauth.access_token_ttl',
            self.env['ir.config_parameter'].sudo().get_param(
                'ont_mcp_oauth.access_token_ttl', 900),
        )
        return min(ttl + _RETIRE_LEEWAY_SECONDS, _RETIRE_MAX_SECONDS)

    def write(self, vals):
        """Stamp the retirement deadline when a key is deactivated.

        The deadline is fixed at retirement time rather than recomputed later.
        Deriving it from ``write_date`` and the current TTL meant that lowering
        the TTL retroactively killed tokens issued under the old one, and that
        any incidental write to the row silently extended how long a possibly
        leaked key kept verifying.

        Retirement is one-way: a retired key can never be brought back. It may
        be retired precisely because it leaked, and un-archiving it would put
        it back in the signing pool (and clear its deadline, so it could never
        be collected). Rotate to a fresh key instead.

        ``active`` is tested for truthiness rather than identity: callers -
        JSON-RPC in particular - legitimately send 0/1, and ``0 is False`` is
        False, which would have retired a key without stamping a deadline.
        """
        retiring = self.browse()
        if 'active' in vals:
            if vals['active']:
                if self.filtered(lambda key: not key.active):
                    raise UserError(_(
                        'A retired signing key cannot be reactivated — it may '
                        'have been retired because it leaked. Rotate to a new '
                        'key instead.'))
                vals = dict(vals, retire_until=False)
            elif 'retire_until' not in vals:
                retiring = self.filtered(lambda key: key.active and not key.retire_until)
        result = super().write(vals)
        if retiring:
            deadline = fields.Datetime.add(
                fields.Datetime.now(), seconds=self._retention_seconds())
            super(OntMcpOAuthSigningKey, retiring).write({'retire_until': deadline})
        return result

    # ------------------------------------------------------------------
    # Key material
    # ------------------------------------------------------------------
    @api.model
    def _get_active_key(self):
        """Return the current active signing key, generating one on first use.

        Uses a Postgres advisory lock to serialise the first-generation race
        between workers.
        """
        key = self.sudo().search([('active', '=', True)], limit=1)
        if key:
            return key
        self.env.cr.execute('SELECT pg_advisory_xact_lock(%s)', (_LOCK_KEY,))
        key = self.sudo().search([('active', '=', True)], limit=1)
        if key:
            return key
        return self._new_key()

    @api.model
    def _new_key(self):
        private_pem, public_pem = jwt_tools.generate_rsa_keypair()
        return self.sudo().create({
            'kid': secrets.token_urlsafe(8),
            'private_pem': private_pem,
            'public_pem': public_pem,
            'active': True,
        })

    @api.model
    def _rotate(self):
        """Retire the active key(s) and return a fresh active one.

        Takes the same advisory lock as ``_get_active_key``: without it, two
        concurrent rotations each fail to see the other's uncommitted insert
        and both commit a new active key, leaving an orphan that is never
        selected for signing, never retired, and never collected — a live
        private key sitting in the JWKS indefinitely.
        """
        self.env.cr.execute('SELECT pg_advisory_xact_lock(%s)', (_LOCK_KEY,))
        self.sudo().search([('active', '=', True)]).write({'active': False})
        # Flush the retirement before inserting the successor: the one-active
        # partial unique index is checked per statement, and the ORM would
        # otherwise emit the INSERT while the UPDATE still sits in cache.
        self.env.flush_all()
        return self._new_key()

    # ------------------------------------------------------------------
    # Lookups — must see retired keys, hence active_test=False
    # ------------------------------------------------------------------
    @api.model
    def _all_keys(self):
        """Every key row, active or retired.

        ``active`` is Odoo's magic archive field, so a bare ``search()`` would
        silently filter retired keys out. That would make rotation a hard
        outage: tokens signed by the outgoing key would fail with unknown_kid
        the instant it was deactivated.
        """
        return self.sudo().with_context(active_test=False)

    @api.model
    def _live_domain(self):
        """Keys that may still verify a signature: active, or within grace.

        Enforced at lookup time rather than relying on ``_gc_signing_keys``,
        which is an autovacuum job: a key whose deadline passed hours ago must
        stop verifying immediately, not whenever cron next happens to run.
        """
        return ['|', ('active', '=', True),
                ('retire_until', '>', fields.Datetime.now())]

    @api.model
    def _get_jwks(self):
        """Return the public JWK set (active + still-valid retired keys).

        A key whose PEM cannot be parsed is skipped rather than allowed to
        raise: this endpoint is the discovery path every client depends on, so
        one bad row must not take the whole authorization server offline.
        """
        jwks = []
        for key in self._all_keys().search(self._live_domain()):
            if not key.public_pem:
                continue
            try:
                jwks.append(jwt_tools.public_jwk(key.public_pem, key.kid))
            except Exception:  # noqa: BLE001 - never break discovery
                _logger.warning(
                    'Skipping unparseable signing key %s in JWKS', key.kid, exc_info=True)
        return {'keys': jwks}

    @api.model
    def _find_by_kid(self, kid):
        """Return the key record for a kid, or an empty recordset.

        A retired key past its ``retire_until`` no longer resolves, so tokens
        it signed stop authenticating on time regardless of cron health.
        """
        if not kid:
            return self.browse()
        return self._all_keys().search(
            [('kid', '=', kid)] + self._live_domain(), limit=1)

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------
    def action_rotate(self):
        """Start signing with a fresh key, retiring the current one gracefully."""
        self._rotate()
        return {'type': 'ir.actions.client', 'tag': 'reload'}

    def action_revoke_now(self):
        """Delete these retired keys immediately — for a suspected compromise.

        ``action_rotate`` only *retires* a key: it stays in the JWKS and keeps
        verifying its own signatures until ``retire_until`` passes. If the
        private key may have leaked, retiring is not enough — an attacker
        holding it can still mint valid access tokens. Deleting the row is what
        actually stops that, at the cost of invalidating any live token it
        signed.
        """
        if any(key.active for key in self):
            raise UserError(_(
                'Rotate first: the active signing key cannot be revoked, or the '
                'server would have no key left to sign tokens with.'))
        self.unlink()
        # Return to the list: a plain reload would land on the form of the
        # record we just deleted.
        return self.env['ir.actions.act_window']._for_xml_id(
            'ont_mcp_oauth.action_ont_mcp_oauth_signing_key')

    @api.autovacuum
    def _gc_signing_keys(self):
        """Delete retired keys once every token they signed has expired.

        Rows with no deadline — retired before ``retire_until`` existed, or
        created directly as inactive — are backfilled from ``write_date``,
        their effective retirement moment, **not** from now. Stamping them
        with a fresh full window would hand a long-retired (possibly leaked)
        key another lease of life; deriving from write_date means an old row
        is stamped and reaped in the same pass.
        """
        now = fields.Datetime.now()
        retention = self._retention_seconds()
        retired = self._all_keys().search([('active', '=', False)])
        for key in retired.filtered(lambda k: not k.retire_until):
            key.retire_until = fields.Datetime.add(
                key.write_date or now, seconds=retention)
        retired.filtered(lambda k: k.retire_until and k.retire_until < now).unlink()
