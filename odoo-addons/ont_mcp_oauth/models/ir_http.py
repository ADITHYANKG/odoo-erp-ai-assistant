from __future__ import annotations

import logging
import re

import werkzeug.exceptions

from odoo import SUPERUSER_ID, api, fields, models
from odoo.http import request

from odoo.addons.muk_mcp.tools.rate_limit import rate_limiter
from odoo.addons.ont_mcp_oauth.tools import jwt as jwt_tools

_logger = logging.getLogger(__name__)
_BEARER_RE = re.compile(r'^bearer\s+(.+)$', re.IGNORECASE)

# Ceiling on JWT verification attempts per IP per minute. Bounds the CPU cost
# of unauthenticated RSA verifications and blind jti probing, which muk's own
# rate limiter cannot cover because it only runs *after* successful auth.
# Generous by default so several busy agents behind one NAT are unaffected;
# tune with the ont_mcp_oauth.jwt_attempts_per_minute config parameter.
_JWT_ATTEMPTS_PER_MINUTE = 300


class _OAuthKeyAdapter:
    """Duck-types ``muk_mcp.key`` for the five members the MCP code consumes.

    muk_mcp's controller/tool code only touches ``.id``, ``.name``,
    ``.key_prefix``, ``.scope`` and ``._check_rate_limit()`` on
    ``request._mcp_key`` — so an
    OAuth-issued identity presents this lightweight object instead of a real
    key record.
    """

    #: muk_mcp 17 logs ``key.id`` into ``muk_mcp.log.key_id`` on every request;
    #: an OAuth identity has no key row, so it reports a falsy id (stored NULL).
    __slots__ = ('id', 'name', 'key_prefix', 'scope', '_dbname', '_client_id', '_rate_limit')

    def __init__(self, dbname, client_name, client_id, jti, scope, rate_limit):
        self.id = False
        self.name = 'OAuth: %s' % client_name
        self.key_prefix = (jti or '')[:8]
        self.scope = scope  # 'read' or 'write'
        self._dbname = dbname
        self._client_id = client_id
        self._rate_limit = rate_limit

    def _check_rate_limit(self) -> bool:
        # 0 => unlimited, matching muk_mcp.key semantics.
        return rate_limiter.check((self._dbname, 'oauth', self._client_id), self._rate_limit, 60)


class IrHttp(models.AbstractModel):
    _inherit = 'ir.http'

    # ------------------------------------------------------------------
    # Auth
    # ------------------------------------------------------------------
    @classmethod
    def _auth_method_mcp(cls) -> None:
        """Authenticate an /mcp request, trying an OAuth JWT before the muk key.

        A bearer value with exactly two dots is treated as a JWT and verified
        here; on any JWT failure we raise (never silently fall back, so a bad
        JWT can't be probed against the key path). Opaque tokens fall through to
        muk_mcp's original key-hash authentication.
        """
        if request.httprequest.method == 'OPTIONS':
            return
        header = request.httprequest.headers.get('Authorization', '')
        match = _BEARER_RE.match(header)
        token = match and match.group(1).strip()
        if token and token.count('.') == 2:
            return cls._ont_auth_jwt(token)
        return super()._auth_method_mcp()

    @classmethod
    def _ont_auth_jwt(cls, token: str) -> None:
        env = api.Environment(request.env.cr, SUPERUSER_ID, {})
        SigningKey = env['ont_mcp_oauth.signing.key']
        Token = env['ont_mcp_oauth.token']
        base = env['ir.config_parameter'].get_param('web.base.url', '').rstrip('/')
        issuer = base
        audience = base + '/mcp'

        def deny(reason):
            cls._ont_audit_denied('mcp/auth', reason)
            raise werkzeug.exceptions.Unauthorized()

        # Bound unauthenticated verification work per IP before doing any crypto.
        # Throttling MUST answer 429, not 401: a 401 tells the client its token
        # is bad, so it reacts by refreshing or restarting OAuth — turning a
        # rate limit into a re-auth storm (all clients behind one NAT/proxy IP
        # share this bucket). 429 + Retry-After (added in _handle_error) tells
        # it to back off instead.
        limit = int(env['ir.config_parameter'].get_param(
            'ont_mcp_oauth.jwt_attempts_per_minute', _JWT_ATTEMPTS_PER_MINUTE))
        if not rate_limiter.check(
                (request.env.cr.dbname, 'oauth_jwt', request.httprequest.remote_addr),
                limit, 60):
            cls._ont_audit_denied('mcp/auth', 'jwt_rate_limited')
            raise werkzeug.exceptions.TooManyRequests()

        try:
            header = jwt_tools.peek_header(token)
        except jwt_tools.JWTError:
            return deny('malformed_jwt')
        if header.get('alg') != 'RS256':
            return deny('unexpected_alg')
        key = SigningKey._find_by_kid(header.get('kid'))
        if not key:
            return deny('unknown_kid')
        try:
            payload = jwt_tools.decode(
                token, key.public_pem, issuer=issuer, audience=audience,
            )
        except jwt_tools.JWTError as exc:
            _logger.warning('MCP JWT rejected: %s (expected iss=%s aud=%s)', exc, issuer, audience)
            return deny('invalid_jwt')

        jti = payload.get('jti')
        access_row = Token.sudo().search(
            [('kind', '=', 'access'), ('token_hash', '=', jti)], limit=1,
        )
        if not access_row or access_row.revoked:
            return deny('revoked_or_unknown_jti')

        sub = payload.get('sub')
        try:
            uid = int(sub)
        except (TypeError, ValueError):
            return deny('bad_sub')
        user = env['res.users'].sudo().browse(uid)
        if not user.exists() or not user.active:
            return deny('inactive_user')
        # MCP is for internal users only. muk's bearer-key path is implicitly
        # gated this way (its key model ACL is base.group_user); the OAuth path
        # must not widen that boundary to portal/share users.
        if not user._is_internal():
            return deny('non_internal_user')

        # Scope maps to muk's enforcement: only an explicit mcp:write grants
        # write. Fail CLOSED — an empty/garbled scope claim is read-only.
        scope_str = payload.get('scope') or ''
        scopes = set(scope_str.split())
        scope = 'write' if 'mcp:write' in scopes else 'read'

        client_id = payload.get('client_id') or ''
        client = env['ont_mcp_oauth.client'].sudo().search(
            [('client_id', '=', client_id)], limit=1,
        )
        if not client or not client.active:
            return deny('unknown_client')

        request.update_env(user=user.id)
        request._mcp_key = _OAuthKeyAdapter(
            request.env.cr.dbname, client.name, client_id, jti, scope, client.rate_limit,
        )
        request.update_env(context={'mcp_name': request._mcp_key.name})
        request.session.can_save = False
        # Best-effort last_used stamp — throttled and isolated in a savepoint.
        # MCP clients overlap requests (closing one session while opening the
        # next); two stamps of this row under REPEATABLE READ raise
        # "could not serialize access due to concurrent update" at FLUSH time,
        # i.e. later, inside whatever handler flushes first — past this
        # try/except — poisoning the whole request ("current transaction is
        # aborted") and breaking tools/call for OAuth users. Flushing inside a
        # savepoint confines any conflict to the stamp itself, and the
        # throttle makes conflicts rare to begin with.
        try:
            now = fields.Datetime.now()
            last_used = client.last_used
            if not last_used or (now - last_used).total_seconds() >= 60:
                with env.cr.savepoint():
                    client.sudo().last_used = now
                    client.sudo().flush_recordset(["last_used"])
        except Exception:  # noqa: BLE001 - stamping must never fail auth
            pass
        return None

    @classmethod
    def _ont_audit_denied(cls, method, reason):
        try:
            ip = request.httprequest.remote_addr
            # Sample denial audits per IP. Every denied bearer writes an audit
            # row on a fresh pooled connection, so an unauthenticated junk
            # flood would otherwise convert free HTTP requests into DB
            # connection churn plus 30 days of stored junk. 60/min per IP
            # keeps real incidents fully visible; overflow goes to the server
            # log only.
            if not rate_limiter.check(
                    (request.env.cr.dbname, 'oauth_denied_audit', ip), 60, 60):
                _logger.info('mcp auth denied (audit sampled): %s from %s', reason, ip)
                return
            request.env['muk_mcp.log'].sudo().log(
                method=method, status='denied', error_message=reason, ip_address=ip,
            )
        except Exception:  # noqa: BLE001 - auditing must never raise
            pass

    # ------------------------------------------------------------------
    # Error rendering — attach WWW-Authenticate on /mcp 401s
    # ------------------------------------------------------------------
    @classmethod
    def _handle_error(cls, exception):
        response = super()._handle_error(exception)
        try:
            code = getattr(exception, 'code', None)
            path = request.httprequest.path if request else ''
            # Only the protected resource itself advertises the challenge — the
            # OAuth endpoints under /mcp/oauth/* have their own RFC 6749 errors.
            is_mcp_resource = path == '/mcp'
            if code == 401 and is_mcp_resource and hasattr(response, 'headers'):
                base = request.env['ir.config_parameter'].sudo().get_param(
                    'web.base.url', '',
                ).rstrip('/')
                meta = f'{base}/.well-known/oauth-protected-resource/mcp'
                response.headers['WWW-Authenticate'] = f'Bearer resource_metadata="{meta}"'
            if code == 429 and is_mcp_resource and hasattr(response, 'headers'):
                # Pair with the JWT rate limiter: tell SDKs when to come back.
                response.headers['Retry-After'] = '60'
        except Exception:  # noqa: BLE001 - never break error handling
            pass
        return response
