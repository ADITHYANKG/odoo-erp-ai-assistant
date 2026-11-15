from __future__ import annotations

import base64
import binascii
import json
import secrets
import uuid
from html import escape
from urllib.parse import unquote, urlencode, urlsplit, urlunsplit

import werkzeug.exceptions
import werkzeug.utils

from odoo import fields, http
from odoo.http import request
from odoo.tools import str2bool

from odoo.addons.muk_mcp.tools.rate_limit import rate_limiter
from odoo.addons.ont_mcp_oauth.models.res_config_settings import clamp_param
from odoo.addons.ont_mcp_oauth.tools import jwt as jwt_tools
from odoo.addons.ont_mcp_oauth.tools import pkce as pkce_tools
from odoo.addons.ont_mcp_oauth.tools import uri as uri_tools
from .metadata import _issuer

_VALID_SCOPES = {'mcp:read', 'mcp:write'}
_MAX_BODY = 8192
# Generous: state-as-encrypted-JWT patterns routinely exceed 512 bytes, and a
# rejected state cannot legally be echoed back (RFC 6749 wants state in error
# redirects), so an over-tight cap strands the client mid-flow.
_MAX_STATE = 2048
_TOKEN_AUTH_METHODS = ('none', 'client_secret_post', 'client_secret_basic')


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _param(config_key, default):
    return request.env['ir.config_parameter'].sudo().get_param(config_key, default)


def _int_param(config_key, default):
    """Read an integer config parameter, clamped to its documented safe range.

    Never trust the stored value: System Parameters are editable directly, so a
    hand-edited TTL must not be able to mint an effectively immortal token.
    """
    return clamp_param(config_key, _param(config_key, default))


def _json_error(error, description=None, status=400, headers=None):
    body = {'error': error}
    if description:
        body['error_description'] = description
    all_headers = [('Content-Type', 'application/json'), ('Cache-Control', 'no-store')]
    if headers:
        all_headers.extend(headers)
    return request.make_response(json.dumps(body), headers=all_headers, status=status)


def _rate_limited_response(endpoint):
    """429 with Retry-After so SDKs can schedule a retry instead of dying.

    The window is 60 s (see _rate_limited), hence the fixed value. Note
    RFC 6749's token-error registry has no perfect code for throttling;
    clients key off the 429 status, which is why the header matters more
    than the JSON error name.
    """
    _audit(endpoint, 'rate_limited', 'rate_limit')
    return _json_error(
        'temporarily_unavailable', 'Too many requests.', 429,
        headers=[('Retry-After', '60')],
    )


def _client_credentials(form):
    """Resolve client credentials from Basic auth or the form body.

    RFC 6749 §2.3.1 makes HTTP Basic support a MUST for servers, and it is the
    DEFAULT client-auth style of mainstream OAuth libraries (openid-client,
    Authlib, Spring, requests-oauthlib). Both halves are percent-decoded per
    the RFC. Returns (client_id, client_secret, via_header).
    """
    header = request.httprequest.headers.get('Authorization', '')
    if header[:6].lower() == 'basic ':
        try:
            decoded = base64.b64decode(header[6:].strip(), validate=True).decode('utf-8')
            client_id, _sep, secret = decoded.partition(':')
            return unquote(client_id), unquote(secret), True
        except (binascii.Error, UnicodeDecodeError, ValueError):
            return None, None, True  # malformed header: fail as bad client auth
    return form.get('client_id'), form.get('client_secret'), False


def _invalid_client(via_header):
    """401 invalid_client; advertise Basic when the creds came by header."""
    headers = [('WWW-Authenticate', 'Basic realm="mcp-oauth"')] if via_header else None
    return _json_error('invalid_client', 'Client authentication failed.', 401,
                       headers=headers)


def _json_ok(payload, status=200):
    return request.make_response(
        json.dumps(payload),
        headers=[('Content-Type', 'application/json'), ('Cache-Control', 'no-store')],
        status=status,
    )


def _audit(method, status, reason=None, client=None, user_id=None):
    try:
        request.env['muk_mcp.log'].sudo().log(
            method=method, status=status, error_message=reason,
            ip_address=request.httprequest.remote_addr,
            key_name=client.name if client else None,
            key_prefix=(client.client_id[:8] if client else None),
            user_id=user_id,
        )
    except Exception:  # noqa: BLE001 - auditing must never raise
        pass


def _rate_limited(bucket, limit):
    return not rate_limiter.check((request.env.cr.dbname, bucket, request.httprequest.remote_addr), limit, 60)


def _form():
    """POST parameters from the request BODY only.

    OAuth credential-bearing endpoints (token, revoke) must not accept
    parameters from the query string — that would leak codes, verifiers and
    tokens into access logs and proxies (RFC 6749 §3.2 requires the body).
    """
    return request.httprequest.form


def _redirect_with_params(redirect_uri, params):
    """Append params to a (already-validated) redirect URI, safely serialized."""
    parts = urlsplit(redirect_uri)
    query = parts.query
    extra = urlencode({k: v for k, v in params.items() if v is not None})
    new_query = f'{query}&{extra}' if query else extra
    return werkzeug.utils.redirect(
        urlunsplit((parts.scheme, parts.netloc, parts.path, new_query, '')), code=303,
    )


def _error_page(title, message, status=400):
    # Escape even though every current call site passes a literal: this keeps a
    # future edit that interpolates user input from becoming an XSS.
    title = escape(title)
    message = escape(message)
    html = (
        '<!DOCTYPE html><html><head><meta charset="utf-8">'
        f'<title>{title}</title></head><body style="font-family:sans-serif;'
        'max-width:32rem;margin:4rem auto;text-align:center">'
        f'<h2>{title}</h2><p>{message}</p></body></html>'
    )
    return request.make_response(
        html, status=status, headers=[
            ('Content-Type', 'text/html'),
            ('Cache-Control', 'no-store'),
            ('X-Frame-Options', 'DENY'),
            ('Content-Security-Policy', "frame-ancestors 'none'"),
        ],
    )


class OntMcpOAuth(http.Controller):

    # ------------------------------------------------------------------
    # RFC 7591 — Dynamic Client Registration
    # ------------------------------------------------------------------
    @http.route('/mcp/oauth/register', type='http', auth='public', methods=['POST'],
                csrf=False, save_session=False, cors='*')
    def register(self, **kw):
        if not str2bool(_param('ont_mcp_oauth.dcr_enabled', 'False'), default=False):
            _audit('oauth/register', 'denied', 'dcr_disabled')
            return _json_error('access_denied', 'Dynamic client registration is disabled.', 403)
        if _rate_limited('oauth_dcr', 5):
            return _rate_limited_response('oauth/register')

        raw = request.httprequest.get_data(as_text=True) or ''
        if len(raw) > _MAX_BODY:
            return _json_error('invalid_client_metadata', 'Request body too large.')
        try:
            meta = json.loads(raw) if raw else {}
        except (json.JSONDecodeError, ValueError):
            return _json_error('invalid_client_metadata', 'Body is not valid JSON.')
        if not isinstance(meta, dict):
            return _json_error('invalid_client_metadata', 'Body must be a JSON object.')

        redirect_uris = meta.get('redirect_uris')
        if not isinstance(redirect_uris, list) or not redirect_uris:
            return _json_error('invalid_redirect_uri', 'redirect_uris is required.')
        for u in redirect_uris:
            if not uri_tools.is_valid_redirect_uri(u):
                return _json_error('invalid_redirect_uri', f'Invalid redirect URI: {u}')

        # RFC 7591 §2: the default for an OMITTED token_endpoint_auth_method is
        # client_secret_basic, not 'none'. Public MCP clients (ChatGPT,
        # Claude.ai, mcp-remote) all send 'none' explicitly, so honouring the
        # spec default only affects generic libraries — which expect it.
        auth_method = meta.get('token_endpoint_auth_method', 'client_secret_basic')
        if auth_method not in _TOKEN_AUTH_METHODS:
            return _json_error('invalid_client_metadata', 'Unsupported token_endpoint_auth_method.')
        # RFC 7591's default is authorization_code alone, but real MCP clients
        # routinely omit grant_types and still expect refresh; only withhold
        # refresh when a client explicitly lists grants without it.
        grant_types = meta.get('grant_types') or ['authorization_code', 'refresh_token']
        if not set(grant_types).issubset({'authorization_code', 'refresh_token'}):
            return _json_error('invalid_client_metadata', 'Unsupported grant_types.')

        # Honour the requested scope rather than always granting the write
        # ceiling: a client that only asks to read can never be consented into
        # write, even by mistake.
        requested_scope = set((meta.get('scope') or 'mcp:read mcp:write').split())
        if not requested_scope.issubset(_VALID_SCOPES):
            return _json_error('invalid_client_metadata', 'Unsupported scope requested.')
        ceiling = 'write' if 'mcp:write' in requested_scope else 'read'
        allow_refresh = 'refresh_token' in grant_types

        confidential = auth_method in ('client_secret_post', 'client_secret_basic')
        client = request.env['ont_mcp_oauth.client'].sudo().create({
            'name': (meta.get('client_name') or 'DCR client')[:128],
            'client_type': 'confidential' if confidential else 'public',
            'redirect_uris': '\n'.join(redirect_uris),
            'scope': ceiling,
            'allow_refresh': allow_refresh,
            'dcr': True,
            'dcr_ip': request.httprequest.remote_addr,
        })
        response = {
            'client_id': client.client_id,
            'client_id_issued_at': int(fields.Datetime.now().timestamp()),
            'client_name': client.name,
            'redirect_uris': redirect_uris,
            'token_endpoint_auth_method': auth_method,
            'grant_types': grant_types,
            'response_types': ['code'],
            'scope': ' '.join(sorted(client._scope_ceiling())),
        }
        if confidential:
            secret = secrets.token_urlsafe(32)
            client._set_secret(secret)
            response['client_secret'] = secret
            response['client_secret_expires_at'] = 0
        _audit('oauth/register', 'ok', client=client)
        return _json_ok(response, status=201)

    # ------------------------------------------------------------------
    # RFC 6749 — Authorization endpoint (GET) + consent
    # ------------------------------------------------------------------
    @http.route('/mcp/oauth/authorize', type='http', auth='public', methods=['GET'],
                csrf=False, save_session=False)
    def authorize(self, **kw):
        if _rate_limited('oauth_authorize', 30):
            response = _error_page('Too many requests', 'Please try again shortly.', status=429)
            response.headers['Retry-After'] = '60'
            return response
        Client = request.env['ont_mcp_oauth.client'].sudo()

        client_id = kw.get('client_id')
        client = Client.search([('client_id', '=', client_id), ('active', '=', True)], limit=1)
        if not client:
            _audit('oauth/authorize', 'denied', 'unknown_client')
            return _error_page('Invalid client', 'Unknown or inactive client.')

        redirect_uri = kw.get('redirect_uri')
        if not client._allows_redirect_uri(redirect_uri):
            _audit('oauth/authorize', 'denied', 'bad_redirect_uri', client=client)
            return _error_page('Invalid redirect URI', 'This redirect URI is not registered.')

        # From here, errors may redirect (redirect_uri is trusted).
        state = kw.get('state') or ''
        issuer = _issuer()
        if len(state) > _MAX_STATE or any(c in state for c in '\r\n'):
            return _redirect_with_params(redirect_uri, {'error': 'invalid_request', 'iss': issuer})

        if kw.get('response_type') != 'code':
            return _redirect_with_params(
                redirect_uri, {'error': 'unsupported_response_type', 'state': state, 'iss': issuer})

        challenge = kw.get('code_challenge')
        if kw.get('code_challenge_method') != 'S256' or not pkce_tools.is_valid_challenge(challenge):
            return _redirect_with_params(
                redirect_uri, {'error': 'invalid_request',
                               'error_description': 'PKCE S256 required', 'state': state, 'iss': issuer})

        requested = set((kw.get('scope') or 'mcp:read').split())
        if not requested or not requested.issubset(_VALID_SCOPES):
            return _redirect_with_params(
                redirect_uri, {'error': 'invalid_scope', 'state': state, 'iss': issuer})
        requested &= client._scope_ceiling()
        if not requested:
            return _redirect_with_params(
                redirect_uri, {'error': 'invalid_scope', 'state': state, 'iss': issuer})

        resource = kw.get('resource')
        if resource and uri_tools.canonical_resource(resource) != \
                uri_tools.canonical_resource(f'{issuer}/mcp'):
            return _redirect_with_params(
                redirect_uri, {'error': 'invalid_target', 'state': state, 'iss': issuer})

        # Require an authenticated Odoo user; else send to login and back.
        if not request.session.uid:
            login_url = '/web/login?' + urlencode({'redirect': request.httprequest.full_path})
            return werkzeug.utils.redirect(login_url, code=303)
        user = request.env.user

        # MCP is for internal users only (muk's key path is implicitly gated the
        # same way by its ACL). Deny portal/share users without telling the
        # client app what kind of account it is.
        if not user._is_internal():
            _audit('oauth/authorize', 'denied', 'non_internal_user',
                   client=client, user_id=user.id)
            return _redirect_with_params(
                redirect_uri, {'error': 'access_denied', 'state': state, 'iss': issuer})

        # Per-USER ceiling: a user outside "MCP: Allow Write" is silently
        # narrowed to read here, BEFORE the remembered-consent check — so an
        # old remembered write consent can never mint a write code for a user
        # who no longer qualifies, and the consent screen below only renders
        # the checkboxes the user may actually grant.
        requested &= user._mcp_scope_ceiling()
        if not requested:
            _audit('oauth/authorize', 'denied', 'user_scope_ceiling',
                   client=client, user_id=user.id)
            return _redirect_with_params(
                redirect_uri, {'error': 'invalid_scope', 'state': state, 'iss': issuer})

        # Skip consent if already granted.
        if request.env['ont_mcp_oauth.consent'].sudo()._covers(user, client, requested):
            code = self._issue_code(client, user, requested, challenge, redirect_uri, resource)
            _audit('oauth/authorize', 'ok', client=client, user_id=user.id)
            return _redirect_with_params(redirect_uri, {'code': code, 'state': state, 'iss': issuer})

        # Otherwise render the consent screen.
        authreq = request.env['ont_mcp_oauth.authrequest'].sudo().create({
            'client_id': client.id,
            'user_id': user.id,
            'scope': ' '.join(sorted(requested)),
            'code_challenge': challenge,
            'redirect_uri': redirect_uri,
            'resource': resource or False,
            'state': state or False,
            'expires_at': request.env['ont_mcp_oauth.authrequest']._default_expiry(),
        })
        values = {
            'client_name': client.name,
            'req_token': authreq.req_token,
            'redirect_uri': redirect_uri,
            'scope_read': 'mcp:read' in requested,
            'scope_write': 'mcp:write' in requested,
            'csrf_token': request.csrf_token(),
        }
        response = request.render('ont_mcp_oauth.consent', values)
        response.headers['X-Frame-Options'] = 'DENY'
        response.headers['Content-Security-Policy'] = "frame-ancestors 'none'"
        response.headers['Cache-Control'] = 'no-store'
        return response

    @http.route('/mcp/oauth/authorize', type='http', auth='user', methods=['POST'])
    def authorize_consent(self, **kw):
        user = request.env.user
        if not user._is_internal():  # defence in depth; the GET already gates
            return _error_page('Not permitted', 'MCP access requires an internal user.', status=403)
        snap = request.env['ont_mcp_oauth.authrequest'].sudo()._consume(kw.get('req'), user)
        if not snap:
            return _error_page('Request expired', 'This authorization request is invalid or expired.')

        client = snap['client_id']
        redirect_uri = snap['redirect_uri']
        state = snap['state'] or ''
        issuer = _issuer()

        if kw.get('action') != 'approve':
            _audit('oauth/authorize', 'denied', 'user_denied', client=client, user_id=user.id)
            return _redirect_with_params(
                redirect_uri, {'error': 'access_denied', 'state': state, 'iss': issuer})

        requested = set(snap['scope'].split())
        granted = set()
        if kw.get('scope_read') and 'mcp:read' in requested:
            granted.add('mcp:read')
        if kw.get('scope_write') and 'mcp:write' in requested:
            granted.add('mcp:write')
        granted &= client._scope_ceiling()
        # The pending request was already user-narrowed at the GET, but the
        # snapshot could predate a demotion (or the POST could be crafted) —
        # re-apply the user ceiling so scope_write=on cannot be smuggled in.
        granted &= user._mcp_scope_ceiling()
        if not granted:
            _audit('oauth/authorize', 'denied', 'empty_grant', client=client, user_id=user.id)
            return _redirect_with_params(
                redirect_uri, {'error': 'access_denied', 'state': state, 'iss': issuer})

        request.env['ont_mcp_oauth.consent'].sudo()._remember(user, client, granted)
        code = self._issue_code(
            client, user, granted, snap['code_challenge'], redirect_uri, snap['resource'])
        _audit('oauth/authorize', 'ok', client=client, user_id=user.id)
        return _redirect_with_params(redirect_uri, {'code': code, 'state': state, 'iss': issuer})

    def _issue_code(self, client, user, scopes, challenge, redirect_uri, resource):
        Code = request.env['ont_mcp_oauth.code'].sudo()
        raw = secrets.token_urlsafe(48)
        Code.create({
            'code_hash': Code._hash(raw),
            'client_id': client.id,
            'user_id': user.id,
            'scope': ' '.join(sorted(scopes)),
            'code_challenge': challenge,
            'redirect_uri': redirect_uri,
            'resource': resource or False,
            'expires_at': Code._default_expiry(),
        })
        return raw

    # ------------------------------------------------------------------
    # RFC 6749 — Token endpoint
    # ------------------------------------------------------------------
    @http.route('/mcp/oauth/token', type='http', auth='public', methods=['POST'],
                csrf=False, save_session=False, cors='*')
    def token(self, **kw):
        if _rate_limited('oauth_token', 30):
            return _rate_limited_response('oauth/token')
        form = _form()  # body-only; ignore any query-string params
        creds = _client_credentials(form)
        grant_type = form.get('grant_type')
        if not grant_type:
            return _json_error('invalid_request', 'grant_type is required.')
        if grant_type == 'authorization_code':
            return self._token_authorization_code(form, creds)
        if grant_type == 'refresh_token':
            return self._token_refresh(form, creds)
        return _json_error('unsupported_grant_type', 'Unsupported grant_type.')

    def _resolve_client(self, creds):
        """Look up the client from resolved credentials and authenticate it.

        :return: (client, error_response) — exactly one is set. Distinguishes
            'unknown client' (invalid_client 401, so a misconfigured connector
            surfaces as such) from grant problems (invalid_grant), which
            clients handle by silently discarding tokens and looping through
            re-authorization.
        """
        client_id, secret, via_header = creds
        client = request.env['ont_mcp_oauth.client'].sudo().search(
            [('client_id', '=', client_id), ('active', '=', True)], limit=1)
        if not client:
            _audit('oauth/token', 'denied', 'unknown_client')
            return None, _invalid_client(via_header)
        if client.client_type == 'confidential' and not client._verify_secret(secret):
            _audit('oauth/token', 'denied', 'client_auth_failed', client=client)
            return None, _invalid_client(via_header)
        return client, None  # public clients rely on PKCE

    def _token_authorization_code(self, kw, creds):
        Code = request.env['ont_mcp_oauth.code'].sudo()
        Token = request.env['ont_mcp_oauth.token'].sudo()

        client, error = self._resolve_client(creds)
        if error:
            return error
        raw_code = kw.get('code')
        if not raw_code:
            return _json_error('invalid_request', 'code is required.')

        # Detect replay of an already-consumed code BEFORE consuming.
        existing = Code._find_by_hash(raw_code)
        code = Code._consume(raw_code)
        if not code:
            if existing and existing.consumed_at:
                # Code replay: revoke exactly the token families minted from
                # this code (precise, not a user+client guess).
                Token._revoke_families_of_code(existing.code_hash)
                _audit('oauth/token', 'denied', 'code_replay', client=client, user_id=existing.user_id.id)
            else:
                _audit('oauth/token', 'denied', 'code_invalid', client=client)
            return _json_error('invalid_grant', 'Authorization code is invalid or expired.')

        if code.client_id.id != client.id:
            # Client tried to redeem another client's code: burn any families
            # already derived from it, deny.
            Token._revoke_families_of_code(code.code_hash)
            _audit('oauth/token', 'denied', 'code_client_mismatch', client=client)
            return _json_error('invalid_grant', 'Code was issued to a different client.')
        if code.redirect_uri != kw.get('redirect_uri'):
            _audit('oauth/token', 'denied', 'redirect_mismatch', client=client)
            return _json_error('invalid_grant', 'redirect_uri mismatch.')
        resource = kw.get('resource')
        if resource and (code.resource or '') and uri_tools.canonical_resource(resource) != \
                uri_tools.canonical_resource(code.resource):
            return _json_error('invalid_target', 'resource mismatch.')

        verifier = kw.get('code_verifier')
        if not pkce_tools.verify(verifier, code.code_challenge):
            _audit('oauth/token', 'denied', 'pkce_failed', client=client, user_id=code.user_id.id)
            return _json_error('invalid_grant', 'PKCE verification failed.')

        scopes = set(code.scope.split())
        family_id = uuid.uuid4().hex
        return self._issue_tokens(
            client, code.user_id, scopes, code.resource, family_id, code_hash=code.code_hash)

    def _token_refresh(self, kw, creds):
        Token = request.env['ont_mcp_oauth.token'].sudo()

        client, error = self._resolve_client(creds)
        if error:
            return error
        raw = kw.get('refresh_token')
        if not raw:
            return _json_error('invalid_request', 'refresh_token is required.')
        if not client.allow_refresh:
            _audit('oauth/token', 'denied', 'refresh_grant_not_allowed', client=client)
            return _json_error('unauthorized_client', 'This client may not use refresh_token.')

        existing = Token._find_refresh(raw)
        row = Token._consume_refresh(raw)
        if not row:
            # Distinguish benign in-grace retry from a genuine replay attack.
            if existing and existing.consumed_at:
                grace = _int_param('ont_mcp_oauth.refresh_grace_seconds', '30')
                age = (fields.Datetime.now()
                       - fields.Datetime.to_datetime(existing.consumed_at)).total_seconds()
                if (age <= grace and not existing.revoked
                        and existing.client_id.id == client.id
                        and existing.expires_at > fields.Datetime.now()):
                    # Lost-response retry (Auth0/Okta "reuse interval"): the
                    # client consumed this token moments ago but never received
                    # the successor — a mobile timeout, a dropped connection, a
                    # double-fire. Returning invalid_grant here would strand a
                    # healthy connector into forced re-authorization, so mint a
                    # fresh pair in the same family instead. The successor the
                    # client never saw stays valid (a concurrent double-fire
                    # may be using it); family revocation still covers both
                    # chains, and a replay AFTER the window still burns the
                    # family below.
                    _audit('oauth/token', 'ok', reason='refresh_grace_reissue',
                           client=client, user_id=existing.user_id.id)
                    return self._issue_tokens(
                        client, existing.user_id, set((existing.scope or '').split()),
                        existing.resource, existing.family_id,
                        code_hash=existing.code_hash)
                if age > grace:
                    existing._revoke_family()
                    _audit('oauth/token', 'denied', 'refresh_replay', client=client,
                           user_id=existing.user_id.id)
            return _json_error('invalid_grant', 'Refresh token is invalid, expired, or already used.')
        if row.client_id.id != client.id or row.revoked:
            row._revoke_family()
            _audit('oauth/token', 'denied', 'refresh_client_mismatch', client=client)
            return _json_error('invalid_grant', 'Refresh token does not match client.')

        scopes = set((row.scope or '').split())
        requested = kw.get('scope')
        if requested:
            narrowed = set(requested.split())
            if not narrowed.issubset(scopes):
                return _json_error('invalid_scope', 'Refresh cannot widen scope.')
            scopes = narrowed
        return self._issue_tokens(
            client, row.user_id, scopes, row.resource, row.family_id, predecessor=row)

    def _issue_tokens(self, client, user, scopes, resource, family_id,
                      predecessor=None, code_hash=None):
        # Single choke point for both grants: never mint for an account that
        # has been deactivated or demoted out of internal access since the
        # grant was made. Without this, a refresh token kept working for its
        # full 30-day life after the employee was offboarded — the access
        # tokens it produced were refused at /mcp, but the token endpoint
        # itself happily kept issuing them, and would have gone live again the
        # moment the account was re-enabled.
        if not user.active or not user._is_internal():
            if predecessor is not None:
                predecessor._revoke_family()
            _audit('oauth/token', 'denied', 'user_not_eligible',
                   client=client, user_id=user.id)
            return _json_error(
                'invalid_grant', 'The account is no longer eligible for MCP access.')

        # Per-USER scope ceiling, enforced at the mint itself so it covers
        # every grant path at once (code exchange, refresh, grace reissue):
        # losing "MCP: Allow Write" narrows the very next token to read-only,
        # even though the code/refresh row still carries the old write scope.
        scopes = set(scopes) & user._mcp_scope_ceiling()
        if not scopes:
            _audit('oauth/token', 'denied', 'user_scope_ceiling',
                   client=client, user_id=user.id)
            return _json_error(
                'invalid_grant', 'The account is no longer eligible for the granted scope.')

        Token = request.env['ont_mcp_oauth.token'].sudo()
        SigningKey = request.env['ont_mcp_oauth.signing.key'].sudo()
        issuer = _issuer()
        now = int(fields.Datetime.now().timestamp())
        access_ttl = _int_param('ont_mcp_oauth.access_token_ttl', '900')
        refresh_days = _int_param('ont_mcp_oauth.refresh_token_ttl_days', '30')
        # On rotation, carry the originating code linkage forward so the whole
        # chain stays revocable via the code that started it.
        if code_hash is None and predecessor is not None:
            code_hash = predecessor.code_hash

        key = SigningKey._get_active_key()
        jti = uuid.uuid4().hex
        scope_str = ' '.join(sorted(scopes))
        claims = {
            'iss': issuer,
            'sub': str(user.id),
            'aud': f'{issuer}/mcp',
            'client_id': client.client_id,
            'scope': scope_str,
            'jti': jti,
            'iat': now,
            'nbf': now,
            'exp': now + access_ttl,
        }
        access_token = jwt_tools.encode(claims, key.private_pem, key.kid)
        Token.create({
            'kind': 'access', 'token_hash': jti, 'family_id': family_id,
            'client_id': client.id, 'user_id': user.id, 'scope': scope_str,
            'resource': resource or False, 'code_hash': code_hash or False,
            'expires_at': fields.Datetime.add(fields.Datetime.now(), seconds=access_ttl),
        })

        body = {
            'access_token': access_token,
            'token_type': 'Bearer',
            'expires_in': access_ttl,
            'scope': scope_str,
        }
        # Only mint a refresh token for clients allowed to redeem one.
        if client.allow_refresh:
            raw_refresh = secrets.token_urlsafe(48)
            refresh_row = Token.create({
                'kind': 'refresh', 'token_hash': Token._hash(raw_refresh), 'family_id': family_id,
                'client_id': client.id, 'user_id': user.id, 'scope': scope_str,
                'resource': resource or False, 'code_hash': code_hash or False,
                'expires_at': fields.Datetime.add(fields.Datetime.now(), days=refresh_days),
            })
            if predecessor is not None:
                predecessor.successor_id = refresh_row.id
            body['refresh_token'] = raw_refresh

        _audit('oauth/token', 'ok', client=client, user_id=user.id)
        return _json_ok(body)

    # ------------------------------------------------------------------
    # RFC 7009 — Revocation (always 200)
    # ------------------------------------------------------------------
    @http.route('/mcp/oauth/revoke', type='http', auth='public', methods=['POST'],
                csrf=False, save_session=False, cors='*')
    def revoke(self, **kw):
        if _rate_limited('oauth_revoke', 60):
            return request.make_response('', status=200)
        Token = request.env['ont_mcp_oauth.token'].sudo()
        raw = _form().get('token')
        if raw:
            refresh = Token._find_refresh(raw)
            if refresh:
                refresh._revoke_family()
                _audit('oauth/revoke', 'ok', reason='refresh_revoked')
            elif raw.count('.') == 2:
                try:
                    header = jwt_tools.peek_header(raw)
                    key = request.env['ont_mcp_oauth.signing.key'].sudo()._find_by_kid(header.get('kid'))
                    if key:
                        issuer = _issuer()
                        payload = jwt_tools.decode(
                            raw, key.public_pem, issuer=issuer, audience=f'{issuer}/mcp', leeway=86400)
                        if payload.get('jti'):
                            Token._revoke_jti(payload['jti'])
                            _audit('oauth/revoke', 'ok', reason='access_revoked')
                except jwt_tools.JWTError:
                    pass
        return request.make_response(
            '', headers=[('Cache-Control', 'no-store')], status=200,
        )
