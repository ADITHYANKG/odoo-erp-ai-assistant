from __future__ import annotations

from odoo import api
from odoo.tests import tagged

from odoo.addons.muk_mcp.core.tool import invalidate_registry_cache, mcp_tool
from odoo.addons.ont_mcp_oauth.tools import jwt as jwt_tools

from .common import OAuthHttpCase


@api.model
@mcp_tool(name='ont_probe_read', description='probe read', category='read')
def _probe_read(self):
    return {'ok': True}


@api.model
@mcp_tool(name='ont_probe_write', description='probe write', category='write')
def _probe_write(self):
    return {'ok': True}


@tagged('post_install', '-at_install')
class TestJwtAuth(OAuthHttpCase):

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.mixin_cls = type(cls.env['muk_mcp.mixin'])
        cls.mixin_cls._probe_read = _probe_read
        cls.mixin_cls._probe_write = _probe_write
        invalidate_registry_cache(cls.env)

    @classmethod
    def tearDownClass(cls):
        for attr in ('_probe_read', '_probe_write'):
            if hasattr(cls.mixin_cls, attr):
                delattr(cls.mixin_cls, attr)
        invalidate_registry_cache(cls.env)
        super().tearDownClass()

    def _mint(self, **kw):
        _client, _v, _code, tok = self.full_flow(**kw)
        return tok['access_token']

    def test_valid_jwt_calls_read_tool(self):
        jwt_token = self._mint()
        sid = self.mcp_handshake(token=jwt_token)
        result = self.mcp_call_tool('ont_probe_read', {}, session_id=sid, token=jwt_token)
        self.assertNotIn('error', result, result)

    def test_read_scope_blocked_from_write_tool(self):
        jwt_token = self._mint(scope_write=False)  # read-only token
        sid = self.mcp_handshake(token=jwt_token)
        result = self.mcp_call_tool('ont_probe_write', {}, session_id=sid, token=jwt_token)
        # muk surfaces MCPScopeDenied as an isError tool result.
        self.assertTrue(
            result.get('isError') or 'error' in result
            or 'read-only' in str(result).lower(),
            result,
        )

    def test_session_is_not_usable_by_another_user(self):
        # Per-user isolation is a product requirement: each executive gets
        # their own chat instance and must not be able to drive someone
        # else's. muk binds muk_mcp.session to user_id and re-adds that
        # predicate on every lookup; this pins that behaviour.
        a_token = self._mint()
        sid = self.mcp_handshake(token=a_token)
        self.assertTrue(sid)
        # A second user with their own valid token.
        self.env['res.users'].create({
            'name': 'Other Exec', 'login': 'oauth_other_exec',
            'password': 'oauth_other_exec_pw_123',
            'groups_id': [(6, 0, [self.env.ref('base.group_user').id])]})
        client = self.make_client()
        verifier, challenge = self.pkce_pair()
        self._login('oauth_other_exec', 'oauth_other_exec_pw_123')
        resp = self.authorize_get(client, challenge)
        code = self.code_from_redirect(resp)
        if not code:
            req, csrf = self.scrape_consent(resp.text)
            code = self.code_from_redirect(self.consent_post(req, csrf))
        b_token = self.token_request(
            grant_type='authorization_code', code=code, client_id=client.client_id,
            code_verifier=verifier,
            redirect_uri='https://client.example/cb').json()['access_token']
        # B authenticates fine on their own, but must not inherit A's session.
        # Use a session-bound method: ping is answered without one.
        r = self.mcp_post(
            {'jsonrpc': '2.0', 'id': 9, 'method': 'tools/list'},
            token=b_token, session_id=sid)
        self.assertNotEqual(
            r.status_code, 200, 'another user must not drive this session')
        # A's session is untouched and still belongs to A.
        session = self.env['muk_mcp.session'].sudo().search([('session_id', '=', sid)])
        self.assertEqual(session.user_id, self.oauth_user)

    def test_token_survives_key_rotation(self):
        # End-to-end proof for the rotation regression: because `active` is
        # Odoo's magic archive field, a bare search() hid retired keys, so
        # rotating instantly 401'd every token already in the wild with
        # unknown_kid. Rotation must be a graceful hand-over.
        jwt_token = self._mint()
        self.env['ont_mcp_oauth.signing.key'].sudo()._rotate()
        self.env.flush_all()
        r = self.mcp_post(self.mcp_ping(), token=jwt_token)
        self.assertEqual(
            r.status_code, 200,
            'a token signed before rotation must still authenticate')

    def test_revoked_key_stops_its_tokens(self):
        # The flip side: revoking a retired key (leak response) must actually
        # invalidate the tokens it signed.
        jwt_token = self._mint()
        Key = self.env['ont_mcp_oauth.signing.key'].sudo()
        Key._rotate()
        Key.with_context(active_test=False).search(
            [('active', '=', False)]).action_revoke_now()
        self.env.flush_all()
        r = self.mcp_post(self.mcp_ping(), token=jwt_token)
        self.assertEqual(r.status_code, 401)

    def test_tampered_jwt_rejected(self):
        jwt_token = self._mint()
        h, p, s = jwt_token.split('.')
        forged = f'{h}.{p}.{s[:-2]}xx'
        r = self.mcp_post(self.mcp_ping(), token=forged)
        self.assertEqual(r.status_code, 401)

    def test_wrong_aud_rejected(self):
        # Sign a token with a valid key but wrong audience.
        key = self.env['ont_mcp_oauth.signing.key'].sudo()._get_active_key()
        base = self.env['ir.config_parameter'].sudo().get_param('web.base.url').rstrip('/')
        import time
        now = int(time.time())
        claims = {'iss': base, 'sub': str(self.oauth_user.id), 'aud': 'https://evil/mcp',
                  'client_id': 'x', 'scope': 'mcp:read', 'jti': 'aud-test',
                  'iat': now, 'nbf': now, 'exp': now + 900}
        bad = jwt_tools.encode(claims, key.private_pem, key.kid)
        r = self.mcp_post(self.mcp_ping(), token=bad)
        self.assertEqual(r.status_code, 401)

    def test_revoked_jti_rejected(self):
        jwt_token = self._mint()
        payload = jwt_tools.decode(
            jwt_token, self.env['ont_mcp_oauth.signing.key'].sudo()._get_active_key().public_pem,
            issuer=self.env['ir.config_parameter'].sudo().get_param('web.base.url').rstrip('/'),
            audience=self.env['ir.config_parameter'].sudo().get_param('web.base.url').rstrip('/') + '/mcp',
        )
        self.env['ont_mcp_oauth.token'].sudo()._revoke_jti(payload['jti'])
        r = self.mcp_post(self.mcp_ping(), token=jwt_token)
        self.assertEqual(r.status_code, 401)

    def test_401_has_www_authenticate(self):
        r = self.mcp_post(self.mcp_ping(), token='aa.bb.cc')
        self.assertEqual(r.status_code, 401)
        self.assertIn('resource_metadata', r.headers.get('WWW-Authenticate', ''))

    def test_empty_scope_claim_is_read_only(self):
        # Fail-closed: a token with no/garbled scope must NOT get write access.
        key = self.env['ont_mcp_oauth.signing.key'].sudo()._get_active_key()
        base = self.env['ir.config_parameter'].sudo().get_param('web.base.url').rstrip('/')
        client = self.make_client()
        import time
        now = int(time.time())
        jti = 'emptyscope-jti'
        claims = {'iss': base, 'sub': str(self.oauth_user.id), 'aud': f'{base}/mcp',
                  'client_id': client.client_id, 'scope': '', 'jti': jti,
                  'iat': now, 'nbf': now, 'exp': now + 900}
        token = jwt_tools.encode(claims, key.private_pem, key.kid)
        # Register the jti so the registry check passes.
        self.env['ont_mcp_oauth.token'].sudo().create({
            'kind': 'access', 'token_hash': jti, 'family_id': 'fam-empty',
            'client_id': client.id, 'user_id': self.oauth_user.id, 'scope': '',
            'expires_at': self.env['ont_mcp_oauth.code']._default_expiry()})
        sid = self.mcp_handshake(token=token)
        result = self.mcp_call_tool('ont_probe_write', {}, session_id=sid, token=token)
        self.assertTrue(
            result.get('isError') or 'error' in result or 'read-only' in str(result).lower(),
            f'empty scope must be treated as read-only: {result}')

    def test_opaque_muk_key_still_works(self):
        # Regression: the muk bearer-key path is untouched.
        sid = self.mcp_handshake(token=self.mcp_token)
        result = self.mcp_call_tool('ont_probe_read', {}, session_id=sid, token=self.mcp_token)
        self.assertNotIn('error', result, result)
