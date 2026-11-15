from __future__ import annotations

from odoo.tests import tagged

from .common import OAuthHttpCase


@tagged('post_install', '-at_install')
class TestAuthorizationCodeFlow(OAuthHttpCase):

    def test_happy_path(self):
        _client, _v, _code, tok = self.full_flow()
        self.assertEqual(tok['token_type'], 'Bearer')
        self.assertIn('access_token', tok)
        self.assertIn('refresh_token', tok)
        self.assertEqual(tok['access_token'].count('.'), 2)
        self.assertEqual(set(tok['scope'].split()), {'mcp:read', 'mcp:write'})

    def test_wrong_verifier_rejected(self):
        client = self.make_client()
        _v, challenge = self.pkce_pair()
        self._login()
        resp = self.authorize_get(client, challenge)
        req, csrf = self.scrape_consent(resp.text)
        redir = self.consent_post(req, csrf)
        code = self.code_from_redirect(redir)
        bad = self.token_request(
            grant_type='authorization_code', code=code, client_id=client.client_id,
            code_verifier='x' * 64, redirect_uri='https://client.example/cb')
        self.assertEqual(bad.status_code, 400)
        self.assertEqual(bad.json()['error'], 'invalid_grant')

    def test_plain_pkce_method_rejected(self):
        client = self.make_client()
        _v, challenge = self.pkce_pair()
        self._login()
        from urllib.parse import urlencode
        params = {'response_type': 'code', 'client_id': client.client_id,
                  'redirect_uri': 'https://client.example/cb', 'code_challenge': challenge,
                  'code_challenge_method': 'plain', 'scope': 'mcp:read', 'state': 's'}
        resp = self.url_open('/mcp/oauth/authorize?' + urlencode(params), allow_redirects=False)
        self.assertEqual(self.error_from_redirect(resp), 'invalid_request')

    def test_reused_code_revokes_tokens(self):
        client, verifier, code, tok = self.full_flow()
        # Replay the same code.
        replay = self.token_request(
            grant_type='authorization_code', code=code, client_id=client.client_id,
            code_verifier=verifier, redirect_uri='https://client.example/cb')
        self.assertEqual(replay.json()['error'], 'invalid_grant')
        # The originally-issued family must now be revoked.
        access = self.Token.sudo().search([('kind', '=', 'access'), ('client_id', '=', client.id)])
        self.assertTrue(access)
        self.assertTrue(all(a.revoked for a in access), 'code replay must revoke the family')

    def test_redirect_uri_mismatch_at_token(self):
        client = self.make_client()
        verifier, challenge = self.pkce_pair()
        self._login()
        resp = self.authorize_get(client, challenge)
        req, csrf = self.scrape_consent(resp.text)
        code = self.code_from_redirect(self.consent_post(req, csrf))
        bad = self.token_request(
            grant_type='authorization_code', code=code, client_id=client.client_id,
            code_verifier=verifier, redirect_uri='https://client.example/OTHER')
        self.assertEqual(bad.json()['error'], 'invalid_grant')

    def test_state_and_iss_passthrough(self):
        client = self.make_client()
        _v, challenge = self.pkce_pair()
        self._login()
        resp = self.authorize_get(client, challenge, state='mystate123')
        req, csrf = self.scrape_consent(resp.text)
        redir = self.consent_post(req, csrf)
        from urllib.parse import parse_qs, urlsplit
        q = parse_qs(urlsplit(redir.headers['Location']).query)
        self.assertEqual(q['state'][0], 'mystate123')
        self.assertIn('iss', q)

    def test_deny(self):
        client = self.make_client()
        _v, challenge = self.pkce_pair()
        self._login()
        resp = self.authorize_get(client, challenge)
        req, csrf = self.scrape_consent(resp.text)
        redir = self.consent_post(req, csrf, approve=False)
        self.assertEqual(self.error_from_redirect(redir), 'access_denied')

    def test_scope_narrowing(self):
        # Uncheck write on consent → read-only token.
        _client, _v, _code, tok = self.full_flow(scope_write=False)
        self.assertEqual(set(tok['scope'].split()), {'mcp:read'})

    def test_code_replay_revokes_only_its_own_family(self):
        # Two grants for the same user+client → two independent token families.
        client = self.make_client()
        _c, _v1, code1, tok1 = self.full_flow(client=client)
        _c, _v2, _code2, tok2 = self.full_flow(client=client)  # 2nd skips consent
        jti1 = self._jti(tok1['access_token'])
        jti2 = self._jti(tok2['access_token'])
        # Replay the FIRST code.
        replay = self.token_request(
            grant_type='authorization_code', code=code1, client_id=client.client_id,
            code_verifier='x' * 64, redirect_uri='https://client.example/cb')
        self.assertEqual(replay.json()['error'], 'invalid_grant')
        # Family 1 must be dead; family 2 must survive.
        row1 = self.Token.sudo().search([('kind', '=', 'access'), ('token_hash', '=', jti1)])
        row2 = self.Token.sudo().search([('kind', '=', 'access'), ('token_hash', '=', jti2)])
        self.assertTrue(row1.revoked, 'replayed code family must be revoked')
        self.assertFalse(row2.revoked, 'unrelated family must NOT be revoked')

    def test_token_endpoint_ignores_query_string(self):
        # Secrets in the URL must be ignored (RFC 6749 §3.2 requires the body).
        client = self.make_client()
        verifier, challenge = self.pkce_pair()
        self._login()
        resp = self.authorize_get(client, challenge)
        req, csrf = self.scrape_consent(resp.text)
        code = self.code_from_redirect(self.consent_post(req, csrf))
        from urllib.parse import urlencode
        # Everything supplied ONLY in the query string, empty body.
        qs = urlencode({'grant_type': 'authorization_code', 'code': code,
                        'client_id': client.client_id, 'code_verifier': verifier,
                        'redirect_uri': 'https://client.example/cb'})
        # Non-empty body forces a real POST; the body has no grant_type, so the
        # query-string grant_type must be ignored.
        r = self.url_open('/mcp/oauth/token?' + qs, data={'ignored': '1'}, allow_redirects=False)
        self.assertEqual(r.status_code, 400)
        # A MISSING grant_type is invalid_request per RFC 6749 §5.2 — and it
        # being "missing" here is exactly the point: the query string carried
        # one and must be ignored.
        self.assertEqual(r.json()['error'], 'invalid_request')

    @staticmethod
    def _jti(access_token):
        import json
        import base64
        p = access_token.split('.')[1]
        p += '=' * (-len(p) % 4)
        return json.loads(base64.urlsafe_b64decode(p))['jti']

    def test_consent_skip_on_second_authorize(self):
        client, _v, _code, _tok = self.full_flow(client=self.make_client())
        # Second authorize with same-or-narrower scope: no consent form, direct code.
        _v2, challenge2 = self.pkce_pair()
        resp = self.authorize_get(client, challenge2, scope='mcp:read')
        self.assertIn(resp.status_code, (303, 302))
        self.assertTrue(self.code_from_redirect(resp), 'should skip consent and return a code')
