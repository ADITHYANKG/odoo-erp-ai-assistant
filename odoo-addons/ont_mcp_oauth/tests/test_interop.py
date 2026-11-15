"""Interoperability regression tests: real-world client behaviours.

Pins the deployment-readiness fixes: HTTP Basic client auth (RFC 6749 §2.3.1
MUST), CORS for browser-based clients (MCP Inspector), Retry-After on 429,
RFC-correct token error codes, OpenID-Connect-style discovery aliases, the
in-grace refresh reissue for lossy networks, and resource canonicalization.
"""
from __future__ import annotations

import base64

from odoo.tests import tagged

from odoo.addons.muk_mcp.tools.rate_limit import rate_limiter
from odoo.addons.ont_mcp_oauth.tools import uri as uri_tools

from .common import OAuthHttpCase


@tagged('post_install', '-at_install')
class TestClientSecretBasic(OAuthHttpCase):

    def _confidential_flow(self):
        """Authorize + consent against a confidential client; return (client, secret, code, verifier)."""
        client = self.make_client(client_type='confidential')
        action = client.action_regenerate_secret()
        secret = action['params']['message'].strip().rsplit('\n', 1)[-1].strip()
        verifier, challenge = self.pkce_pair()
        self._login()
        resp = self.authorize_get(client, challenge)
        code = self.code_from_redirect(resp)
        if not code:
            req, csrf = self.scrape_consent(resp.text)
            code = self.code_from_redirect(self.consent_post(req, csrf))
        return client, secret, code, verifier

    @staticmethod
    def _basic(client_id, secret):
        return 'Basic ' + base64.b64encode(f'{client_id}:{secret}'.encode()).decode()

    def test_basic_auth_on_token_endpoint(self):
        client, secret, code, verifier = self._confidential_flow()
        r = self.url_open(
            '/mcp/oauth/token',
            data={'grant_type': 'authorization_code', 'code': code,
                  'code_verifier': verifier,
                  'redirect_uri': 'https://client.example/cb'},
            headers={'Authorization': self._basic(client.client_id, secret)},
            allow_redirects=False)
        self.assertEqual(r.status_code, 200, r.text)
        self.assertIn('access_token', r.json())

    def test_basic_auth_wrong_secret_is_invalid_client_with_challenge(self):
        client, _secret, code, verifier = self._confidential_flow()
        r = self.url_open(
            '/mcp/oauth/token',
            data={'grant_type': 'authorization_code', 'code': code,
                  'code_verifier': verifier,
                  'redirect_uri': 'https://client.example/cb'},
            headers={'Authorization': self._basic(client.client_id, 'wrong')},
            allow_redirects=False)
        self.assertEqual(r.status_code, 401)
        self.assertEqual(r.json()['error'], 'invalid_client')
        self.assertIn('Basic', r.headers.get('WWW-Authenticate', ''))

    def test_unknown_client_is_invalid_client_not_invalid_grant(self):
        # A typo'd client_id (or a GC-reaped DCR client) must surface as a
        # client problem, not a grant problem — clients react to invalid_grant
        # by silently re-running authorization in a loop.
        r = self.token_request(grant_type='authorization_code', code='x',
                               client_id='ont_no_such_client', code_verifier='v' * 43,
                               redirect_uri='https://client.example/cb')
        self.assertEqual(r.status_code, 401)
        self.assertEqual(r.json()['error'], 'invalid_client')

    def test_missing_grant_type_is_invalid_request(self):
        r = self.token_request(code='x', client_id='ont_whatever')
        self.assertEqual(r.json()['error'], 'invalid_request')


@tagged('post_install', '-at_install')
class TestCorsAndDiscovery(OAuthHttpCase):

    def test_token_preflight_succeeds(self):
        # MCP Inspector runs in the browser: without a working OPTIONS
        # preflight + ACAO header the whole connect flow dies client-side.
        r = self.opener.request(
            'OPTIONS', self.base_url() + '/mcp/oauth/token', timeout=30)
        self.assertLess(r.status_code, 300)
        self.assertTrue(r.headers.get('Access-Control-Allow-Origin'))

    def test_metadata_sends_acao(self):
        r = self.url_open('/.well-known/oauth-authorization-server')
        self.assertEqual(r.status_code, 200)
        self.assertTrue(r.headers.get('Access-Control-Allow-Origin'))

    def test_openid_configuration_alias(self):
        r = self.url_open('/.well-known/openid-configuration')
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertIn('token_endpoint', body)
        self.assertIn('subject_types_supported', body)
        self.assertIn('client_secret_basic', body['token_endpoint_auth_methods_supported'])

    def test_mcp_trailing_slash_is_served_by_the_mcp_endpoint(self):
        # muk_mcp 17 registers /mcp with a non-strict slash, so /mcp/ reaches
        # the MCP handler directly rather than 308-redirecting. An
        # unauthenticated POST therefore comes back 401 (the auth layer), which
        # proves the POST and its body reached the endpoint intact.
        r = self.url_open(
            '/mcp/', data='{"jsonrpc": "2.0", "id": 1, "method": "ping"}',
            headers={'Content-Type': 'application/json'}, allow_redirects=False,
        )
        self.assertIn(r.status_code, (200, 401))

    def test_rate_limited_token_carries_retry_after(self):
        # The test HTTP server runs in-process, so priming the shared limiter
        # bucket is equivalent to 30 real requests from 127.0.0.1.
        for _ in range(30):
            rate_limiter.check(
                (self.env.cr.dbname, 'oauth_token', '127.0.0.1'), 30, 60)
        r = self.token_request(grant_type='authorization_code', code='x',
                               client_id='ont_x')
        self.assertEqual(r.status_code, 429)
        self.assertEqual(r.headers.get('Retry-After'), '60')


@tagged('post_install', '-at_install')
class TestGraceReissue(OAuthHttpCase):

    def test_lost_response_retry_heals_within_grace(self):
        # A client that consumed its refresh token but never received the
        # response (mobile timeout) retries the SAME token. Inside the grace
        # window it must get a fresh pair — invalid_grant would strand it into
        # forced re-authorization.
        client, _v, _code, tok = self.full_flow()
        first = self.token_request(grant_type='refresh_token',
                                   refresh_token=tok['refresh_token'],
                                   client_id=client.client_id)
        self.assertEqual(first.status_code, 200)
        retry = self.token_request(grant_type='refresh_token',
                                   refresh_token=tok['refresh_token'],
                                   client_id=client.client_id)
        self.assertEqual(retry.status_code, 200, retry.text)
        healed = retry.json()
        self.assertIn('refresh_token', healed)
        # Both chains stay usable: the winner's successor was never revoked.
        cont = self.token_request(grant_type='refresh_token',
                                  refresh_token=first.json()['refresh_token'],
                                  client_id=client.client_id)
        self.assertEqual(cont.status_code, 200, cont.text)
        # And the family is NOT revoked. Scope to this test's own client:
        # HttpCase commits, so a database templated from a live instance can
        # already hold revoked tokens from real logins, which are not ours.
        live = self.Token.sudo().search(
            [('revoked', '=', True), ('client_id', '=', client.id)])
        self.assertFalse(live, 'grace reissue must not revoke anything')

    def test_replay_after_grace_still_burns_the_family(self):
        client, _v, _code, tok = self.full_flow()
        first = self.token_request(grant_type='refresh_token',
                                   refresh_token=tok['refresh_token'],
                                   client_id=client.client_id)
        self.assertEqual(first.status_code, 200)
        # Collapse the grace window, then replay: attack semantics unchanged.
        self.env['ir.config_parameter'].sudo().set_param(
            'ont_mcp_oauth.refresh_grace_seconds', '0')
        import time
        time.sleep(1)
        replay = self.token_request(grant_type='refresh_token',
                                    refresh_token=tok['refresh_token'],
                                    client_id=client.client_id)
        self.assertEqual(replay.json()['error'], 'invalid_grant')
        dead = self.token_request(grant_type='refresh_token',
                                  refresh_token=first.json()['refresh_token'],
                                  client_id=client.client_id)
        self.assertEqual(dead.json()['error'], 'invalid_grant')


@tagged('post_install', '-at_install')
class TestCanonicalResource(OAuthHttpCase):

    def test_canonicalization(self):
        canon = uri_tools.canonical_resource
        self.assertEqual(canon('https://Host:443/mcp/'), 'https://host/mcp')
        self.assertEqual(canon('https://host/mcp'), 'https://host/mcp')
        self.assertEqual(canon('http://host:80/mcp'), 'http://host/mcp')
        self.assertNotEqual(canon('https://host:8443/mcp'), canon('https://host/mcp'))
        self.assertIsNone(canon('not a uri'))
        self.assertIsNone(canon(''))

    def test_long_state_is_accepted(self):
        # state-as-encrypted-JWT patterns routinely exceed 512 bytes.
        client = self.make_client()
        _v, challenge = self.pkce_pair()
        self._login()
        resp = self.authorize_get(client, challenge, state='s' * 1500)
        # Must reach consent (or code), not bounce with invalid_request.
        self.assertNotEqual(self.error_from_redirect(resp), 'invalid_request')
