from __future__ import annotations

import re
import secrets
from urllib.parse import parse_qs, urlsplit

from odoo.addons.muk_mcp.tests.common import MCPHttpCase
from odoo.addons.muk_mcp.tools.rate_limit import rate_limiter
from odoo.addons.ont_mcp_oauth.tools import pkce as pkce_tools


class OAuthHttpCase(MCPHttpCase):
    """Base for OAuth HTTP tests: helpers to drive the full code+PKCE flow."""

    def setUp(self):
        super().setUp()
        # The per-IP limiter is a process-global sliding window, so every test
        # in the suite shares one bucket and they all arrive from 127.0.0.1.
        # Without a reset, adding tests eventually pushes earlier ones over the
        # 30/min token and authorize limits and they fail for reasons that have
        # nothing to do with what they assert.
        rate_limiter._windows.clear()

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.Client = cls.env['ont_mcp_oauth.client']
        cls.Token = cls.env['ont_mcp_oauth.token']
        # In the "MCP: Allow Write" group so the suite's write-scope flows
        # keep testing what they always tested.
        cls.oauth_user = cls.env['res.users'].create({
            'name': 'OAuth Tester',
            'login': 'oauth_tester',
            'password': 'oauth_tester_pw_123',
            'groups_id': [(6, 0, [
                cls.env.ref('base.group_user').id,
                cls.env.ref('ont_mcp_oauth.group_mcp_write').id,
            ])],
        })
        # A plain internal user OUTSIDE the write group — per-user scope
        # ceiling tests log in as this one.
        cls.reader_user = cls.env['res.users'].create({
            'name': 'OAuth Reader',
            'login': 'oauth_reader',
            'password': 'oauth_reader_pw_123',
            'groups_id': [(6, 0, [cls.env.ref('base.group_user').id])],
        })

    # -- client / pkce helpers ------------------------------------------
    def mcp_call_tool(self, name, arguments=None, session_id=None, token=None,
                      request_id=2):
        """Call one MCP tool and return its result object.

        muk_mcp 19's MCPHttpCase shipped this helper; the 17 base class only
        offers mcp_json/mcp_handshake, so it is provided here. A successful
        call returns the JSON-RPC ``result`` (carrying ``content``/``isError``);
        a JSON-RPC-level failure returns the whole response, so callers can
        test ``'error' in result``.
        """
        response = self.mcp_json(
            {'jsonrpc': '2.0', 'id': request_id, 'method': 'tools/call',
             'params': {'name': name, 'arguments': arguments or {}}},
            session_id=session_id, token=token,
        )
        return response.get('result', response)

    def make_client(self, redirect_uri='https://client.example/cb', scope='write',
                    client_type='public', name='Test Client'):
        return self.Client.create({
            'name': name,
            'redirect_uris': redirect_uri,
            'scope': scope,
            'client_type': client_type,
        })

    @staticmethod
    def pkce_pair():
        verifier = secrets.token_urlsafe(64)[:96]
        return verifier, pkce_tools.compute_challenge(verifier)

    # -- flow drivers ---------------------------------------------------
    def _login(self, login='oauth_tester', password='oauth_tester_pw_123'):
        self.authenticate(login, password)

    def authorize_get(self, client, challenge, redirect_uri='https://client.example/cb',
                      scope='mcp:read mcp:write', state='xyz', resource=None):
        params = {
            'response_type': 'code', 'client_id': client.client_id,
            'redirect_uri': redirect_uri, 'code_challenge': challenge,
            'code_challenge_method': 'S256', 'scope': scope, 'state': state,
        }
        if resource:
            params['resource'] = resource
        from urllib.parse import urlencode
        return self.url_open('/mcp/oauth/authorize?' + urlencode(params), allow_redirects=False)

    def scrape_consent(self, html):
        req = re.search(r'name="req"\s+value="([^"]+)"', html)
        csrf = re.search(r'name="csrf_token"\s+value="([^"]+)"', html)
        return (req.group(1) if req else None), (csrf.group(1) if csrf else None)

    def consent_post(self, req_token, csrf_token, approve=True,
                     scope_read=True, scope_write=True):
        data = {'req': req_token, 'csrf_token': csrf_token,
                'action': 'approve' if approve else 'deny'}
        if scope_read:
            data['scope_read'] = 'on'
        if scope_write:
            data['scope_write'] = 'on'
        return self.url_open('/mcp/oauth/authorize', data=data, allow_redirects=False)

    @staticmethod
    def code_from_redirect(response):
        loc = response.headers.get('Location', '')
        return parse_qs(urlsplit(loc).query).get('code', [None])[0]

    @staticmethod
    def error_from_redirect(response):
        loc = response.headers.get('Location', '')
        return parse_qs(urlsplit(loc).query).get('error', [None])[0]

    def token_request(self, **form):
        return self.url_open('/mcp/oauth/token', data=form, allow_redirects=False)

    def full_flow(self, client=None, scope='mcp:read mcp:write',
                  scope_read=True, scope_write=True, redirect_uri='https://client.example/cb',
                  login='oauth_tester', password='oauth_tester_pw_123'):
        """Run authorize→consent→token, returning the token JSON dict."""
        client = client or self.make_client(redirect_uri=redirect_uri)
        verifier, challenge = self.pkce_pair()
        self._login(login, password)
        resp = self.authorize_get(client, challenge, redirect_uri=redirect_uri, scope=scope)
        code = self.code_from_redirect(resp)  # consent-skip path returns code directly
        if not code:
            req, csrf = self.scrape_consent(resp.text)
            self.assertTrue(req, 'consent form should carry a req token')
            redir = self.consent_post(req, csrf, scope_read=scope_read, scope_write=scope_write)
            code = self.code_from_redirect(redir)
        self.assertTrue(code, 'authorize should return a code')
        tok = self.token_request(
            grant_type='authorization_code', code=code, client_id=client.client_id,
            code_verifier=verifier, redirect_uri=redirect_uri)
        self.assertEqual(tok.status_code, 200, tok.text)
        return client, verifier, code, tok.json()
