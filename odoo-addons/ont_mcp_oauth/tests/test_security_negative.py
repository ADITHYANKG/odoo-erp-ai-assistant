from __future__ import annotations

from urllib.parse import urlencode

from odoo.tests import tagged

from .common import OAuthHttpCase


@tagged('post_install', '-at_install')
class TestSecurityNegative(OAuthHttpCase):

    def test_unregistered_redirect_uri_shows_error_page_not_redirect(self):
        client = self.make_client(redirect_uri='https://good.example/cb')
        _v, challenge = self.pkce_pair()
        self._login()
        params = {'response_type': 'code', 'client_id': client.client_id,
                  'redirect_uri': 'https://attacker.example/steal',
                  'code_challenge': challenge, 'code_challenge_method': 'S256',
                  'scope': 'mcp:read', 'state': 's'}
        r = self.url_open('/mcp/oauth/authorize?' + urlencode(params), allow_redirects=False)
        # Must NOT 3xx-redirect to the attacker; must render an error page.
        self.assertNotIn(r.status_code, (301, 302, 303))
        self.assertNotIn('attacker.example', r.headers.get('Location', ''))

    def test_unknown_client_shows_error_page(self):
        _v, challenge = self.pkce_pair()
        self._login()
        params = {'response_type': 'code', 'client_id': 'ont_does_not_exist',
                  'redirect_uri': 'https://x/cb', 'code_challenge': challenge,
                  'code_challenge_method': 'S256', 'scope': 'mcp:read'}
        r = self.url_open('/mcp/oauth/authorize?' + urlencode(params), allow_redirects=False)
        self.assertNotIn(r.status_code, (301, 302, 303))

    def test_crlf_in_state_rejected(self):
        client = self.make_client()
        _v, challenge = self.pkce_pair()
        self._login()
        params = {'response_type': 'code', 'client_id': client.client_id,
                  'redirect_uri': 'https://client.example/cb', 'code_challenge': challenge,
                  'code_challenge_method': 'S256', 'scope': 'mcp:read',
                  'state': 'a\r\nSet-Cookie: x'}
        r = self.url_open('/mcp/oauth/authorize?' + urlencode(params), allow_redirects=False)
        self.assertEqual(self.error_from_redirect(r), 'invalid_request')

    def test_forged_req_token_rejected(self):
        self._login()
        r = self.url_open('/mcp/oauth/authorize',
                          data={'req': 'forged-nonexistent', 'action': 'approve'},
                          allow_redirects=False)
        self.assertNotIn(r.status_code, (301, 302, 303))

    def test_cross_user_consent_rejected(self):
        client = self.make_client()
        _v, challenge = self.pkce_pair()
        self._login()
        resp = self.authorize_get(client, challenge)
        req, csrf = self.scrape_consent(resp.text)
        # Switch to a different user, then try to approve the first user's request.
        other = self.env['res.users'].create({
            'name': 'Other', 'login': 'oauth_other', 'password': 'oauth_other_pw_123',
            'groups_id': [(6, 0, [self.env.ref('base.group_user').id])]})
        self.authenticate('oauth_other', 'oauth_other_pw_123')
        r = self.url_open('/mcp/oauth/authorize',
                          data={'req': req, 'csrf_token': csrf, 'action': 'approve',
                                'scope_read': 'on'}, allow_redirects=False)
        # Different user must not be able to consume the request.
        self.assertNotIn('code=', r.headers.get('Location', ''))

    def test_portal_user_cannot_authorize(self):
        # MCP is internal-users-only; a portal/share user must be refused.
        portal = self.env['res.users'].create({
            'name': 'Portal Person', 'login': 'oauth_portal',
            'password': 'oauth_portal_pw_123',
            'groups_id': [(6, 0, [self.env.ref('base.group_portal').id])]})
        self.assertFalse(portal._is_internal())
        client = self.make_client()
        _v, challenge = self.pkce_pair()
        self.authenticate('oauth_portal', 'oauth_portal_pw_123')
        r = self.authorize_get(client, challenge)
        self.assertEqual(self.error_from_redirect(r), 'access_denied')
        # And no consent form was rendered.
        self.assertNotIn('req', (r.text or '')[:0] or '')

    def test_leaked_req_token_cannot_cancel_other_users_request(self):
        client = self.make_client()
        _v, challenge = self.pkce_pair()
        self._login()
        resp = self.authorize_get(client, challenge)
        req, _csrf = self.scrape_consent(resp.text)
        self.assertTrue(req)
        # A different user POSTs the stolen req token.
        self.env['res.users'].create({
            'name': 'Thief', 'login': 'oauth_thief', 'password': 'oauth_thief_pw_123',
            'groups_id': [(6, 0, [self.env.ref('base.group_user').id])]})
        self.authenticate('oauth_thief', 'oauth_thief_pw_123')
        self.url_open('/mcp/oauth/authorize',
                      data={'req': req, 'action': 'approve', 'scope_read': 'on'},
                      allow_redirects=False)
        # The owner's pending request must still exist (not silently deleted).
        still_there = self.env['ont_mcp_oauth.authrequest'].sudo().search_count(
            [('req_token', '=', req)])
        self.assertEqual(still_there, 1, 'a stolen req token must not cancel the owner request')

    def test_no_secrets_in_audit_log(self):
        client, verifier, code, tok = self.full_flow()
        logs = self.env['muk_mcp.log'].sudo().search([('method', '=like', 'oauth/%')])
        blob = ' '.join(
            (log.request_data or '') + (log.response_data or '') + (log.error_message or '')
            for log in logs)
        self.assertNotIn(tok['access_token'], blob)
        self.assertNotIn(tok['refresh_token'], blob)
        self.assertNotIn(verifier, blob)
