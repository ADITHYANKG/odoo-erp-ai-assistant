"""Per-user write-scope ceiling: only "MCP: Allow Write" members may ever be
granted mcp:write, no matter what the client requests or its own ceiling says.
"""
from __future__ import annotations

from odoo.tests import tagged

from .common import OAuthHttpCase


@tagged('post_install', '-at_install')
class TestUserScopeCeiling(OAuthHttpCase):

    def _drop_write_group(self, user):
        group = self.env.ref('ont_mcp_oauth.group_mcp_write')
        user.write({'groups_id': [(3, group.id)]})
        self.addCleanup(lambda: user.write({'groups_id': [(4, group.id)]}))

    # -- unit level -----------------------------------------------------
    def test_scope_ceiling_follows_group_membership(self):
        self.assertEqual(
            self.oauth_user._mcp_scope_ceiling(), {'mcp:read', 'mcp:write'})
        self.assertEqual(self.reader_user._mcp_scope_ceiling(), {'mcp:read'})

    def test_administrators_inherit_the_group(self):
        # base.group_system implies the write group: admins qualify without
        # anyone ticking a box.
        admin = self.env.ref('base.user_admin')
        self.assertTrue(admin.has_group('ont_mcp_oauth.group_mcp_write'))
        self.assertEqual(admin._mcp_scope_ceiling(), {'mcp:read', 'mcp:write'})

    # -- authorize + consent -------------------------------------------
    def test_non_member_sees_no_write_checkbox_and_gets_read_token(self):
        client = self.make_client()
        verifier, challenge = self.pkce_pair()
        self._login('oauth_reader', 'oauth_reader_pw_123')
        resp = self.authorize_get(client, challenge)  # requests read+write

        self.assertNotIn('scope_write', resp.text)  # checkbox never rendered
        self.assertIn('scope_read', resp.text)

        req, csrf = self.scrape_consent(resp.text)
        # Crafted POST smuggling scope_write=on must still yield read-only.
        redir = self.consent_post(req, csrf, scope_read=True, scope_write=True)
        code = self.code_from_redirect(redir)
        self.assertTrue(code)
        tok = self.token_request(
            grant_type='authorization_code', code=code, client_id=client.client_id,
            code_verifier=verifier, redirect_uri='https://client.example/cb')
        self.assertEqual(tok.status_code, 200, tok.text)
        self.assertEqual(set(tok.json()['scope'].split()), {'mcp:read'})

    def test_member_still_gets_write(self):
        _client, _v, _code, tok = self.full_flow()  # oauth_tester, in group
        self.assertEqual(set(tok['scope'].split()), {'mcp:read', 'mcp:write'})

    def test_remembered_write_consent_narrows_after_demotion(self):
        # First login remembers a read+write consent; after losing the group,
        # the remembered consent must NOT mint another write token.
        client, _v, _code, tok = self.full_flow()
        self.assertIn('mcp:write', tok['scope'])

        self._drop_write_group(self.oauth_user)
        verifier, challenge = self.pkce_pair()
        self._login()
        resp = self.authorize_get(client, challenge)
        code = self.code_from_redirect(resp)  # consent-skip path (read covered)
        self.assertTrue(code, 'read-only request should be covered by consent')
        tok2 = self.token_request(
            grant_type='authorization_code', code=code, client_id=client.client_id,
            code_verifier=verifier, redirect_uri='https://client.example/cb')
        self.assertEqual(tok2.status_code, 200, tok2.text)
        self.assertEqual(set(tok2.json()['scope'].split()), {'mcp:read'})

    # -- token endpoint (the mint choke point) --------------------------
    def test_refresh_narrows_to_read_after_demotion(self):
        client, _v, _code, tok = self.full_flow()
        self._drop_write_group(self.oauth_user)
        r = self.token_request(
            grant_type='refresh_token', refresh_token=tok['refresh_token'],
            client_id=client.client_id)
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(set(r.json()['scope'].split()), {'mcp:read'})

    def test_write_only_grant_dies_on_demotion(self):
        # A token whose ONLY scope was write cannot narrow — refresh must fail.
        client, _v, _code, tok = self.full_flow(scope_read=False)
        self.assertEqual(set(tok['scope'].split()), {'mcp:write'})
        self._drop_write_group(self.oauth_user)
        r = self.token_request(
            grant_type='refresh_token', refresh_token=tok['refresh_token'],
            client_id=client.client_id)
        self.assertEqual(r.json()['error'], 'invalid_grant')
