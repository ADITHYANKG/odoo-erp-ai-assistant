from __future__ import annotations

import time

from odoo.tests import tagged

from .common import OAuthHttpCase


@tagged('post_install', '-at_install')
class TestRefresh(OAuthHttpCase):

    def _refresh(self, client, refresh_token, scope=None):
        form = {'grant_type': 'refresh_token', 'refresh_token': refresh_token,
                'client_id': client.client_id}
        if scope:
            form['scope'] = scope
        return self.token_request(**form)

    def test_rotation(self):
        client, _v, _code, tok = self.full_flow()
        r1 = self._refresh(client, tok['refresh_token'])
        self.assertEqual(r1.status_code, 200, r1.text)
        new = r1.json()
        self.assertNotEqual(new['refresh_token'], tok['refresh_token'])
        # Old refresh, replayed after grace, must fail and kill the family.
        self.env['ir.config_parameter'].sudo().set_param('ont_mcp_oauth.refresh_grace_seconds', '0')
        time.sleep(1)
        replay = self._refresh(client, tok['refresh_token'])
        self.assertEqual(replay.json()['error'], 'invalid_grant')
        # New refresh should also now be dead (family revoked).
        after = self._refresh(client, new['refresh_token'])
        self.assertEqual(after.json()['error'], 'invalid_grant')

    def test_scope_narrow_on_refresh(self):
        client, _v, _code, tok = self.full_flow()
        r = self._refresh(client, tok['refresh_token'], scope='mcp:read')
        self.assertEqual(set(r.json()['scope'].split()), {'mcp:read'})

    def test_cannot_widen_on_refresh(self):
        client, _v, _code, tok = self.full_flow(scope_write=False)  # read-only grant
        r = self._refresh(client, tok['refresh_token'], scope='mcp:read mcp:write')
        self.assertEqual(r.json()['error'], 'invalid_scope')

    def test_revoke_refresh_kills_family(self):
        client, _v, _code, tok = self.full_flow()
        self.url_open('/mcp/oauth/revoke', data={'token': tok['refresh_token']})
        r = self._refresh(client, tok['refresh_token'])
        self.assertEqual(r.json()['error'], 'invalid_grant')

    def test_revoke_always_200(self):
        r = self.url_open('/mcp/oauth/revoke', data={'token': 'totally-unknown-token'})
        self.assertEqual(r.status_code, 200)
