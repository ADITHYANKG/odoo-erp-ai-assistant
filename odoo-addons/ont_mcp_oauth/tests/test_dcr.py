from __future__ import annotations

import json

from odoo.tests import tagged

from .common import OAuthHttpCase


@tagged('post_install', '-at_install')
class TestDcr(OAuthHttpCase):

    def _register(self, body):
        return self.url_open(
            '/mcp/oauth/register', data=json.dumps(body),
            headers={'Content-Type': 'application/json'}, allow_redirects=False)

    def test_disabled_returns_403(self):
        self.env['ir.config_parameter'].sudo().set_param('ont_mcp_oauth.dcr_enabled', 'False')
        r = self._register({'redirect_uris': ['https://c.example/cb']})
        self.assertEqual(r.status_code, 403)
        self.assertEqual(r.json()['error'], 'access_denied')

    def test_enabled_registers(self):
        self.env['ir.config_parameter'].sudo().set_param('ont_mcp_oauth.dcr_enabled', 'True')
        r = self._register({'redirect_uris': ['https://c.example/cb'], 'client_name': 'DCR App'})
        self.assertEqual(r.status_code, 201, r.text)
        body = r.json()
        self.assertTrue(body['client_id'].startswith('ont_'))
        client = self.Client.sudo().search([('client_id', '=', body['client_id'])])
        self.assertTrue(client.dcr)

    def test_invalid_redirect_rejected(self):
        self.env['ir.config_parameter'].sudo().set_param('ont_mcp_oauth.dcr_enabled', 'True')
        r = self._register({'redirect_uris': ['http://evil.example/cb']})
        self.assertEqual(r.status_code, 400)
        self.assertEqual(r.json()['error'], 'invalid_redirect_uri')

    def test_requested_scope_is_honoured(self):
        # A client asking only to read must get a read-only ceiling.
        self.env['ir.config_parameter'].sudo().set_param('ont_mcp_oauth.dcr_enabled', 'True')
        r = self._register({'redirect_uris': ['https://ro.example/cb'], 'scope': 'mcp:read'})
        self.assertEqual(r.status_code, 201, r.text)
        client = self.Client.sudo().search([('client_id', '=', r.json()['client_id'])])
        self.assertEqual(client.scope, 'read')
        self.assertEqual(client._scope_ceiling(), {'mcp:read'})
        self.assertEqual(r.json()['scope'], 'mcp:read')

    def test_omitted_auth_method_defaults_to_confidential(self):
        # RFC 7591 §2: the default for an omitted token_endpoint_auth_method is
        # client_secret_basic — the client must be registered confidential and
        # receive a secret. (Public MCP clients all send 'none' explicitly.)
        self.env['ir.config_parameter'].sudo().set_param('ont_mcp_oauth.dcr_enabled', 'True')
        r = self._register({'redirect_uris': ['https://basic.example/cb']})
        self.assertEqual(r.status_code, 201, r.text)
        body = r.json()
        self.assertEqual(body['token_endpoint_auth_method'], 'client_secret_basic')
        self.assertIn('client_secret', body)
        self.assertEqual(body['client_name'], 'DCR client')  # RFC 7591 echo
        client = self.Client.sudo().search([('client_id', '=', body['client_id'])])
        self.assertEqual(client.client_type, 'confidential')

    def test_grant_types_are_enforced(self):
        self.env['ir.config_parameter'].sudo().set_param('ont_mcp_oauth.dcr_enabled', 'True')
        r = self._register({'redirect_uris': ['https://nr.example/cb'],
                            'grant_types': ['authorization_code'],
                            'token_endpoint_auth_method': 'none'})
        client = self.Client.sudo().search([('client_id', '=', r.json()['client_id'])])
        self.assertFalse(client.allow_refresh)
        # No refresh token should be issued at all...
        _c, _v, _code, tok = self.full_flow(client=client, redirect_uri='https://nr.example/cb')
        self.assertNotIn('refresh_token', tok)
        # ...and the grant is refused outright.
        bad = self.token_request(grant_type='refresh_token', refresh_token='anything',
                                 client_id=client.client_id)
        self.assertEqual(bad.json()['error'], 'unauthorized_client')

    def test_missing_redirect_uris_rejected(self):
        self.env['ir.config_parameter'].sudo().set_param('ont_mcp_oauth.dcr_enabled', 'True')
        r = self._register({'client_name': 'no uris'})
        self.assertEqual(r.status_code, 400)
