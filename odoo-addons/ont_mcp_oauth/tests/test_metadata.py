from __future__ import annotations

from odoo.tests import tagged

from .common import OAuthHttpCase


@tagged('post_install', '-at_install')
class TestMetadata(OAuthHttpCase):
    def test_protected_resource_metadata(self):
        r = self.url_open('/.well-known/oauth-protected-resource/mcp')
        self.assertEqual(r.status_code, 200)
        doc = r.json()
        self.assertTrue(doc['resource'].endswith('/mcp'))
        self.assertEqual(doc['scopes_supported'], ['mcp:read', 'mcp:write'])
        self.assertIn(doc['authorization_servers'][0], doc['resource'])

    def test_authorization_server_metadata(self):
        r = self.url_open('/.well-known/oauth-authorization-server')
        self.assertEqual(r.status_code, 200)
        doc = r.json()
        self.assertTrue(doc['authorization_endpoint'].endswith('/mcp/oauth/authorize'))
        self.assertTrue(doc['token_endpoint'].endswith('/mcp/oauth/token'))
        self.assertEqual(doc['code_challenge_methods_supported'], ['S256'])
        self.assertTrue(doc['authorization_response_iss_parameter_supported'])

    def test_registration_endpoint_hidden_when_dcr_off(self):
        self.env['ir.config_parameter'].sudo().set_param('ont_mcp_oauth.dcr_enabled', 'False')
        doc = self.url_open('/.well-known/oauth-authorization-server').json()
        self.assertNotIn('registration_endpoint', doc)
        self.env['ir.config_parameter'].sudo().set_param('ont_mcp_oauth.dcr_enabled', 'True')
        doc = self.url_open('/.well-known/oauth-authorization-server').json()
        self.assertIn('registration_endpoint', doc)

    def test_jwks_shape(self):
        # Force a signing key to exist.
        self.env['ont_mcp_oauth.signing.key'].sudo()._get_active_key()
        doc = self.url_open('/mcp/oauth/jwks.json').json()
        self.assertTrue(doc['keys'])
        jwk = doc['keys'][0]
        self.assertEqual(jwk['kty'], 'RSA')
        self.assertEqual(jwk['alg'], 'RS256')
        self.assertIn('n', jwk)
        self.assertIn('e', jwk)
        self.assertNotIn('d', jwk)  # never leak private material
