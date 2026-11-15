from __future__ import annotations

import secrets

from odoo.exceptions import ValidationError
from odoo.tests import TransactionCase, tagged

from odoo.addons.ont_mcp_oauth.tools import jwt as jwt_tools
from odoo.addons.ont_mcp_oauth.tools import pkce as pkce_tools
from odoo.addons.ont_mcp_oauth.tools import uri as uri_tools


@tagged('post_install', '-at_install')
class TestPkce(TransactionCase):
    def test_verifier_lengths(self):
        self.assertFalse(pkce_tools.is_valid_verifier('a' * 42))
        self.assertTrue(pkce_tools.is_valid_verifier('a' * 43))
        self.assertTrue(pkce_tools.is_valid_verifier('a' * 128))
        self.assertFalse(pkce_tools.is_valid_verifier('a' * 129))

    def test_verifier_charset(self):
        self.assertFalse(pkce_tools.is_valid_verifier('a' * 42 + '!'))
        self.assertTrue(pkce_tools.is_valid_verifier('-._~' + 'A9z' * 13))

    def test_roundtrip_and_verify(self):
        v = secrets.token_urlsafe(64)[:80]
        c = pkce_tools.compute_challenge(v)
        self.assertTrue(pkce_tools.is_valid_challenge(c))
        self.assertTrue(pkce_tools.verify(v, c))
        self.assertFalse(pkce_tools.verify(v + 'x', c))
        self.assertFalse(pkce_tools.verify('short', c))


@tagged('post_install', '-at_install')
class TestRedirectUri(TransactionCase):
    def test_matrix(self):
        ok = [
            'https://claude.ai/api/mcp/auth_callback',
            'https://good.example:8443/cb',
            'http://localhost:1234/cb',
            'http://127.0.0.1/cb',
            'com.example.app:/callback',
        ]
        bad = [
            'http://evil.example/cb',           # non-loopback http
            'https://good.com/cb#frag',          # fragment
            'javascript:alert(1)',               # dangerous scheme
            'data:text/html,x',                  # dangerous scheme
            'file:///etc/passwd',                # dangerous scheme
            'https://good.com/cb\r\nX:1',        # CRLF
            '',                                  # empty
            'not a uri',                         # no scheme
        ]
        for u in ok:
            self.assertTrue(uri_tools.is_valid_redirect_uri(u), u)
        for u in bad:
            self.assertFalse(uri_tools.is_valid_redirect_uri(u), u)

    def test_exact_match_only(self):
        registered = ['https://good.example/cb']
        self.assertTrue(uri_tools.matches_registered('https://good.example/cb', registered))
        # Prefix / suffix tricks must NOT match.
        self.assertFalse(uri_tools.matches_registered('https://good.example/cb/extra', registered))
        self.assertFalse(uri_tools.matches_registered('https://good.example.evil.com/cb', registered))


@tagged('post_install', '-at_install')
class TestJwt(TransactionCase):
    def setUp(self):
        super().setUp()
        self.priv, self.pub = jwt_tools.generate_rsa_keypair()
        self.claims = {
            'iss': 'https://odoo.example', 'sub': '7',
            'aud': 'https://odoo.example/mcp', 'jti': 'j1',
            'iat': 1000, 'nbf': 1000, 'exp': 9999999999,
        }

    def test_roundtrip(self):
        tok = jwt_tools.encode(self.claims, self.priv, 'kid1')
        out = jwt_tools.decode(tok, self.pub, issuer='https://odoo.example',
                               audience='https://odoo.example/mcp', now=2000)
        self.assertEqual(out['sub'], '7')

    def test_tampered_payload(self):
        tok = jwt_tools.encode(self.claims, self.priv, 'kid1')
        h, p, s = tok.split('.')
        bad = f'{h}.{p}x.{s}'
        with self.assertRaises(jwt_tools.JWTError):
            jwt_tools.decode(bad, self.pub, issuer='https://odoo.example',
                             audience='https://odoo.example/mcp', now=2000)

    def test_wrong_key(self):
        _p2, pub2 = jwt_tools.generate_rsa_keypair()
        tok = jwt_tools.encode(self.claims, self.priv, 'kid1')
        with self.assertRaises(jwt_tools.JWTError):
            jwt_tools.decode(tok, pub2, issuer='https://odoo.example',
                             audience='https://odoo.example/mcp', now=2000)

    def test_alg_none_and_hs256_rejected(self):
        # Forge a header claiming alg:none / HS256 — decode must reject on alg.
        import base64
        import json
        for alg in ('none', 'HS256'):
            header = base64.urlsafe_b64encode(
                json.dumps({'alg': alg, 'typ': 'JWT'}).encode()).rstrip(b'=').decode()
            payload = base64.urlsafe_b64encode(
                json.dumps(self.claims).encode()).rstrip(b'=').decode()
            forged = f'{header}.{payload}.'
            with self.assertRaises(jwt_tools.JWTError):
                jwt_tools.decode(forged, self.pub, issuer='https://odoo.example',
                                 audience='https://odoo.example/mcp', now=2000)

    def test_aud_iss_exp_nbf(self):
        tok = jwt_tools.encode(self.claims, self.priv, 'kid1')
        with self.assertRaises(jwt_tools.JWTError):
            jwt_tools.decode(tok, self.pub, issuer='https://evil.example',
                             audience='https://odoo.example/mcp', now=2000)
        with self.assertRaises(jwt_tools.JWTError):
            jwt_tools.decode(tok, self.pub, issuer='https://odoo.example',
                             audience='https://odoo.example/other', now=2000)
        expired = dict(self.claims, exp=1500)
        etok = jwt_tools.encode(expired, self.priv, 'kid1')
        with self.assertRaises(jwt_tools.JWTError):
            jwt_tools.decode(etok, self.pub, issuer='https://odoo.example',
                             audience='https://odoo.example/mcp', now=2000)


@tagged('post_install', '-at_install')
class TestModelConstraints(TransactionCase):
    def test_redirect_uri_validation_on_create(self):
        with self.assertRaises(ValidationError):
            self.env['ont_mcp_oauth.client'].create({
                'name': 'bad', 'redirect_uris': 'http://evil.example/cb',
            })

    def test_scope_ceiling(self):
        c = self.env['ont_mcp_oauth.client'].create({
            'name': 'ro', 'redirect_uris': 'https://x.example/cb', 'scope': 'read',
        })
        self.assertEqual(c._scope_ceiling(), {'mcp:read'})

    def test_unique_constraints_exist(self):
        # Security-critical: these UNIQUE constraints must be present in Postgres
        # (Odoo 19 ignores the legacy _sql_constraints style silently).
        self.env.cr.execute("""
            SELECT string_agg(conname, ',') FROM pg_constraint
             WHERE contype = 'u' AND conrelid::regclass::text LIKE 'ont_mcp_oauth%'
        """)
        names = (self.env.cr.fetchone()[0] or '')
        for table in ('ont_mcp_oauth_client', 'ont_mcp_oauth_code',
                      'ont_mcp_oauth_token', 'ont_mcp_oauth_consent'):
            self.env.cr.execute(
                "SELECT count(*) FROM pg_constraint WHERE contype='u' "
                "AND conrelid = %s::regclass", (table,))
            self.assertGreaterEqual(
                self.env.cr.fetchone()[0], 1, f'{table} is missing its UNIQUE constraint ({names})')
