from __future__ import annotations

import json

import werkzeug.utils

from odoo import http
from odoo.http import request


def _issuer() -> str:
    """Canonical issuer URL (no trailing slash). Single source of truth."""
    base = request.env['ir.config_parameter'].sudo().get_param('web.base.url', '')
    return base.rstrip('/')


def _dcr_enabled() -> bool:
    from odoo.tools import str2bool
    return str2bool(
        request.env['ir.config_parameter'].sudo().get_param('ont_mcp_oauth.dcr_enabled', 'False'),
        default=False,
    )


def _json(payload, status=200, max_age=None):
    headers = [('Content-Type', 'application/json')]
    if max_age is not None:
        headers.append(('Cache-Control', f'public, max-age={max_age}'))
    else:
        headers.append(('Cache-Control', 'no-store'))
    return request.make_response(json.dumps(payload), headers=headers, status=status)


class OntMcpOAuthMetadata(http.Controller):

    # All discovery endpoints carry cors='*': MCP Inspector and other
    # browser-based clients fetch them cross-origin, and without the header the
    # browser blocks the response (and 404s the preflight), which presents as
    # "failed to discover OAuth metadata". muk_mcp's own /mcp route already
    # defaults to cors='*'.

    # RFC 9728 — Protected Resource Metadata. Both the bare and /mcp-suffixed
    # paths are served so path-aware clients find it either way.
    @http.route(
        ['/.well-known/oauth-protected-resource',
         '/.well-known/oauth-protected-resource/mcp'],
        type='http', auth='public', methods=['GET'], csrf=False,
        save_session=False, cors='*',
    )
    def protected_resource_metadata(self, **kw):
        issuer = _issuer()
        return _json({
            'resource': f'{issuer}/mcp',
            'authorization_servers': [issuer],
            'scopes_supported': ['mcp:read', 'mcp:write'],
            'bearer_methods_supported': ['header'],
            'resource_documentation': f'{issuer}/mcp',
        }, max_age=300)

    # RFC 8414 — Authorization Server Metadata. The openid-configuration
    # aliases cover clients (older ChatGPT connector flows, generic OpenID
    # Connect libraries) that only probe the OIDC discovery path.
    @http.route(
        ['/.well-known/oauth-authorization-server',
         '/.well-known/oauth-authorization-server/mcp',
         '/.well-known/openid-configuration',
         '/.well-known/openid-configuration/mcp'],
        type='http', auth='public', methods=['GET'], csrf=False,
        save_session=False, cors='*',
    )
    def authorization_server_metadata(self, **kw):
        issuer = _issuer()
        doc = {
            'issuer': issuer,
            'authorization_endpoint': f'{issuer}/mcp/oauth/authorize',
            'token_endpoint': f'{issuer}/mcp/oauth/token',
            'revocation_endpoint': f'{issuer}/mcp/oauth/revoke',
            'jwks_uri': f'{issuer}/mcp/oauth/jwks.json',
            'scopes_supported': ['mcp:read', 'mcp:write'],
            'response_types_supported': ['code'],
            'response_modes_supported': ['query'],
            'grant_types_supported': ['authorization_code', 'refresh_token'],
            'code_challenge_methods_supported': ['S256'],
            'token_endpoint_auth_methods_supported': [
                'none', 'client_secret_post', 'client_secret_basic'],
            'revocation_endpoint_auth_methods_supported': [
                'none', 'client_secret_post', 'client_secret_basic'],
            'authorization_response_iss_parameter_supported': True,
            # OIDC-required members, for strict validators reading the
            # openid-configuration alias. We issue no id_tokens; these merely
            # describe the signing setup so the document parses as OIDC.
            'subject_types_supported': ['public'],
            'id_token_signing_alg_values_supported': ['RS256'],
        }
        if _dcr_enabled():
            doc['registration_endpoint'] = f'{issuer}/mcp/oauth/register'
        return _json(doc, max_age=300)

    # RFC 7517 — JWKS (public signing keys).
    @http.route(
        '/mcp/oauth/jwks.json',
        type='http', auth='public', methods=['GET'], csrf=False,
        save_session=False, cors='*',
    )
    def jwks(self, **kw):
        jwks = request.env['ont_mcp_oauth.signing.key'].sudo()._get_jwks()
        return _json(jwks, max_age=300)

    # Users paste "https://host/mcp/" (trailing slash) into connector UIs;
    # werkzeug does not redirect a slashed request onto the slashless muk
    # route, so it 404s and discovery never bootstraps. 308 preserves the
    # method and body, so POSTed JSON-RPC survives the hop.
    @http.route(
        '/mcp/', type='http', auth='public', csrf=False, save_session=False,
        methods=['GET', 'POST', 'DELETE', 'OPTIONS'], cors='*',
    )
    def mcp_trailing_slash(self, **kw):
        return werkzeug.utils.redirect('/mcp', code=308)
