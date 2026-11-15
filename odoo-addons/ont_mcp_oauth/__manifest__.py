{
    'name': 'MCP OAuth (Ontash)',
    'summary': 'OAuth 2.1 authorization server for the MuK MCP server',
    'description': """
        Adds an OAuth 2.1 authorization server on top of muk_mcp so remote AI
        clients (ChatGPT.com Custom MCP connectors, Claude.ai Custom Connectors,
        custom agents) can sign in to Odoo via the standard authorization-code +
        PKCE S256 flow. Turns the existing /mcp endpoint into a fully discoverable
        authorization server: Dynamic Client Registration (RFC 7591), Protected
        Resource Metadata (RFC 9728), Authorization Server Metadata (RFC 8414),
        RS256 JWT access tokens with RFC 8707 audience binding, rotating refresh
        tokens, and an in-Odoo per-scope consent screen.

        Two scopes: mcp:read and mcp:write. Bearer API keys from muk_mcp keep
        working unchanged for local clients.

        Target: Odoo 17 on-premise and odoo.sh. Not for Odoo Online.
    """,
    'version': '17.0.1.0.0',
    'category': 'Tools/API',
    'license': 'LGPL-3',
    'author': 'Ontash',
    'depends': ['muk_mcp'],
    'external_dependencies': {'python': ['cryptography']},
    'data': [
        'security/groups.xml',
        'security/security.xml',
        'security/ir.model.access.csv',
        'data/ir_config_parameter.xml',
        'views/client.xml',
        'views/signing_key.xml',
        'views/res_config_settings.xml',
        'views/res_users.xml',
        'views/connect.xml',
        'views/templates.xml',
    ],
    'installable': True,
    'application': False,
}
