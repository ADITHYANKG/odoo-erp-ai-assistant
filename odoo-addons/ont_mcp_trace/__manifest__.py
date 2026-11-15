{
    'name': 'MCP Trace (Ontash)',
    'summary': 'Post a chatter note on records changed through MCP write tools',
    'description': """
        After every successful WRITE-category Model Context Protocol (MCP)
        tool call, posts an internal note on the affected record's chatter:
        "via MCP: update_records by <user> - fields: phone". Gives every
        record a human-readable right-side log of agent activity, on top of
        the technical MCP log. Reads stay silent by design.
    """,
    'version': '17.0.1.0.0',
    'category': 'Tools/API',
    'license': 'LGPL-3',
    'author': 'Ontash',
    'depends': ['muk_mcp', 'mail'],
    'data': [],
    'installable': True,
    'application': False,
}
