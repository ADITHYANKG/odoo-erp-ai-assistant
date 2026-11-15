{
    'name': 'MCP Field Guard (Ontash)',
    'summary': 'Drop unknown fields from MCP reads instead of failing the call.',
    'description': """
Sanitises the ``fields`` argument of the muk_mcp read tools (search_read,
read_records). An MCP client — typically an LLM — that names a field which does
not exist on the target model would otherwise make the whole read raise
``Invalid field X on model Y``. This extension drops unknown fields (logging
what it dropped) and returns the valid ones, so a wrong field name is never a
fatal error. Target: Odoo 17.
""",
    'version': '17.0.1.0.0',
    'category': 'Tools/API',
    'license': 'LGPL-3',
    'author': 'Ontash',
    'website': 'https://example.com',
    'depends': ['muk_mcp'],
    'installable': True,
    'application': False,
}
