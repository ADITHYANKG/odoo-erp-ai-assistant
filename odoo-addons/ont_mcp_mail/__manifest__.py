{
    'name': 'MCP Mail (Ontash)',
    'summary': 'Send MCP email with optional PDF or data export attachment',
    'description': """
        Adds a single Model Context Protocol (MCP) tool, ``send_email``, on top
        of muk_mcp: agents can send an email through the company's configured
        outgoing mail server, optionally attaching a record's PDF report
        (quotation, invoice, ...) or a CSV/XLSX data export rendered server-side.

        Write-category tool: requires the mcp:write token scope, and any
        chat-agent confirmation layer built on tool categories applies
        unchanged. The mail is logged in Odoo's email queue and stamped with
        the authenticated user's identity.
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
