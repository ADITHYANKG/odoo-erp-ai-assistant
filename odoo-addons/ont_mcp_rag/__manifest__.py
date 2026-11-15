{
    'name': 'MCP Document Search (Ontash)',
    'summary': 'Ask the AI assistant about policy documents held in Odoo Documents',
    'description': """
        Makes the contents of Odoo Documents answerable through the Model
        Context Protocol (MCP), so a chat assistant can answer "what is the
        harassment policy?" from the company's own documents instead of from
        the language model's general knowledge.

        Folders are opt-in: tick "Searchable by AI" on a folder and a
        scheduled action chunks every document in it, embeds each chunk
        through OpenRouter, and stores the vectors in pgvector alongside the
        document. Uploading a new file needs no further action.

        Retrieval is hybrid - cosine similarity over the embeddings combined
        with PostgreSQL full-text search - because similarity alone misses
        exact terms and full-text alone misses paraphrases. Search runs as
        the requesting user, so Odoo's own record rules decide what each
        person can find.

        Read-category tool: no write scope required.
    """,
    'version': '17.0.1.0.0',
    'category': 'Tools/API',
    'license': 'LGPL-3',
    'author': 'Ontash',
    # attachment_indexation fills ir_attachment.index_content, the only text
    # source this module reads. The pgvector extension is created by the hook.
    'depends': ['muk_mcp', 'documents', 'attachment_indexation'],
    'pre_init_hook': '_pre_init_vector',
    'data': [
        'security/ir.model.access.csv',
        'data/ir_cron.xml',
        'views/documents_folder_views.xml',
    ],
    'installable': True,
    'application': False,
}
