from . import models
from . import mcp


def _pre_init_vector(env):
    """pgvector must exist before ``ont.document.chunk.init()`` adds its column."""
    env.cr.execute('CREATE EXTENSION IF NOT EXISTS vector')
