"""Acme AI-over-ERP agent.

A standalone service that lets company executives query their Odoo data
through chat platforms (Telegram first). It is a pure *client* of the Odoo
stack: it talks to muk_mcp's ``/mcp`` endpoint (Model Context Protocol over
Streamable HTTP), authenticating per user, and calls its large language model
through OpenRouter only.
"""
