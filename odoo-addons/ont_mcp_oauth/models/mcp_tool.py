from __future__ import annotations

from typing import Any

from odoo import _, api, models
from odoo.api import Environment

from odoo.addons.muk_mcp.tools.exception import MCPScopeDenied


class MCPTool(models.Model):
    """Per-user gate on the call_method tool.

    call_method can invoke any public method on an allowed model — the widest
    write surface — so beyond the mcp:write token scope it requires membership
    in the "MCP: Allow Method Calls" group (administrators via implication).

    The check lives in ``_execute`` (not ``_call``) on purpose: raising here
    is inside muk_mcp's audit try/finally, so the attempt is logged with
    status 'denied'. The authenticated user is the ``env`` parameter's user
    (the controller passes ``request.env``), never ``self.env``.
    """

    _inherit = 'muk_mcp.tool'

    @api.model
    def _execute(
        self,
        name: str,
        arguments: dict[str, Any],
        env: Environment,
        enforce_scope: str | None,
    ) -> tuple[Any, dict[str, Any], str | None]:
        if name == 'call_method' and not env.user._mcp_can_call_methods():
            raise MCPScopeDenied(_(
                'Access denied: call_method requires the '
                '"MCP: Allow Method Calls" group.'
            ))
        return super()._execute(name, arguments, env, enforce_scope)
