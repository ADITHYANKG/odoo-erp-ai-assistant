"""call_method is gated on the "MCP: Allow Method Calls" group: even a
write-scoped token cannot invoke it unless the USER is a member."""
from __future__ import annotations

from odoo.tests import tagged

from .common import OAuthHttpCase


@tagged('post_install', '-at_install')
class TestMethodCallGate(OAuthHttpCase):

    def _mint(self, **kw):
        _client, _v, _code, tok = self.full_flow(**kw)
        return tok['access_token']

    def _call_method(self, token):
        sid = self.mcp_handshake(token=token)
        return self.mcp_call_tool(
            'call_method',
            {'model': 'res.partner', 'method': 'search_count', 'args': '[[]]'},
            session_id=sid, token=token,
        )

    def _grant_method_group(self, user):
        group = self.env.ref('ont_mcp_oauth.group_mcp_method_call')
        user.write({'groups_id': [(4, group.id)]})
        self.addCleanup(lambda: user.write({'groups_id': [(3, group.id)]}))

    # -- unit level -----------------------------------------------------
    def test_membership_helper_and_admin_implication(self):
        self.assertFalse(self.oauth_user._mcp_can_call_methods())
        self.assertFalse(self.reader_user._mcp_can_call_methods())
        admin = self.env.ref('base.user_admin')
        self.assertTrue(admin._mcp_can_call_methods())

    # -- HTTP level -----------------------------------------------------
    def test_write_scope_alone_is_not_enough(self):
        # oauth_user holds mcp:write but NOT the method group.
        token = self._mint()
        result = self._call_method(token)
        self.assertIn('Method Calls', str(result), result)

    def test_member_can_call_methods(self):
        self._grant_method_group(self.oauth_user)
        token = self._mint()
        result = self._call_method(token)
        text = str(result)
        self.assertNotIn('Method Calls', text, result)
        self.assertNotIn('Access denied', text, result)

    def test_other_write_tools_unaffected_by_the_gate(self):
        # The gate is call_method-specific: a normal write tool still works
        # for a write-scoped non-member.
        token = self._mint()
        sid = self.mcp_handshake(token=token)
        result = self.mcp_call_tool(
            'create_records',
            {'model': 'res.partner', 'values': [{'name': 'Gate Probe Partner'}]},
            session_id=sid, token=token,
        )
        self.assertNotIn('Method Calls', str(result), result)
        self.assertNotIn('Access denied', str(result), result)
