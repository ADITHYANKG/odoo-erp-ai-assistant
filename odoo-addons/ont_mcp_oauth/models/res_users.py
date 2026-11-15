from __future__ import annotations

from odoo import fields, models


class ResUsers(models.Model):
    _inherit = 'res.users'

    oauth_consent_ids = fields.One2many(
        'ont_mcp_oauth.consent', 'user_id', string='OAuth Connections',
    )

    def write(self, vals):
        """Cut off AI connectors when a user is offboarded or demoted.

        Access tokens are already refused at /mcp for an inactive or
        non-internal user, but the token rows stayed live: re-enabling the
        account would have silently resurrected every connector the person had
        authorised, with no fresh consent. Revoking on the way out makes
        offboarding actually stick.
        """
        result = super().write(vals)
        if 'active' in vals and not vals['active']:
            self._ont_revoke_mcp_access()
        elif 'groups_id' in vals:
            # A demotion to portal/share revokes too — _is_internal() is the
            # same boundary /mcp enforces.
            self.filtered(lambda user: not user._is_internal())._ont_revoke_mcp_access()
        return result

    def _mcp_scope_ceiling(self):
        """Scopes this user may ever be granted, regardless of client.

        The per-user counterpart of ``ont_mcp_oauth.client._scope_ceiling()``:
        the effective grant is requested ∩ client ceiling ∩ user ceiling.
        Membership in "MCP: Allow Write" (administrators via implication) is
        what unlocks ``mcp:write``.
        """
        self.ensure_one()
        if self.has_group('ont_mcp_oauth.group_mcp_write'):
            return {'mcp:read', 'mcp:write'}
        return {'mcp:read'}

    def _mcp_can_call_methods(self):
        """Whether this user may use the call_method MCP tool.

        call_method invokes arbitrary public methods (the widest write
        surface), so it is gated on its own group — administrators have it
        via implication; grant it individually to trusted users.
        """
        self.ensure_one()
        return self.has_group('ont_mcp_oauth.group_mcp_method_call')

    def _ont_revoke_mcp_access(self):
        """Revoke every live OAuth token belonging to these users."""
        if not self:
            return
        self.env['ont_mcp_oauth.token'].sudo().search([
            ('user_id', 'in', self.ids), ('revoked', '=', False),
        ]).write({'revoked': True})

    @property
    def SELF_READABLE_FIELDS(self):
        # Readable only. Revocation goes through the consent record's
        # action_revoke button, which the consent record rule already scopes to
        # the owner — so this never needs to be self-writeable. Keeping it out
        # of SELF_WRITEABLE_FIELDS means a future ACL change that granted write
        # on consent could not turn this one2many into a cross-user handle.
        return super().SELF_READABLE_FIELDS + ['oauth_consent_ids']
