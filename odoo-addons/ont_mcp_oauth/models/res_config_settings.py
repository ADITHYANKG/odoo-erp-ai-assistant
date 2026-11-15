from __future__ import annotations

from odoo import _, api, fields, models
from odoo.exceptions import ValidationError
from odoo.tools import str2bool

# Accepted (minimum, maximum) for each integer config parameter, keyed by the
# parameter name. Unbounded values are a security problem, not just a footgun:
# an over-long access-token TTL makes bearer JWTs effectively immortal (leaving
# revocation resting entirely on the jti registry), and an over-long refresh
# grace window disables rotation-replay detection, so a stolen rotated refresh
# token stays redeemable for the whole window.
PARAM_BOUNDS = {
    'ont_mcp_oauth.access_token_ttl': (60, 86400),
    'ont_mcp_oauth.refresh_token_ttl_days': (1, 365),
    'ont_mcp_oauth.refresh_grace_seconds': (0, 60),
}

# Settings field -> config parameter, for validation messages.
_FIELD_PARAMS = {
    'oauth_access_token_ttl': 'ont_mcp_oauth.access_token_ttl',
    'oauth_refresh_token_ttl_days': 'ont_mcp_oauth.refresh_token_ttl_days',
    'oauth_refresh_grace_seconds': 'ont_mcp_oauth.refresh_grace_seconds',
}


def clamp_param(config_key: str, value) -> int:
    """Coerce a raw config-parameter value to an int inside its safe range.

    System Parameters can be edited directly (Settings -> Technical -> System
    Parameters), which bypasses this model's validation entirely, so every
    consumer clamps on read as well. Belt and braces: the form refuses a bad
    value, and a hand-edited one is ignored.
    """
    lo, hi = PARAM_BOUNDS.get(config_key, (None, None))
    try:
        number = int(value)
    except (TypeError, ValueError):
        number = lo if lo is not None else 0
    if lo is None:
        return number
    return max(lo, min(hi, number))


class ResConfigSettings(models.TransientModel):
    _inherit = 'res.config.settings'

    oauth_access_token_ttl = fields.Integer(
        string='Access Token TTL (seconds)',
        config_parameter='ont_mcp_oauth.access_token_ttl', default=900,
    )
    oauth_refresh_token_ttl_days = fields.Integer(
        string='Refresh Token TTL (days)',
        config_parameter='ont_mcp_oauth.refresh_token_ttl_days', default=30,
    )
    # 30 s, not 5: mobile HTTP timeouts run 10-30 s, so a client retrying a
    # lost token response usually lands OUTSIDE a 5 s window and the grace
    # reissue never applies. Auth0's reuse-interval guidance is ~30 s.
    oauth_refresh_grace_seconds = fields.Integer(
        string='Refresh Grace Window (seconds)',
        config_parameter='ont_mcp_oauth.refresh_grace_seconds', default=30,
    )
    oauth_dcr_enabled = fields.Boolean(
        string='Dynamic Client Registration',
        config_parameter='ont_mcp_oauth.dcr_enabled', default=False,
    )
    oauth_base_url_frozen = fields.Boolean(
        string='Base URL Frozen', compute='_compute_oauth_base_url_frozen',
    )

    @api.depends_context('uid')
    def _compute_oauth_base_url_frozen(self):
        """Whether web.base.url is pinned against Odoo's auto-rewrite.

        Every successful login by a Settings-group user overwrites
        web.base.url from that request's Host header unless this is set
        (odoo/addons/base/models/res_users.py). Our token issuer, the `aud`
        binding and all /.well-known metadata derive from web.base.url, so an
        unfrozen value means one admin login from another hostname silently
        invalidates every live token and republishes the OAuth endpoints on
        that host.
        """
        frozen = str2bool(
            self.env['ir.config_parameter'].sudo().get_param(
                'web.base.url.freeze', 'False'),
            default=False,
        )
        for record in self:
            record.oauth_base_url_frozen = frozen

    def action_freeze_base_url(self):
        """Pin web.base.url to its current value."""
        self.env['ir.config_parameter'].sudo().set_param('web.base.url.freeze', 'True')
        return {'type': 'ir.actions.client', 'tag': 'reload'}

    @api.model
    def default_get(self, fields_list):
        """Load the form with clamped values.

        A value hand-edited into System Parameters must not make the shared
        General Settings page unopenable — show it clamped instead, matching
        what the runtime actually enforces.
        """
        result = super().default_get(fields_list)
        for field_name, config_key in _FIELD_PARAMS.items():
            if field_name in result:
                result[field_name] = clamp_param(config_key, result[field_name])
        return result

    @api.constrains(*_FIELD_PARAMS)
    def _check_oauth_bounds(self):
        for record in self:
            for field_name, config_key in _FIELD_PARAMS.items():
                value = record[field_name]
                lo, hi = PARAM_BOUNDS[config_key]
                if not lo <= value <= hi:
                    raise ValidationError(_(
                        '%(label)s must be between %(low)s and %(high)s.',
                        label=record._fields[field_name].string, low=lo, high=hi,
                    ))
