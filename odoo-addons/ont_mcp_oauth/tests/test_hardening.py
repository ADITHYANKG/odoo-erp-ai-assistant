"""Regression tests for the third-audit hardening fixes.

Each test here pins a specific security property that was missing before:
strict base64url, single-use consent requests, bounded TTLs, and a signing-key
retirement window short enough that rotation actually contains a key leak.
"""
from __future__ import annotations

from odoo import fields
from odoo.exceptions import UserError, ValidationError
from odoo.tests import TransactionCase, tagged

from odoo.addons.ont_mcp_oauth.models.res_config_settings import clamp_param
from odoo.addons.ont_mcp_oauth.tools import jwt as jwt_tools

from .common import OAuthHttpCase


@tagged('post_install', '-at_install')
class TestStrictBase64Url(TransactionCase):
    """A token must have exactly one valid string spelling."""

    def test_non_alphabet_characters_rejected(self):
        # 'abcd\n' is the subtle one: regex '$' also matches just before a
        # trailing newline, so an anchored ^...$ match would let it through as
        # an alias. Only fullmatch rejects it.
        for bad in ('ab!c', 'ab c', 'ab=c', 'ab+c', 'ab/c', 'ab\nc', 'abcd\n', 'abcd\n\n'):
            with self.assertRaises(jwt_tools.JWTError, msg=bad):
                jwt_tools.b64url_decode(bad)

    def test_alphabet_still_decodes(self):
        self.assertEqual(jwt_tools.b64url_decode(jwt_tools.b64url_encode(b'hello')), b'hello')

    def test_token_alias_is_rejected(self):
        # Padding characters injected into a segment used to be silently
        # dropped, giving the same token unlimited distinct spellings.
        priv, pub = jwt_tools.generate_rsa_keypair()
        claims = {'iss': 'https://o.example', 'sub': '1', 'aud': 'https://o.example/mcp',
                  'jti': 'j', 'iat': 1000, 'nbf': 1000, 'exp': 9999999999}
        token = jwt_tools.encode(claims, priv, 'k1')
        head, payload, sig = token.split('.')
        with self.assertRaises(jwt_tools.JWTError):
            jwt_tools.decode(f'{head}.{payload}=.{sig}', pub, issuer='https://o.example',
                             audience='https://o.example/mcp', now=2000)


@tagged('post_install', '-at_install')
class TestAuthRequestSingleUse(TransactionCase):

    def setUp(self):
        super().setUp()
        self.AuthRequest = self.env['ont_mcp_oauth.authrequest']
        self.user = self.env['res.users'].create({
            'name': 'Consent User', 'login': 'ont_consent_user',
            'groups_id': [(6, 0, [self.env.ref('base.group_user').id])]})
        self.client = self.env['ont_mcp_oauth.client'].create({
            'name': 'App', 'redirect_uris': 'https://app.example/cb'})

    def _make(self, user=None, expires_at=None):
        return self.AuthRequest.create({
            'client_id': self.client.id,
            'user_id': (user or self.user).id,
            'scope': 'mcp:read',
            'code_challenge': 'x' * 43,
            'redirect_uri': 'https://app.example/cb',
            'expires_at': expires_at or self.AuthRequest._default_expiry(),
        })

    def test_consumed_exactly_once(self):
        req = self._make()
        token = req.req_token
        first = self.AuthRequest._consume(token, self.user)
        self.assertTrue(first)
        self.assertEqual(first['scope'], 'mcp:read')
        self.assertEqual(first['client_id'], self.client)
        # A second submission of the same token must yield nothing, so one
        # consent interaction can never mint two authorization codes.
        self.assertIsNone(self.AuthRequest._consume(token, self.user))
        self.assertFalse(self.AuthRequest.sudo().search_count([('req_token', '=', token)]))

    def test_other_user_cannot_consume_or_delete(self):
        req = self._make()
        thief = self.env['res.users'].create({
            'name': 'Thief', 'login': 'ont_consent_thief',
            'groups_id': [(6, 0, [self.env.ref('base.group_user').id])]})
        self.assertIsNone(self.AuthRequest._consume(req.req_token, thief))
        # The owner's request must survive a stolen-token probe.
        self.assertEqual(
            self.AuthRequest.sudo().search_count([('req_token', '=', req.req_token)]), 1)
        self.assertTrue(self.AuthRequest._consume(req.req_token, self.user))

    def test_expired_is_refused(self):
        req = self._make(expires_at=fields.Datetime.subtract(fields.Datetime.now(), seconds=1))
        self.assertIsNone(self.AuthRequest._consume(req.req_token, self.user))

    def test_blank_token_refused(self):
        self.assertIsNone(self.AuthRequest._consume('', self.user))
        self.assertIsNone(self.AuthRequest._consume(None, self.user))


@tagged('post_install', '-at_install')
class TestParameterBounds(TransactionCase):

    def test_clamp_param(self):
        self.assertEqual(clamp_param('ont_mcp_oauth.access_token_ttl', 999999999), 86400)
        self.assertEqual(clamp_param('ont_mcp_oauth.access_token_ttl', 1), 60)
        self.assertEqual(clamp_param('ont_mcp_oauth.access_token_ttl', 900), 900)
        self.assertEqual(clamp_param('ont_mcp_oauth.refresh_grace_seconds', 99999), 60)
        # Garbage falls back to the low bound rather than exploding.
        self.assertEqual(clamp_param('ont_mcp_oauth.access_token_ttl', 'abc'), 60)
        self.assertEqual(clamp_param('ont_mcp_oauth.access_token_ttl', None), 60)

    def test_settings_reject_out_of_range(self):
        with self.assertRaises(ValidationError):
            self.env['res.config.settings'].create({'oauth_access_token_ttl': 999999999})
        with self.assertRaises(ValidationError):
            self.env['res.config.settings'].create({'oauth_refresh_grace_seconds': 3600})

    def test_settings_form_loads_hand_edited_value_clamped(self):
        # A value edited straight into System Parameters must not break the
        # shared General Settings page.
        self.env['ir.config_parameter'].sudo().set_param(
            'ont_mcp_oauth.access_token_ttl', '999999999')
        defaults = self.env['res.config.settings'].default_get(['oauth_access_token_ttl'])
        self.assertEqual(defaults['oauth_access_token_ttl'], 86400)


@tagged('post_install', '-at_install')
class TestSelfServiceRevocation(TransactionCase):
    """A user must be able to revoke their own AI connection, and only theirs."""

    def setUp(self):
        super().setUp()
        self.Consent = self.env['ont_mcp_oauth.consent']
        self.client = self.env['ont_mcp_oauth.client'].create({
            'name': 'Connector', 'redirect_uris': 'https://c.example/cb'})
        group_user = self.env.ref('base.group_user').id
        self.alice = self.env['res.users'].create({
            'name': 'Alice', 'login': 'ont_alice', 'groups_id': [(6, 0, [group_user])]})
        self.bob = self.env['res.users'].create({
            'name': 'Bob', 'login': 'ont_bob', 'groups_id': [(6, 0, [group_user])]})
        for user in (self.alice, self.bob):
            self.Consent.create({
                'user_id': user.id, 'client_id': self.client.id, 'scope_read': True})

    def test_preferences_view_exposes_the_panel(self):
        # Regression: the panel used to exist only on the admin Users form, so
        # no non-admin could reach it and self-revocation was impossible.
        view = self.env.ref('ont_mcp_oauth.view_users_preferences_oauth_consents')
        self.assertEqual(view.inherit_id, self.env.ref('base.view_users_form_simple_modif'))
        self.assertIn('oauth_consent_ids', view.arch)

    def test_field_is_self_readable_but_not_self_writeable(self):
        users = self.env['res.users']
        self.assertIn('oauth_consent_ids', users.SELF_READABLE_FIELDS)
        self.assertNotIn('oauth_consent_ids', users.SELF_WRITEABLE_FIELDS)

    def test_user_sees_and_revokes_only_their_own(self):
        as_alice = self.Consent.with_user(self.alice)
        visible = as_alice.search([])
        self.assertEqual(len(visible), 1)
        self.assertEqual(visible.user_id, self.alice)
        visible.action_revoke()
        self.assertFalse(visible.exists())
        # Bob's grant is untouched.
        self.assertEqual(
            self.Consent.sudo().search_count([('user_id', '=', self.bob.id)]), 1)

    def test_user_cannot_reach_another_users_consent(self):
        bobs = self.Consent.sudo().search([('user_id', '=', self.bob.id)])
        self.assertFalse(
            self.Consent.with_user(self.alice).search([('id', '=', bobs.id)]))


@tagged('post_install', '-at_install')
class TestConfidentialClientSecret(TransactionCase):

    def setUp(self):
        super().setUp()
        self.Client = self.env['ont_mcp_oauth.client']

    def test_confidential_client_starts_without_a_secret(self):
        client = self.Client.create({
            'name': 'Conf', 'redirect_uris': 'https://c.example/cb',
            'client_type': 'confidential'})
        # Visible in the form so the broken state is not silent.
        self.assertFalse(client.has_secret)
        self.assertFalse(client._verify_secret('anything'))

    def test_regenerate_secret_makes_it_usable(self):
        client = self.Client.create({
            'name': 'Conf', 'redirect_uris': 'https://c.example/cb',
            'client_type': 'confidential'})
        action = client.action_regenerate_secret()
        secret = action['params']['message'].strip().rsplit('\n', 1)[-1].strip()
        self.assertTrue(secret)
        client.invalidate_recordset()
        self.assertTrue(client.has_secret)
        self.assertTrue(client._verify_secret(secret))
        self.assertFalse(client._verify_secret(secret + 'x'))

    def test_regenerate_rotates_the_old_secret_out(self):
        client = self.Client.create({
            'name': 'Conf', 'redirect_uris': 'https://c.example/cb',
            'client_type': 'confidential'})
        first = client.action_regenerate_secret()['params']['message'].strip().rsplit('\n', 1)[-1].strip()
        client.action_regenerate_secret()
        self.assertFalse(client._verify_secret(first))

    def test_public_client_refuses_secret(self):
        client = self.Client.create({
            'name': 'Pub', 'redirect_uris': 'https://c.example/cb'})
        with self.assertRaises(UserError):
            client.action_regenerate_secret()


@tagged('post_install', '-at_install')
class TestOffboardingRevokesAccess(TransactionCase):
    """Deactivating or demoting a user must cut off their AI connectors."""

    def setUp(self):
        super().setUp()
        self.Token = self.env['ont_mcp_oauth.token']
        self.client = self.env['ont_mcp_oauth.client'].create({
            'name': 'Connector', 'redirect_uris': 'https://c.example/cb'})
        self.user = self.env['res.users'].create({
            'name': 'Leaver', 'login': 'ont_leaver',
            'groups_id': [(6, 0, [self.env.ref('base.group_user').id])]})

    def _token(self, kind='refresh'):
        return self.Token.create({
            'kind': kind, 'token_hash': f'h-{kind}-{self.user.id}',
            'family_id': 'fam1', 'client_id': self.client.id,
            'user_id': self.user.id, 'scope': 'mcp:read',
            'expires_at': fields.Datetime.add(fields.Datetime.now(), days=30)})

    def test_deactivating_revokes_live_tokens(self):
        token = self._token()
        self.assertFalse(token.revoked)
        self.user.write({'active': False})
        self.assertTrue(token.revoked, 'offboarding must revoke live tokens')

    def test_demotion_to_portal_revokes(self):
        token = self._token()
        self.user.write({'groups_id': [(6, 0, [self.env.ref('base.group_portal').id])]})
        self.assertFalse(self.user._is_internal())
        self.assertTrue(token.revoked)

    def test_ordinary_write_does_not_revoke(self):
        token = self._token()
        self.user.write({'name': 'Still Here'})
        self.assertFalse(token.revoked)


@tagged('post_install', '-at_install')
class TestIneligibleUserCannotRefresh(OAuthHttpCase):
    """The token endpoint must refuse an offboarded account, not just /mcp."""

    def test_refresh_refused_after_deactivation(self):
        client, _v, _code, tok = self.full_flow()
        self.assertIn('refresh_token', tok)
        self.oauth_user.write({'active': False})
        response = self.token_request(
            grant_type='refresh_token', refresh_token=tok['refresh_token'],
            client_id=client.client_id)
        self.assertEqual(response.json().get('error'), 'invalid_grant', response.text)
        # And the family is burned, so re-enabling the account does not
        # silently resurrect the connector.
        row = self.Token.sudo().search([('family_id', '!=', False),
                                        ('user_id', '=', self.oauth_user.id)], limit=1)
        self.assertTrue(row.revoked)


@tagged('post_install', '-at_install')
class TestSigningKeyRetirement(TransactionCase):

    def setUp(self):
        super().setUp()
        self.Key = self.env['ont_mcp_oauth.signing.key']
        # The partial unique index allows at most one active key DB-wide, so
        # clear whatever earlier flows committed before creating our own.
        self.Key.with_context(active_test=False).search([]).unlink()

    def _retired_key(self, kid, valid_for_seconds):
        """A retired key whose grace window ends in ``valid_for_seconds``."""
        return self.Key.create({
            'kid': kid, 'private_pem': 'x', 'public_pem': '', 'active': False,
            'retire_until': fields.Datetime.add(
                fields.Datetime.now(), seconds=valid_for_seconds),
        })

    def test_rotation_keeps_the_old_key_resolvable(self):
        # THE regression: `active` is Odoo's magic archive field, so a bare
        # search() hides retired keys. That turned every rotation into an
        # instant outage - tokens signed by the outgoing key died with
        # unknown_kid, and the JWKS dropped it so nobody could verify them.
        self.Key.create({
            'kid': 'ont_outgoing', 'private_pem': 'x', 'public_pem': '', 'active': True})
        self.Key._rotate()
        found = self.Key._find_by_kid('ont_outgoing')
        self.assertTrue(found, 'a retired key must still resolve for verification')
        self.assertFalse(found.active)

    def test_rotation_keeps_the_old_key_in_the_jwks(self):
        priv, pub = jwt_tools.generate_rsa_keypair()
        self.Key.create({
            'kid': 'ont_jwks_old', 'private_pem': priv, 'public_pem': pub, 'active': True})
        self.Key._rotate()
        kids = [k['kid'] for k in self.Key._get_jwks()['keys']]
        self.assertIn('ont_jwks_old', kids)

    def test_jwks_survives_an_unparseable_key(self):
        # Discovery must not 500 because one row has a corrupt PEM.
        priv, pub = jwt_tools.generate_rsa_keypair()
        self.Key.create({
            'kid': 'ont_good', 'private_pem': priv, 'public_pem': pub, 'active': True})
        self.Key.create({
            'kid': 'ont_corrupt', 'private_pem': 'x', 'public_pem': 'not-a-pem',
            'active': False, 'retire_until': fields.Datetime.add(
                fields.Datetime.now(), seconds=600)})
        kids = [k['kid'] for k in self.Key._get_jwks()['keys']]
        self.assertIn('ont_good', kids)
        self.assertNotIn('ont_corrupt', kids)

    def test_rotation_stamps_a_deadline_from_the_current_ttl(self):
        self.env['ir.config_parameter'].sudo().set_param(
            'ont_mcp_oauth.access_token_ttl', '900')
        key = self.Key.create({
            'kid': 'ont_stamped', 'private_pem': 'x', 'public_pem': '', 'active': True})
        self.Key._rotate()
        key.invalidate_recordset()
        self.assertTrue(key.retire_until)
        window = (key.retire_until - fields.Datetime.now()).total_seconds()
        # 900s TTL + 1h leeway, allowing for clock drift during the test.
        self.assertAlmostEqual(window, 4500, delta=120)

    def test_deadline_is_not_recomputed_when_ttl_shrinks(self):
        # Tokens were issued under the old, longer TTL; hardening the TTL
        # afterwards must not retroactively kill them.
        self.env['ir.config_parameter'].sudo().set_param(
            'ont_mcp_oauth.access_token_ttl', '86400')
        key = self.Key.create({
            'kid': 'ont_long', 'private_pem': 'x', 'public_pem': '', 'active': True})
        self.Key._rotate()
        key.invalidate_recordset()
        original = key.retire_until
        self.env['ir.config_parameter'].sudo().set_param(
            'ont_mcp_oauth.access_token_ttl', '900')
        self.Key._gc_signing_keys()
        self.assertTrue(key.exists(), 'shrinking the TTL must not reap a live key')
        self.assertEqual(key.retire_until, original)

    def test_within_window_is_kept_and_past_window_is_reaped(self):
        live = self._retired_key('ont_live', 600)
        dead = self._retired_key('ont_dead', -600)
        self.Key._gc_signing_keys()
        self.assertTrue(live.exists())
        self.assertFalse(dead.exists())

    def test_unstamped_legacy_key_gets_a_deadline_before_reaping(self):
        legacy = self.Key.create({
            'kid': 'ont_legacy', 'private_pem': 'x', 'public_pem': '', 'active': False})
        self.assertFalse(legacy.retire_until)
        self.Key._gc_signing_keys()
        self.assertTrue(legacy.exists(), 'must not be deleted on first sight')
        self.assertTrue(legacy.retire_until)

    def test_unstamped_deadline_derives_from_retirement_not_now(self):
        # An old unstamped row must not be handed a fresh full window; its
        # deadline comes from write_date, so it is stamped and reaped at once.
        old = self.Key.create({
            'kid': 'ont_ancient', 'private_pem': 'x', 'public_pem': '', 'active': False})
        self.env.cr.execute(
            'UPDATE ont_mcp_oauth_signing_key SET write_date = %s WHERE id = %s',
            (fields.Datetime.subtract(fields.Datetime.now(), days=30), old.id))
        old.invalidate_recordset()
        self.Key._gc_signing_keys()
        self.assertFalse(old.exists(), 'a long-retired key must not be revived')

    def test_retired_key_cannot_be_resurrected(self):
        # A key is often retired *because* it leaked. Un-archiving it put it
        # back in the signing pool and cleared its deadline, so it could never
        # be collected - and left two keys active at once.
        key = self._retired_key('ont_resurrect', 600)
        with self.assertRaises(UserError):
            key.write({'active': True})
        self.assertFalse(key.active)
        self.assertTrue(key.retire_until, 'the deadline must survive the attempt')

    def test_integer_boolean_still_stamps_a_deadline(self):
        # JSON-RPC callers send 0/1, and `0 is False` is False - the identity
        # check used to retire a key without ever stamping a deadline.
        key = self.Key.create({
            'kid': 'ont_intbool', 'private_pem': 'x', 'public_pem': '', 'active': True})
        key.write({'active': 0})
        self.assertFalse(key.active)
        self.assertTrue(key.retire_until)

    def test_second_retire_does_not_extend_the_deadline(self):
        key = self._retired_key('ont_twice', 600)
        original = key.retire_until
        key.write({'active': False})
        self.assertEqual(key.retire_until, original)

    def test_mixed_recordset_stamps_only_the_newly_retired(self):
        already = self._retired_key('ont_already', 600)
        original = already.retire_until
        live = self.Key.create({
            'kid': 'ont_live_one', 'private_pem': 'x', 'public_pem': '', 'active': True})
        (already | live).write({'active': False})
        self.assertEqual(already.retire_until, original)
        self.assertTrue(live.retire_until)

    def test_past_deadline_key_stops_verifying_before_gc_runs(self):
        # Expiry must not depend on autovacuum having run.
        expired = self._retired_key('ont_past', -60)
        self.assertFalse(expired.exists() and self.Key._find_by_kid('ont_past'))
        self.assertNotIn('ont_past', [k['kid'] for k in self.Key._get_jwks()['keys']])
        self.assertTrue(expired.exists(), 'still on disk, just no longer usable')

    def test_revoke_now_deletes_retired_key(self):
        key = self._retired_key('ont_leaked', 600)
        key.action_revoke_now()
        self.assertFalse(key.exists())

    def test_revoke_now_refuses_active_key(self):
        active = self.Key.create({
            'kid': 'ont_active', 'private_pem': 'x', 'public_pem': '', 'active': True})
        with self.assertRaises(UserError):
            active.action_revoke_now()
        self.assertTrue(active.exists())
