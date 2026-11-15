-- Executed automatically by Odoo's database neutralization (odoo.sh runs it on
-- every staging/dev copy of production; `odoo-bin neutralize` runs it locally).
--
-- A staging branch copies the production database, which would otherwise carry
-- over the LIVE OAuth trust material: the RSA private signing keys (so a
-- production access token replayed at staging would verify until it expires),
-- every token family, pending codes/consents, and — because web.base.url is
-- frozen in production — discovery metadata pointing OAuth clients at the
-- PRODUCTION endpoints. Wipe it all: staging must never be able to validate or
-- mint production credentials.

DELETE FROM ont_mcp_oauth_token;
DELETE FROM ont_mcp_oauth_code;
DELETE FROM ont_mcp_oauth_authrequest;
DELETE FROM ont_mcp_oauth_signing_key;

-- Keep client registrations and consents (harmless without keys/tokens), but
-- make sure nobody can self-register clients against a neutralized copy.
UPDATE ir_config_parameter
   SET value = 'False'
 WHERE key = 'ont_mcp_oauth.dcr_enabled';
