# TASK-902: Google Home capture-only experiment

Project: `bawthub-00a0bf`. This is a Cloud-to-cloud test, not Gemini Enterprise
Agent Platform; no Google model API key or ADC is needed.

## Current status — 2026-09-23

Implemented: disabled-by-default fulfillment, browser consent, one-use
OAuth authorization codes, access/refresh tokens and atomic unlink revocation.
The earlier hypothetical introspection adapter was replaced by a scoped local
issuer because the installed OAuth2-Proxy authenticates browsers but does not
issue downstream OAuth credentials. Existing browser login is reused; no new
password system or OAuth2-Proxy configuration changes are needed.

**Active (owner explicitly authorized):** runtime settings enabled
for the owner's exact sign-in email; encrypted app client credential provisioned and backed by
agent-vault item `integrations/google-home-probe`
(`63cb42e2-254c-47c2-be85-cb88c72f7b2d`). PostgreSQL linking schema created and
app-container imports/config verified. NPM host 37 exact-path rules applied via
API, `meta.nginx_online=true`; HTTP rate/log directives installed. NPM API reload
picked up these rules, so no manual/container NPM restart was needed. Main app
and `/v1/admin/` retain 302 SSO gating. App reload job
`39432de8d8644ffea4ff158d35fa2dd8` succeeded, exit 0; app healthy and `/health` 200.
No Google console setup, actual browser/Nest acceptance or commit yet.

Backups: Unraid `http_top.conf.bak-TASK-902-20260923` beside HTTP config; original
NPM host JSON `npm-host-37-before.json` in the deployment's probe backup directory.
Initial proxy apply rejected a new log file on permissions; corrected to reuse
`/config/log/proxy-host-37_access.log` with the query-free probe format; API
readback confirms successful configuration. No permissions loosened.

Live public checks passed: anonymous fulfillment 401 without redirect; wrong
client 401; valid authorization query shows login page 200; token GET 403;
oversized body 413. Synthetic admin-created authorization grant exercised actual
PostgreSQL + public edge: exchange 200, refresh 200, SYNC 200 with probe device,
DISCONNECT 200, revoked access 401 and revoked refresh 400 invalid_grant. All
smoke grants revoked. This is NOT browser consent or Google speech proof.

Existing Cloudflare edge rejects Python-urllib User-Agent with 403 code 1010;
curl, Mozilla and `Google` User-Agent reach handlers. No WAF changes made. Actual
Google linking still must be tested; if blocked, inspect the specific Cloudflare
event and make only a narrow approved endpoint exception, never disable site-wide
protection. NPM full configuration passes syntax validation (two unrelated
pre-existing CIDR warnings); test authorization state absent from proxy access log.

Next: the owner completes Google console linking fields and signs in through Google
Home; client secret is in the agent vault, never chat.

### Actual identity/ingress discovery (read-only)

Deployment addresses, identity values and installation paths are intentionally
omitted; discover the current values from the authorized deployment configuration.
The configuration example below is illustrative, not a live readback.

- OAuth2-Proxy-Bawthub on Unraid, configured internal address/port, Google provider.
- Config `oauth2_proxy.cfg` in the deployment's OAuth2-Proxy configuration directory.
- Cookie domain `.bawthub.com`, secure cookies, `set_xauthrequest=true`.
- Login: `https://auth.bawthub.com/oauth2/start`.
- Internal cookie check: configured OAuth2-Proxy origin plus `/oauth2/auth`; anonymous returns 401.
- Its configured allowed domain is NOT an owner identity allowlist.
  **The owner confirmed one exact email as the sole allowed sign-in email.**
  The owner explicitly authorized activation and required app/proxy reloads.
- NPM app host ID 37, certificate ID 17, `app.bawthub.com`, existing upstream
  address/port from the host configuration. Keep it unchanged; add only the three exact path exceptions.
- Authelia docs are stale for BawtHub. No fake Authelia issuer is used.

## Implemented contract

| Path under `/integrations/google-home` | Method | Authentication |
|---|---|---|
| `/authorize` | GET | Existing SSO cookie verified directly with OAuth2-Proxy |
| `/authorize` | POST | Verified SSO + consent cookie + one-use nonce + exact Origin |
| `/token` | POST | Exact client ID + constant-time secret validation in form body |
| `/fulfillment` | POST | Short-lived opaque Bearer access token |

All routes return 503 while disabled/unconfigured. Authorization presents a
BawtHub login link if needed, then an explicit Agree and link / Cancel screen.
Browser-supplied identity headers are never trusted. Only the configured exact
email may link, mapped to the explicitly configured local subject.

Only project-bound HTTPS Google production/sandbox redirect URIs are allowed:

- `https://oauth-redirect.googleusercontent.com/r/bawthub-00a0bf`
- `https://oauth-redirect-sandbox.googleusercontent.com/r/bawthub-00a0bf`

State is preserved. Consent expires after 10 minutes; authorization codes after
5 minutes, single-use; access tokens after 1 hour. Refresh tokens remain stable
until unlink (Google's documented contract). Codes/tokens are random 256-bit
opaque values and only their SHA-256 hashes are stored. Client secrets use the
existing encrypted CredentialStore. No auth bodies or credentials are logged by
these handlers; sensitive responses are no-store/no-referrer and consent pages
forbid framing/scripts. Client credentials in HTTP Basic are deliberately not
supported: leave Google's Basic-auth checkbox OFF. PKCE parameters are rejected
rather than silently ignored; Google's documented web OAuth flow does not require
PKCE. This is a single-client experiment, not a general-purpose OAuth server.

`google_home_probe_links` serializes per-link mutations via PostgreSQL row locks.
`google_home_probe_tokens` persists consent/code/access/refresh hashes across
process restarts. Codes are consumed with atomic DELETE RETURNING. DISCONNECT
rechecks the bearer under the link lock, changes generation and deletes ALL
pending/issued credentials, independently of capture availability; returns empty
200. A stale DISCONNECT cannot revoke a subsequently linked generation.

Expired tokens are pruned on consent/token activity. Refresh grants persist until
unlink. Public ingress must enforce rate/body/time limits; this is not a
multi-user production authorization service.

## Capture behavior

SYNC advertises one virtual TV, **BawtHub TV**, with AppSelector and a known
application **Probe Baseline**. QUERY reports static test state. EXECUTE captures
fields but returns `functionNotSupported`: audible Google errors can coexist
with successful capture. Unknown devices return `deviceNotFound`.

No bot dispatch, chat execution, HA action, public capture reader or public admin
endpoint. Headers/tokens/device customData are not captured. Protocol fields,
including forwarded spoken parameters, are encrypted using existing provider
crypto in `google_home_probe_captures`. Duplicate subject/request ID preserves
first capture; max body 32 KiB, max 1,000 rows. Older-than-24h captures are hidden
from reads and pruned on the next capture write, NOT deleted on a timer while idle.
Internal inspection: `ProbeCaptureStore.recent()`; do not expose a reader publicly.

## Runtime configuration example (illustrative)

Global setting `google_home_probe`; substitute deployment-owned values before use.
The reserved example email and service hostname below are not installed settings:

```json
{
  "enabled": true,
  "project_id": "bawthub-00a0bf",
  "client_id": "bawthub",
  "allowed_subject": "example-owner",
  "allowed_email": "owner@example.com",
  "required_scope": "bawthub.home",
  "public_origin": "https://app.bawthub.com",
  "sso_auth_url": "http://oauth2-proxy.example:4180/oauth2/auth",
  "sso_signin_url": "https://auth.bawthub.com/oauth2/start"
}
```

Generate a 32+ byte random client secret only at provisioning time, put it in
`CredentialStore` provider `google-home-probe`, `secret.client_secret`, status
`connected`, and deliver to the owner through the established secret store for console
entry. Never put it in source, docs, shell arguments, tool output or chat.

## Public activation runbook — applied; browser acceptance pending

Applied configuration templates:

- `docker/google-home-probe-http.conf.example`: rate zone + query-free log format.
- `docker/google-home-probe-server.conf.example`: three exact path locations.

Standalone `nginx -t` using the real NPM binary passed. This does not establish
full-config merge compatibility or live external behavior.

At authorized activation:

1. Confirm owner email; provision secret/settings with probe disabled.
2. Read ops runbook and discover live operation catalog; app code needs a targeted
   app reload, no dependency rebuild. Preserve/review other bots' working-tree edits.
3. Back up persistent NPM config. Merge HTTP directives into `http_top.conf`;
   merge path locations into ONLY host 37 via NPM API, preserving existing host
   fields/advanced config. NEVER replace generated proxy_host/37.conf directly or
   put these routes into the global all-host include.
4. Validate merged Nginx config and API `meta.nginx_online`; apply using approved
   ops action when available. Do not bypass unsupported/denied operations. NPM's
   documented custom-include changes may require full container restart; obtain
   explicit authorization for manual runbook fallback if catalog doesn't cover it.
5. Enable config and verify public TLS/path behavior. These three paths use
   endpoint-level authentication; everything else keeps existing SSO. No broad
   `/integrations/`, `/v1/`, `/docs`, `/openapi.json` or app-port exposure.
6. Verify unauthorized fulfillment gives 401, not login redirects; wrong client
   gets OAuth error; unsupported methods fail; oversized/rate-limited requests
   fail; public arbitrary `/v1/*` is not forwarded to llm-bawt.
7. Complete browser login/consent and token exchange through the actual edge.
   Verify unlink rejects access AND refresh; confirm access logs contain no
   tokens/body/query strings. Backend Uvicorn may log authorization request query
   (client ID/state, not code/token); never log response Location or bodies.

## Google Home console values — live URLs

| Field | Value |
|---|---|
| Project | `bawthub-00a0bf` |
| Integration | Operator-chosen display name |
| Flow | OAuth authorization code |
| Authorization URL | `https://app.bawthub.com/integrations/google-home/authorize` |
| Token URL | `https://app.bawthub.com/integrations/google-home/token` |
| Fulfillment URL | `https://app.bawthub.com/integrations/google-home/fulfillment` |
| Client ID | `bawthub` |
| Client secret | Agent vault item `integrations/google-home-probe`, password field |
| Scope | `bawthub.home` |
| Send client credentials via Basic auth | OFF (form body) |
| App Flip / seamless linking | OFF; use web OAuth |

Google's September 2026 linking migration leaves this core web OAuth contract
unchanged. Its iOS `/a/com.google.Chromecast` redirect is documented for App Flip
allowlists; do not add it to this web-only project-bound experiment unnecessarily.

## Device acceptance — owner must perform

Link the integration in the **Google Home app**, not Google Assistant settings. Use the
account controlling the Nest speakers; developer/tester access must be available
to that account. No access to the owner's developer console is established here.

1. Verify authenticated SYNC exposes only the probe.
2. Known baseline: "Open Probe Baseline on BawtHub TV."
3. Try app search/select with several unfamiliar multiword names; compare actual
   captures against exact utterances. AppSelector documents en-US support.
4. A hand-built HTTP EXECUTE is NOT evidence of speech forwarding. If Google
   rejects unfamiliar names before webhook, record that outcome. Arbitrary
   "ask <bot> to ..." routing remains unproven, not a promised feature.
5. Unlink in Google Home and verify revocation.

## Verification

Verified on echo: **69 passed** across `test_google_home_probe.py`,
`test_google_home_oauth.py`, and `test_provider_api_keys.py`; Ruff clean for all
probe source/tests; `git diff --check` clean. NPM standalone draft syntax passed.
Broader run: 74 passed / 1 failed in untouched
`test_schema_bootstrap_guard.py::test_turn_log_orchestration_does_not_repeat_nested_schema`:
its FakeConnection cannot be inspected by changed_files_store's migration. The
failure reproduces alone without importing/running the Google probe tests; not
modified as part of this task.

Focused tests: `uv run --no-sync pytest tests/test_google_home_probe.py tests/test_google_home_oauth.py -q`.
SQLite tests cover durable store recreation, code/consent replay, expiry, scope/
subject/client isolation, token-type confusion, CSRF, exact redirect allowlist,
SSO failure/spoofing, form limits, client authentication, full mocked link-refresh-
capture-unlink, and disabled defaults. PostgreSQL concurrency and actual
browser/edge/Nest acceptance remain activation-time checks.

References:
- https://developers.home.google.com/cloud-to-cloud/project/authorization
- https://developers.home.google.com/cloud-to-cloud/account-linking-migration
- https://developers.home.google.com/cloud-to-cloud/traits/appselector
- https://developers.home.google.com/cloud-to-cloud/intents/execute
