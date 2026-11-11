# External integration connections (TASK-906)

Separate from inference providers. Current catalog: `google-keep`, official
Workspace Keep API, read/write authorization. No polling, task import, note mutation,
or bot execution yet. The connection is prepared for supported future operations.

The official discovery document (`https://keep.googleapis.com/$discovery/rest?version=v1`)
exposes notes.create/delete/get/list, permission batchCreate/batchDelete, and media.download.
There is **no notes.update/patch or checklist-item mutation method**. Full Keep permission
supports reading, creating and deleting whole notes, not editing existing checklist items.
Do not substitute delete-and-recreate for list editing: it changes IDs and risks data loss.

## Workspace Keep setup (current supported UI)

Keep is explicitly Workspace-only. The integration registry remains generic;
`auth_method=service_account_delegation` selects its setup card, while the OAuth
card remains available for redirect-based integrations. Existing Web OAuth client
configuration is retained; it is not reused as a service-account credential.

1. Select the intended Google Cloud project. Enable Google Keep API.
2. IAM & Admin → Service Accounts → Create service account, e.g. `bawthub-keep`.
   No Cloud project IAM role is needed solely for Workspace Keep delegation.
3. Service account → Keys → Add key → Create new key → JSON. If organization
   policy blocks key creation, stop; don't weaken policy or try alternative key
   creation commands to bypass it.
4. BawtHub → Tools → Integrations → Google Keep · Workspace. Paste JSON directly
   into the protected configuration form (never chat) and enter the Workspace
   user email. Save. The private key is encrypted with the existing credential
   store and never returned; the form clears it after save.
5. Copy the numeric service-account client ID displayed by BawtHub. Workspace
   Admin → Security → Access and data control → API controls → Manage domain-wide
   delegation → Add new. Grant only `https://www.googleapis.com/auth/keep`.
   This is NOT the Web OAuth client ID. No OpenID/email scopes or callback URI
   are needed for this route. Changes may take up to 24 hours to propagate.
6. Verify Keep access. The backend signs an RS256 assertion with fixed Google
   audience, service-account issuer, the configured user's subject, and Keep
   scope; it exchanges it for a short-lived access token, then calls notes.list.
   No browser redirect, refresh token, or note mutation. Only successful API
   access creates a connected record. A 403 still needs entitlement/policy diagnosis.

**Authority:** delegation can impersonate domain users within its granted scope.
The configured single-user restriction is application behavior, not an IAM bound.
The Workspace-only direction was approved after that distinction was explained.

Routes: PUT `/v1/integrations/{id}/workspace` (service_account_json, subject),
POST `/v1/integrations/{id}/workspace/verify`. Reuse the existing same-origin proxy.
`integration_connection:workspace:google-keep` contains public service-account
identity/target and encrypted private key. Disconnect deletes this key as well as
connection tokens. Changed Workspace configuration invalidates its previous
verified connection. In-flight verification uses the atomic pending-flow claim;
a disconnect/reconfigure/new attempt prevents stale completion.

Read-only note discovery for TASK-907: GET `/v1/integrations/google-keep/notes`
uses a fresh delegated token, paginates at most 20 pages, returns note resource
names, titles, types and checklist item counts. GET `.../notes/{note_id}` returns
only unchecked, flat item texts for an explicitly named list; it never writes
Google notes or BawtHub tasks. In live discovery on 2026-09-24, `notes.list`
returned four notes (one list, three text). That list initially had zero items;
a voice-created test item subsequently appeared as one unchecked item in both
`notes.list` and `notes.get`. Google Keep ListItem has `text`, `checked` and nested
child items, but no stable per-item ID. The selected-list import is opt-in and
baselines existing unchecked texts before manually importing future new texts.
Exact trimmed text in one list is one logical request forever; identical repeat
texts are not imported twice, and editing text creates a new logical request.
Nested items are ignored until their semantics are explicitly designed. No bot
is assigned or dispatched by this slice. The BawtHub receipt and task are created
atomically; sync is manual and duplicate receipt checks run under a Postgres
transaction-level advisory lock. No polling or automatic bot delivery yet.

This slice does not run background refresh or import notes. Future consumers must
use an app-owned token service, obtaining fresh signed-assertion access tokens
from the stored configuration; they must not treat the verification token as
permanent or accept arbitrary per-request impersonation subjects.

## Legacy redirect UI and protocol (retained, not the Keep default)

Tools → Integrations (also Settings → Integrations), or the optional Integrations
panel on setup's Providers step. Add integration → Google Keep. First configure
the deployment's Google OAuth **Web application** client; subsequent connections
use Continue with Google → consent → callback → verified connection persisted.

The OAuth application setup cannot be created by a user consent redirect. The
client ID, secret, registered callback and Workspace admin permissions must exist.
Do not reuse Google Home's `bawthub` account-linking client: it is an OAuth client
of *our* server, not a Google-issued OAuth application credential.

1. Use the Cloud project containing the integration's OAuth client. Enable the
   Google Keep API there and configure the intended Workspace audience. Do not
   assume the Google Home project is the project containing this OAuth client.
2. Create/use a Google Web application OAuth client. Register exactly:
   `https://app.example/api/chat/proxy/v1/integrations/google-keep/callback`
   after replacing `app.example` with the deployment's hostname. Each deployment
   hostname needs its corresponding registered URI.
   This slice supports one active callback origin and account per integration.
3. Read Google's official [Keep authorization overview](https://developers.google.com/workspace/keep/api/guides).
   It documents domain-wide delegation with an OAuth client ID: the administrator
   approves scopes but each user authenticates and consents. Workspace edition,
   API entitlement and organization policy can still block access. Do not promise
   all Workspace subscriptions can use this API because Keep's UI works.
4. A Workspace super administrator opens Security → Access and data control →
   API controls → **Manage domain-wide delegation** → Add new. Enter the integration
   OAuth client ID (not the network login client) and these comma-separated scopes:
   `https://www.googleapis.com/auth/keep,openid,https://www.googleapis.com/auth/userinfo.email`.
   This exactly matches the authorization request. Keep is read/write, including
   permanent note deletion; OpenID/email identify the signed-in account. The
   Configured apps access-policy list is not domain-wide delegation. Changes can
   take up to 24 hours. Administrator preapproval remains mandatory. The flow
   always uses `prompt=select_account`. Live testing on 2026-09-24 confirmed that
   forcing `prompt=consent select_account` returns `invalid_scope` for Keep even
   after the same client previously completed token exchange. Never reintroduce
   forced consent as refresh-token recovery. The first Keep 403 discarded an
   issued refresh token in the original implementation; retention is now fixed,
   but recovery of that lost grant and the original API denial remain unresolved.
   Do not ask users to cycle retries or revoke a shared app grant without a
   verified recovery path.
   Existing read-only connections are flagged for reauthorization, not silently upgraded.
5. Enter the client credentials in Configure. They are encrypted server-side;
   the secret is never returned. Blank secret preserves it only for the same ID.
6. Continue with Google from the same origin as the configured callback. The
   browser must be able to reach that origin after consent (dev is LAN-only).
7. Callback verifies token scopes, offline refresh access, verified Google account,
   and an actual read-only Keep `notes.list(pageSize=1)` call. Note contents are
   discarded. Only then is the connection committed and shown as connected.

Google errors produce bounded error codes/instructions; never save a nominally
connected record merely because account login worked. A failed reconnect keeps
the prior credential. Revoking access through Google's account settings remains
available; Disconnect locally deletes stored tokens and invalidates pending auth.

## Live acceptance blocker and documented alternative (2026-09-24)

Observed in this deployment, not inferred from mocked tests:

- The OAuth client with `prompt=select_account` completed token exchange, full
  Keep-scope validation and identity verification, then Keep returned HTTP 403.
- The initial implementation discarded the refresh token on that 403. No usable
  credential remains in either the connected or unverified store.
- Forcing `prompt=consent select_account` subsequently returned `invalid_scope`.
  Do not oscillate between these prompts and ask the operator to retry blindly.
- The original 403 reason remains unknown. No Workspace-edition diagnosis has
  been established. A different authentication method cannot guarantee entitlement.

Google's official [Keep Java quickstart](https://developers.google.com/workspace/keep/api/guides/java)
uses a service account authorized through Workspace domain-wide delegation. The
[service-account protocol](https://developers.google.com/identity/protocols/oauth2/service-account)
obtains short-lived access tokens from signed assertions, avoiding browser consent
and the lost refresh-token loop. The operator subsequently approved this Workspace-only
direction; it is implemented above but still awaits real service-account credentials
and a successful live Keep API test.

A domain-wide-delegated service account can impersonate domain users within its authorized scopes. Pinning
BawtHub to one account limits application behavior, not the credential's domain
privilege. Never silently add this route, grant broad Cloud roles, or repurpose
network-login credentials. A read-only `notes.list` acceptance test must still
succeed before claiming Keep API access works.

## Persistence / future consumer contract

Uses the existing runtime_settings table + provider Fernet encryption. No new
schema migration or env-file credentials. Keys, global deployment scope:

- `integration_connection:client:google-keep`: public client ID/callback, encrypted client secret.
- `integration_connection:pending:google-keep`: one bounded encrypted pending flow,
  ten-minute expiry, hashed OAuth state and browser binding, PKCE verifier.
- `integration_connection:unverified:google-keep`: encrypted OAuth credentials retained
  when Keep verification/offline authorization fails, never advertised as connected.
  Matched by client ID, verified Google subject and required scope before reuse.
  Removed on successful connection, Disconnect or client configuration change.
  Writes use the same atomic callback claim as successful connections; a stale
  callback cannot restore them after disconnect. Do not advise revoking the whole
  Google app grant casually: clients share project branding and other logins may
  be affected. Probe Keep even when no refresh token is returned, so an offline
  error does not hide the underlying API denial.
- `integration_connection:google-keep`: connection state/account, Google subject,
  granted scopes, verification time, access-token expiry, encrypted access/refresh tokens.

`IntegrationConnections(config).descriptor('google-keep')` gives a safe public
view; backend consumers may use `IntegrationStore(config).load('google-keep')`.
Never expose `.secret` through generic config tools. `connected` means account
access was verified during authorization, not continuously monitored token health.
The later ingestion slice must add a single app-owned refresh/health service;
consumers must not independently refresh the stored credential or treat the
original access token as permanent. No scheduled refresh worker in this slice.

Public management routes under `/v1/integrations`: GET catalog, PUT `/{id}/client`,
POST `/{id}/connect`, GET `/{id}/callback`, DELETE `/{id}`. Browser routes use the
existing `/api/chat/proxy` prefix. The scoped Hono proxy requires same-origin HTTPS
mutations, forwards only the OAuth-binding cookie and Origin, preserves Set-Cookie
and redirects, and never follows callback redirects server-side. Configuration
is tenant-wide like existing provider accounts, behind the existing deployment
boundary; it is not per-user multi-tenant authorization.

OAuth state is DB-backed, single use, and bound to a Secure/HttpOnly/SameSite=Lax
cookie. Atomic claim plus conditional final commit prevents callback replay or
an in-flight callback resurrecting a disconnected/replaced flow. Starting a new
flow supersedes a prior tab; failed claims remain bounded to a single row.

Operations: app source reload needed to register backend routes; frontend is dev
HMR. No production frontend release implied. Do not log callback query strings
(codes/state) in external access logs. Actual Workspace consent and live Keep
access are separate acceptance checks; mocked Google tests cannot establish them.
