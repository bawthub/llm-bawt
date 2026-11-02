# Scoped Crawl4AI bearer rekey

This is a **source-only procedure**, not authorization or a live rotation receipt.
Use the existing settings API, exporter and named-service operations. Do not
rotate the provider encryption root, provider OAuth, unrelated derived labels,
bridge gate tokens, or manually edit shared bearer files. No new secret store,
platform, hot-reload loop, generic executor or catalog action is needed.

## Mechanism and compatibility

`crawl4ai.token_generation` is a **nonsecret global integer** in 0..2147483647,
written through `PUT /v1/settings` (or its existing batch route). For example:

```json
{"scope_type":"global","key":"crawl4ai.token_generation","value":1}
```

Read back this key alone through the existing filtered settings API. Record the
old/new counters, not bearer values, prefixes, hashes, root material or request
headers. Before the first rotation, no row and generation zero retain the exact
`crawl4ai-api-token/v1` derivation. A positive counter adds `/generation/<n>` to
that label only. Use a fresh unused increasing counter for each rekey; deleting
the row or reusing a counter can restore a previously compromised bearer.

Only the exact global row is read; bot/request overrides and resolver caches do
not participate. Invalid JSON/type/range, ambiguous rows and configured DB
failure do not fall back to zero. Standalone use with no DB configured retains
legacy behavior. At app startup a derivation failure withdraws stale exported
files through the existing exporter. An explicit command failure returns false;
it never repairs the setting or silently rolls back to an earlier generation.

**Updating the setting is not publication, activation or revocation.** The app
normally exports at startup. A separately authorized one-time command inside
the app's existing environment can publish without restarting the app:

```
python -m llm_bawt.service.crawl4ai_rekey export <expected-generation>
```

Arguments are public counters only. No token, root, env dump, shell expansion,
Authorization header or copied credential may enter argv, stdin scripts, logs,
chat, task artifacts or process listings. The command derives once, uses the
existing atomic-0600/reader-UID publisher for both configured targets and emits
only booleans. Both targets must report written and the DB counter must still
match. The two replacements are individually atomic, **not a transaction across
files/services**. A false result can mean partial publication; reconcile it
before another action. No automatic retries or rollback are performed.

## Smallest coordinated window (owner/operator, separate authority)

1. Publish the reviewed source through the normal integrator. Verify the exact
   exporter/probe source is available in the app's supported mounted release or
   image; do not point a live service at an author's worktree. Existing loaded
   app code is not assumed to hot-reload. Before the next app startup it must
   include the new generation-aware exporter, otherwise it can overwrite files
   with generation zero. Source/image activation remains separately authorized.
2. Discover current enabled catalog rows and named targets. Select the existing
   approval path from an unaffected approver; a possibly exposed capability is
   not authority to approve its own remediation. A scoped card is the permission
   boundary, not a second duplicate consent request. No blanket cutover-plan,
   broad app/bridge/Redis restart, tag/profile change or provider rotation.
3. Checkpoint affected consumers and arrange bounded quiet points for each
   bridge that actually caches Crawl4AI MCP headers. Do not require idle bots
   with no affected process. Keep the interrupting bridge action last and use
   durable callbacks. Preflight existing server/bridge recovery/start actions,
   export directory mounts (not inode-pinning single-file mounts), reader UID
   access and the stable persisted provider root without printing any contents.
   If a non-file legacy override or unrelated custom MCP credential is in use,
   stop and disposition that exact consumer; do not rewrite it blindly.
4. Record the previous public counter and choose an unused new counter. Write
   and read back the new global setting. Under the reviewed manual maintenance
   authority, execute the fixed exporter command **inside the app environment**
   (operator may use a named-container fixed-argv exec; never generic repository
   shell execution through ops). It reads existing configured DB/root/files;
   do not fetch/copy credentials through the host. Require all booleans true.
   Do not run the command just because source review passed.
5. Reload the named Crawl4AI server with its enabled scoped action, e.g.
   `llm-bawt.restart-aux {"service":"crawl4ai"}`. It reads the server export at
   boot. A config reconcile may be **verify-only** and must not be substituted
   for a proven reload. If container configuration itself requires correction,
   separately use `llm-bawt.reconcile-crawl4ai-config` under its own card; it
   binds the current image, not an upgrade. Check its actual terminal receipt.
6. Verify using the fixed app-environment command:

   ```
   python -m llm_bawt.service.crawl4ai_rekey probe <previous-generation> <expected-generation>
   ```

   It derives old/new in memory, checks both export files against the new value,
   and sends three bounded GETs to the configured server's protected `/schema`:
   new must return 200, old and anonymous must each return 401 or 403. Redirects,
   timeouts, errors or a changed counter are not proof. It emits booleans only,
   never response bodies, URLs, credentials, exception text or tracebacks. This
   proves only the specified previous generation is rejected; it is not a
   survey of other unknown credentials or all endpoints.
7. App web-fetch rereads its file on each call. Already-loaded Claude MCP
   servers are cached in `self._mcp_servers` and copied into options; Codex
   registrations are also bootstrapped into config. Rewriting the token file,
   starting a new conversation, or inspecting a new config file does **not**
   prove the running MCP transport refreshed. Use the existing scoped
   `llm-bawt.restart-bridge {"bridge":"<affected-bridge>"}` at that bridge's
   quiet point to rerun its bootstrap, then exercise its actual Crawl4AI MCP
   connection with one harmless tool call and record only success/failure.
   Refresh each actually-cached bridge individually; do not infer one covers
   both or add a hot-reload system to avoid the normal maintenance window.
8. Run the app's existing `web_fetch` cutover probe (protected `/schema`, no
   external crawl), require success, and retain boolean/terminal job receipts
   for every affected server/app/bridge consumer. A direct probe from a new
   subprocess alone is not evidence of an existing bridge transport's headers.
   If a bridge cannot be safely refreshed, report a partial cutover and retain
   its concrete quiet-point dependency; do not call remediation complete.

## Partial failure and recovery

- **Setting write/readback uncertain:** query that single nonsecret key; reconcile
  the outcome before publishing. Do not assume the previous generation.
- **One file failed:** preserve the chosen counter. Correct that target's
  mount/UID/access problem under applicable authority, then explicitly republish
  the **same** generation to both targets and recheck. No new counter is needed.
- **Server restart failed/unknown:** reconcile the job/container once, use its
  documented recovery/start path, and keep both exports at the chosen counter.
  Restarting an old server container still reads its configured server export;
  prove its accepted generation, not just its health URL. Health alone says
  nothing about revocation.
- **One bridge remains cached:** keep the new counter/server, checkpoint and
  recover only that bridge through its existing operation. Temporary Crawl4AI
  failures are preferable to silently restoring a known exposed key. Other
  tools/services/root credentials do not need rotation.
- **Code rollback:** preserve generation-aware export behavior or hold app
  startup/export until that is possible. Rolling back to the old fixed-label
  exporter can silently republish zero; it is not a safe remediation rollback.
- **Operator elects old-key restoration:** this is a separate, explicitly
  approved containment tradeoff. Record that the exposed credential is accepted
  again and remediation is **not complete**; coordinate setting, both exports,
  server and actual caches consistently, then plan a fresh-generation cutover.
  Never describe availability restoration as revocation.

Final evidence is exact source pin/review/tests plus chosen nonsecret counters,
per-action terminal IDs, export/probe booleans and actual consumer receipts.
Source publication/unit tests alone do not prove server rejection or completion.
