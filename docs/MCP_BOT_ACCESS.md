# MCP bot-data access (TASK-1222)

The registered callable is the enforcement boundary for ordinary and approved
MCP execution. An ordinary call also checks before approval processing, so a
denied target does not create an approval request. Approval execution re-reads
current policy; approval is not a private-data grant.

## Policy and callers

`bot_access.BotPolicyResolver(engine).resolve(caller=None)` reads a fresh,
bounded, read-only bot-tag snapshot (`bot_tags.TagSnapshot`, TASK-1329: every
profile with its tags, plus the catalog, in one statement). No code names a bot:
a bot is private when it carries the `private` tag, or while the legacy
`include_in_global_search=false` column still says so; a bot with no profile row
is private. An absent/invalid policy or tag snapshot, unknown trusted caller, or
ambiguous partition mapping fails closed. The result is immutable.
`readable_bots` (known, not private) gates targeted tools: `require_target(bot_id)`
authorizes a readable bot or trusted self. Unknown and private-other targets both
return `access_denied`. The resolver does not provision partitions.

`aggregate_scope(bot_id=None, require_user=False)` returns positive partitions:
`searchable_bots` (not private, not `no_global_search`; private implies
`no_global_search`) as GlobalScope UNION the caller's own CallerScope when the
caller is outside that set. An explicit bot filter only narrows this set. A memory aggregate requires a trusted user.
`PartitionScope` retains canonical bot slug separately from its SQL partition.

Trusted identity comes from verified task-turn capability, server-owned approved
caller context, or the purpose-separated service envelope. All supplied
credentials are checked; conflicting bot/user identities, blank/duplicate/bad
headers, and expired credentials are errors rather than anonymous fallback.
No tool argument establishes caller identity.

The HTTP `MemoryClient` signs each exact tool/argument invocation with its
construction-bound bot/user principal (purpose `mcp-service`, version 1, TTL
120 seconds). This cannot bypass ASK/DENY policy or mint task-turn authority.
Embedded cross-bot client methods resolve and bind the same principal inside
the async adapter. Ordinary bot-bound embedded persistence remains an internal
service API. Request-local bindings are always reset.

A verified **user-less service principal**, unlike a turn or approved caller,
retains operator read/close filters only for its own bot. The shared
`is_service_self(access, target)` predicate validates the caller, then requires
`kind == "service"`, `user_id is None`, and `target == caller.bot_id`.
`SERVICE_SELF_TOOLS` explicitly permits only `messages_get`, `sessions_get`,
`sessions_close`, `sessions_list`, and `sessions_get_active`;
`SERVICE_SELF_STORAGE_OPS` lists the corresponding storage operations. This
exception does not authorize another bot, bypass session bot ownership, accept
another bot's pointer, or satisfy the trusted-user requirement of aggregate
memory. An explicit user is a filter here, never an authorship claim. All
non-allowlisted operations keep their existing user pins (including message,
memory, fact and session-creation writes); user-bound service callers stay pinned.

All six `sessions_*` MCP tools require an explicit `bot_id`; the nonexistent
`default` namespace is no longer advertised as an optional scope. The app's
MemoryClient supplies its constructor-bound bot for every session invocation.

## Scope and SQL

The explicit inventory is `mcp_server/bot_access_manifest.py`; a registration
test requires every tool to have a classification. Per-bot memory, message,
profile, context and session tools authorize before executing their handler.
`self_fwd` authorizes its source. Session IDs are checked against the explicit
bot; session get/close also bind ownership in SQL. Private-bot session operations
pin the trusted user except for the allowlisted service-self filters above.
Session and source discovery lists use positive SQL
filters before LIMIT/counts. Receipt get/list constrain BOTH sender and target
before materializing rows, without routing guarded reads through operator REST.

Aggregate memory uses `MemoryProvenance(ScopedMessageAuthority.for_partition(
scope, bot_id))`. Every source ID must resolve in the same canonical partition,
with canonical session bot/user/status and the applicable effective-author rule.
Nonarrays, empty arrays, nonstring elements, missing sources and arrays longer
than 64 fail eligibility. Searchable memory uses GlobalScope; a private self uses
strict MESSAGE_AUTHORITY, not the broader self history authority. Both semantic
query branches apply eligibility before candidate LIMIT and access-count writes.
Tag filters are bound arrays in text and embedding searches, never interpolated
SQL: a tag cannot override the provenance predicate or other WHERE filters.
A backend failure raises `source_unavailable`; it cannot return empty or partial
aggregate success. Ranking/decay and existing date/relevance postfilters remain
unchanged. See TASK-1323's normative retrieval specification for author rules.

`EffectiveAuthorRule` is the shared SQL classifier. The legacy display resolver
is not a retrieval-authority fallback. The new bounded worker retains its slot
until the underlying thread finishes even if the awaiting request is cancelled.

## Deliberate exclusions

Nick's 2026-10-23 ruling excludes new send/waited-send/router/reset/compact/cancel
mutation restrictions. Availability and router discovery hide private and
inactive bots (TASK-1329 S3). A private bot can still be messaged or reset; a new
send to an `inactive` bot is refused with `target_inactive`.
Existing recall and working-memory self-namespace controls remain unchanged.
TASK-1323 owns message source-author filtering and ranking/parser changes.

LAN/operator REST and direct UI authority remain unchanged. This is an MCP data
boundary, not a sandbox against agents with shell/network/database access. No
production data backfill, schema migration, restart, or deployment is included.

Tests use synthetic `private`-tagged fixture profiles and an isolated private
PostgreSQL schema. The HTTP proof exercises the actual ASGI MCP transport, bound
service client, history/context assembler and native memory tool executor; it
does not invoke a vendor model or a production private bot.
