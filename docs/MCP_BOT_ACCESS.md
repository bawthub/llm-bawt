# MCP bot-data access (TASK-1222)

The registered callable is the enforcement boundary for ordinary and approved
MCP execution. An ordinary call also checks before approval processing, so a
denied target does not create an approval request. Approval execution checks
current in-memory policy again; approval is not a private-data grant.

## Policy and callers

`bot_access.BotPolicyResolver(engine).resolve(caller=None)` uses a validated,
immutable process-local snapshot; **neither authorization stage reads policy
from the database or fills a cold cache** (TASK-1907). App and MCP lifespans own
one background refresher per process/engine, including approved execution,
HTTP and stdio. A standalone embedded `MemoryClient(server_url=None)` acquires
and warms the same lifecycle at construction, before any operations; use it as
a context manager or call `close()` to release its reference (a GC finalizer
also releases it). The last owner stops the worker and invalidates its view;
closing one client does not dispose shared storage/pools or stop other owners.
No cross-bot request starts a worker or performs a lazy policy read.
The background read uses `bot_tags.TagSnapshot` (TASK-1329:
every profile with its tags and catalog in one read-only statement).

Nick selected **bounded revocation delay**, not immediate global revocation:
- Freshness expires **five seconds from before the snapshot read**, including
  connection/read latency. An observed notification invalidates the old view
  immediately and requests refresh. Cold, failed, expired or disconnected views
  fail closed; requests never fall back to stale data or query on a miss.
- Profile create/delete/privacy, tag assignments (including grants), and catalog
  membership changes publish a schema-scoped PG transactional notification.
  Borrowed-session writes notify only after the owning transaction commits;
  rollbacks notify nobody. No bot identities enter the notification payload.
- Background reconciliation every two seconds repairs lost hints and direct SQL
  changes. This is not TTL-only invalidation or a per-write revision check.
- The deadline uses Linux `CLOCK_BOOTTIME`, including host/process suspension
  and independent of wall-clock/NTP steps. Platforms without a suspend-aware
  clock fail closed (`snapshot_clock`), rather than weaken the bound.
- Final admission checks freshness again after asynchronous ownership work;
  pending and approved retries reauthorize. Already admitted work may finish.
  Stale grants can therefore be admitted for up to five seconds after an
  unobserved change; there is no immediate-revocation or in-flight-drain claim.
- Diagnostics expose bounded reason/state, freshness remaining and refresh/failure
  counts, never row values, bot identities or raw database errors. No reader
  registration, global reader locks, durable lease or trigger schema is added.

No code names a bot:
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
For the exact durable conversation in an existing verified turn capability,
its matching bot/user tuple supplies session ownership in memory at both
stages. This is not an SDK `session_key`, and service-signed arbitrary arguments
do not establish ownership.

Runtime streaming history (including an internally supplied `/new` successor)
uses service identity, not agent tool headers. SDK binding already fetches the
session row. Only when that existing row's **id, bot and user** match the runtime
binding does it mark the request-local binding as verified. The worker scopes
the existing signed turn proof to its history writes; `MemoryClient` includes
it inside the exact-invocation service envelope only for matching `messages_add`
arguments. Both authorization stages revalidate both signatures, expiry and
exact session/bot/user. Caller kind remains `service`, without supplying task or
approval headers or bypassing approval policy. The context is cleared on success
and failure, and never retained on a cached client. No owner query or new query
was added to SDK binding. A token minted around an arbitrary selected session,
a missing/mismatched row, or a raw owner tuple is **not** sufficient provenance.

Native-chat `/new` (including durable stream execution of a nonstream request)
can instead attest its exact **committed keyed successor**. Only when the
rotation backend's captured bot/user match the runtime caller does the prepared
reset carry private owner provenance. The MCP history adapter exposes that
identity from its immutable service principal, not mutable request arguments.
After rotation succeeds, the worker signs the existing turn-context format for
that successor and scopes it inside the existing service envelope. Native
stream and nonstream executor history writes therefore use the same two-stage
in-memory ownership checks. This token is not passed as an agent/task header or
to the native model. An uncommitted/failed rotation, arbitrary selected session,
unknown/mismatched backend owner, or signing failure retains the secure lookup.

Agent nonstream persistence uses the same already fetched exact SDK-owner
provenance before its initial history write. It signs an early history-only
proof without moving reset preparation or rotation. That scope ends before
seed/reset/model execution. Final history separately uses the **final** binding
and existing final agent proof, so initial ownership never carries across a
rotation. Both scopes live inside the executor thread and clear on errors;
they never mutate the cached MemoryClient. An arbitrary selected id or a later
task token without a verified binding still requires authoritative lookup.

Other explicit-session callers retain the existing `sessions(bot_id,user_id)`
lookup: this remains unapproved and is **not** a zero-database-reads-all-writes
claim. These include standalone/archive/service callers without a verified
runtime binding and approved replay. Approved execution
reauthorizes with its server-owned caller and does not inherit runtime-service
ownership proof. Lifecycle/SDK reads and storage's own work are separate from
zero policy/ownership authorization SELECTs during a proven history write. Seed
transcript construction is distinct from conversation history persistence.

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
