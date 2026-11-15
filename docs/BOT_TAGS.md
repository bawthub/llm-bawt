# Bot tags

Current implementation reference, reconciled from the original tag design.
Numbered sections are stable because source comments cite them. Examples use
fictional identities; installed bots, assignments and role instructions are
operator data, not product defaults.

## 1. Purpose and boundaries

Bot behavior comes from catalog tags rather than bot-name branches. Restricting
behavior tags constrain discovery, content access or background work; role tags
provide instructions and capability tokens. Neither replaces authenticated caller
identity or a tool's own authorization checks.

`private` is a content/discovery boundary, not a blanket prohibition on messaging
or resetting a bot. `inactive` rejects new inter-bot deliveries. An operator UI
can still display private bots. Per-user hidden-bot preferences are separate.

## 2. Implementation map

Paths below are relative to `src/llm_bawt/`:

| Area | Source |
|---|---|
| Behavior vocabulary | `bot_tags/vocabulary.py` |
| Schema, seed, migration, assignment transactions | `bot_tags/catalog.py`, `runtime_settings.py` |
| Catalog, instructions, grants | `bot_tags/store.py`, `bot_tags/grants.py` |
| Validated tags and SQL exclusion | `bot_tags/facade.py` |
| Caller-scoped content access and refresh | `bot_access/policy.py`, `bot_access/refresh.py` |
| Role prompt context | `bot_tags/context.py`, `core/prompt_sections.py` |
| REST surfaces | `service/routes/bot_tags.py`, `service/routes/bot_grants.py` |
| Storage protection | `orphan_partitions.py`, `runtime_settings.py` |

## 3. Vocabulary

`BehaviorTag` defines code-owned behavior rows:

| Tag | Effective meaning |
|---|---|
| `private` | Excluded from bot-facing discovery and cross-bot content reads, and from private-excluding background consumers. Implies `no_global_search`. Also protects retained storage (§6). |
| `no_global_search` | Excluded from aggregate search/recall, even when another bot explicitly filters to it; targeted per-bot reads remain possible (§5.1). |
| `inactive` | Hidden from normal lists and excluded from routing/dispatch consumers; new inter-bot deliveries are refused. Content access and retained data are not removed. |
| `external` | Opt-in behavior; absence or unavailable policy grants nothing. It is not a grant-locked tag. |
| `approve_ops` | Opt-in delegated ops approval grant; membership uses the separate authenticated grant path (§8). |

Unknown profile identities are effectively private and no-global-search, not
implicitly inactive or opted in. Role slugs are catalog data, not new behavior
constants. Starter roles map `coordinator` to `coordinate`, `coder` to `implement`,
`reviewer` to `review`, and `arbiter` to `ship_check` and `mediate`.

## 4. Schema and bootstrap

`bot_tag_catalog` stores slug, name, description, kind (`behavior` or `role`),
nullable `prompt_key`, JSONB capability array, optional color, system flag and
timestamps. Slugs match `^[a-z][a-z0-9_]{0,63}$`; prompt keys require role kind
and a `role.` prefix. System rows must be behaviors.

`bot_profile_tags` stores `(bot_slug, tag_slug)`, assignment time and actor.
Its catalog foreign key restricts deletion/renaming; deliberately no foreign
key points to `bot_profiles`, so retained assignments can outlive a profile.

Profile-store bootstrap uses the current schema's relation ownership, not just
search-path visibility. It creates/updates the schema and upserts code-owned
behavior definitions. If the assignment table did not exist before bootstrap,
it inserts starter roles without overwriting existing role choices and migrates
legacy false flags to both `private` and `no_global_search`. Migration is
transactional and does not run again merely because a bot was untagged.
Starter seeding creates no bot assignments.

### 4.1 Legacy flag compatibility

Effective private status is the `private` tag **or**
`include_in_global_search = false`. The latter remains a restrictive fallback.

Assignment and legacy-flag writes share a per-bot transaction lock:

- An explicit `tags` list replaces the full set, preserves kept assignment
  metadata, and derives the legacy column from absence of `private`.
- Without `tags`, changing the flag to false adds both legacy privacy tags;
  changing an existing profile's flag to true removes both.
- An unchanged flag leaves assignments alone; a whole-profile save is not a
  general mirror-repair operation.
- Recreating a slug does not strip inherited privacy tags.
- Removing an individual `private` assignment sets the column true when a
  profile exists; removing another tag does not rewrite the column.

Full replacements cannot add or remove grant membership. Optional
`expected_tags` detects a stale edit under the same lock and returns
`tags_changed` with the current set. Tags are exact catalog slugs, not normalized
free text; use `[]`, not `null`, to request an empty set.

## 5. Resolver and content scope

### 5.1 Contract

`BotPolicyResolver` reads a process-local immutable snapshot. The lifecycle
refresh owner loads at startup/reconnect and on committed policy-change events;
request paths do no policy database I/O. There is no periodic refresh or runtime
expiry lease. Direct SQL or a missed hint requires explicit administrative
refresh, not an assumption that a later request will reload policy.

`BotAccess.require_target` permits a known, non-private bot or the trusted
caller itself. `aggregate_scope` spans searchable bots plus the caller.
An explicit bot filter can narrow that scope, never widen it.

For example, another bot can target `sample-quiet` carrying only
`no_global_search`, but cannot include it by naming it in aggregate search.
The bot itself retains the self exception. Caller-less operator aggregate
search has no self exception. Content scope also carries the caller's user
identity; tags alone do not authorize cross-user reads.

### 5.2 Failure semantics

| Condition | Result |
|---|---|
| Cold, invalidated, disconnected or failed content-policy snapshot | `BotAccessError("policy_unavailable")`, never an unfiltered successful result. |
| Sync `BotTags` facade cannot read/validate its snapshot | Restricting behaviors apply; opt-in behaviors grant nothing. Role answers cannot be guessed and raise. |
| Unknown bot in an available snapshot | Private and no-global-search; no role or opt-in behavior. Targeted access denies an unknown partition. |
| Unknown tag in an available catalog | Raises rather than silently matching nothing. Missing opt-in behavior rows grant nothing. |
| Malformed rows, unknown assigned tags, inconsistent catalog or over 256 profiles | Snapshot validation fails. |

The sync facade still performs a bounded database read, unlike the
lifecycle-managed content resolver. It uses a caller transaction's savepoint
when present, otherwise a read-only transaction with statement/lock timeouts.
Policy views recheck that they are current before guarded access. Refresh
failures get a bounded event-triggered retry budget, not healthy polling.

### 5.3 SQL-side exclusion

`sql_allowed(column, exclude, prefix=...)` keeps exclusion in the same statement
as a sensitive read. It accepts a qualified column such as `d.target_bot_id`;
unqualified columns and unsafe prefixes are refused. Exclusions must be behavior
tags and cannot be empty.

The predicate uses correlated tag lookups, so no profile join is required.
Excluding `no_global_search` also excludes `private`. Excluding `private`
additionally requires an existing profile whose legacy flag is true. Missing
tables or failed reads raise; they never become an empty protected set.
Use distinct prefixes when filtering both delivery sender and recipient.

Consumers include idle candidates, delivery cancellation notices and
working-memory health. Routing uses a single sync tag snapshot before
eligibility/activity reads instead; private/inactive identities are excluded
or reported with their exclusion reason.

## 6. Storage protection

A retained `private` assignment is the storage protection list, even without
a profile. The protected set is the union of private assignments and profiles
whose legacy flag is false. The assignment branch deliberately has no profile
join.

Current profile deletion removes non-privacy assignments but preserves
`private` and `no_global_search` by default. The internal
`drop_privacy_tags` option is for teardown after the bot's data was purged.
Recreating the same slug inherits retained privacy assignments.

The orphan planner does not count protected partition rows. Execution locks
profiles and assignments in `SHARE MODE NOWAIT`, rechecks the protected set
and rejects drift before dropping a partition. The shared-row purge takes the
same locks before reading known/protected slugs on its non-dry-run path.
Missing policy data or lock conflicts abort rather than weakening protection.
Plan/digest confirmation remains a separate layer.

Removing a retained privacy assignment is explicit, including through the
catalog's profile-less `used_by` entry, and requires a fresh cleanup plan.
Removing `no_global_search` alone does not remove private storage protection.

## 7. Roles, instructions and discovery

Roles have operator-defined capability tokens and optional instructions.
Discovery returns the sorted capability union from assigned roles and can
filter by tag/capability. Unknown filters raise. Private/inactive bots are
excluded; live state is advisory, with unavailable sources identified rather
than reported idle. See `mcp_server/inter_bot_tools.py`.

Agent prompt context renders own-role instructions and visible roster/catalog
information. Instructions live in versioned global registry prompts with the
`bot_tag` category. Missing/blank instructions fall back to the role description;
prompt-context rendering failure skips that section, not the privacy policy.

An atomic role rename moves assignments but preserves its existing prompt key.
Treat keys as opaque references, not permanently derived from the current slug.
Instruction saves allocate a free `role.<slug>` key when needed. Clearing
instructions detaches the key without deleting registry history; a later save
can allocate a suffixed key. Catalog writes reject shared prompt ownership.

Capability tokens are opaque to discovery, but consumers can give configured
tokens real authority, such as grant management, model switching or house
assignment. A role description alone grants none of those actions.

## 8. Mutation surfaces

| Surface | Purpose |
|---|---|
| `GET/POST /v1/bot-tags` | Read catalog/usage or create an operator role. |
| `PATCH/DELETE /v1/bot-tags/{slug}` | Edit/rename or delete an operator role. |
| `PUT/DELETE /v1/bot-tags/{slug}/instructions` | Save or detach role instructions. |
| `DELETE /v1/bot-tags/{slug}/assignments/{bot}` | Remove one non-grant assignment, including profile-less usage. |
| `PATCH /v1/bots/{slug}/profile` | Replace the bot's full tag set, subject to grant protection and optional stale-edit check. |
| `GET /v1/bot-grants`, `PUT /v1/bot-grants/{tag}/bots/{slug}` | Inspect or change audited grants. |

Behavior definitions are code-owned; ordinary catalog mutations cannot edit or
delete them. Assigned roles cannot be deleted, including profile-less usage.
Role metadata includes optional chip color; grant styling is reserved.

`approve_ops` changes use the authenticated grant path with a reason and an
existing target profile. Bot grant managers need signed caller context and
the configured manager capability; they cannot change their own grants or
bot-manage private targets. Without a manager capability configured, management
is person-only. These are product authorization rules, not a claim of an
adversarial security boundary on a shared host.

MCP is not read-only for tags: `bots_profile_patch` permits constrained peer-tag
edits, while `bots_grant_set` uses the separate grant path. Peer edits cannot
change own tags, private state or protected management capabilities. Callers
must follow the specific tool contract rather than bypassing it with raw writes.

## 9. Operator UI

The bot editor uses catalog tags instead of a separate global-search toggle.
Normal lists hide inactive bots; the editor can include them.
The catalog UI manages roles/instructions and exposes assignment usage,
including identities with no remaining profile. UI presentation is not a
substitute for resolver checks or store validation.

## 10. Verification map

Existing coverage lives in `tests/test_bot_tags.py`, `test_bot_tags_pg.py`,
`test_bot_tag_api_pg.py`, `test_bot_tag_instructions_pg.py`,
`test_bot_tag_context.py`, `test_bot_tag_discovery.py`, `test_bot_access_*.py`
and `test_orphan_partitions_pg.py`. Use synthetic identities and isolated data;
do not test against installed private bots.

This reference adds no rollout, gate or live-data action. It supersedes the
original design's pending decisions, delivery slices and historical hardcode
inventory; source remains authoritative for exact request/error shapes.
