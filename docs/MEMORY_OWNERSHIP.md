# Stored-memory ownership rollout

## Contract

`memories.owner_user_id` is nullable, immutable metadata for trusted personal
writes. It supplements existing bot-access and author/source predicates; it does
not grant access to another bot. Backend authority is immutable and per call, not
stored on a cached per-bot backend. Omitted backend authority is **not** operator
authority. The existing internal `delete_memory(..., user_id=None)` operator
shorthand remains for compatibility; explicit `NO_MEMORY_AUTHORITY` is unbound.
MCP resolves the registered caller before touching storage. Internal user-less
MemoryClients and service-credential HTTP operations remain operator calls;
personal clients are constructed with the trusted `user_id`.

A durable `memory_ownership_activations` row, keyed by canonical storage identity,
controls enforcement. Existing unmarked bots retain legacy behavior until an
operator deliberately activates each one. A genuinely new profile is marked in
its creation transaction, including direct/import creation. Rediscovery or
partition recreation never marks an existing profile. Retained data, conflicting
storage identities and retired aliases cannot silently become a fresh identity.

On a marked bot:

- Personal reads exclude explicit foreign owners. Ambiguous legacy NULL rows
  retain the old entry point's read eligibility; ownership does not make a manual
  memory eligible for sourced-only recall or extraction.
- Personal mutation and dedup accept the explicit owner, or a NULL row whose
  complete provenance resolves to that user's non-legacy sessions and registered
  non-service human profile. Service or unknown identities never prove ownership.
  New supplied live sources must meet the same strict proof. Session/message locks retain that proof
  through insertion. Extraction stays sourced-only, excluding manual rows.
- Dedup happens within the appropriate owner group before LIMIT. The returned
  memory is the persisted survivor, not incoming metadata paired with its ID.
- Operator access does not merge distinct owners. NULL is a separate dedup and
  consolidation group. Consolidation checks the originals before inserting,
  preserves their owner, then verifies survivor content/ownership and original
  timestamps/ownership in one replacement transaction. Changed originals remain.
- Source-message deletion is intentionally different from by-memory-ID deletion:
  every supplied live or forgotten message ID must belong to the caller's session,
  **including legacy-backfill sessions**, then **any** memory citing those IDs is
  removed, including mixed-source memories. This deletion-only permission preserves
  forget semantics after attestation without turning guessed sources into proof
  for writes, dedup or by-memory-ID mutation.
- Supersede is still unsupported and returns false. No new supersede feature is
  introduced by this rollout.

## Operator phases

Use the installation's approved database tooling and planned maintenance window.
Never put database URLs/passwords on a command line. The helper uses the normal
application configuration; do not repoint live profiles to rehearse. Tests run
only on disposable synthetic PostgreSQL databases. Publishing code is not
permission to activate existing bots.

The helper is `python -m scripts.memory_ownership`. It has no all-bots/wildcard
mode and does not auto-select a user. Its output is counts, refusal/remedy,
operation status and archive ID only, never stored facts or a user roster.

1. **Expand**: `expand --apply` adds the nullable column and activation table.
   There is no default, backfill or activation in DDL. Runtime bootstrap applies
   the same expansion, never the data migration.
2. **Index**: `index --apply`, outside the activation transaction, creates the
   partitioned index with `ON ONLY`, builds each existing child concurrently,
   then attaches it. Pause partition creation for this phase. Rerunning resumes
   completed children and rebuilds invalid interrupted concurrent indexes.
   Do not replace this with an ordinary parent index build over populated data.
3. **Plan one bot**: `activate --bot <canonical-slug> --actor <operator-id>` is
   dry-run by default. Use approved live SQL tools (for example the installation's
   `pgq` wrapper) for live inspection; the helper does not bypass installation
   access policy. Retain a counts-only preview and review refusals before apply.
4. **Attest only when justified**: the same command accepts
   `--attest-user <human-id>`. It refuses another proven human or explicit owner.
   Never infer that service/bot authors or guessed legacy sessions prove a human
   owner. Missing evidence is not permission to use a default user. Alternatively,
   explicitly select `--mark-only` on the same one-bot command to activate without
   stamping **any** NULL row, even with multi-human legacy/service/unknown evidence.
   This is mutually exclusive with attestation and is never an inference fallback.
   Counts include `restricted_memories` with disjoint pre-stamp reasons, in priority
   order: `restricted_unsourced`, `restricted_legacy`, `restricted_service_or_unknown`
   (also missing/malformed sources), and `restricted_mixed_human`. These describe
   NULL rows without one proven human owner, not read visibility; attestation may
   subsequently stamp them. Mark-only records mode `mark_only` and zero stamps.
5. **Apply in the window**: repeat the exact selected bot/mode command
   with `--apply` through the approved executor. Evidence, stamping and marking
   are one READ COMMITTED transaction under writer-excluding locks across
   profiles, sessions, live/forgotten messages, memories and the marker table.
   The helper times out on unavailable locks instead of waiting indefinitely.
   Stop/retry after writers drain; do not remove the locks. Inference stamps only
   when one human is proven. A proven multi-human bot is marked without stamping
   ambiguous NULL rows. Repeated activation never reinterprets historical rows.
6. **Verify**: `status` reports total profiles and unmarked counts. Check the
   selected durable receipt's mode, actor, time and counts using approved SQL.
   Verify personal two-user isolation with synthetic fixtures on the activated
   surface, then clean up the fixtures. Do not use another user's live memories
   as acceptance data.

If the installation requires all live SQL through `pgq`, keep the Python helper
on the disposable rehearsal database until the operator has an explicitly
reviewed execution procedure for the live window. A rehearsal receipt alone is
not live-mutation authority.

## Lifecycle and rollback

Purge retains and inventories the activation record, including combined
profile-and-data deletion. HTTP creation refuses retained activation identity.
Do not delete this record merely to make a previously used slug reusable.

There is no online bot rename operation. A manual rename/move must carry the
activation row to the new sanitized storage identity **in the same maintenance
transaction** as the profile/data move, refusing any destination activation,
profile, alias or retained storage collision. Preserve the original receipt's
mode, actor, timestamp and counts. If the sanitized identity does not change,
leave the marker in place. A profile-only rename without the marker/storage move
is not a supported procedure. Retired aliases still reserve the original name.

Prefer **code rollback while retaining the additive column and marker table**.
This preserves recoverable ownership evidence. Roll back code before any schema
removal; code that expects the column must not continue running.

If a complete down migration is required, take the normal recoverable backup and
stop writers. `archive-down --apply` atomically copies explicit per-row ownership
and full activation receipts to `memory_ownership_rollback`, returns an archive
ID, then drops the index, column and marker table. The archive remains in the DB;
retain it and the backup. Repeat down is a no-op. Re-expansion creates nullable
metadata, **not** an automatic archive restore: an operator must deliberately
restore the chosen archive or re-run the reviewed activation procedure. Never
silently reactivate using stale or merged archives.

The obsolete `scripts/rebuild_profile.py` predates these APIs and is unsupported
for owned-memory stores. Do not use it as an ownership migration or repair path.

## Verification map

- `test_memory_crud_owner_pg.py`: two users, exact/semantic dedup, durable
  provenance, manual memories, reads before LIMIT, source deletion and concurrent
  ownership/cutover/source-session changes.
- `test_memory_maintenance_owner_pg.py`: meaning edits, recurrence grouping,
  pruning, scoped recall, owner-preserving consolidation and stale replacement.
- `test_memory_ownership_migration_pg.py`: inferred/attested activation, legacy
  refusal, writer exclusion, idempotence, profile lifecycle, online indexes and
  recoverable down migration.
- `test_memory_crud_authority.py`: registered MCP caller forwarding and unbound
  caller refusal. Existing extraction/forget/consolidation tests remain required.
