# TASK-1149 — source schema convergence and bounded operational PLAN

**PLAN ONLY: do not paste/run this document as a script.** Nothing here is an
approval for operational DDL, vector upgrade, deployment or restart. Snark
owns operational coordination. Use `pgq` for all live SQL; never use inline
credentials.

## Reference and invariant

Build a **new, empty, isolated, disposable PostgreSQL 16 + pgvector database**,
assert its `_test` identity at Config and actual connection before app-owned
bootstrap, and run each supported store bootstrap in its creation order. Do
not reuse/reset the shared `llm_bawt_test` reference, and do not copy live rows,
bots or protected partitions into the reference. Run the private gate
`make test-pg-acceptance`; its synthetic `test_schema_convergence_pg.py`
proves both FKs and their delete paths. The reference must include exactly the
source-owned shared tables, no test fixture partitions in the final comparison.
Snark already retired both July backup tables under Nick's exact authorization;
no further top-level table-drop work remains in this plan.

`scripts/schema_catalog.py capture --mode test --expected-db <name> --out
<new-path>.jsonl` is read-only and refuses a non-`_test` database. Operational
capture uses `capture --mode operational --out <new-path>.jsonl` and `pgq`
read-only only. Capture queries are single-statement MVCC snapshots; output
includes a source-byte digest, Git HEAD/dirty flag, server version, catalog
hash, and record count. `diff --reference <fresh>.jsonl --operational
<live>.jsonl --out <new-path>.jsonl` is offline and deterministic. Column
ordinal-only drift is ignored; `entity_profiles.summary` TEXT and unbounded
VARCHAR are equivalent **only for that column**. All other missing/extra
objects and definition/validity changes remain visible, including runtime
partitions and extension-version mismatch. Record runtime partition
bounds/registered identities as explicit reviewed exceptions while deployed;
never infer a bot from a truncated partition name. This tool is an inventory,
**not** a dependency/approval oracle for destructive DDL.

The pre-cleanup and post-partition-cleanup catalogs are historical evidence.
Snark's post-backup-table-retirement catalog
`TASK-1149-phase1/resume-20261018T1653/post-backup-table-retirement/operational-catalog.jsonl`
has 1,800 records and **49 live top-level tables**, matching the old reference
set with zero missing/extra. It is not a fresh reference against this new
source; regenerate both captures before migration. Snark's current receipts
cover 327/327 partition outcomes and the exact guarded retirement of both
unused backup tables. Do **not** repeat either action for schema drift.

## Pristine reference and private legacy rehearsal (2026-10-18)

`tests/test_schema_full_store_pg.py` composes 21 actual store bootstrap groups in
recorded dependency order on **new, randomly named** databases inside the
networkless pytest-owned pgvector cluster. The reference has exactly 49 shared
top-level tables; the three empty `schema_full_fixture` partition children are
removed with RESTRICT before capture. Neither fixture copies live rows or
connects to operational PostgreSQL. `scripts/schema_shared_upgrade.py` is
**deliberately test-only**: it refuses a non-`*_test` database at URL and actual
connection, checks table ownership, expected definitions/index collisions and
full-table zero-NULL counts, runs separately bounded ALTER transactions and
individually nontransactional concurrent index builds, and stops on drift.
It is executable SQL on synthetic legacy state, **not** a production command or
a blanket authorization to run any of its statements against `askllm`.

Reproduce on the host with `make test-pg-acceptance`; for immutable catalog
receipts supply a **new** `TASK1149_EVIDENCE_DIR` outside git to the two-case
pytest module. Saved example:
`TASK-1149-phase1/fullstore-rehearsal-20261018-final/`. The pristine reference
contains 1,705 catalog records, and its 21-entry bootstrap manifest and source
hash are in `pristine-full-store.jsonl.meta.json`. The synthetic legacy catalog
contains 1,690 records and reports **65** differences; after the explicit
private upgrade, it contains 1,705 records and is **byte-identical** to its
pristine reference (zero comparator differences). This tests the shared
nullable/width drift, five absent indexes, dispatch predicate, four historical
constraint/index names and event-sequence name; it does **not** rehearse vector
extension upgrade, runtime partition modifications, data-bearing NULL repairs,
or an operational quiet-window lock load.

Offline diff against the **saved** post-retirement operational baseline reports
1,386 raw differences: nine shared columns, the known shared index/constraint/
sequence signatures, 33 legitimate registered bot partitions with their child
columns/indexes/constraints, extension-version difference, and ownership and
physical dependency records. In particular, that historical 1,800-record live
catalog was captured *before* the current tool recorded owner/ACL and dependency
records; missing records of those kinds do **not** establish absent DB objects.
The 49 shared relation names match. Do not misreport 1,386 as all actionable
DDL, or claim zero live drift from the synthetic result. A fresh operational
read-only capture and narrow runtime/environment classification remain required.

For operational execution Snark must produce the corresponding reviewed,
identity-fenced SQL, validate the current owner/ACL/dependencies and existing
index definitions against a **fresh** live capture, prove full-table NULL
preflights, and decide backup/restore/quiet-window and vector gates. The test
runner's failure handling intentionally does **not** auto-retry INVALID
concurrent indexes or resume a partially applied migration; interrupted work
requires a new catalog and operator judgment. No operational DDL is authorized.

## Mandatory gates before any operational mutation

1. Match a **fresh** reference, source fingerprint and operational capture to
   the exact reviewed DDL list. Preflight owner/ACL, indexes valid/ready,
   constraints/sequence ownership and dependencies; reject collisions or
   changed definitions. Require verified current full backup and a tested
   restoration decision/rollback window; a listed dump is not a restore test.
2. Lock/availability review: all five concurrent builds are on unpartitioned
   shared tables. No parent index, partition detach, FK rebuild on `messages`,
   protected-bot row action or surprise CASCADE. Every ALTER takes a table
   lock; choose a quiet window, short lock timeout, bounded statement timeout,
   and stop on mismatch/contention. Concurrent index statements must be each
   executed outside a transaction. No automatic retry of INVALID indexes.
3. For bot flags and pending-question harness, an operator must prove **table-
   wide** zero NULL values and make an explicit data decision if not zero.
   Existing source checks excluding the protected bot are insufficient. Do not
   fill NULLs or rewrite protected records to force NOT NULL. Widening tool
   columns is non-lossy, but verify index/dependency effects and lock impact.
4. Record step-level start/end/catalog receipts and reverse decisions; compare
   fresh source-vs-live semantic catalog after each group. An interruption
   requires catalog inspection and a reviewed resume, not blind replay.

## Source-aligned shared columns — in-place proposal

Existing columns are not retrofitted silently on
startup. Source now creates `bot_profiles.endpoint_id BIGINT` (same as its
referenced `model_endpoints.id`); three bot flags and
`chat_pending_questions.origin_harness` have matching Python/server defaults
and NOT NULL in **fresh** and **missing-column additive** creation;
`entity_profiles.summary` is TEXT; the four tool-call ID/status columns are
unbounded VARCHAR in fresh/additive paths; `result_payload_available` has a
server default FALSE. Source startup does not alter already-existing column
width, default or nullability. Live endpoint_id and summary already match the
target and must not be rewritten.

After zero-NULL proof for *every* affected column, in short bounded, separately
reviewed transactions (one table at a time; no blanket transaction across all
steps), proposed SQL is:

```sql
-- On public.bot_profiles; no WHERE-based data repair is implied.
SET LOCAL lock_timeout = '1s';
SET LOCAL statement_timeout = '30s';
ALTER TABLE public.bot_profiles
  ALTER COLUMN tts_mode SET DEFAULT false,
  ALTER COLUMN include_summaries SET DEFAULT true,
  ALTER COLUMN include_in_global_search SET DEFAULT true,
  ALTER COLUMN tts_mode SET NOT NULL,
  ALTER COLUMN include_summaries SET NOT NULL,
  ALTER COLUMN include_in_global_search SET NOT NULL;
-- In a separate bounded transaction, after all origin_harness NULLs audited:
ALTER TABLE public.chat_pending_questions
  ALTER COLUMN origin_harness SET DEFAULT 'claude',
  ALTER COLUMN origin_harness SET NOT NULL;
-- In separate bounded transaction(s), after dependency/index check:
ALTER TABLE public.tool_call_records
  ALTER COLUMN tool_use_id TYPE VARCHAR,
  ALTER COLUMN parent_tool_use_id TYPE VARCHAR,
  ALTER COLUMN approval_request_id TYPE VARCHAR,
  ALTER COLUMN approval_status TYPE VARCHAR;
-- result_payload_available already has live DEFAULT false; do not touch.
```

`SET LOCAL` must be inside each operator-created transaction; these lines are
**not executable as pasted**. On preflight failure, stop for a user-approved
row-specific decision. No application startup drift-repair loop.

## Five missing indexes and dispatch predicate

Preflight exact names absent and definitions/validity of existing targets.
Create each independently **CONCURRENTLY**, outside transaction blocks, with
operator-managed timeouts/receipts. `IF NOT EXISTS` would mask a wrong or
INVALID definition, so do not use it as proof of convergence.

```sql
CREATE INDEX CONCURRENTLY ix_bot_profiles_endpoint_id ON public.bot_profiles (endpoint_id);
CREATE INDEX CONCURRENTLY ix_bot_profiles_prompt_override_id ON public.bot_profiles (prompt_override_id);
CREATE INDEX CONCURRENTLY ix_entity_profiles_email ON public.entity_profiles (email);
CREATE INDEX CONCURRENTLY ix_tool_call_records_tool_use_id ON public.tool_call_records (tool_use_id);
CREATE INDEX CONCURRENTLY ix_turn_logs_ended_at ON public.turn_logs (ended_at);
-- Build replacement BEFORE removing the old QUEUED+DISPATCHING index.
CREATE INDEX CONCURRENTLY ix_inter_bot_delivery_dispatch_next
  ON public.inter_bot_deliveries (target_bot_id, ordinal)
  WHERE status IN ('QUEUED', 'STEERING', 'DISPATCHING');
```

Inspect `pg_get_indexdef`, `indisvalid`, `indisready`, uniqueness, predicate,
and owned dependencies; failed concurrent builds may leave INVALID artifacts.
Only after verifying the temporary replacement, in **one short bounded
transaction**, `ALTER INDEX public.ix_inter_bot_delivery_dispatch RENAME TO
ix_inter_bot_delivery_dispatch_previous; ALTER INDEX
public.ix_inter_bot_delivery_dispatch_next RENAME TO
ix_inter_bot_delivery_dispatch;` Verify canonical definition again; then in
its own nontransactional step `DROP INDEX CONCURRENTLY
public.ix_inter_bot_delivery_dispatch_previous;`. Stop rather than guessing
if any name/definition changed. Reverse a completed swap by rebuilding the old
predicate concurrently under a temporary name and repeating the verified
swap; do not assume an aborted concurrent build rolled back completely.

## Historical names; FK/index already present

After equivalence/collision checks, rename the four existing constraints,
which renames their backing indexes; **do not duplicate unique indexes**:

```sql
ALTER TABLE public.agent_runs RENAME CONSTRAINT openclaw_runs_pkey TO agent_runs_pkey;
ALTER TABLE public.agent_events RENAME CONSTRAINT openclaw_events_pkey TO agent_events_pkey;
ALTER TABLE public.agent_events RENAME CONSTRAINT openclaw_events_event_dedupe_key_key TO agent_events_event_dedupe_key_key;
ALTER TABLE public.agent_session_state RENAME CONSTRAINT openclaw_session_state_pkey TO agent_session_state_pkey;
ALTER SEQUENCE public.openclaw_events_id_seq RENAME TO agent_events_id_seq;
```

Sequence rename preserves OID/OWNED BY/grants/current value; inspect the
`agent_events.id` regclass default afterwards and never reset/recreate it.
The existing `idx_sessions_one_active_per_bot_user` is already unique with
`NULLS NOT DISTINCT WHERE status='active'`: source creates it on new databases,
so **do not** build it again. Existing live `messages.fk_messages_session` is
validated NO ACTION, NOT DEFERRABLE; existing live
`tool_call_records.fk_toolcalls_turn` is validated ON DELETE SET NULL, NOT
DEFERRABLE. Source creates both only on fresh parents, in dependency order.
Do not drop/rebuild them live merely to match fresh bootstrap. If a future
catalog lacks either, stop and plan separate explicit validation/DDL after
preflight of dependent rows; no implicit startup FK repair.

## Completed retirement and remaining acceptance

Snark independently verified that the two July backup tables had no
application references/inheritance/incoming FKs/triggers, and Nick authorized
exact guarded retirement. Receipt `cmve31tvf00340mp4wf2r2c55` records an
atomic `DROP RESTRICT` of **both** and post-catalog proof of only those two
relations plus their 34 columns disappearing. Neither table is a remaining
migration step or a permanent exception. No live protected bot/history/memory
objects were changed in that retirement; no content was read by me.

The vector extension 0.8.1→0.8.6 difference is a **separate explicit upgrade
and compatibility rehearsal**; do not silently add `ALTER EXTENSION` to this
plan or treat it as an accepted final exception. Its downgrade is not a
reliable rollback. Before claiming "nothing extra," obtain the vector
decision, fresh reference + live semantic diff, verified backup/restore posture
and clean app health after approved changes. No production data-copy/cutover
is justified by these metadata-only drifts.
