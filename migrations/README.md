# Schema migrations (Alembic)

Status: **partial ownership (TASK-631 slice 5).** Head is `0005_playground`.
`0001_baseline` performs no DDL. `0002_ops_store` owns the ops store family
(`ops_operations`, `ops_jobs`, `ops_operation_revisions`, `ops_bell_outbox`,
`ops_release_runs`, `ops_release_events`, `quiesce_run_sightings`, `ops_host_access`).
`0003_working_memory` owns the working memory family (`working_memory_namespaces`,
`working_memory_items`, `working_memory_task_suppressions`,
`working_memory_task_purge_audit`, `working_memory_changes`); it has no data transitions.
`0004_gates` owns the gate family (`system_gates`, `gate_members`, `gate_escalations`,
`ops_resume_notices`); its one contract step drops the retired `gate_members`
control-pump columns only where they still exist, as the runtime bootstrap does.
`0005_playground` owns `playground_artifacts`, including its indexes and the
nullable archive/trash timestamp additions; it has no data transitions.
Every other table is still created and altered by the runtime bootstrap (`ensure_*`
and `create_all` units guarded by `utils/schema.SchemaBootstrapGuard`).

## Startup behavior and activation

The app checks the database's Alembic state at startup, under the runtime schema
advisory lock:

- **No `alembic_version` table:** compatibility mode. Nothing runs; every runtime
  bootstrap behaves exactly as before.
- **Stamped at a known revision:** the app runs `alembic upgrade head` on that one
  locked connection, then skips the runtime DDL that a revision now owns. The ops
  store's current-revision repair still runs every start.
- **Invalid** (empty table, several rows, a revision this source doesn't know): the
  app refuses to start rather than guess.

So **stamping `0001_baseline` activates migrations**: the next app start applies
every later revision up to head. A database already stamped
needs no new stamp: its next app start on a release that ships a new revision
applies it. On a database the runtime has already bootstrapped this is a no-op
(the revision calls the same idempotent bootstraps), and its one-time data transitions
run only when their column is being added. A failed upgrade rolls back together with
the lock, and the app does not serve.

What this slice does NOT provide:
- **Parity.** `target_metadata` is None, so autogenerate is off. The SQLModel
  metadata covers only part of the schema; diffing it would propose dropping real tables.
- **Full reconstruction.** `alembic upgrade head` on an empty database creates only
  the eight ops-family, five working-memory, four gate and one Playground table;
  every other table still needs the runtime bootstrap.
- **Drift detection.** That comes in a later slice, along with schema reconstruction
  and retiring runtime DDL.

## Stamping a database

Preconditions:
1. The database has finished the supported runtime bootstrap. The app has started
   against it at least once on the release that ships this revision.
2. You have confirmed the target is the database you intend. Settings come from the
   app's `POSTGRES_*` config, or from `LLM_BAWT_ALEMBIC_URL` if set.
3. `make migrate-current` shows **no revision**. Never use the baseline stamp to
   relabel a database that is already at a later revision.

Then run `make migrate-stamp-baseline`. Inside the app container, `make` isn't
shipped, so run the CLI from `/app`, where `alembic.ini` lives:

    docker compose exec -w /app app alembic current
    docker compose exec -w /app app alembic stamp 0001_baseline

The app images copy `alembic.ini` and `migrations/` into `/app`, and dev compose
bind-mounts both read-only. The new mounts need a container recreate, not a plain
restart. The only schema change is the
`alembic_version` table, which holds one row.

Rollback: drop `alembic_version`. That returns the database to compatibility mode,
and the runtime bootstraps take the ops tables back; they are idempotent against
the `0002` schema, and `0002` drops nothing on downgrade.

## Baseline schema reference

`baseline_schema.sql` is a schema-only `pg_dump` of the runtime-bootstrapped schema
at master 9dfbd3c9. It is documentation, not a restore recipe. Regenerate it with
`make migrate-schema-snapshot`.

- **Kept:** the extensions (`vector`, `pg_trgm`), all ordinary tables, the
  partitioned parents (`messages`, `memories`, `forgotten_messages`) with their
  `PARTITION BY LIST (bot_id)` keys, and the constraints, functions, triggers and
  indexes defined on those tables.
- **Excluded:** per-bot child partitions (`messages_p_*`, `memories_p_*`,
  `forgotten_messages_p_*`). The runtime creates these per bot, so their
  child-local indexes and cloned triggers are not listed. Also excluded is
  `task2061_auto_title_backup`, a one-off backup table left over from one
  deployment. It is not product schema.
- **Normalized:** `\restrict`/`\unrestrict` tokens and the `-- Dumped from/by`
  version header are stripped, so snapshots compare cleanly. So is
  `SET transaction_timeout`, which pg_dump 17 emits and a PostgreSQL 16
  server rejects.
