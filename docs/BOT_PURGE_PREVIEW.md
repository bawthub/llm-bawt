# Bot purge: preview and confirmed execution

`GET /v1/bots/{slug}/purge-preview[?delete_profile=true]` is a read-only
manifest of one action:

- `scope: purge_data` (default) — the standalone `POST /v1/bots/{slug}/purge-data`
  selection; the profile and its tags are kept.
- `scope: delete_profile_and_purge_data` — `DELETE /v1/bots/{slug}/profile?purge=true`;
  the same selection plus the `bot_profiles` row and every `bot_profile_tags` row.

Both destructive routes require `manifest_digest` from a preview of the **same
action** (`execution_enabled: true`). Profile-only `DELETE` (no `purge`) is
unaffected and needs no digest.

## Execution contract (`bot_purge_execute.execute_bot_purge`)

One PostgreSQL REPEATABLE READ transaction, `lock_timeout` 3s:

1. `LOCK bot_profiles, bot_profile_tags IN SHARE ROW EXCLUSIVE MODE`, then the
   prompt-purge writer guard. Table discovery uses a separate connection so the
   snapshot is taken **after** the locks.
2. Recompute the manifest; a different digest refuses with 409 `stale_preview`.
3. Refuse, deleting nothing:
   - 423 `protected` — reserved identities (`system`, `*`), or message storage
     not tied to this bot alone (ambiguous or profile-less storage, or a
     storage id reserved by an alias);
   - 409 `aliases` — retired slug aliases still point at the bot;
   - 423 `active` — a non-terminal turn started in the last 6 hours, or an
     undelivered inter-bot delivery to the bot;
   - 423 `busy` — lock timeout, serialization failure or deadlock;
   - 409 `dependent` — another bot's row references selected rows (FK);
   - 404 `not_found` — combined delete of a missing profile;
   - 503 `unavailable` — no PostgreSQL, a required store missing, or ownership
     evidence incomplete.
4. Delete the selected rows (and, for the combined action, tags then profile)
   and commit. Partition caches are evicted afterwards.

The receipt is `{slug, action, manifest_digest, profile_deleted, deleted_rows,
retained[]}`; `retained` lists the manifest's retained and unknown entries so
nothing outside the purge is presented as erased. Any refusal rolls back the
whole transaction: there is never a partial profile delete.

`runtime_settings.purge_bot_data` remains an unconfirmed internal primitive for
tests and maintenance; no HTTP route reaches it without the guard.

Orphan storage (no profile) is not purged here; System → Clean up orphans
handles profile-less, non-private storage.

## Wire contract

The response and its errors use `Cache-Control: no-store`.

- `schema_version: 2`, `scope_version: bot-purge/3`, `scope` and
  `scope_fingerprint` identify the inventory/predicate contract.
- `slug` is trimmed/lowercased; `storage_id` uses the existing partition sanitizer.
  Legacy punctuation and one-character slugs are supported; blank/control/overlong
  slugs fail with 422 before resolving a database.
- `profile_version` has `state: present|absent|unknown` and a fingerprint of the
  observed profile row when present. Profile content/settings/secrets are not
  returned. An absent profile is supported, not a 404 or a tombstone proof.
- `storage_ownership` reports raw-owner evidence, a candidate fingerprint/count
  (not the identities), missing sources and attribution uncertainty. A deleted
  profile or a raw alias sharing the storage key is not exclusive ownership.
- `inventory[]` contains `key`, `store`, `table`, `disposition:
  deleted|retained|unknown`, nullable `count`, `coverage` and a reason.
  Missing tables/columns and uninspected stores are **null, never invented zero**.
  Ambiguous partition rows additionally expose `observed_storage_count`; their
  target-owned deletion count remains null.
- `manifest_digest` is lowercase SHA-256 over UTF-8 canonical JSON of the response
  **excluding `manifest_digest`**: sorted object keys, compact separators,
  ASCII escaping, finite JSON numbers. No volatile generation timestamp is added.
  Scope, profile state/fingerprint, storage evidence, counts, coverage and reasons
  participate. Identical observations produce identical digests.
- An unavailable local PostgreSQL snapshot returns generic 503 with no driver
  SQL/parameters. A missing optional local relation returns an unknown entry,
  without bootstrapping it. External services are not contacted.

The production wrapper opens one **REPEATABLE READ, READ ONLY** PostgreSQL
transaction. It uses SQL reflection and SELECTs, not store constructors, writer
locks, temporary tables, DDL, executor calls, cache invalidation or events.
The injected collector leaves its caller's transaction alone.

## Meaning of the categories

`deleted` means *rows selected if the current executor's guards and transaction
succeed*, not completed deletion, successful FK validation, or permission to run.
This includes raw-slug sessions/configuration/BOT-or-bot entities/prompts/turns,
owned tool records and their legacy payloads, and jobs/companions through their
**parent job's ownership**. Messages, memories and forgotten rows use the
sanitized storage key; ambiguous attribution is explicitly unknown.

`retained` does **not** mean historical, harmless or drained. Standalone
purge leaves profiles/tags (the combined action deletes them), working memory, changed files, approvals, routing,
ops jobs, delivery receipts and session outbox records. The preview separately
counts nonterminal deliveries and pending session events as overlapping retained
operational subsets. Shared/other-job runs, including `RUNNING` markers, remain;
no run cancellation is inferred. Profile-session and foreign-tool-turn pointer
rows are retained but can be changed by installed SET NULL FKs; those impacts are
listed separately. All retained subsets can overlap: **do not sum the counts**.

`unknown` includes uninspected external BawtHub links, object/media/attachment
ownership and physical bytes, Redis/bridge/provider state, backups, and indirect
or unmapped references. Shared or user-owned blobs are not reclassified as
exclusive bot data. This inventory is intentionally not a complete erasure claim.

## Deliberate limits

The digest is **not a row-level preimage**: an equal-count change can preserve
it, so it proves the confirmed counts and scope, not each row. Observed raw-owner
sources are not an exhaustive historical registry. The active-turn check sees
recorded turn logs and deliveries, not in-flight provider or bridge state, and
stores outside local PostgreSQL (media, Redis, backups, external services) are
reported unknown and left untouched — there is no cross-store atomicity.

Tests use disposable SQLite and test-owned PostgreSQL fixtures only.
