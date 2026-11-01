# Bot purge impact preview (read-only)

`GET /v1/bots/{slug}/purge-preview` describes the **existing standalone
`POST /v1/bots/{slug}/purge-data` selection**, not a new purge implementation.
It does not describe profile-only deletion or authorize combined deletion.
The GET is advisory and always returns `execution_enabled: false`; no execution
endpoint accepts its digest. Existing destructive routes are unchanged.

## Wire contract

The response and its errors use `Cache-Control: no-store`.

- `schema_version: 1`, `scope_version: standalone-purge-data/1` and
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

`retained` does **not** mean historical, harmless or drained. Current standalone
purge leaves profiles/tags, working memory, changed files, approvals, routing,
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

## Deliberate limits / later slices

The digest is **not a row-level preimage or a fresh confirmation capability**.
An equal-count row/value change outside the profile can preserve it. A writer can
commit immediately after the snapshot; observed raw-owner sources are not an
exhaustive historical registry. This GET establishes no writer fence, active or
protected-bot clearance, alias-reuse safety, FK-catalog proof, drain, transport
cancellation, stale-confirmation rejection or cross-store atomicity. Those
requirements in TASK-1897's broader design remain later explicitly authorized
slices, not claims made by this initial preview.

No frontend destructive enablement, schema migration, service activation or live
bot query/purge is part of this implementation/test slice. Tests use disposable
SQLite and test-owned PostgreSQL fixtures. The PostgreSQL regression checks actual
RR/read-only mode, DB refusal of an accidental write, and consistency while a
separate fixture writer commits.
