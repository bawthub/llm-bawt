# Product defaults, tenant overrides and approved promotion

## Ownership and lifecycle

Product defaults are portable, reviewed release content. A developer database is
not the public fixture source merely because it is live. Tenant configuration
and overrides remain authoritative for that tenant.

| Domain | Owner and natural key | Apply/update/reset | Version and rollback |
|---|---|---|---|
| Schema | Owning application; migration identity | Existing schema/migration engine; not fixture export | Migration history; use its recovery procedure |
| Prompts | Product catalog by key; tenant override by `(key, scope_type, scope_id)` | `TenantSeeder.apply()` seeds missing defaults and advances proven default-managed rows. Ordinary saves create overrides. Reset deletes the selected override and inherits immediately. | Product content SHA-256 is independent of append-only DB edit version. Revert the reviewed fixture to roll back managed defaults; overrides stay unchanged. |
| Ops catalog | Product starter specifications by slug; tenant/operator owns activation and configuration | Existing `seed_all` inserts missing disabled operations. Do not overwrite an enabled/customized operation on upgrade. Review an explicit catalog reconciliation separately. | Catalog/source revisions plus operation audit; restore configuration under the owning operational procedure |
| Approval policies | Product starter rules; tenant/operator owns effective approval decisions | Existing policy store uses deterministic default identities and inserts missing rules. Tenant edits remain in place; changing authorization is not a prompt-default upgrade. | Policy history/audit and source revision; explicitly restore the intended rule, never bulk reseed over edits |
| Historical agent/project/task seeds | Installer/developer fixture, not tenant defaults | BawtHub `frontend/prisma/seed-agents.ts` is historical sample data. Do not import its people, bots or tasks into a tenant. First bot/model setup follows provider connection. | No automatic refresh/reset of user work |
| Voice catalog | Voice service/package owns supported runtime voices | Runtime `voices.yaml` catalog is deployed with the service. Tenant selection is configuration, not a DB snapshot to promote. | Roll back compatible service/catalog artifacts; keep tenant selection |
| Credentials, personas, profiles, user data and environment-specific configuration | Tenant only | Never export as portable prompt defaults. Provision and back up through their existing owners. | Tenant backup/recovery, not public fixtures |

Use the existing SQLModel seeder and packaged UTF-8 prompt bodies/JSON manifest.
Prisma is already the frontend schema owner, not a replacement backend seeder.
A new fixture/seed library would not solve override intent, scope or portability;
no additional framework is needed for this contract.

## Prompt resolution and explicit intent

Resolution is exact scoped override, then global tenant value, then the current
packaged product default. All seven Repos templates are in the canonical prompt
catalog and therefore the shared tenant seeder; consumers resolve those bodies,
not a separately maintained frontend string.

Seed/sync-written global rows with matching tracked hashes follow upgrades.
Legacy rows with proven seed/sync provenance can follow too. Human-edited,
ambiguous and bot-scoped rows are not adopted merely because their body equals a
later product release. A client cannot opt into following by returning an old
`default_sha256` or claiming default intent in metadata.

Every ordinary save, including a title-only/no-op save, records an explicit
override on the server and removes inherited tracking. Sync classification is a
separate explicit API operation; a normal save preserves its stored value.
`sync=product` means eligible for selected, approved export, not automatically
published and not permission to overwrite tenant intent. Private scopes and
categories remain excluded regardless of that flag.

Reset uses the current fallback immediately and follows future product releases.
For a bot it removes only that bot's override, restoring the global fallback.
Nonempty prompt bodies are required by the save API; empty does not mean
"inherit" or "disabled". Use reset for inheritance. Empty built-in install-notes
defaults are a documented no-additional-instructions value, not an override
deletion signal. Disabling another feature belongs to that feature's setting.

Responses expose `provenance`, `default_version`, `effective_version` and
`default_outdated`. Product versions are content hashes, not DB edit numbers.
Legacy `source` retains its storage meaning for existing clients.

## One live edit, one explicit promotion command

1. Edit/validate the selected live global prompt once through the registry.
   Keep tenant/private text install-only. Editing a prompt does not itself
   authorize a product release.
2. Once that exact portable change is approved for promotion, run in the isolated
   source checkout, selecting the source API and every approved key explicitly:

   ```sh
   python3 -I scripts/prompt_sync_export.py src export \
     --api "$LLM_BAWT_API" \
     --root src/llm_bawt/prompt_defaults/synced \
     --key repos.workspace.commit_task
   ```

   Repeat `--key` only for other approved keys. Add installation-specific names
   and hostnames with `--install-term` or `PROMPT_SYNC_INSTALL_TERMS`.
   The exporter reads only selected exact global rows, never enumerates bots,
   writes allowlisted public fields deterministically, validates declared
   placeholders/metadata, and holds private/credential-like content. It neither
   writes the live database nor publishes source. Do not manually copy the same
   prompt body into another seed or frontend fallback.
3. Inspect the fixture diff for private prose as well as behavior, run focused
   checks, then use the normal review and exact-tip publication process. Pattern
   checks cannot prove arbitrary prose contains no private facts. A held prompt
   is not exported; remove private material in the live edit or keep it local.
4. Promotion changes product defaults, not any tenant's override intent.
   After the approved default is actually shipped, explicitly reset the source
   tenant's override if it should follow that product again. Never reset before
   the new release is available; that would restore the old default. Reset/live
   writes require their usual authority, not permission inferred from this file.

## Verification and drift

Offline CI tests fixture registration, portable metadata, placeholders, export
selection/privacy holds and deterministic output. SQLite fixtures test successive
default releases, retained tenant and scoped overrides, conservative legacy
migration, concurrent-save CAS fencing, reset and immediate/future inheritance.
Fresh Repos seed/render/fallback tests use the same canonical bodies.

Offline validation cannot inspect a live database. An authorized operator can
run the read-only live probe:

```sh
python3 -I scripts/prompt_sync_export.py src check --api "$LLM_BAWT_API"
```

It requests hashes/provenance, not bodies. The API reports stale managed defaults
separately from intentional overrides and optional promotion candidates. Health
only flags lagging managed defaults, not tenant edits that happen to differ.
Use filtered exact prompt reads instead of an installation-wide probe when a
tenant has protected scopes you must not inspect.

Tests do not authorize live writes, restart, image release or deployment. Keep
source evidence separate from usable activation evidence and respect active
operator holds.
