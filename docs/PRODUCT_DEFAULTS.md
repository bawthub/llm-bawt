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

The shipped stock Setup role definitions explicitly marked `code_only: true`
are an exception to automatic seeding. Cold startup and tenant seeding leave
both their templates and history untouched, so withdrawing these definitions
restores role-description fallback where no tenant prompt exists. An explicit
save still overrides the shipped body, including an equal-body save. Existing
rows, including copies seeded by an earlier build, still take precedence and
are not reclassified or removed by recovery. The flag is source policy, not
accepted from tenant exports.

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

## Stage the Setup roster in the same product sync run

Use the same scheduled job, with an app-capable Python environment and approved
read-only source configuration. This is a source candidate generator, not a
customer install step. From a clean checkout of the release base:

```sh
# The operator supplies reviewed install terms, exclusions and input bytes.
mkdir -m 700 "$PRODUCT_SYNC_STAGE"
python -m llm_bawt.seeding.product_sync --stage "$PRODUCT_SYNC_STAGE" --setup-catalog \
  --portable-input "$PRODUCT_SYNC_PORTABLE_INPUT" \
  --portable-sha256 "$PRODUCT_SYNC_APPROVED_SHA256" \
  --excluded-bots "$PRODUCT_SYNC_EXCLUDED_BOTS" \
  --excluded-roles "$PRODUCT_SYNC_EXCLUDED_ROLES"
```

Add `--api "$LLM_BAWT_API" --key role.example` (repeat `--key` as needed) only
when the exact global prompt changes are approved for export. No `--key` means
no prompt API read. The read-only tenant projection returns only eligible
bot slugs, structural role membership and one-way revisions computed by
PostgreSQL, never raw IDs or timestamps. Private and hidden profiles are
filtered before the role closure. Keep protected slugs in private operator
inputs (`--excluded-bots` and `--excluded-roles`, explicitly empty only if
there are none). Only role keys from approved portable definitions can be
resolved; explicit denials win even for shared assignments. An unavailable
required role holds the entire snapshot without returning its key or details.
Only known product behavior assignments are ignored. Product sync never
reads tenant profile/role prose, assets, grants, bindings or prompt overrides.
The one-shot `read_setup_structure_pgq` operator helper accepts the same private
scope (including a trusted non-live role allowlist), sends it over stdin, and
uses the same read-only, repeatable-read projection as the app reader.

Use the private structural snapshot to bind existing approved portable text to
current revisions; do not export live prose for approval. Input schema `1` has
`profiles` and `roles` objects keyed by slug. Each profile has `revision`, the
reviewed `roles` list, and `preset` fields including starter `system_prompt`.
Each role has `revision`, `name`, `description`, `color`, and `prompt_key` plus
`prompt_sha256` (both null for a reviewed descriptive role). Pin refs to the
canonical metadata and body digest from `portable_role_prompt_revisions` in
the chosen product prompt source. This never duplicates a role prompt body.
The approval digest is supplied by independent exact-byte review, not computed
from whichever file happens to be on disk at run time. Source revision tokens
are private and not published; edits that advance row timestamps, including
operational edits, conservatively require a refreshed binding even if the
portable text itself has not changed.

The JSON result describes each stage. `held` or `failed` makes the command
exit nonzero. A failed prompt stage does not prevent a catalog that can stand
on shipped defaults, but a held or failed selected role-prompt change cannot
silently fall back to the older shipped version. Use only files named by the
current result: a previous catalog on disk after an error is not current.
Serialize runs sharing a staging directory. Each `prompts-*` directory is an
overlay on the exact release base (changed bodies and the merged manifest),
not a replacement for the prompt defaults directory. Review all outgoing text
and the matching dependency changes before copying them into release source.
The product catalog at `src/llm_bawt/seeding/setup_catalog.json` contains
only general-use starters. Its `stock/` refs are separate from the previous
`starter/` choices. A generated roster that identifies a particular installation
stays in private staging, even if its prose and role dependencies are portable.
Do not copy its entries over the stock catalog or add policy exceptions for them.
Only independently reviewed general-use content belongs in the distributed
catalog. Remove private staging files when no longer needed. No timer,
publication, or deployment action is performed here.

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
