---
name: tenant-tools
description: Query the configured tenant PostgreSQL database or manage its encrypted vault using bundled credential-safe helpers.
---

# Tenant tools

Bundled helpers live in this skill's `scripts/` directory. Resolve their paths
relative to this SKILL.md, not the current working directory or a global mount.
They use the configured `LLM_BAWT_POSTGRES_*` variables, then standard `PGHOST`,
`PGPORT`, `PGUSER`, `PGDATABASE` and pgpass/`PGPASSWORD` configuration. Do not put credentials in argv.

- `scripts/pgq -c 'SELECT ...'`: authorized SQL on this tenant's database.
Legacy only: `scripts/vault` works only where an operator has separately
provisioned `BAWTHUB_MASTER_KEY` and the `vault.secrets` schema. A fresh tenant
bootstrap provisions neither, so it exits before querying there.

- `scripts/vault list`: names and update times only.
- `scripts/vault get NAME`: prints a secret; use only when explicitly needed and
  never paste plaintext into task receipts, logs or model-facing output.
- `scripts/vault set NAME`: reads the value from stdin, never a value argument.
- `scripts/vault rm NAME`: removes a secret; require appropriate authorization.

The vault requires psql 16+ for `\getenv`, `BAWTHUB_MASTER_KEY`, and the
`pgcrypto`/`vault.secrets` schema; current tenant bootstrap does not provision them.
An external database needs equivalent operator-provisioned schema. Do not create
or migrate production schemas merely because a helper reports them missing.

Environment variables avoid command-line disclosure but are not an OS security
boundary. Restrict the process/container and database role. These helpers do not
bypass tool approvals or confer permission to inspect unrelated data.
