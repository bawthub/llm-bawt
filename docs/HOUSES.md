# House pool and previews

Houses are optional capped bridge containers, one tenant/harness per house.
The default pool is empty. They share the mansion filesystem; task worktrees,
not copied checkouts or container layers, hold persistent code and sessions.
This file owns image/pool/startup instructions; routing and lease behavior is
owned by the `agent_bridge` house router.

## Moving in and out

A bot moves itself with the `self_house` MCP tool (`/v1/houses/self`, signed
turn only, no bot argument; rules in `llm_bawt/house_tenancy.py`):

- `list`: the configured pool (`BAWT_HOUSE_POOL_FILE`, this deployment's pool
  file mounted read-only) plus any house holding a lease or tenant, their
  tenants (private bots masked) and a status for the caller's harness: `free`,
  `occupied`, `unconfigured` (not in the pool, or no pool), `incompatible`
  (another harness), `down` (no live lease), `recycling` (asked to restart at
  the current lease epoch), `draining` (a turn running, or a send unread or
  unacked on its stream), or `retiring` (reserved for count-based removal).
- `move_in`: a named house, or any free one; only `free` is claimable, so an
  occupied house is always refused. A bot housed elsewhere must move out
  first. Never pins. Applies from the bot's next turn.
- `move_out`: leaves the caller's own house, idempotent in the mansion; a pin
  bound to that tenancy goes with it. An emptied house recycles.

Every write of a bot's `house` (profile PATCH, `/new`, the idle janitor, self
moves) takes one advisory lock in `BotProfileStore.upsert`, so a race has one
winner. Recycling an emptied house rechecks for tenants under that lock, so a
claim racing a release either cancels the recycle or is refused. Unreadable
Redis refuses rather than guesses.

Only `claude-code` bots are eligible for now (by runtime, not name); others
get `unsupported_harness`.
Bots never patch another bot's `house` field. Pinning, and assigning a bot by
profile, stay with a coordinator or the user.

## One count, one reconciliation path

For a new tenant, set `HOUSES=3` in `deploy/tenant.env`. Use
`deploy/bootstrap.sh houses` to inspect the read-only plan and
`deploy/bootstrap.sh houses --apply` to reconcile. Normal bootstrap `up`
uses the same reconciler after starting its named core services. Both the app
image and installer must contain this feature before the first apply.

An extended Compose deployment uses BawtHub's
`deploy/house_reconcile_host.py` with `layout: "extends"`. Keep these nonsecret
settings outside product source control:

```json
{
  "layout": "extends",
  "project": "tenant",
  "compose_command": [
    "/deployment/llm-bawt/scripts/compose-secrets",
    "--project-dir", "/deployment/llm-bawt", "--",
    "--project-name", "tenant",
    "-f", "/deployment/llm-bawt/docker-compose.yml",
    "-f", "/private/houses/houses.compose.json"
  ],
  "compose_file": "/private/houses/houses.compose.json",
  "pool_file": "/private/houses/houses.json",
  "routes_file": "/watched/traefik/houses.local.yml",
  "journal_file": "/private/houses/reconcile-state.json",
  "renderer": {
    "base_compose": "/deployment/llm-bawt/docker-compose.yml",
    "route_template_file": "/deployment/bawthub/deploy/traefik/dynamic/routes.yml",
    "origin_template": "https://{id}.preview.example.test",
    "pool": {
      "image": "example-house:pinned",
      "base_image": "example-bridge:pinned",
      "memory": "4g",
      "preview_network": "tenant-frontend",
      "service_token_volume": "tenant-service-token",
      "frontend_env_file": "/private/houses/frontend.env"
    }
  }
}
```

The command array must preserve **all** existing project overlays in their
existing order, with the published house overlay last. The example is not a
replacement for a deployment's full stack. Preserve its image pins, mounts,
frontend settings and optional `mansion_origin` in `renderer.pool`; do not put
a second `houses` list there. The existing app and its published house overlay
must be available for the app-side renderer. Activate the new app code through
the deployment's normal approved procedure before using its runtime command.

```sh
HOUSES=3 python3 /deployment/bawthub/deploy/house_reconcile_host.py \
  --settings /private/houses/reconcile.json
# Review the same plan before the approved service change:
HOUSES=3 python3 /deployment/bawthub/deploy/house_reconcile_host.py \
  --settings /private/houses/reconcile.json --apply
```

The count derives `h1` through `hN` (0–16). Planning reads actual Compose
ownership, service hashes, app occupancy and DNS; it writes only disposable
rendering scratch. Apply adopts matching healthy houses and starts only
missing/stopped named services, with `--no-deps --no-build --no-recreate`.
Definition drift refuses the whole change; this is not an image-upgrade tool.
No whole-stack `up`, `down`, orphan cleanup or volume deletion is used.

Lower the same count to remove surplus houses. Assigned bots, active turns,
unread/unacked sends, another reconciliation or unreadable runtime state refuse
removal. The app reserves all targets under the assignment lock before any
deletion, and **every** profile assignment writer checks that reservation.
Pool publication preserves an existing file's inode for single-file binds.
Health, the expected harness lease and preview HTTP/TLS are checked before
success. A rerun does not recreate already matching houses.

Failures leave a durable journal and any removal reservations in place. Resume
with the same count and rendered definitions. Do not delete the journal to
bypass a refusal. Recovery that restores a retained house must restore its
configuration, start and verify it, then run the app-side runtime command's
`cancel` action for the journal's exact `reconcile_id` and house IDs. Release
after removal requires an empty target absent from the advertised pool.
Both actions take bounded JSON on stdin via the deployment wrapper's
`exec -T app python -m llm_bawt.house_reconcile_runtime`. Never clear
reservations with host database edits.

## DNS and existing proxy policy

Optional `edge` settings support only AdGuard rewrites and UniFi static A
records. Configure `answer`, plus `adguard` (`base_url`, `username_env`,
`password_env`) and/or `unifi` (`base_url`, `site`, `api_key_env`). The named
environment variables come from the deployment's credential helper, never
command arguments or committed files. TLS verification is on; use
`insecure_tls: true` only for an explicitly accepted local API endpoint.
Exact records are adopted, conflicts refuse, and only records created and
recorded by this reconciler are eligible for later deletion. Adopted records
and existing certificates/proxy policy are never removed or rewritten.

Proxy routes clone the deployment's existing Traefik rules. Certificate
issuance and ingress authentication/LAN-only policy are not inferred from DNS.
Verify them explicitly and record the nonsecret receipt as
`edge.proxy_policy_receipt`; without it, output lists this remaining manual
check. No NPM restart, public DNS change or new certificate is implicit.

## Advanced explicit-pool configuration

Copy `docker/houses.example.json` outside source control. All images, resource
limits, hostnames, network/volume names and secrets-file paths are deployment
data. An empty `houses` list renders no containers, networks or volumes. Add:

```json
{"id":"h1","harness":"claude-code","origin":"https://dev1.example.test","memory":"4g"}
```

`harness` can instead be `codex`, using the same image and definition. IDs match
`^[a-z0-9][a-z0-9-]{0,30}$` (TASK-2077). Set `cgroup_parent` to the locally
provisioned aggregate slice when available; empty means the engine default.
Every house always has its own positive `mem_limit` and `init: true`.

Use the *same base Compose files, project name and .env as the mansion*. The
renderer extends its Claude service to inherit identical `/home/bridge/dev`,
host-identical dev root, `.claude`, SSH, scripts and broker-token mounts, then
adds explicit host Codex config/sessions binds at `/home/bridge/.codex`.
Neither harness needs a separate `codex-bridge` mansion service. A GPT bot can
use `claude-code` with the proxy; `codex` selects the legacy native harness,
not the GPT model family.
The dev source and optional signed Git/GitHub account-config mount are included
explicitly: extending the base file does not inherit later dev overlays.
Do not use a new Compose project: the house must join the existing app/Redis
network. The frontend network and service-token volume are external references,
not new copies. Inspect `docker compose config` without publishing credentials.

```sh
python -m agent_bridge.house_pool compose /private/houses.json \
  --base-compose /deployment/llm-bawt/docker-compose.yml > /private/houses.compose.yml
python -m agent_bridge.house_pool routes /private/houses.json \
  --route-template /deployment/bawthub/deploy/traefik/dynamic/routes.yml \
  > /private/houses.routes.yml
```

Review the rendered files; they are JSON, also valid YAML. Use the Compose output
as a final overlay after the deployment's existing files. Pin `base_image` to
the running mansion bridge image for builds. Build only `house-h1`; start only
that service with `up -d --no-deps`, never recreate the mansion to start a house.
`CODEX_MODEL` is intentionally empty: bot/endpoint configuration supplies models.
Houses do not allocate themselves or write bot profiles/Redis leases. Do not
route bots into them before the house-reader/fencing change is active.

## Self-hosted tenants

A tenant installed with `bawthub/deploy/bootstrap.sh` does not use the
renderers above. Its installer writes its own house services (`house-<id>`,
the tenant's pinned agent-bridge image, which carries `house-preview`) and a
pool file holding only `houses`; the app checks just those rows
(`validate_houses`). A tenant preview origin may be `http` and carry a port:
each house publishes its Vite port on the tenant's own host. The Traefik route
renderer still accepts only bare HTTPS origins. See `deploy/README.md` there.

## Preview frontend

The private `frontend_env_file` contains the same shared frontend `DATABASE_URL`
and nonsecret application configuration as regular dev, and nothing else
decides how a preview signs in: a preview authenticates exactly like the shared
dev site (TASK-2135). If dev runs without sign-in sessions, so do previews, and
`/api/auth` and `/api/setup` answer 404, so the owner-setup wizard can't run or
write to the shared DB. Never add an owner account to the shared DB for a
preview. The existing exported service-token volume is mounted read-only.
Never copy secrets into pool JSON, generated routes, logs or source.

The house runs its preview itself (`agent_bridge.house_preview_supervisor`,
enabled by `BAWT_HOUSE_PREVIEW=on` in the rendered service). Port 5173 always
answers:

- Empty house, or an occupant with no BawtHub task worktree: a built-in page
  says so, names the occupant and explains how a preview starts. No 502.
- Occupied: the checkout is the occupant's choice, else its most recently used
  `.worktrees/.wt-bawthub-<bot>-TASK-N`. The house installs from the lockfile
  (`pnpm install --frozen-lockfile --prefer-offline`, shared store at
  `<dev>/.pnpm-store`, linked rather than copied), runs `prisma generate`, then
  starts the normal Vite + Hono pair on 5173/3001. The page shows progress, or
  the log tail if the install or the preview fails, and it retries with backoff.
- After any house or container restart the same preview comes back by itself.
  An empty house forgets the occupant's choice.

To serve a different checkout, the occupant runs either of:

```sh
house-preview "${FRONTEND_CHECKOUT:?set the absolute task frontend checkout}"
python -m agent_bridge.house_preview_supervisor use "${FRONTEND_CHECKOUT:?}"
```

`python -m agent_bridge.house_preview_supervisor clear` goes back to the latest
task worktree. Don't start a second Vite on
5173 by hand. No Prisma migrate/db push or schema change runs; Hono is part of
the frontend; llm-bawt app/router, voice, schedulers and DB stay shared. Do not
start per-house Python backend apps from a preview. Campus links only to a live
preview; otherwise it says the preview is starting or not running.

Install the generated route file alongside the existing watched Traefik file.
It clones the owning frontend/frontend-api rules so API prefixes stay in sync;
unclaimed `/api` traffic still reaches the existing shared voice/backend route.
Attach houses to the configured frontend network so Traefik resolves them.
Page assets and HMR WebSockets reach Vite; `/api/auth` reaches house-local Hono.
The browser auth client stays same-origin. There is no broad CORS/origin bypass.

For the deployment's preview hostnames, provision LAN DNS (wildcard where
supported), a covering TLS certificate and an edge/NPM route forwarding to
Traefik with the original Host and WebSocket upgrade headers. Keep previews
LAN/VPN-only at the edge; DNS alone is not access control. Check hostname
collisions before reserving dev1..N. Do not overwrite another tenant's hostname.

## Verification and removal

`pytest` uses `/opt/house-tests`, separate from the pinned bridge Python. It holds
the app with its `dev,service,mcp,search` extras, installed from `pyproject.toml`
rather than the frozen lock, which is enough for focused tests. It is not a
full-unit or full-gate environment: no memory extra, offline tokenizer cache,
local Redis or private PostgreSQL. The house also inherits Node/pnpm/git/uv/dev
tools. Use task-local mutable test environments where a project requires its own
lock. A full pre-check is the integrator's solo host `make gate`, whose
exact-SHA receipt is reused for publication (see `HEAVY_QUEUE.md`).

Verify both harness startup commands, identical mounts/session visibility,
healthy bridge, per-container limit/init, and later the aggregate slice. Final
acceptance requires HMR on the configured HTTPS origin, with the same sign-in
behavior as the shared dev site.
Before the aggregate slice lands, temporary smoke houses must be capped and
removed after checking; do not assign bots. Remove only named house containers,
not shared volumes, sessions, worktrees, app or mansion bridges. Keep receipts
outside disposable containers.
