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
  the current lease epoch) or `draining` (a turn running, or a send unread or
  unacked on its stream).
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

## Local configuration

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
house-preview /home/bridge/dev/.worktrees/<task>/frontend
python -m agent_bridge.house_preview_supervisor use /home/bridge/dev/.worktrees/<task>/frontend
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

`pytest` uses `/opt/house-tests` with full CPU-only app/test dependencies, separate
from the pinned bridge Python. The house also inherits Node/pnpm/git/uv/dev tools.
Use task-local mutable test environments where a project requires its own lock.
Do not run concurrent full gates merely because pytest is now available.

Verify both harness startup commands, identical mounts/session visibility,
healthy bridge, per-container limit/init, and later the aggregate slice. Final
acceptance requires HMR on the configured HTTPS origin, with the same sign-in
behavior as the shared dev site.
Before the aggregate slice lands, temporary smoke houses must be capped and
removed after checking; do not assign bots. Remove only named house containers,
not shared volumes, sessions, worktrees, app or mansion bridges. Keep receipts
outside disposable containers.
