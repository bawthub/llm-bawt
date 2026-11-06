# House pool and previews

Houses are optional capped bridge containers, one tenant/harness per house.
The default pool is empty. They share the mansion filesystem; task worktrees,
not copied checkouts or container layers, hold persistent code and sessions.
This file owns image/pool/startup instructions; routing and lease behavior is
owned by the `agent_bridge` house router.

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
adds the mansion's Codex config/sessions mounts at `/home/bridge/.codex`.
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

## Preview frontend

The private `frontend_env_file` contains the same shared frontend `DATABASE_URL`
and nonsecret application configuration as regular dev. The existing exported
service-token volume is mounted read-only. Never copy secrets into pool JSON,
generated routes, logs or source. Each house overrides `BAWTHUB_AUTH_URL` to its
own HTTPS origin and requires sign-in sessions; cookies remain origin-scoped.

Inside the house, install dependencies in its task worktree using the normal
frontend workflow, then run:

```sh
house-preview /home/bridge/dev/.worktrees/<task>/frontend
```

This starts the normal Vite + Hono pair on 5173/3001. A conflicting port fails,
not a silent move to 5174. No dependency install, Prisma migrate/db push or
schema change runs in the launcher. Hono is part of the frontend; llm-bawt
app/router, voice, schedulers and DB stay shared. Do not start per-house Python
backend apps or perform shared schema mutations from a preview.

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
acceptance requires HMR and real sign-in on the configured HTTPS origin.
Before the aggregate slice lands, temporary smoke houses must be capped and
removed after checking; do not assign bots. Remove only named house containers,
not shared volumes, sessions, worktrees, app or mansion bridges. Keep receipts
outside disposable containers.
