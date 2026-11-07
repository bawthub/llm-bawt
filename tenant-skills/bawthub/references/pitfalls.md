# Common pitfalls (learned the hard way) — full detail

> Part of the bawthub skill. Read when a symptom below matches. The SKILL.md index names each pitfall; this file has the full triage/diagnosis.

### Mobile-only jitter, keyboard, scroll, or focus behavior

Use [mobile-troubleshooting.md](mobile-troubleshooting.md) before guessing at a
layout fix: capture bounded geometry from the actual phone via `?viewportTrace=1`,
read `/api/clog`, and distinguish native viewport movement from transcript UI
mounting/unmounting. Includes the TASK-1097 iPhone-confirmed queue-row case and how
to adapt the opt-in recorder to other mobile-only interactions.

### Blank / empty screen — a BUG to fix, never a user cache problem (TASK-1027)

**Hard rule:** a blank home screen (or any blank page) means our code crashed or hung. Diagnose and fix it in code. **Forbidden responses:** "clear your cache", "clear site data", "reinstall the app/browser", "try incognito", "hard refresh". If wiping storage "fixes" it, that PROVES the bug is ours: we persisted state the current code can't read. The user has heard these excuses enough. Don't give them.

**Get the real error. Don't guess:**
1. `src/app/RootErrorBoundary.tsx` wraps the whole app in `main.tsx`. On any render crash it shows a visible error card and POSTs the error, the JS stack and the component stack to `/api/clog`. Read it (`HOST` = the Traefik host's LAN address):
   `curl -s -H "Host: <dev-host>" "http://$HOST/api/clog" | python3 -c "import json,sys;[print(e['ua'],e['url'],e['text'][:4000]) for e in json.load(sys.stdin)['entries'] if 'RootErrorBoundary' in e['text']]"`
   (The buffer is in-memory and resets on server restart. Ask the user to reload once on the device if it's empty.)
2. Never remove the boundary or make it render `null`. A blank screen also kills `useVersionGate`, which lives inside the tree, so a broken client can never self-heal.

**Usual root causes, in order:**
- **Persisted client state in an old shape** (localStorage/sessionStorage/IndexedDB) fed into the FIRST render, e.g. as TanStack `initialData`. It throws before the network fetch can overwrite it, so the crash repeats on every load until storage is wiped. TASK-1027: `bawthub-navigation-v1` held pre-TASK-966 `aliases: string[]`, and `alias.from.split()` crashed `AppMainNav`. **Fix pattern:** version the key (bump it whenever the cached shape changes, delete the legacy key) AND shape-validate on read, dropping anything invalid (see `isUsableCatalog` in `lib/navigation/client.ts`). Every new persisted cache must do both.
- **Stale lazy chunk** after a deploy or dev restart (`Failed to fetch dynamically imported module` / `Importing a module script failed`). Handled by the boundary plus `vite:preloadError`, both of which reload once with a loop guard. If it still happens, fix the guard; don't tell users to reload.
- **Browser-incompatible syntax** taking a whole module down on one engine (TASK-1026: GFM autolink lookbehind on Safari).

### "<dev-host> is down / slow / 404" — triage in THIS order (do not reflex-restart)
Traefik routes from `deploy/traefik/dynamic/routes.yml` (file provider, since
TASK-1173) and ignores container health: a stopped or crashed upstream answers
**502**, an unhealthy-but-running one still gets traffic. A 404 means no router
matched (wrong host or a missing `/api/*` prefix in routes.yml). Check the
container first; don't reflex-restart Traefik, which has no labels to reload.

Triage (on the host; from a bridge, over SSH to the host):
```bash
docker ps --filter name=bawthub-frontend-1 --format '{{.Status}}'   # exited/restarting? → 502
docker stats bawthub-frontend-1 --no-stream --format '{{.MemUsage}}' # climbing fast = a leak/spiral
docker inspect bawthub-frontend-1 --format '{{json .Config.Healthcheck.Test}}'  # should probe a cheap URL
docker logs bawthub-frontend-1 --tail 50                              # vite/tsx startup or crash?
```
- Keep the healthcheck on a **cheap** endpoint (Vite serves a static `index.html` at
  `/`, so a `/` probe is fine now — there's no SSR render behind it anymore). If you
  see repeated OOM/restart, capture `docker stats` before restarting — a `docker
  restart` only clears memory (maintenance), it's not a root-cause fix.

> Historical note: under the old Next.js dev server this failure was a
> healthcheck-driven **SSR render spiral** (probing the server-rendered `/`
> dashboard, overlapping renders sharing one Prisma connection, GC-thrash → OOM →
> unhealthy → 404). That mechanism is gone — the Vite SPA is static, no per-request
> server render — so the `--max-old-space-size` / Next-soft-restart tuning that used
> to live here no longer applies.

### Hydration mismatches in shared UI components
Any component that branches render output on `typeof window !== "undefined"` /
`typeof CSS !== "undefined"` hydrate-mismatches in React 19 and bails out of patching,
breaking interactions. Example: `Input.tsx` gating `fieldSizing: "content"` on a
module-level `CSS.supports` check. **Pattern:** emit identical server/client markup;
put browser-feature gates in `useEffect`. Unknown CSS props are ignored by browsers,
so emit them unconditionally.

### Stale-async races on selection-driven pages
`useEffect(() => { fetch(...id...) }, [id])` gets out-of-order responses if the
selection flips faster than the fetch resolves — older response lands last, form
flashes then empties (e.g. `/tools/profile`). **Pattern:** snapshot the selection at
request time, compare against a ref of the latest on resolution, drop if changed.

### Config / server edits restart the dev server (not HMR)
`vite.config.ts`, `tsconfig.json`, `package.json`, `.env` edits restart the Vite
dev server; edits under `server/**` restart the Hono API via `tsx watch`. Either
drops SSE/WebSocket connections and the chat page logs a reconnect storm (one
reconnect is normal). If a user reports unexplained SSE drops, check
`docker logs bawthub-frontend-1` for a vite restart or a `tsx` server reload.

### Vite HMR websocket must reach the dev host
Vite serves HMR over a websocket from the dev server. If a dev hostname isn't
accepted by Vite (`server.allowedHosts` / `server.hmr` / `server.cors` in
`vite.config.ts`) or a proxy in front drops the ws upgrade, the page loads over
HTTP but HMR silently dies (edits don't reflect, no reconnect). Add every dev
hostname (`<dev-host>` + any new dev-only host) to the Vite server config.

Dev is normally reached at **`<dev-host>`** (LAN HTTPS, no OAuth), which sits
in front of the Vite ws cleanly. Historical note: under Next.js there was an
OAuth-gated-HMR outage on the OAuth-gated prod host where NPM's `auth_request`
killed the long-lived `wss://…/_next/webpack-hmr` and looped full reloads — fixed
with an NPM `location` override. That `/_next/webpack-hmr` path no longer exists
(Vite uses its own ws path). If HMR-over-proxy trouble ever recurs on an
OAuth-gated host under Vite, find Vite's actual ws path and allowlist THAT — do not
resurrect the `/_next/` block. Simplest workaround remains the no-OAuth
`<dev-host>` / direct LAN host.

## Quick-symptom index (moved from SKILL.md, TASK-1640)

- **"<dev-host> is down/slow"** → 502 = upstream container down (check `docker ps --filter name=bawthub-frontend-1` first); 404 = no route in `deploy/traefik/dynamic/routes.yml`. Do not reflex-restart Traefik.
- **Hydration mismatch** → components branching render on `typeof window`/`CSS.supports` break React 19 patching; emit identical markup, gate in `useEffect`.
- **Stale-async race** on selection-driven pages → snapshot selection at request time, compare on resolve, drop if changed.
- **Config/server edits restart the dev server** (not HMR) → SSE/ws drop + one reconnect is normal; check container logs for a vite/tsx restart.
- **HMR silently dead** → the dev hostname isn't in Vite's `server.allowedHosts`/`hmr`/`cors`, or a proxy drops the ws upgrade. Use `<dev-host>`; never resurrect the old `/_next/webpack-hmr` block.
