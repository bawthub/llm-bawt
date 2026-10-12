# Platform Node runtime — TASK-1062

Target: **Node 24.21.0 LTS**, bundled npm/npx **11.19.0**, explicitly installed
Corepack **0.36.0**. BawtHub selects pnpm **10.32.1** via `packageManager`.
This extends TASK-1061's Claude-only activation to the platform. REA installation
is TASK-1060 and is not included.

## Inventory and rollout receipt (2026-10-12)

| Target | Before | Verified after |
| --- | --- | --- |
| Echo host default | 22.22.1 | 24.21.0 |
| Ubuntu gateway host default / running gateway | 22.22.0 | 24.21.0 |
| Claude bridge | 20.19.2 (TASK-1061) | 24.21.0, healthy |
| Codex and OpenClaw bridges | 20.19.2 | 24.21.0, healthy |
| BawtHub dev frontend | 22.22.2 | 24.21.0, healthy, DB connected |
| Running tenant-e2e frontend | 22.22.2 | 24.21.0, healthy, DB connected |
| Playwright MCP | 22.23.1 | 24.21.0 live; Chromium and RTX 5080 WebGL verified |
| BawtHub production | 22.23.3 | 24.21.0, v0.1.68, healthy, DB connected |
| BawtHub CI / release gates | 20 / 22 | 24.21.0; managed release gates passed |

Python app/ops/model/voice services, Redis, Postgres and other non-Node services
are not Node upgrade targets. Crawl4AI has no Node on PATH and its upstream-owned
browser payload is not rewritten. Vendor appliances and Electron-bundled engines
are not replaced arbitrarily. The powered-off tenant VM remains off and is not
claimed upgraded. Retained rollback images and distro `/usr/bin/node` are not
active platform defaults.

## Rebuild contract

- `Dockerfile.agent-bridge`: official checksummed tarball; all bridge Python
  dependencies constrained to the pre-upgrade versions. See `AGENT_BRIDGE_NODE.md`.
- `Dockerfile.playwright`: pinned upstream MCP/browser payload plus pinned Node
  Debian stage; replace npm's directory, never merge dependency trees.
- BawtHub `.node-version`, frontend `engines`, both frontend Dockerfiles and both
  setup-node workflows align to Node24. `scripts/test_node_runtime.py` prevents drift.
- Dev Dockerfile now copies `patches/` before frozen dependency installation.
- BawtHub `frontend/Dockerfile.node-runtime` is ONLY for retained tenant snapshots.
  It preserves application/node_modules/entrypoint. Canonical prod always uses
  `bawthub.release-prod`; never use the tenant overlay as a production shortcut.

## Host toolchains

`scripts/install-node-runtime.sh` installs a verified official tarball at
`/opt/node-v24.21.0`; `/usr/local/bin/{node,npm,npx,corepack,pnpm,pnpx}` resolve into
that installation. Distro packages remain untouched: removing them would remove
129 reverse dependencies on echo. Existing local commands are preserved under
`/var/lib/llm-bawt/runtime-backups/TASK-1062-<UTC timestamp>/` (including an absent
command manifest). Do not remove distro Node as cosmetic cleanup.

Ubuntu's `openclaw-gateway` user unit now starts `/usr/local/bin/node`; its actual
running executable resolves into `/opt/node-v24.21.0`. A unit restart was explicitly
authorized during this rollout. Future gateway/bridge interruptions still need
their own authorization.

## Image rollback identities

| Surface | Candidate image ID prefix | Preserved rollback tag |
| --- | --- | --- |
| All agent bridges | `ef94417af61e` | `llm-bawt-agent-bridge:task-1061-rollback` (`59d7ff8e9b92`) |
| Dev frontend | `4ccf5119abd5` | `bawthub-frontend:task-1062-rollback` (`1091132d901d`) |
| Playwright | `d98ae9d3cdc6` | `llm-bawt-playwright-gpu:task-1062-rollback` (`32542444b9fc`) |
| tenant-e2e frontend | `c5ed4df2eb78` | `bawthub-tenant-frontend:task-1062-rollback` (`7040ac74ce2c`) |

Before any future rollback, recheck full image IDs and exact target configuration.
Use ops where supported, obtain exact authorization, recreate only the affected
service and verify health. Never restart the whole stack as a shortcut.
Tenant host files `/home/nick/tenant-e2e/docker-compose.prod.yml` and `.env` have
mode-0600 `.task-1062-backup` copies. The compose frontend accepts `FRONTEND_IMAGE`
and `.env` selects `bawthub-tenant-frontend:task-1062-node24`; application version and
DB schema were not upgraded. These two host-file edits were made through SSH,
not tracked Edit calls.

## Validation

- Mount-free bridge script verifies Node/npm/npx/pnpm, ESM/fetch/WebSocket/crypto,
  Python imports and 49 constrained versions, both harness CLI versions and `llm`.
- Dev candidate passed client/server typechecks against existing read-only mounted
  dependencies and a sharp native operation. Active dev passes both typechecks
  and HTTP health/DB connection checks.
- Tenant candidate passes tsx, Prisma client load and sharp; live health is healthy.
- Isolated Playwright candidate launches bundled Chromium and reads page content.
- Four Node runtime contract tests and 26 total release-tool tests pass.
- Full JS suite: **1073 pass / 64 fail**, identically under Node24 and retained
  Node22.22.1. Failed test names match exactly; failures need DB fixtures or Node's
  experimental module-mock flag. This is not an all-green suite and no test gates
  were weakened for the upgrade.

## Host GPU incident — resolved, without CPU fallback

At activation, the NVIDIA prestart hook failed before Playwright could start:
loaded kernel module **595.91.07**, installed NVML and on-disk module **595.99.02**.
Nick separately authorized one echo reboot AFTER production verification. That
reboot loaded **595.99.02**; host `nvidia-smi` now succeeds. Boot identity changed
from `7940e1d4-9cd4-45c2-b123-7cf48c06893a` to
`a443066f-68ed-4593-9e50-baef4489217e`. No driver package change, module unloading,
or CPU downgrade was performed.

The never-started Playwright container remained Created after reboot. Approved
`llm-bawt.restart-aux` job **f35b80aa83824daf8a80359ddc4b2373** then succeeded,
exit 0. Live Node is 24.21.0. Chromium **149.0.7827.22**, launched inside the running
container with its unchanged GPU configuration, passed DOM evaluation and dev
navigation (HTTP 200, title BawtHub). WebGL2 reports:
`ANGLE (NVIDIA, Vulkan 1.4.329 (NVIDIA NVIDIA GeForce RTX 5080 (0x00002C02)), NVIDIA)`.
A read-only MCP protocol readiness test successfully initialized the service,
listed **23 tools** (including browser navigation/evaluation), and closed its test
session. A harness session established while the server was down may retain its
failed connection until the next connection attempt/session.

## Final production and recovery receipts

- Managed release **8a3cddaf238241c5bfea458f164fd5e1** and child deploy
  **3423d6f1d327465daf0700e3e177e27c** succeeded with exit 0.
- Production **v0.1.68**, source **800c65630d82a3c55de807b24d6ba57be215442c**,
  GitHub run **38322573189**, image digest
  `sha256:3b07bdcf365ecbecfa487670de12b868cb487292ab97dcc3fd86a5f4a1f7687d`.
- Post-reboot production, dev and tenant health APIs are healthy/DB connected;
  app `/health` is OK. All three agent bridges and the model/voice/database/cache
  containers recovered healthy. All first-party active Node targets in the table
  were rechecked at **24.21.0**; the Ubuntu gateway remains active on its explicit
  `/opt/node-v24.21.0/bin/node` executable.
- The powered-off tenant VM stayed off. Rollback images remain available.
- The queued self-verification delivery did not start after reboot; Nick's resume
  message began verification instead. It was cancelled while still QUEUED to avoid
  duplicate work. No claim of automatic callback completion is made.
- TASK-1062 is submitted for review after recovery, not before it.
