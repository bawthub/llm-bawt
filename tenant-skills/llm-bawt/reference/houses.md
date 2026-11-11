# Houses and where tests run

Owning reference for bot houses and test placement. Other skills link here
instead of restating it. Product details: llm-bawt `docs/HOUSES.md` (image,
pool, preview launcher) and `docs/HEAVY_QUEUE.md` (queue containment).

## What a house is

A house is a capped bridge container for one harness (`claude-code` or
`codex`) on the configured shared dev filesystem. Worktrees, sessions and
skills are the same files the mansion sees; nothing persistent lives in the
container. The mansion is the shared Claude bridge, which runs Anthropic and
proxy-routed GPT bots. The separate native Codex mansion service is retired;
optional `codex` houses remain supported and extend the Claude Compose parent,
not a removed Codex service.

## Routing

- A bot's `house` profile field picks its house; empty means the mansion.
  IDs match `^[a-z0-9][a-z0-9-]{0,30}$`; the profile API rejects others.
- A house bridge runs with `BAWT_HOUSE=<id>` and reads only
  `agent:commands:house:<id>`. The mansion reads `agent:commands`.
- Each house bridge holds `agent:house:<id>:lease:<harness>` (TTL 30 s,
  renewed every 10 s) and a start epoch. New turns go to the house only while
  that lease exists.
- Lease missing: new turns run in the mansion, and the user gets one push/bell
  notice per outage. In-flight turns never move; a command from an older epoch
  is refused, never run late or twice.
- Steer, approval, answer and abort follow the turn's recorded location, not
  the bot's current setting.
- Run exactly one container per house ID. Takeover by a newer copy is not
  atomic; it is safe only because of that singleton rule.

## Assigning a bot

Pool, image and rebuild steps are deployment-specific; see the
local-infrastructure skill for this installation's pointer.

A claude-code bot moves itself with the `self_house` MCP tool: `list` shows
each house's status (`free`, `occupied`, `unconfigured`, `incompatible`,
`down`, `recycling`, `draining`), `move_in` takes a named or any free house,
`move_out` leaves its own. Only a free house in the configured pool is
claimable; occupied is always refused, and a bot housed
elsewhere moves out first. Self-moves never pin. The move applies from the
next turn. Codex/native and chat bots are refused for now.

Bots never patch another bot's `house`. Assigning someone else or setting a
pin is a coordinator's or the user's call, via the profile API
(`PATCH /v1/bots/<slug>/profile`, between that bot's turns). Every `house`
write shares one lock, so it cannot race a self-move.

Lifecycle: `/new` and the idle janitor vacate unpinned tenants (unless the
reset asked to keep the house); a pinned tenant stays until it moves out,
which also clears its pin. An emptied house recycles to a fresh container.

Verify a move: the next turn appears on the house stream and is acked by its
group. Test with disposable bots on spare houses, never assuming a house is
empty; delete the bots and their retained rows afterwards.

## Where tests run

| Work | Where |
|---|---|
| Focused tests (named files/nodes), lint on changed files | In place, in your task worktree, from your bridge or house. Houses have pytest (`/opt/house-tests`) and Node/pnpm. |
| Full suites, project-wide `tsc`, mypy, broad pytest/vitest | The heavy queue (`heavy-queue submit … -- <cmd>`), which runs one capped job at a time and calls you back. Bridge guards refuse these locally. |
| Exhaustive gate with private PostgreSQL | The integrator's `make gate` / exact-SHA batch gate. Not the queue. |
| Live UI for your change | Your house preview: `house-preview <worktree>/frontend`, then the house's HTTPS origin with real sign-in. Shared `<dev-host>` shows integrated `main`, not branch work. |
| Browser checks and screenshots | Playwright MCP against the house origin or the deployed dev surface, at phone and desktop width. |

Rules:

- Focused means real filenames. Globs, directories and whole-suite lists are
  heavy and go through the queue.
- Do not ssh to the host to run suites, and do not start a per-house Python
  backend or shared schema change from a preview. App, router, voice,
  schedulers and the database stay shared.
- Mutable test environments stay task-local; do not borrow another task's.
- The queue callback carries status and paths only; read `run.log` for output.
  After an uncertain exit, reconcile the named container before clearing any
  hold or marker.
