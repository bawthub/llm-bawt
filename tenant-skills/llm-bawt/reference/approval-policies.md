# Approval-gated tool policies — "This action requires user approval"

> Part of the llm-bawt skill. Read when an agent bot hits the approval gate, before editing `tool_approval_policies` rules, or when a benign command false-positives.

**"This action requires user approval" is llm-bawt's gate, not a reason to
change harness `settings.json`.** First-party bawthub MCP tools are gated in the
app MCP server. Claude native/external MCP tools are gated in its PreToolUse hook;
first-party MCP is deliberately not double-gated. Codex first-party MCP calls
carry request-local signed correlation; Codex native tools and direct
clients remain outside a universal bridge hook.

### TASK-861 architecture and rollout

- `approval_policies.py` is a facade over model, policy-store, request-store,
  validation and defaults modules. Policies have immutable create/update/delete
  revisions. Preview is non-executing; status reports coverage, not installed
  bridge versions. Repo `docs/approval-policies.md` is the current contract.
- `GET /v1/tool-approval-policies/{id}/revisions?limit=50&offset=0` and
  `/v1/tool-approval-decisions?limit=50&offset=0` return real totals (limit 1–200).
- Reload returns `published`/`publish_failed`, Redis subscriber count and
  `bridge_installed: "unknown"`. Publication (even with recipients) is NOT proof
  of installation. Claude also refreshes its TTL cache; Codex has no
  native enforcement to reload.
- Claude emits allow/deny/require-approval decisions on the existing Redis run
  stream, with policy id/version, bundle etag, exact invocation hash and
  request/session/tool-use identity. App adds bot/thread context and commits
  decisions independently of UI streaming. Grant consumption is a separate
  `grant_allowed` outcome, not an execution receipt.
- New MCP/bridge decision subjects are fully redacted and bounded; full inputs,
  cwd and capabilities are not added to this audit. Existing request/turn logs
  are not retroactively scrubbed. Bridge snapshots are resolved via revisions.
- Durability begins at app DB commit. Bridge delivery is best-effort; app retries
  commits three times and logs explicit loss. Redis outage, timeout/expiry or app
  death can lose events; no durable audit outbox guarantees complete replay.
  Audit failure does not change a bridge gate decision. MCP commits inline.
- Claude authorization binds fully qualified tool + full JSON input + cwd, the
  original session/approval, and the server-owned continuation request. Subject
  matching is NOT an authorization fingerprint. Current hard deny still wins.
- **Fresh approvals required on rollout/restart**: old subject-only/unscoped
  requests cannot acquire new exact authority; process-local pending records
  vanish on restart. Re-attempt to create a fresh gated request, then approve.
  Do not replay old keys. Source edits need coordinated deployment; never
  restart bridges hosting active agents.
- MCP resolution persists one terminal choice, claims immutable execution,
  records outcomes and server-owned continuation state. Uncertain side effects
  are not blindly retried. Policy/operation edits affect new evaluations, not
  already-approved MCP snapshots.
- Interactive MCP approval rows require verified bot/user/turn/trigger/session,
  backend, server, tool-use, invocation-hash, caller-context and continuation
  identity. Missing/forged/sentinel context fails before persistence/fanout.
  Codex uses `env_http_headers` plus a signed request envelope bound to the MCP
  JSON-RPC request ID; `shell_environment_policy` excludes the source variables
  from model-invoked subprocesses. Deterministic IDs deduplicate transport
  retries. Schema bootstrap supersedes legacy unroutable pending rows while
  retaining audit.

**Where it lives (all in `llm-bawt`):**
- Engine (pure, the matching semantics): `src/agent_bridge/approval.py`
- Storage + request audit + seed defaults: `src/llm_bawt/approval_policies.py`
- Admin/resolve routes: `src/llm_bawt/service/routes/approval_policies.py`
- Rules table: **`tool_approval_policies`** in the **`askllm`** DB
  (`pgq askllm`). Audit log of every gated call: `tool_approval_requests`.
- Admin UI: bawthub **`/tools/approval-policies`**. Inline approve/deny card in
  chat: `ApprovalCard.tsx` + `/api/chat/approvals/{id}/resolve`.

**Evaluation semantics (`evaluate()` in `approval.py`):**
- A rule targets `backend_scope` (`*` = any bridge) + `tool_name` (`*` = any;
  MCP-tail aware, so `mcp__x__Bash` matches a `Bash` rule).
- It derives a **subject** string from the tool input via `field` (blank →
  per-tool default; **for `Bash` the subject is the full `command` string**),
  then tests it with `matcher_type` ∈ `always|exact|prefix|contains|glob|regex`.
- `action` ∈ `require_approval | allow | deny`. **First match wins** over rules
  sorted by `(order, id)`; **no match → allow** (default-allow). Put a low-order
  `allow` rule above a broad `require_approval` to carve a safe exception.

**Inspect / change rules** (API is proxied through the frontend at
`/api/chat/proxy/v1/*`; reach it from the bridge with `-H "Host: dev.bawthub.com"
http://$HOST`, `HOST` = the Traefik host's LAN address):
- List: `GET /v1/tool-approval-policies` · bundle: `GET …/bundle`
- Create `POST`, update `PATCH /v1/tool-approval-policies/{id}`, delete `DELETE`
- **After any change, `POST /v1/admin/reload-tool-approval-policies`** requests
  Claude cache invalidation (the UI does this); inspect publication status,
  never assume every bridge installed it.
- Raw read: `pgq askllm -c 'SELECT order_index,enabled,tool_name,matcher_type,pattern,action FROM tool_approval_policies ORDER BY order_index;'`

**The subject = whole Bash command string gotcha (this WILL bite you):**
because the `Bash` subject is the entire command line, a rule's regex also
matches trigger words that appear in an `ssh …`-wrapped remote command, in a
quoted heredoc, or as literal test data in a script. So a benign
`ssh host docker exec c kill -SIGUSR1 …`, or even a python script that merely
*mentions* `docker compose down` in a string, can trip a docker/make rule. To
handle a false positive, inspect the matched policy and improve matching with
regression tests; do not conceal a gated action in a script to evade approval.
Supported service mutations belong in ops (`ops-actions.md`). The source matcher
now recognizes a narrow subset of literal direct rg/grep searches, preserving
wrappers, substitutions and ambiguous shell syntax. Source edits still require
coordinated deployment before the live bridge gains that behavior.

Approval replay arguments must remain byte-exact. Do not shorten home paths or
normalize commands in stored authority/replay payloads; formatting belongs only
in display data. A 2026-09-08 regression saved `/home/bridge/...` as `~/...`, causing
the continuation to re-gate despite an approved original invocation.

**Historical deployed rules (2026-06; not a current inventory):** two
`Bash`/`require_approval`/high rules below. TASK-861 includes catalog operation
seeds as well; use the API for actual current rules, not this historical list.
- `order 35` (git): gates switching the repo off `main`/`master`
  (`git checkout -b` / `switch -c` / checkout|switch to a non-main ref).
- `order 70` (docker): gates container lifecycle — `docker [compose] … (up|down|
  restart|stop|start|kill|rm|pause|unpause)` **excluding `docker exec …`** — and
  service-mutating `make` targets (`up|down|restart|rebuild|build|run|stop|start|
  prod-mode|dev-mode|snapshot|recreate|docker-dev`). Read-only `make` targets
  (`mode-status`, `logs*`) and `docker exec/logs/inspect/stats` are NOT gated.
  (Tightened 2026-06 from a version that matched `docker exec … kill` and bare
  `make ` — both false-positived routine bridge ops.)

## Summary formerly in the parent SKILL.md (TASK-1640)

## Approval-gated tool policies

**"This action requires user approval" on an agent bot's tool call is THIS system (llm-bawt), NOT the Claude Code harness `settings.json`** — don't edit harness settings to fix it. Rules live in `tool_approval_policies` (askllm DB), engine in `agent_bridge/approval.py`, admin UI at bawthub `/tools/approval-policies`; after changes `POST /v1/admin/reload-tool-approval-policies`. **Gotcha that WILL bite:** Bash policies inspect command text; wrappers and literal data can false-positive. Inspect the matched rule and fix matching with regression coverage; never conceal a gated action in a script to evade approval. Supported service actions belong in ops. The source matcher excludes proven inert heredocs and narrow direct rg/grep arguments, but requires deployment before live behavior changes. Semantics, current rules, API: **`reference/approval-policies.md`**.
