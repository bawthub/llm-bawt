# Security model — deliberate design decisions (NOT vulnerabilities)

> Part of the llm-bawt skill. Read BEFORE writing up any security finding against llm-bawt.

llm-bawt is a **self-hosted, LAN-only** service whose security model is a network trust boundary, not per-request auth. Before writing up a security "finding," know these are **intentional** and are the correct design for this system — flagging them as bugs is applying least-privilege/fail-closed dogma where it doesn't belong:

- **No app-level auth on `/v1/*` is BY DESIGN.** It's an integration API for home services on the LAN; the network boundary IS the auth. The app API (`:8642`) and MCP server (`:8001`) are **not** internet-reachable (NAT + no port-forward + split-horizon DNS — verify with the exposure checks in the `bridge-container-network` skill). Do NOT propose bolting a bearer/API-key gate onto the whole surface.
- **The approval gate's default-allow is BY DESIGN.** `agent_bridge/approval.py` `evaluate()` returns ALLOW on no-match on purpose — the gate exists to catch **specific operator-defined dangerous patterns** (via regex rules), NOT to block every command an agent runs in its sandbox. "Default-allow / no-match → allow" is the intended behavior, not a fail-open bug.
- **Approvals and turn capabilities are coordination, not a security boundary.** All bots share one uid in one container and could act outside the approval system; the fleet runs on the honor system. Approval cards make sure a restart fits what else is happening. Exposure of internal tokens (turn context, approval grants, LAN service keys) gets a short note to Nick, not an incident investigation or a rotation rollout.
- **The GitHub git-credential endpoint being open on the internal network is BY DESIGN.** It's a secret-broker that mints short-lived, repo-scoped GitHub App installation tokens on demand so no static secret sits on the tenant disk. `BRIDGE_GIT_CREDENTIAL_TOKEN` (`X-Bridge-Token`) is an *optional* defense-in-depth knob, not the primary control.
- **What IS genuinely sound (verified):** provider-credential encryption at rest — `service/providers/crypto.py` uses real Fernet (no hardcoded key; env `LLM_BAWT_SECRET_KEY` → 0600 keyfile), and the write path always encrypts.

## Approval lifecycle boundaries (TASK-861)

Network trust and deliberate no-match allow remain unchanged. Enforcement is
server-side for first-party bawthub MCP and hook-side for Claude native/external
MCP tools. Codex native tools and direct clients are **uncovered**.
Preview/status/reload APIs are operational controls, not new authentication.
Reload confirms Redis publication only; installed bridge versions remain unknown.

Exact Claude grants bind tool/full JSON input/cwd, session/original approval and
the server-owned continuation request; no subject-only or global grant fallback.
MCP approvals execute stored arguments/immutable operation snapshots with claimed
execution and server-owned continuation state. Terminal decisions are immutable;
uncertain execution does not justify blind replay. Current Claude hard-deny
rules still block a granted retry.

New decision audit subjects are fully redacted and bounded; invocation hashes
and policy id/version permit correlation without full tool inputs. This does
not scrub legacy rows, request arguments required for execution, or turn logs.
Claude decisions use the existing Redis event pipeline, committed at the app
consumer with three bounded retries. Durability starts at commit; delivery is
best-effort, and Redis outage/app loss/expired streams can leave gaps. No audit
outbox or all-backend coverage is claimed. MCP commits inline; bridge audit
failure never changes enforcement. See `approval-policies.md` for contracts.

**Upgrade/restart requires fresh harness approvals.** Legacy/unscoped approvals
cannot be promoted, and bridge process-local pending authority is lost safely on
restart. Re-attempt the operation to create a new approval request; do not replay
old approvals. Coordinate deployment rather than restarting active agents.

**Rule for audits:** a real finding here requires tracing *intent* + the *deployment* (bootstrap scripts, NAT/DNS/proxy topology), not pattern-matching a route handler or a `docker-compose ports:` line. If the "vuln" dissolves once you understand the trust boundary, it wasn't one.

## Summary formerly in the parent SKILL.md (TASK-1640)

## Security model — deliberate design, NOT vulnerabilities

LAN-only service; the network boundary IS the auth. **Intentional, do not "fix" or file findings on:** no app-level auth on `/v1/*`; approval gate default-allow on no-match; open internal git-credential endpoint (short-lived scoped tokens by design). A real finding requires tracing intent + deployment topology, not pattern-matching a route. Full rationale: **`reference/security-model.md`**.
