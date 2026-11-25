# Security model — authentication and coordination boundaries

> Part of the llm-bawt skill. Read BEFORE writing up any security finding against llm-bawt.

llm-bawt is self-hosted, but network placement does not replace the requested caller authentication. Keep user sessions, service authentication and fleet coordination distinct:

- **API caller authentication is intentional.** `/v1/*` uses the `api_auth_mode` rollout (`off`, `log`, `enforce`); MCP has its own `mcp_auth_mode`. A per-caller API key identifies a program; existing verified turn, service and bridge credentials retain their distinct principals and route-specific checks. `log` observes missing credentials without enforcing the outer gate. Inspect actual modes and `GET /v1/admin/auth-observations` rather than assuming source publication or elapsed observation time enables enforcement. See the repository's `api_auth/` implementation and `docs/MCP_SERVER.md`.
- **BawtHub user sign-in is separate.** Enrollment/password sessions and trusted-proxy identity determine the UI user; a server API key does not sign that user in. A route's 401 can mean missing legacy identity, not an enabled password/session gate. Check `/api/chat/auth` and the installation's auth configuration; see [BawtHub authentication](../../bawthub/references/auth-and-personas.md).
- **The approval gate's default-allow is BY DESIGN.** `agent_bridge/approval.py` `evaluate()` returns ALLOW on no-match on purpose — the gate exists to catch **specific operator-defined dangerous patterns** (via regex rules), NOT to block every command an agent runs in its sandbox. "Default-allow / no-match → allow" is the intended behavior, not a fail-open bug.
- **Approvals and turn capabilities are coordination, not a security boundary.** All bots share one uid in one container and could act outside the approval system; the fleet runs on the honor system. Approval cards make sure a restart fits what else is happening. Exposure of internal tokens (turn context, approval grants, LAN service keys) gets a short note to the operator, not an incident investigation or a rotation rollout.
- **The GitHub git-credential endpoint is a credential broker.** It mints short-lived, repo-scoped GitHub App installation tokens. Inspect its current bridge gate and outer API authentication instead of assuming LAN reachability makes it anonymous.
- **What IS genuinely sound (verified):** provider-credential encryption at rest — `service/providers/crypto.py` uses real Fernet (no hardcoded key; env `LLM_BAWT_SECRET_KEY` → 0600 keyfile), and the write path always encrypts.

## Approval lifecycle boundaries (TASK-861)

Deliberate approval no-match allow remains unchanged; it is independent of API
caller authentication. Approval enforcement is
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

**Rule for audits:** trace intended behavior, effective auth modes and deployment topology (bootstrap scripts, NAT/DNS/proxy configuration), not just a route handler or a `docker-compose ports:` line. Distinguish a staged log-only rollout from enforcement and from unrelated approval policy.
