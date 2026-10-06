# Approval-Gated Tool Policies

## Architecture and coverage (TASK-861)

The app owns policies, immutable revisions, decisions, approval requests, MCP
execution claims and continuation state. `approval_policies.py` is the compatibility
facade over `approval_models.py`, `approval_policy_store.py`,
`approval_request_store.py`, `approval_validation.py` and `approval_defaults.py`.
The pure matcher is `agent_bridge/approval.py`.

- First-party bawthub MCP tools are gated at the app MCP server. The Claude hook
  stamps exact invocation context; Codex sends a signed request envelope that
  the server binds to the MCP JSON-RPC request ID. Both deliberately skip a
  second bridge gate.
- Claude Code native tools and external MCP tools use the PreToolUse bridge gate
  (and the equivalent permission callback when applicable).
- Codex, OpenClaw and direct clients do **not** enforce this layer on their native
  tools. Calling the first-party MCP server still reaches its server gate.
- AskUserQuestion uses its separate interactive-question path, not this gate.

## Rules and management

First enabled match wins, ordered by `(order, id)`; no match allows by design.
`ops_run` additionally requires approval by default when no rule matches.
Rules select `backend_scope`, `tool_name`, input `field`, and a matcher
(`always`, `exact`, `prefix`, `contains`, `glob`, `regex`). Actions are `allow`,
`deny`, `require_approval`; `severity`, `category`, and `approval_prompt` supply
operator-facing metadata. MCP-qualified tool names support tail matching.
Validation rejects invalid matcher/action/severity values and regexes before save.

For Bash, matching uses the command subject, with proven inert quoted-heredoc
bodies and literal arguments to direct `rg`/`grep` searches removed by the matcher.
Search handling recognizes a deliberately narrow command-list grammar; wrappers,
substitutions, pipelines, executable search hooks and ambiguous syntax retain
literal text and may still cause false positives. This is not a general shell
parser. Do not add broad `allow` rules for commands merely starting with a search.
Approval-required events preserve full tool arguments verbatim: home-directory
shortening is display formatting and must never modify the stored replay payload.
Changing the subject or whitespace normalization must never loosen authorization:
Claude grants bind the fully qualified tool, **full JSON input and cwd** exactly.

| Method | Path | Purpose |
|---|---|---|
| GET/POST | `/v1/tool-approval-policies` | List/create rules |
| GET/PATCH/DELETE | `/v1/tool-approval-policies/{id}` | Read/update/delete |
| POST | `/v1/tool-approval-policies/seed-defaults` | Insert missing defaults |
| GET | `/v1/tool-approval-policies/bundle` | Compiled bundle, conditional etag |
| POST | `/v1/tool-approval-policies/preview` | Evaluate saved/candidate policies without execution |
| GET | `/v1/tool-approval-policies/status` | Source availability and honest coverage notes |
| GET | `/v1/tool-approval-policies/{id}/revisions?limit=50&offset=0` | Immutable create/update/delete revisions, real total |
| GET | `/v1/tool-approval-decisions?limit=50&offset=0` | Decision audit, real total |
| GET | `/v1/tool-approval-requests` | Gated request lifecycle |
| POST | `/v1/admin/reload-tool-approval-policies` | Publish cache invalidation |
| POST | `/v1/chat/approvals/{request_id}/resolve` | Approve/deny/cancel/respond |

Revision and decision pagination clamp limit to 1–200 and offset to at least zero;
empty pages retain the full matching total. Revisions survive policy deletion.
Reload reports `published`/`publish_failed`, `published`, Redis `subscribers`, and
`bridge_installed: "unknown"`. Zero subscribers is successful Redis publication,
not delivery or installation. There is no bridge-installed-version ACK protocol.
Claude also refreshes after `CLAUDE_CODE_APPROVAL_BUNDLE_TTL` (default 15 seconds).
On fetch failure it keeps cached/empty policies; the optional
`CLAUDE_CODE_APPROVAL_FAIL_CLOSED` instead pauses tools.

## Decision audit and outage semantics

Claude emits additive `approval_decision` events on its **existing Redis run
stream** for allow, deny and require-approval evaluations. A consumed grant emits
an additional `grant_allowed` outcome; these are evaluations, not tool execution
receipts. Fail-closed source-unavailable denials have no matched policy/hash.
Each event carries policy id/version, evaluated bundle etag, exact invocation
hash and request/session/tool-use identity. The app run consumer adds bot and
thread-session context and commits into `tool_approval_decisions` independently
of SSE delivery, including non-streaming calls. Replayed events are deduplicated
by run/event identity. Bridge policy snapshots are recovered by id/version from
immutable revisions, not from whatever policy is current at ingest time.

Audit subjects are **fully redacted and bounded**, for both new MCP and bridge
rows. Arbitrary commands/JSON cannot be reliably secret-scrubbed with regexes.
No full arguments, cwd, task capability or tool results are added to this audit.
Existing approval-request storage still retains arguments needed for execution;
this is not a retroactive scrub of old rows or of ordinary tool/turn logs.

**Durable means the app database commit succeeded**, not merely event emission.
Bridge emission is best-effort and never adds a per-tool HTTP/DB blocking call.
The app retries commit three times, logs explicit loss on exhaustion, and does
not change the gate decision on audit failure. Redis outage, stream expiry,
consumer timeout, or app death can leave gaps: there is no durable audit outbox
or guaranteed replay for ordinary turns. MCP audits commit inline at its app
boundary and a write failure prevents proceeding normally. Do not advertise
complete all-backend audit coverage or exactly-once execution.

An interactive MCP approval is persisted only with complete verified routing:
bot, user, turn, trigger message, session, backend, MCP server, tool-use ID,
invocation hash, caller context, and continuation capability. Blank or sentinel
identities such as `unknown` are rejected in the domain layer and by a database
constraint for pending MCP rows. Missing or forged context returns
`approval_context_missing` / `approval_context_invalid` before persistence or
fanout. Stable per-invocation IDs make retries idempotent; only the transaction
that creates the row publishes the live approval event. Legacy unroutable
pending MCP rows are superseded and retained as terminal audit records during
schema bootstrap.

## Resolution and rollout

Resolve atomically chooses a terminal decision once. MCP approval claims execute
stored arguments (operations use immutable snapshots), record outcomes, and queue
server-owned continuations. Execution/dispatch uncertainty is represented in the
persisted state rather than blindly rerunning side effects. Editing or deleting
an operation/policy does not rewrite an already-approved MCP snapshot; changes
apply to new evaluations. Claude re-evaluates current rules on retry, so a current
hard deny still wins over an old grant.

### `/new` while an approval is pending (TASK-939)

Session rotation atomically records `reset_successor_id` on the old session and
`reset_predecessor_ids` on the fresh session, alongside archiving/creating those
rows. The new session does not inherit the old SDK resume key. These links survive
app and bridge restarts and preserve other session metadata.

An MCP result continuation follows this explicit reset chain at dispatch, with
bot/user ownership and reciprocal-link checks. It uses the successor's normal
SDK binding and seeded message history, not the archived transcript. The original
approval context, approved arguments, execution count, parent-turn audit link and
deterministic continuation identity remain unchanged. Claim validation rechecks
the target, so another reset before validation causes a retry rather than delivery
to the stale session. An already-started continuation cannot be replayed under the
same message IDs in a different session; it requires manual reconciliation.

Ordinary archived/task threads without reset lineage are not redirected just
because another thread is active. Pre-existing archived sessions have no inferred
successor: the new behavior applies to rotations that record these links.

Redirection applies only to a reset that happened *while* the approval was
pending. Rotation stamps the archived session with an absolute `reset_at` epoch;
if that precedes the approval's `created_at`, the user deliberately reopened the
archived thread, so the result (or native grant) stays in that thread. Rotations
recorded before `reset_at` existed keep the follow-the-chain fallback.

Native harness grants are not transferable across resets: they fail closed and
require a fresh tool approval rather than resuming the retired SDK context.

Harness approvals use process-local pending authority bound to the original
session and approval request. A one-shot grant can be consumed only by the exact
invocation in its server-owned continuation request; unsolicited, cross-session,
expired, duplicate, legacy subject-only grants do not authorize execution.

**Rollout requires fresh approvals.** Old unscoped pending approvals cannot be
promoted to exact authority, and a bridge restart loses process-local pending
authority safely. Ask the agent to attempt the operation again, creating a fresh
gated request, then approve that request. Do not replay old approvals or inject
legacy grant keys. Updating source alone does not install running code: coordinate
app/bridge deployment externally; do not restart a bridge hosting active agents.
