# Native Codex Bridge (legacy)

## Service retirement and rollback

The separate `codex-bridge` mansion service is retired from base/dev Compose
and the restart/configuration ops catalog (TASK-2117). GPT bots, including the
Codex personality, run on `claude-code` / `claude-proxy` without changing their
OpenAI model or endpoint. Provider and harness are independent choices.
The Python native backend and optional Codex-capable houses remain supported
compatibility code; the implementation notes below do not imply a deployed
mansion service.

Before the approved, named-container stop, retain the running container and
record its immutable image ID, mounts and pre-retirement Compose revision.
Do not remove images, session/config binds, shared bridge tokens or volumes.
A stopped container preserves its writable layer as the simplest rollback;
`unless-stopped` keeps an intentionally stopped container down.

Rollback is an operator-controlled maintenance action:

1. Keep the Codex bot on its working proxy route until the native bridge is
   healthy. Preserve any in-flight turn; migrate only between turns.
2. Restore the `codex-bridge` service and dev overlay from the recorded
   pre-retirement revision, leaving unrelated service changes intact. Reuse the
   retained image ID; do not pull or rebuild. For the retained container, start
   only that container through an approved, fixed-target `start` operation. If
   it was removed, recreate only the restored service with `--no-deps`,
   `--no-build` and `--pull never`, pinned to that image.
3. Verify its own `/health` on port 8682 and Redis connectivity. The app-owned
   provider broker remains the credential source; do not copy host auth files.
4. Through the bot profile API restore `agent_backend="codex"`,
   `harness="codex"` and `agent_backend_config={"codex_transport":"app_server"}`,
   retaining the bot's exact model/endpoint, identity, memory and tags. Use a
   deliberate new session boundary, then verify a real review.
5. Restore catalog restart schema/quiesce map and reconciliation/cutover
   targets only if native service management is intentionally being restored.
   Seeds do not overwrite live rows; use the catalog API and normal approvals.

The Codex bridge runs bots through the OpenAI Codex SDK. The service entrypoint
is [src/codex_bridge/__main__.py](../src/codex_bridge/__main__.py)
and the bridge logic is
[src/codex_bridge/bridge.py](../src/codex_bridge/bridge.py).

## What it does

- Consumes Redis `chat.send` commands for bots whose `agent_backend` is `codex`.
- Uses `~/.codex/auth.json` OAuth credentials, not API keys.
- Persists Codex thread IDs in `agent_backend_config.session_key` and tracks
  the model in `session_model`.
- Reuses a shared SDK transport and serializes turns per session.
- Builds a request-scoped SDK handle when trusted MCP correlation is present;
  the base environment and shared SDK handle are never mutated.
- Emits provider-aware native Codex tool events instead of pretending they are
  Claude tool names.
- Stages repo-managed local plugins into the Codex home when enabled.

## Authentication model

The bridge is OAuth-only. On startup it:

- scrubs `OPENAI_API_KEY` and `CODEX_API_KEY` from the environment
- materializes a container-local OAuth bundle from the app's
  `GET /v1/providers/openai_chatgpt/bundle` broker endpoint
- validates that file (`CODEX_AUTH_PATH`, default `/home/bridge/.codex/auth.json`)
- points `CODEX_HOME` at its containing directory

**The host `~/.codex/auth.json` is NOT the deployed bridge credential source.**
TASK-636 removed its bind mount. A host CLI can fail with stale credentials
while native Codex bots and the OpenAI proxy both work normally.

Before diagnosing authentication, inspect the actual bridge's CODEX_HOME,
CODEX_AUTH_PATH, executable and mounts without printing tokens. Test through
that execution context. An unrelated host probe does not establish a bridge
failure. Check the app-owned provider connection and broker materialization;
recover through the provider connection flow only if that path actually fails.
Do not advise host login, copy/refresh another bundle, or restart a bridge as a
shortcut. The app owns the refresh chain; do not introduce a second refresher.

## Important environment variables

| Variable | Default | Purpose |
|---|---|---|
| `CODEX_AUTH_PATH` | `/home/bridge/.codex/auth.json` | ChatGPT OAuth bundle |
| `CODEX_HOME` | `/home/bridge/.codex` | Codex home used by the SDK |
| `CODEX_MODEL` | `gpt-5.4` | Fallback model when the request omits one |
| `CODEX_BACKEND_NAME` | `codex` | Redis backend filter |
| `CODEX_BRIDGE_HEALTH_PORT` | `8682` | `/health` TCP listener |
| `CODEX_BRIDGE_LOG_LEVEL` | `INFO` | Bridge logging |
| `CODEX_BRIDGE_REQUEST_TIMEOUT` | `900` | Per-turn timeout |
| `CODEX_BRIDGE_CWD` | `/home/bridge/dev` | Codex working directory |
| `CODEX_BIN` | bundled | Override Codex binary path |
| `CODEX_LOCAL_PLUGINS_ENABLED` | `1` | Install repo-managed local plugins |
| `CODEX_LOCAL_PLUGINS_SRC` | `/home/bridge/dev/agent-skills/codex` | Local plugin source root |
| `CODEX_DEV_ROOT` | `/home/bridge/dev` | Repo root used for local plugin resolution |

## Trusted BawtHub MCP correlation

The app mints `task_turn_capability` for Codex agent turns and carries it through
the existing Redis command. The bridge creates a fresh subprocess environment
containing that capability plus a signed request envelope bound to
`request_id`, `session_key`, and backend. `local_plugins.py` configures Codex's
`env_http_headers` so only the BawtHub MCP request receives:

- `X-LLM-Bawt-Task-Turn-Context`
- `X-LLM-Bawt-MCP-Request-Context`

Both source environment variables are excluded from model-invoked shell
processes through `shell_environment_policy`; they remain available only to the
Codex process that materializes the MCP HTTP headers.

The MCP server verifies both headers and binds the exact tool name/arguments to
FastMCP's JSON-RPC request ID. The derived tool-use and approval-request IDs are
deterministic, so a retry returns the existing row and does not republish the
approval event. Missing, expired, forged, cross-session, or sentinel context
fails before tool execution, approval persistence, or event publication.

Raw MCP clients remain able to call tools allowed by policy. If a policy
requires approval, an uncorrelated raw request returns
`approval_context_missing` and creates no approval row.

## Session behavior

- `session_key` stores the Codex thread ID.
- `session_model` tracks which model that thread was created with.
- If the model changes, the bridge clears the old thread and starts a new one.
- `/new` and explicit reset RPCs clear the saved thread ID.

## Compose layout

The main compose stack mounts:

- `${HOME}/dev` at `/home/bridge/dev`
- **No host auth.json mount**; the bridge materializes its container-local file
  from the app broker at startup
- `${HOME}/.codex/config.toml` at `/home/bridge/.codex/config.toml`
- `${HOME}/.codex/sessions` at `/home/bridge/.codex/sessions`

That matches the bridge's expected runtime layout.

## Related files

- [src/llm_bawt/agent_backends/codex.py](../src/llm_bawt/agent_backends/codex.py)
- [src/codex_bridge/local_plugins.py](../src/codex_bridge/local_plugins.py)
- [src/codex_bridge/mcp_context.py](../src/codex_bridge/mcp_context.py)
- [src/codex_bridge/transport.py](../src/codex_bridge/transport.py)
