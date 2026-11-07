# Anthropic-compatible proxy (TASK-270) — non-Anthropic providers on the Claude Agent SDK

> Part of the llm-bawt skill. Read before touching `src/claude_code_bridge/proxy/`, adding a provider adapter, flipping a bot to the proxy path, or debugging ChatGPT-OAuth/Responses-API issues.

The `claude_code_bridge` daemon now hosts an in-process FastAPI proxy that
accepts Anthropic Messages API requests and forwards them to other providers
via the OpenAI Responses API. This lets codex-personality (and any future)
bots use the Claude Agent SDK's richer tool/chat surface instead of being
locked to the Codex SDK.

**Architecture:**

```
codex-personality bot
  │  endpoint_id → model/access path; harness = claude-proxy
  ▼
ClaudeCodeBridge — detects "<provider>/<upstream>" prefix in model name
  │  spawns Claude Agent SDK subprocess with:
  │    ANTHROPIC_BASE_URL = http://127.0.0.1:<ephemeral>
  │    CLAUDE_CODE_OAUTH_TOKEN = ""   (force the SDK off api.anthropic.com)
  │    ANTHROPIC_AUTH_TOKEN = "proxy-routed"   (sentinel — proxy ignores)
  ▼
FastAPI proxy (in same process)
  │  POST /v1/messages → split provider/upstream → adapter registry
  ▼
ProviderAdapter (e.g. OpenAIChatGPTAdapter)
  │  authorize() returns (bearer, base_url) + extra_headers()/prepare_request()
  │  call() targets the ChatGPT backend Responses endpoint:
  │      https://chatgpt.com/backend-api/codex/responses
  ▼
Translates SSE back to Anthropic Messages SSE events.
```

> **⚠️ Verified-the-hard-way facts (TASK-270 live smoke, 2026-06-15).** The
> original design assumed `api.openai.com/v1/responses`. That is WRONG: a
> ChatGPT *subscription* OAuth token carries scopes `[openid, profile, email,
> offline_access]` only — no `api.responses.write` — so it **401s** against
> the platform API. It authenticates against the **ChatGPT backend** instead
> (`chatgpt.com/backend-api/codex/responses`, same surface `codex exec` uses).
> The adapter therefore:
> - sets `base_url = https://chatgpt.com/backend-api/codex`
> - sends the **`chatgpt-account-id`** header (mandatory; pulled from the
>   bundle's `tokens.account_id`) plus `OpenAI-Beta`/`originator`/`session_id`
> - **strips** `temperature` / `top_p` / `max_output_tokens` — the backend
>   rejects them with HTTP 400 `Unsupported parameter` (the Claude SDK always
>   sends `max_tokens` → must be dropped)
> - guarantees a non-empty `instructions` (backend hard-requires it) and
>   `stream:true` / `store:false`
> - **TASK-636 supersedes the original file-based auth implementation:** the
>   adapter gets access tokens from the app broker and never reads/writes a
>   host auth.json or refreshes OAuth itself. The app owns the refresh chain.
>
> **TASK-931 corrects the earlier reasoning claim:** successful tool execution
> with `store:false` did not prove reasoning continuity. Native Codex carries
> `reasoning.encrypted_content` in its replayed history even with `store:false`.
> The proxy implementation now preserves compatible OpenAI reasoning items in
> versioned, provider/model/account-scoped SDK signature envelopes. Legacy raw
> signatures remain non-replayable rather than guessed. Standard Responses omits
> `reasoning.context`; Lite sets `all_turns`, matching the inspected Codex source.
> Claude Code still owns compaction. See llm-bawt `docs/OPENAI_PROXY_REASONING.md`
> for verification, exact-model compatibility limits and rollout caveats; source
> changes require an explicitly authorized bridge restart before they are live.

**Authentication source — check this before any probe or recovery:**

- App-owned, encrypted provider connection is authoritative (TASK-636).
- Claude proxy requests `GET /v1/providers/openai_chatgpt/token`; no auth.json
  access and no local OAuth refresh.
- Native Codex bridge materializes its container-local auth.json from
  `GET /v1/providers/openai_chatgpt/bundle` at startup. Host auth.json is NOT
  bind-mounted. Host config.toml and sessions mounts do not imply an auth mount.
- A host `codex` command can fail `refresh_token_reused` while both bridges work.
  That proves the host credential path failed, not that the provider or bridge
  failed. Inspect actual container mounts, CODEX_HOME/CODEX_AUTH_PATH and binary;
  run bridge tests in that context without printing tokens.
- Do not recommend host login, copy credentials, add a refresher or restart
  services to fix an unverified failure. Verify the app-owned provider/broker
  path first; use its provider reconnect flow only when recovery is necessary.

**Flipping a bot to the proxy path:**

1. Check the existing normalized endpoint's eligibility for `claude-proxy`
   through the model catalog API. Keep its exact endpoint ID when compatible;
   changing the harness does not require changing the GPT model, access path
   or credentials. Legacy model-alias creation is not the current migration path.
2. Between turns, PATCH `/v1/bots/<slug>/profile` with
   `agent_backend="claude-code"`, `harness="claude-proxy"` and the retained
   `endpoint_id`. Clear native-only transport configuration while preserving
   identity, memory and tags. The profile API supports these fields and
   `expected_fields`/`expected_tags` compare-and-set; the compact
   `bots_profile_patch` MCP tool does not expose backend/harness fields.
3. Use a deliberate supported session boundary and verify a real turn/tool
   result on that endpoint. The bridge resolves the endpoint to its internal
   provider/upstream model string. Profile-only changes do not need a restart.
   Adapter/translator source changes do: the proxy lives in the Claude bridge,
   not the app. Use the matching approved bridge operation from `ops-actions.md`.
   Bind mounts expose files; they do not reload already-imported Python code.

**Rollback (Codex SDK is preserved):**

The native package remains, but the separate `codex-bridge` mansion service
and its catalog targets are retired. A profile flip alone is not rollback:
restore/start the retained native service and verify its health first, then
move the bot between turns with its original model/endpoint and native
transport configuration. Follow llm-bawt `docs/CODEX_BRIDGE.md` for the
retained-container/image and catalog restoration procedure.

**Adding a new provider (e.g. Grok, Kimi):**

If the provider speaks OpenAI Responses API:

```python
# src/claude_code_bridge/proxy/adapters/xai.py
import os
from .base import ProviderAdapter

class GrokAdapter(ProviderAdapter):
    name = "xai"
    async def authorize(self) -> tuple[str, str]:
        key = os.getenv("XAI_API_KEY")
        if not key:
            raise RuntimeError("XAI_API_KEY required for xai provider")
        return key, "https://api.x.ai/v1"
```

Then `register(GrokAdapter())` in `proxy/adapters/__init__.py`. That's it —
the default `call()` on the base ABC handles translation, streaming, and
tool round-tripping. A provider that doesn't speak Responses API can
override `call()` directly.

**Env-var knobs (`__main__.py`):**

| Var | Default | Purpose |
|-----|---------|---------|
| `CLAUDE_CODE_BRIDGE_PROXY_PORT` | `0` (ephemeral) | Pin a fixed port for curl-testing |
| `CLAUDE_CODE_BRIDGE_PROXY_DISABLED` | unset | Set `1`/`true` to skip starting the proxy |
| `LLM_BAWT_API_URL` | deployment-configured | App token broker; replaces the retired proxy `CODEX_AUTH_PATH` file wiring |
| `OPENAI_BASE_URL` | `https://chatgpt.com/backend-api/codex` | Override the upstream Responses host (the `openai` client appends `/responses`) |
| `OPENAI_CHATGPT_REASONING_EFFORT` | `high` | Reasoning effort for GPT-5.4 (`none`/`low`/`medium`/`high`/`xhigh`; `minimal` is rejected). The adapter sends `high` by default, matching Codex CLI. An inbound Anthropic `thinking` block (driven by the bot's `effort`) overrides this per-request. |
OAuth refresh endpoint configuration belongs to the app credential owner, not this adapter.

**Retry hardening (TASK-714, shipped 2026-08-10, `67ff5c1`):** the proxy now has a bounded retry state machine (`proxy/retry.py`) keyed to what has already been forwarded downstream — retries freely before visible output and during thinking, fails upward honestly after visible text, NEVER retries after a forwarded `tool_use` (hard invariant). Splice hygiene keeps retried streams wire-valid; in-band `response.failed`/`response.error` events route through the same classifier. Tests: `tests/test_proxy_retry.py` (54 hermetic).

## Summary formerly in the parent SKILL.md (TASK-1640)

## Anthropic-compat proxy (TASK-270/714)

The claude-code bridge hosts an in-process FastAPI proxy (`127.0.0.1`, ephemeral port, injected as `ANTHROPIC_BASE_URL`) that translates Anthropic `/v1/messages` ↔ OpenAI Responses API, so non-Anthropic providers (ChatGPT-subscription OAuth, Grok, z.ai, Kimi, …) ride the Claude Agent SDK. Model string `provider/upstream` (e.g. `openai_chatgpt/gpt-5.4`) in a `type: claude-code` catalog row selects the adapter. New Responses-API provider = subclass `ProviderAdapter`, override `authorize()`, register — the base `call()` handles translation/streaming/retries. Bounded retry state machine (TASK-714) never replays after a forwarded tool call.

**→ `reference/anthropic-proxy.md`** before touching `proxy/`, adding an adapter, flipping a bot to the proxy path, or debugging ChatGPT-OAuth 401s/400s (ChatGPT backend URL not platform API, `chatgpt-account-id` header, stripped params, and app-owned token broker). The proxy does not read/write host auth.json or refresh OAuth; probe the actual bridge credential path, not a retired host login.

Cache caveat: for ChatGPT-proxy latency, inspect the bridge's per-request `Responses usage` log (cache-hit %) before blaming context size. Caching needs an exact prefix; >~15 req/min per prefix/key can overflow to other machines and reduce hits.
