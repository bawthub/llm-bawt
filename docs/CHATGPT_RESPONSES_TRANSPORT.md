# ChatGPT Responses transport (TASK-864, TASK-872, TASK-896)

## Scope

ChatGPT Responses transport is endpoint policy owned by
`model_endpoints.serving_config.responses_transport`, not inferred from a model
name. Supported values are:

- `sse` — ordinary supervised Responses HTTP streaming (the safe default);
- `lite_ws` — the dedicated WebSocket client with Responses Lite payloads.

The selected scalar follows the resolved endpoint through app bot config, the
Redis `chat.send` command, and the Claude CLI custom-header channel into the
proxy request context. The ChatGPT adapter resolves it once and uses that same
policy for both stream selection and retry budget. Missing or invalid bridge
metadata falls back to `sse`; catalog writes reject invalid values.

Endpoint `gpt-6-astra@openai-oauth` is configured as `sse`. This enables native
parallel tool calls while retaining the ordinary Responses stream supervisor.
No proxy code branches on `gpt-6-astra` or any other model identity.

TASK-884: transport choice no longer controls liveness protection. Ordinary
Responses HTTP streams (including Sol and API-key Responses adapters using the
shared base) use the same first-event and productive-idle deadlines below.
The first-event budget includes waiting for HTTP headers. They keep their
existing non-Lite payloads and reconnect once over a fresh ordinary SSE stream
only before output commitment. Underlying SDK automatic retries are disabled so
the proxy owns the budget; HTTP/auth errors retain their existing classification
and broker-refresh path. No new WebSocket routing or provider configuration is
introduced. Passthrough adapters with their own `call()` remain unchanged.
Cancellation telemetry records status 499 rather than falsely reporting 200.

Protocol reference: public `openai/codex`, tag `rust-v0.153.4`:
`codex-rs/core/src/client.rs`, `codex-api/src/common.rs`,
`codex-api/src/endpoint/responses_websocket.rs`, and
`codex-api/src/sse/responses.rs`.

- Upgrade `/backend-api/codex/responses` using the existing broker bearer and
  account/session headers; beta `responses_websockets=2026-02-06`.
- Send flattened `response.create`, with Lite enabled by
  `x-openai-internal-codex-responses-lite: true` and corresponding request metadata.
- In `lite_ws` mode, move tools to an `additional_tools` developer input item
  and instructions to a developer message. Stable IDs derive from conversation
  and visible content. Lite requires parallel tool calls to be disabled and also
  strips image detail fields.
- Both transports use `reasoning.context=all_turns` and request
  `reasoning.encrypted_content`, preserving reasoning continuity under
  `store:false`. Existing explicit reasoning settings win; fallback effort comes
  from the selected endpoint's `serving_config.reasoning_effort`, with a
  model-agnostic high safety default when absent.
- Ordinary `sse` does not force `parallel_tool_calls=false`; the upstream can
  batch independent calls in one response.
- Capture `x-codex-turn-state` from upgrade/HTTP headers or `response.metadata`;
  replay in reconnect headers and WS create metadata.
- Explicit HTTP 426 upgrade rejection selects SSE with the same Lite payload.
  A successful upgrade that then emits no event within 60 seconds discards the
  socket and makes the one safe outer retry use Lite over SSE. Authentication,
  quota, malformed request and server errors are not hidden by fallback. The
  existing output-aware retry state machine remains authoritative.
- A valid terminal ends consumption immediately. EOF without terminal errors;
  incomplete responses retain the translator's max-token semantics. No transport
  layer silently retries generation. Never replay after forwarded tool calls.

### Productive-progress supervision

TASK-872 separates transport activity from model progress. Non-empty reasoning,
reasoning-summary, text, refusal, and tool-argument deltas are productive. Tool
output-item creation/completion and terminal success/failure events are also
productive. Lifecycle frames (`response.created`, `response.in_progress`),
metadata, pings, empty deltas, and other bookkeeping do not extend the productive
deadline.

Each sampling attempt is bounded by the earliest applicable deadline:

- 60 seconds to the first transport event;
- 240 seconds between transport events;
- 90 seconds without productive model progress.

There is no absolute response-duration cap. Productive text, reasoning, or tool
argument streams may run beyond four minutes; the same idle bounds still apply
if progress stops. This prevents large report-writing calls from being truncated
while their arguments are still arriving. `tests/test_responses_long_progress.py`
simulates ten minutes of productive text, reasoning, and tool arguments on both
WebSocket and ordinary SSE, followed by either completion or a real idle stall.

A stalled attempt before committed output is closed and retried once over a
fresh HTTPS/SSE stream (ordinary SSE reconnects; WebSocket falls back to Lite
SSE). Unfinished thinking summaries may also recover this way: the proxy closes
the abandoned thinking block and resumes with monotonically increasing block
indices inside the same Anthropic message. The upstream request/history and
cache identity remain unchanged; completed tools from earlier model requests
are history, not actions to execute again. Before that retry, the proxy emits a
structured `upstream_status` event (`reconnecting`, transport, fallback,
attempt, stall phase, elapsed time, and productive-idle time). The first
non-failure productive event on the fallback emits `recovered`. The bridge
publishes these onto the active Redis run; the app forwards them on unified SSE;
the frontend renders the reconnect state inside the in-flight assistant bubble.
Status text is transient and never enters persisted assistant content.

Completed native reasoning (a persisted replayable signature), visible answer
text, and tool dispatch still make replay unsafe. A stall after any of those
fails upward as `api_error` without retry. Merely streaming an unsigned thinking
summary is **not** that commitment barrier (TASK-950). The reasoning-summary
idle limit remains 90 seconds: it detects absence of observable progress, not
proof that upstream computation has stopped.

The shared retry policy owns the supervised limit: at most two total attempts,
respecting any smaller adapter budget. A second stall fails as `api_error`
without handing the request back to the Claude CLI for another replay. A stalled
request is not restarted as a new agent turn. In every final stalled path, the
unfinished routing lease and underlying connection are discarded rather than
returned to the pool. Structured transport and retry logs record the stall
phase, timings, attempt, selected transport, fallback, decision, and final
disposition. Cancellation still propagates without retry.

`tests/test_proxy_progress_recovery.py` reproduces Snark's thinking-only stall
on ordinary SSE and WS-to-SSE fallback, checks recovery through the real
Anthropic streaming parser, preserves tool-result history and native reasoning
scope, and covers exhaustion, cancellation, and every commitment barrier.
These are hermetic regressions; activation still requires an explicitly
approved Claude bridge restart.

## Ownership and bounds

The adapter owns the client and closes it at proxy shutdown. Exclusive leases
are keyed by endpoint, account, hashed bearer, durable conversation, bridge
request/turn, and model. Parallel requests receive separate sockets; anonymous
calls cannot reuse a lease. Token rotation cannot reuse an old authenticated socket.

At most 32 active leases and 32 idle entries are retained. Idle entries expire
after 60 seconds; connect/pool waits are bounded at 15 seconds and sends at 60.
The first event after `response.create` is bounded at 60 seconds; later event
inactivity is bounded at 240 seconds, with the productive-idle bound
above applied independently. Astra gets one safe proxy retry. A fully silent
WebSocket plus fallback is bounded near 120 seconds; lifecycle chatter plus
fallback is bounded near 180 seconds, both below the bridge's 300-second SDK
watchdog and the app's 600-second inactivity watchdog. Cancellation and
incomplete/error responses close the socket; scoped sticky routing can survive
only long enough to select the safe SSE retry.

Full history is sent each hop. `previous_response_id`/delta optimization is
intentionally not used: Claude SDK histories can branch and omit upstream
reasoning/output items. No cross-turn warm pool, Codex tool namespaces,
attestation, or claim of complete Codex runtime parity.

## Verification receipts (2026-09-07 ET)

- 147 hermetic tests passed across the new transport suite and existing retry,
  concurrency, Claude proxy, and passthrough suites. Scoped Ruff and diff checks
  passed.
- Isolated live raw transport: Astra produced an `echo_check(value=ping)` tool
  call at 5.40s; on the reused WS connection the supplied tool result produced
  `pong` at 8.24s total. Both hops emitted `response.completed`.
- Isolated live full adapter: Anthropic request -> WS/Lite -> Anthropic tool_use
  at 3.20s; supplied tool_result -> `pong` / end_turn at 4.95s total. Usage was
  captured for both hops. No real tool execution or bot session was involved.
- Raw local smoke logs: `/tmp/task864-smoke.log` and
  `/tmp/task864-adapter-smoke.log`. Durable summary lives in TASK-864.

Nick activated the client by restarting the Claude bridge at
2026-09-08T00:54:52Z. Normal Loopy chat confirmed WebSocket transport, connection
reuse on the next tool hop, and 98.5% prompt-cache hits on that follow-up.

A live Snark turn later exposed a missing bound: request
`req_f8987b7a0f684e1a9189e0be5c0deae0` completed many WS/Lite tool hops, then a
reused `response.create` at 2026-09-08T01:34:58Z emitted zero events. The generic
three-attempt policy could spend 900 seconds on 300-second waits, so the bridge's
600-second watchdog cancelled it first. The follow-up mitigation adds the
60-second first-event bound, Lite SSE recovery, Astra's one-retry limit, terminal
outer-CLI handling at exhaustion, and explicit timeout telemetry. Verification:
173 proxy/bridge regressions passed; isolated live adapter tool round-trip reached
`echo_check` at 5.42s and `pong` at 9.47s. This proves the recovery behavior
hermetically and preserves live protocol compatibility; long-running reliability
still requires observation after bridge activation. Pre-existing orphaned UI turns
remain outside this transport scope.

## TASK-872 verification and activation

Hermetic coverage drives silent and endlessly chattering WebSocket/SSE streams,
successful fallback recovery,
fallback exhaustion, and stalls after reasoning/text/tool commitment. Bridge and
frontend tests cover the structured status path and its transient bubble mapping.
The exact command receipts live in TASK-872.

No services were restarted while implementing TASK-872. The source-mounted
Claude bridge must reload to activate the backend supervisor for normal chat.
Only restart
`claude-code-bridge` with Nick's explicit approval for that run; do not restart
other services. The frontend changes are compatible with older event producers
and unknown event consumers ignore the additive event kind. No dependency/image
rebuild is required. Rollback is a source revert of the TASK-872 changes plus the
same approved bridge reload; model profiles and credentials are unchanged.
